"""Train ETMC from configured train/validation manifests; test data is never loaded."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset.video_dataset import FakeAVCelebDataset
from models.etmc import ETMC
from training.trainer import Trainer, seed_everything

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="config.yaml"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); seed=int(cfg["project"].get("seed",42))
    seed_everything(seed,bool(cfg["project"].get("deterministic",False)))
    data=cfg["data"]; train_path=data.get("train_manifest"); val_path=data.get("validation_manifest")
    if not train_path or not val_path: raise SystemExit("Configure data.train_manifest and data.validation_manifest first. Test manifest is not used during training.")
    train_df=pd.read_csv(train_path); val_df=pd.read_csv(val_path)
    if "group_id" in train_df and "group_id" in val_df and set(train_df.group_id.astype(str)) & set(val_df.group_id.astype(str)):
        raise SystemExit("Identity/source group leakage between train and validation manifests")
    dc=cfg["data"]; tr=cfg["training"]
    train_ds=FakeAVCelebDataset(train_df,dc.get("video_num_frames",8),dc.get("image_size",256),dc.get("audio_sample_rate",16000),dc.get("audio_num_samples",64000)/dc.get("audio_sample_rate",16000))
    val_ds=FakeAVCelebDataset(val_df,dc.get("video_num_frames",8),dc.get("image_size",256),dc.get("audio_sample_rate",16000),dc.get("audio_num_samples",64000)/dc.get("audio_sample_rate",16000))
    strategy=tr.get("imbalance",{}).get("strategy","class_weights")
    sampler=None
    if strategy=="balanced_sampler":
        labels=train_df.overall_label.astype(int).to_numpy(); counts=np.bincount(labels,minlength=2)
        sample_weights=1/np.maximum(counts[labels],1)
        sampler=WeightedRandomSampler(torch.as_tensor(sample_weights,dtype=torch.double),len(sample_weights),replacement=True,
                                       generator=torch.Generator().manual_seed(seed))
    elif strategy not in ("class_weights","none"): raise SystemExit(f"Unknown imbalance strategy {strategy}")
    device=cfg["project"].get("device","auto"); device=None if device=="auto" else device
    model_cfg=cfg["model"]
    model=ETMC(pretrained_visual=bool(model_cfg["visual"].get("pretrained",True)),freeze_visual=True,
       rawnet_checkpoint=model_cfg["audio"].get("checkpoint"),visual_checkpoint=model_cfg["visual"].get("checkpoint"),prior=float(model_cfg["heads"].get("dirichlet_prior",1.)))
    # Stage 1 trains projections/evidential heads only.
    for p in model.audio_encoder.parameters():p.requires_grad=False
    model.visual_encoder.freeze()
    workers=int(tr.get("num_workers",0)); batch=int(tr.get("batch_size",8))
    train_loader=DataLoader(train_ds,batch_size=batch,shuffle=sampler is None,sampler=sampler,num_workers=workers,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available())
    output=Path(cfg["project"].get("output_dir","artifacts"))/"training"
    Trainer(model,train_loader,val_loader,cfg,output,device).fit()

if __name__=="__main__":main()
