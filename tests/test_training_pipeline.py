import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.etmc import ETMC
from training.trainer import Trainer
from training.checkpoint import load_checkpoint

class AudioIdentity(nn.Module):
    def forward(self,x): return x
class VisualIdentity(nn.Module):
    def forward(self,x): return x
class Tiny(Dataset):
    def __len__(self): return 4
    def __getitem__(self,i):
        label=i%2; a=torch.zeros(1024); v=torch.zeros(2560)
        a[0]=v[0]=(-1. if label==0 else 1.)
        return {"audio":a,"frames":v,"audio_label":label,"visual_label":label,"overall_label":label}

def test_one_epoch_and_checkpoint_roundtrip(tmp_path):
    model=ETMC(audio_encoder=AudioIdentity(),visual_encoder=VisualIdentity(),pretrained_visual=False)
    cfg={"training":{"epochs":1,"learning_rate":1e-3,"weight_decay":0.,"mixed_precision":False,
      "gradient_clip_norm":1.,"evidential_kl_weight":.01,"kl_warmup_epochs":1,
      "loss_weights":{"audio":1.,"visual":1.,"pseudo":1.,"fused":1.},"class_weights":{},
      "early_stopping_patience":2,"num_workers":0},"project":{"seed":9}}
    loader=DataLoader(Tiny(),batch_size=2)
    history=Trainer(model,loader,loader,cfg,tmp_path,device="cpu").fit()
    assert len(history)==1 and (tmp_path/"best.pt").is_file() and (tmp_path/"latest.pt").is_file()
    restored=ETMC(audio_encoder=AudioIdentity(),visual_encoder=VisualIdentity(),pretrained_visual=False)
    state=load_checkpoint(tmp_path/"best.pt",restored)
    x=Tiny()[0]; expected=model(x["audio"].unsqueeze(0),x["frames"].unsqueeze(0)).fused_probabilities
    actual=restored(x["audio"].unsqueeze(0),x["frames"].unsqueeze(0)).fused_probabilities
    assert state["epoch"]==1 and torch.allclose(expected,actual)
