"""Reproducible staged trainer using explicitly supplied train/validation loaders."""
from __future__ import annotations
import csv, json, logging, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .checkpoint import save_checkpoint
from .losses import LossWeights, annealed_kl, multimodal_loss
from .metrics import classification_metrics, log_diagnostics

def seed_everything(seed:int,deterministic:bool=False)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=deterministic; torch.backends.cudnn.benchmark=not deterministic

class Trainer:
    """Train against train/validation loaders only. Test data is never accepted."""
    def __init__(self,model:nn.Module,train_loader:DataLoader,validation_loader:DataLoader,
                 config:dict,output_dir:str|Path,device:str|None=None):
        self.model=model; self.train_loader=train_loader; self.validation_loader=validation_loader
        self.cfg=config; self.train_cfg=config["training"]; self.out=Path(output_dir); self.out.mkdir(parents=True,exist_ok=True)
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model.to(self.device)
        self.logger=logging.getLogger(f"etmc.train.{id(self)}")
        self.logger.setLevel(logging.INFO); self.logger.propagate=False
        fmt=logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh=logging.FileHandler(self.out/"training.log",encoding="utf-8"); fh.setFormatter(fmt); self.logger.addHandler(fh)
        sh=logging.StreamHandler(); sh.setFormatter(fmt); self.logger.addHandler(sh)
        self.optimizer=self._optimizer()
        self.unfrozen={id(p) for group in self.optimizer.param_groups for p in group["params"]}
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode="max",patience=int(self.train_cfg.get("lr_patience",2)),factor=float(self.train_cfg.get("lr_factor",.5)))
        self.amp=bool(self.train_cfg.get("mixed_precision",True) and self.device.type=="cuda")
        self.scaler=torch.amp.GradScaler("cuda",enabled=self.amp)
        self.weights=LossWeights(**self.train_cfg.get("loss_weights",{}))
        self.class_weights=self.train_cfg.get("class_weights",config.get("class_weights",{}))
        self.unfreeze_schedule={int(k):v for k,v in self.train_cfg.get("unfreeze_schedule",{}).items()}
        self.unfrozen=set()

    def _optimizer(self):
        groups=[]
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                lr=float(self.train_cfg["learning_rate"])
                if name.startswith("audio_encoder."): lr=float(self.train_cfg.get("audio_encoder_lr",lr*.1))
                if name.startswith("visual_encoder."): lr=float(self.train_cfg.get("visual_encoder_lr",lr*.1))
                groups.append({"params":[p],"lr":lr})
        return torch.optim.AdamW(groups,lr=float(self.train_cfg["learning_rate"]),weight_decay=float(self.train_cfg["weight_decay"]))

    def _unfreeze(self,epoch):
        setting=self.unfreeze_schedule.get(epoch)
        if not setting:return
        if not isinstance(setting,dict): setting={"visual_last_blocks":int(setting)}
        enc=getattr(self.model,"visual_encoder",None)
        if enc is not None and setting.get("visual_last_blocks"):
            enc.unfreeze_last(int(setting["visual_last_blocks"]))
        if setting.get("audio_encoder"):
            for p in self.model.audio_encoder.parameters():p.requires_grad=True
        new=[p for p in self.model.parameters() if p.requires_grad and id(p) not in self.unfrozen]
        if new:
            self.optimizer.add_param_group({"params":new,"lr":float(self.train_cfg.get("encoder_finetune_lr",self.train_cfg["learning_rate"]*.1))})
            self.unfrozen.update(map(id,new))
            self.logger.info("Epoch %d: added %d encoder parameters to optimizer",epoch,len(new))

    def _move(self,batch):
        return {k:(v.to(self.device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

    def _epoch(self,loader,epoch,training):
        self.model.train(training); total=0.; steps=0; losses={}; targets={k:[] for k in ("audio","visual","pseudo","fused")}; probs={k:[] for k in targets}
        kl=annealed_kl(epoch,int(self.train_cfg.get("kl_warmup_epochs",10)),float(self.train_cfg.get("evidential_kl_weight",.1)))
        for raw in loader:
            batch=self._move(raw)
            if training:self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training),torch.autocast(device_type=self.device.type,enabled=self.amp):
                output=self.model(batch.get("audio"),batch.get("frames",batch.get("video")))
                loss, parts=multimodal_loss(output,batch,self.weights,kl,self._class_weight_tensors())
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            if training:
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),float(self.train_cfg.get("gradient_clip_norm",1.)))
                self.scaler.step(self.optimizer); self.scaler.update()
            total+=float(loss.detach()); steps+=1
            for name,val in parts.items():losses[name]=losses.get(name,0.)+float(val.detach())
            for branch,prob,target in (("audio",output.audio_probabilities,"audio_label"),("visual",output.visual_probabilities,"visual_label"),
                 ("pseudo",output.pseudo_probabilities,"overall_label"),("fused",output.fused_probabilities,"overall_label")):
                if prob is not None and target in batch:
                    if not torch.isfinite(prob).all():raise FloatingPointError(f"Nonfinite {branch} probabilities")
                    probs[branch].append(prob.detach().float().cpu().numpy());targets[branch].append(batch[target].detach().cpu().numpy())
        if not steps:raise ValueError("DataLoader is empty")
        metrics={"loss":total/steps,"kl_weight":kl}
        metrics.update({f"loss_{k}":v/steps for k,v in losses.items()})
        for branch in probs:
            if probs[branch]:
                bm=classification_metrics(np.concatenate(targets[branch]),np.concatenate(probs[branch]))
                for key,val in bm.items():metrics[f"{branch}_{key}"]=val
                log_diagnostics(bm,branch,self.logger.warning if "WARNING" in "" else self.logger.info)
        return metrics

    def _class_weight_tensors(self):
        result={}
        for branch,values in self.class_weights.items():result[branch]=torch.as_tensor(values,dtype=torch.float32,device=self.device)
        return result

    def fit(self):
        """Train epochs, choose best strictly by validation fused macro-F1."""
        best=-float("inf"); patience=0; history=[]; epochs=int(self.train_cfg["epochs"])
        (self.out/"config_snapshot.yaml").write_text(__import__('yaml').safe_dump(self.cfg,sort_keys=False),encoding="utf-8")
        for epoch in range(1,epochs+1):
            self._unfreeze(epoch)
            tr=self._epoch(self.train_loader,epoch,True); val=self._epoch(self.validation_loader,epoch,False)
            row={"epoch":epoch,**{f"train_{k}":v for k,v in tr.items()},**{f"validation_{k}":v for k,v in val.items()}}
            history.append(row); self.scheduler.step(val.get("fused_f1",val.get("fused_balanced_accuracy",0.)))
            score=val.get("fused_f1",-float("inf"))
            save_checkpoint(self.out/"latest.pt",self.model,self.optimizer,self.scheduler,epoch,best,self.cfg)
            if score>best:
                best=score; patience=0
                save_checkpoint(self.out/"best.pt",self.model,self.optimizer,self.scheduler,epoch,best,self.cfg)
            else:patience+=1
            with (self.out/"history.csv").open("w",newline="",encoding="utf-8") as f:
                keys=sorted({k for r in history for k in r}); writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(history)
            self.logger.info("Epoch %d/%d train_loss=%.4f val_loss=%.4f val_fused_f1=%.4f",epoch,epochs,tr["loss"],val["loss"],score)
            if patience>=int(self.train_cfg.get("early_stopping_patience",8)):
                self.logger.info("Early stopping at epoch %d",epoch);break
        self.close()
        return history

    def close(self):
        """Flush and close run-owned logger handlers (important on Windows)."""
        for handler in list(self.logger.handlers):
            handler.flush(); handler.close(); self.logger.removeHandler(handler)
