"""Atomic model checkpoint and configuration/history persistence."""
from __future__ import annotations
import os
from pathlib import Path
import torch

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, config):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict() if scheduler else None,"epoch":epoch,
                "best_metric":best_metric,"config":config},tmp)
    os.replace(tmp,path)

def load_checkpoint(path,model,optimizer=None,scheduler=None,map_location="cpu"):
    state=torch.load(path,map_location=map_location,weights_only=False)
    model.load_state_dict(state["model"],strict=True)
    if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"): scheduler.load_state_dict(state["scheduler"])
    return state
