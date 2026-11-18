"""Torchvision EfficientNet-B7 pooled feature encoder."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

class EfficientNetB7Encoder(nn.Module):
    """B7 features + global average pooling (2560-D), with staged unfreezing."""
    feature_dim=2560
    def __init__(self, pretrained: bool = True, checkpoint: str | Path | None = None,
                 freeze: bool = True, progress: bool = True):
        super().__init__()
        weights=EfficientNet_B7_Weights.DEFAULT if pretrained else None
        self.network=efficientnet_b7(weights=weights, progress=progress)
        self.network.classifier=nn.Identity()
        self._frozen_blocks=0
        if freeze: self.freeze()
        if checkpoint is not None: self.load_checkpoint(checkpoint)
    def freeze(self)->None:
        for p in self.network.parameters(): p.requires_grad=False
        self._frozen_blocks=len(self.network.features)
    def unfreeze_last(self, num_blocks:int=1)->None:
        """Unfreeze the last N feature stages; progressive calls can grow N."""
        if not 1<=num_blocks<=len(self.network.features): raise ValueError("num_blocks out of range")
        boundary=len(self.network.features)-num_blocks
        for i,block in enumerate(self.network.features):
            for p in block.parameters(): p.requires_grad=(i>=boundary)
        self._frozen_blocks=boundary
    def unfreeze_all(self)->None:
        for p in self.network.features.parameters(): p.requires_grad=True
        self._frozen_blocks=0
    def load_checkpoint(self,path:str|Path)->None:
        payload=torch.load(path,map_location="cpu",weights_only=True)
        state=payload.get("state_dict",payload) if isinstance(payload,dict) else payload
        if not isinstance(state,dict): raise ValueError("Visual checkpoint must contain state dict")
        state={k.removeprefix("module.").removeprefix("network."):v for k,v in state.items()}
        own=self.network.state_dict(); missing=set(own)-set(state); extra=set(state)-set(own)
        bad=[k for k in own.keys()&state.keys() if own[k].shape!=state[k].shape]
        if missing or extra or bad: raise ValueError(f"Incompatible EfficientNet-B7 checkpoint: missing={sorted(missing)}, extra={sorted(extra)}, shape_mismatch={bad}")
        self.network.load_state_dict(state,strict=True)
    def forward(self,images:Tensor)->Tensor:
        if images.ndim==5:
            b,t,c,h,w=images.shape; images=images.reshape(b*t,c,h,w)
            pooled=self.network(images)
            return pooled.reshape(b,t,-1).mean(dim=1)
        if images.ndim!=4: raise ValueError(f"Expected [B,3,H,W] or [B,T,3,H,W], got {tuple(images.shape)}")
        return self.network(images)
