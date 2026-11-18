"""ETMC-inspired multimodal evidential model (audio, visual, pseudo-view)."""
from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor, nn
from .rawnet import RawNet2Encoder
from .visual_encoder import EfficientNetB7Encoder
from .evidential import EvidentialHead, expected_probabilities
from .fusion import fuse_alpha

@dataclass
class ETMCOutput:
    """Stable named outputs. Missing modality values are None in modality-only runs."""
    audio_alpha: Tensor | None
    visual_alpha: Tensor | None
    pseudo_alpha: Tensor | None
    intermediate_alpha: Tensor | None
    fused_alpha: Tensor | None
    audio_probabilities: Tensor | None
    visual_probabilities: Tensor | None
    pseudo_probabilities: Tensor | None
    intermediate_probabilities: Tensor | None
    fused_probabilities: Tensor | None
    conflicts: dict[str, Tensor]

class ETMC(nn.Module):
    """RawNet2 + EfficientNet-B7 with three evidential views and DS fusion."""
    def __init__(self, pretrained_visual: bool = True, freeze_visual: bool = True,
                 rawnet_checkpoint: str | None = None, visual_checkpoint: str | None = None,
                 prior: float = 1.0, audio_encoder: nn.Module | None = None,
                 visual_encoder: nn.Module | None = None):
        super().__init__()
        self.audio_encoder=audio_encoder or RawNet2Encoder(checkpoint=rawnet_checkpoint)
        self.visual_encoder=visual_encoder or EfficientNetB7Encoder(pretrained_visual,visual_checkpoint,freeze_visual)
        self.audio_projection=nn.Linear(1024,1024)
        self.visual_projection=nn.Linear(2560,1024)
        self.audio_head=EvidentialHead(1024,2,prior)
        self.visual_head=EvidentialHead(1024,2,prior)
        self.pseudo_head=EvidentialHead(2048,2,prior)
    @staticmethod
    def _check_batch(a:Tensor|None,v:Tensor|None)->None:
        if a is not None and v is not None and a.shape[0]!=v.shape[0]: raise ValueError("Audio and visual batch sizes differ")
    def forward(self,audio:Tensor|None=None,video:Tensor|None=None)->ETMCOutput:
        if audio is None and video is None: raise ValueError("Provide audio, video, or both")
        self._check_batch(audio,video)
        aa=va=pa=ia=fa=None
        ap=vp=pp=ip=fp=None
        conflicts={}
        af=vf=None
        if audio is not None:
            af=self.audio_encoder(audio)
            if af.ndim!=2 or af.shape[1]!=1024: raise ValueError(f"Audio encoder must return [B,1024], got {tuple(af.shape)}")
            _,aa,ap=self.audio_head(self.audio_projection(af))
        if video is not None:
            vf=self.visual_encoder(video)
            if vf.ndim!=2 or vf.shape[1]!=2560: raise ValueError(f"Visual encoder must return [B,2560], got {tuple(vf.shape)}")
            _,va,vp=self.visual_head(self.visual_projection(vf))
        if af is not None and vf is not None:
            _,pa,pp=self.pseudo_head(__import__('torch').cat((self.audio_projection(af),self.visual_projection(vf)),dim=1))
            ia,conflicts["audio_visual"]=fuse_alpha(aa,va)
            fa,conflicts["intermediate_pseudo"]=fuse_alpha(ia,pa)
            ip,fp=expected_probabilities(ia),expected_probabilities(fa)
        elif aa is not None: fa,fp=aa,ap
        else: fa,fp=va,vp
        return ETMCOutput(aa,va,pa,ia,fa,ap,vp,pp,ip,fp,conflicts)
