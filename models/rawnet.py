"""RawNet2-style raw-waveform speaker/audio encoder.

Architecture source: Jung et al., "Improved RawNet with Feature Map Scaling for
Text-independent Speaker Verification using Deep Residual Networks", Interspeech
2020 (RawNet2), https://arxiv.org/abs/2004.00578. This implementation follows
the RawNet2 design principles (learned sinc front end, residual 1-D blocks,
GRU temporal aggregation) but is a documented adaptation, not an official
authors' checkpoint-compatible reproduction. No pretrained weights are bundled.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class SincConv1d(nn.Module):
    """Learnable band-pass sinc filter bank inspired by RawNet2 frontends."""
    def __init__(self, out_channels: int = 64, kernel_size: int = 251, sample_rate: int = 16000):
        super().__init__()
        if kernel_size % 2 != 1: raise ValueError("Sinc kernel_size must be odd")
        self.kernel_size, self.sample_rate = kernel_size, sample_rate
        low = torch.linspace(40, sample_rate / 2 - 100, out_channels)
        self.low_hz = nn.Parameter(low)
        self.band_hz = nn.Parameter(torch.full((out_channels,), 100.))
        self.register_buffer("window", torch.hamming_window(kernel_size))
        self.out_channels = out_channels
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2: raise ValueError("RawNet expects waveform [B,samples]")
        low = self.low_hz.abs().clamp(20, self.sample_rate / 2 - 100)
        high = (low + self.band_hz.abs().clamp_min(50)).clamp(max=self.sample_rate / 2 - 1)
        t = torch.arange(-(self.kernel_size//2), self.kernel_size//2+1, device=x.device, dtype=x.dtype) / self.sample_rate
        filters=[]
        for lo, hi in zip(low, high):
            filt = 2*hi*torch.sinc(2*hi*t)-2*lo*torch.sinc(2*lo*t)
            filt = filt * self.window.to(dtype=x.dtype)
            filters.append(filt / filt.abs().sum().clamp_min(1e-6))
        return F.conv1d(x.unsqueeze(1), torch.stack(filters).unsqueeze(1), padding=self.kernel_size//2)

class ResidualBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.body=nn.Sequential(nn.BatchNorm1d(in_ch), nn.LeakyReLU(.3), nn.Conv1d(in_ch,out_ch,3,stride,padding=1,bias=False),
             nn.BatchNorm1d(out_ch), nn.LeakyReLU(.3), nn.Conv1d(out_ch,out_ch,3,padding=1,bias=False))
        self.skip=nn.Conv1d(in_ch,out_ch,1,stride,bias=False) if in_ch!=out_ch or stride!=1 else nn.Identity()
    def forward(self,x:Tensor)->Tensor:return self.body(x)+self.skip(x)

class RawNet2Encoder(nn.Module):
    """Adapted RawNet2 encoder returning a 1024-D waveform representation."""
    feature_dim = 1024
    def __init__(self, sample_rate: int = 16000, checkpoint: str | Path | None = None):
        super().__init__()
        self.frontend=SincConv1d(64,251,sample_rate)
        self.blocks=nn.Sequential(ResidualBlock1d(64,128),ResidualBlock1d(128,128),ResidualBlock1d(128,256),ResidualBlock1d(256,256))
        self.gru=nn.GRU(256,256,batch_first=True,bidirectional=True)
        self.projection=nn.Sequential(nn.Linear(512,1024),nn.LayerNorm(1024))
        if checkpoint is not None: self.load_checkpoint(checkpoint)
    def load_checkpoint(self,path:str|Path)->None:
        """Strictly validate names and shapes before loading a state dict."""
        payload=torch.load(path,map_location="cpu",weights_only=True)
        state=payload.get("state_dict",payload) if isinstance(payload,dict) else payload
        if not isinstance(state,dict): raise ValueError("Checkpoint must be a state dict or contain state_dict")
        state={k.removeprefix("module."):v for k,v in state.items()}
        own=self.state_dict(); missing=set(own)-set(state); unexpected=set(state)-set(own)
        mismatched=[f"{k}: checkpoint {tuple(state[k].shape)} != model {tuple(own[k].shape)}" for k in own.keys()&state.keys() if own[k].shape!=state[k].shape]
        if missing or unexpected or mismatched: raise ValueError(f"Incompatible RawNet checkpoint; missing={sorted(missing)}, unexpected={sorted(unexpected)}, shape_errors={mismatched}")
        self.load_state_dict(state,strict=True)
    def forward(self,waveform:Tensor)->Tensor:
        if waveform.ndim==3 and waveform.shape[1]==1: waveform=waveform[:,0]
        if waveform.ndim!=2 or waveform.shape[1]<512: raise ValueError(f"Expected waveform [B,samples>=512], got {tuple(waveform.shape)}")
        x=F.max_pool1d(torch.abs(self.frontend(waveform)),3,3)
        x=self.blocks(x).transpose(1,2)
        x,_=self.gru(x)
        return self.projection(x.mean(dim=1))
