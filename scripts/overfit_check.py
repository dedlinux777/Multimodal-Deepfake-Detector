"""Tiny synthetic memorization check for model/loss wiring (no dataset/training run)."""
from __future__ import annotations
import torch
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.etmc import ETMC
from training.losses import LossWeights,multimodal_loss

class AudioFeatures(nn.Module):
    def forward(self,x):return x
class VideoFeatures(nn.Module):
    def forward(self,x):return x

def main():
    torch.manual_seed(17)
    model=ETMC(audio_encoder=AudioFeatures(),visual_encoder=VideoFeatures(),pretrained_visual=False)
    # Tiny, separable synthetic representations test all four supervision paths.
    y=torch.tensor([0,1,0,1]); audio=torch.randn(4,1024)*.05; visual=torch.randn(4,2560)*.05
    audio[:,0]=torch.where(y==0,-2.,2.); visual[:,0]=torch.where(y==0,-2.,2.)
    batch={"audio_label":y,"visual_label":y,"overall_label":y}
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    with torch.no_grad(): initial=multimodal_loss(model(audio,visual),batch,LossWeights(),0.)[0].item()
    for _ in range(60):
        opt.zero_grad(); loss,_=multimodal_loss(model(audio,visual),batch,LossWeights(),0.); loss.backward();opt.step()
    final,_=multimodal_loss(model(audio,visual),batch,LossWeights(),0.)
    assert torch.isfinite(final) and final.item()<initial, f"Overfit check did not reduce loss: {initial} -> {final.item()}"
    print(f"Tiny synthetic overfit check passed: loss {initial:.4f} -> {final.item():.4f}")
if __name__=="__main__":main()
