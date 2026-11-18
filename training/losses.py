"""Multi-view evidential objective.

For target class y, expected categorical NLL under Dir(alpha) is
psi(S)-psi(alpha_y). KL regularization is against the uniform Dirichlet after
removing target-class evidence (alpha_tilde_y=1), discouraging unsupported
confidence. A linear annealing coefficient warms the KL term from zero.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

def dirichlet_kl_uniform(alpha: Tensor) -> Tensor:
    k=alpha.shape[-1]; s=alpha.sum(-1)
    log_b = torch.lgamma(s)-torch.lgamma(alpha).sum(-1)
    log_b_uniform = -torch.lgamma(torch.tensor(float(k),device=alpha.device,dtype=alpha.dtype))
    return log_b-log_b_uniform+((alpha-1)*(torch.digamma(alpha)-torch.digamma(s.unsqueeze(-1)))).sum(-1)

def evidential_loss(alpha: Tensor, target: Tensor, kl_weight: float=0., class_weights: Tensor|None=None) -> Tensor:
    """Mean expected cross entropy plus KL-to-uniform regularizer."""
    if alpha.ndim!=2 or target.ndim!=1 or alpha.shape[0]!=target.shape[0]: raise ValueError("Expected alpha [B,K], target [B]")
    if not torch.isfinite(alpha).all() or (alpha<=0).any(): raise ValueError("alpha must be finite and positive")
    if (target<0).any() or (target>=alpha.shape[1]).any(): raise ValueError("Target class out of range")
    s=alpha.sum(-1)
    expected_ce=torch.digamma(s)-torch.digamma(alpha.gather(1,target[:,None]).squeeze(1))
    adjusted=alpha.clone(); adjusted.scatter_(1,target[:,None],1.)
    per_sample=expected_ce+float(kl_weight)*dirichlet_kl_uniform(adjusted)
    if class_weights is not None:
        if class_weights.numel()!=alpha.shape[1]: raise ValueError("class_weights length must equal class count")
        weights=class_weights.to(device=alpha.device,dtype=alpha.dtype)[target]
        return (per_sample*weights).sum()/weights.sum().clamp_min(1e-12)
    return per_sample.mean()

@dataclass
class LossWeights:
    audio: float=1.
    visual: float=1.
    pseudo: float=1.
    fused: float=1.

def annealed_kl(epoch:int, warmup_epochs:int, maximum:float)->float:
    """Linear epoch-level KL ramp, clamped to [0, maximum]."""
    if warmup_epochs<=0:return float(maximum)
    return float(maximum)*min(1.,max(0.,epoch/warmup_epochs))

def multimodal_loss(output, batch:dict, weights:LossWeights, kl_weight:float,
                    class_weights:dict[str,Tensor]|None=None)->tuple[Tensor,dict[str,Tensor]]:
    """Compute supervised losses only for available outputs/labels."""
    specs=(("audio_alpha","audio_label","audio"),("visual_alpha","visual_label","visual"),
           ("pseudo_alpha","overall_label","pseudo"),("fused_alpha","overall_label","fused"))
    values={}; total=None
    for output_key,target_key,branch in specs:
        alpha=getattr(output,output_key); weight=getattr(weights,branch)
        if alpha is None or target_key not in batch or weight==0: continue
        loss=evidential_loss(alpha,batch[target_key],kl_weight,(class_weights or {}).get(branch))
        values[branch]=loss; total=loss*weight if total is None else total+loss*weight
    if total is None: raise ValueError("No supervised branch loss could be computed")
    values["total"]=total
    return total,values
