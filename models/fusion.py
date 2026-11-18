"""Subjective-logic opinions and normalized Dempster-Shafer combination."""
from __future__ import annotations
from torch import Tensor
import torch

def alpha_to_opinion(alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Map Dirichlet alpha to singleton belief masses and ignorance u=K/S."""
    if alpha.ndim < 1 or (alpha <= 0).any() or not torch.isfinite(alpha).all():
        raise ValueError("alpha must be finite and strictly positive")
    k = alpha.shape[-1]
    strength = alpha.sum(-1, keepdim=True)
    belief = (alpha - 1).clamp_min(0) / strength
    uncertainty = (k / strength).clamp(0, 1)
    # Enforce mass conservation against floating point drift.
    total = belief.sum(-1, keepdim=True) + uncertainty
    belief, uncertainty = belief / total.clamp_min(torch.finfo(alpha.dtype).tiny), uncertainty / total.clamp_min(torch.finfo(alpha.dtype).tiny)
    return belief, uncertainty

def combine_opinions(b1: Tensor, u1: Tensor, b2: Tensor, u2: Tensor,
                     conflict_epsilon: float = 1e-7) -> tuple[Tensor, Tensor, Tensor]:
    """Combine singleton opinions. K=sum_{i!=j} b1_i*b2_j.

    Yager-style numerical fallback: when 1-K is below epsilon, return vacuous
    ignorance (uniformly no committed belief) rather than amplify conflict.
    Otherwise apply normalized D-S mass equations and clamp roundoff.
    """
    if b1.shape != b2.shape or b1.shape[-1] < 2: raise ValueError("Belief tensors must have matching [...,K] shapes")
    if u1.shape != b1.shape[:-1] + (1,) or u2.shape != u1.shape: raise ValueError("Uncertainty shape must be [...,1]")
    if conflict_epsilon <= 0: raise ValueError("conflict_epsilon must be positive")
    if not all(torch.isfinite(x).all() for x in (b1,u1,b2,u2)): raise ValueError("Opinions must be finite")
    conflict = (b1.sum(-1, keepdim=True) * b2.sum(-1, keepdim=True) - (b1*b2).sum(-1, keepdim=True)).clamp(0, 1)
    denom = 1 - conflict
    raw = (b1*b2 + b1*u2 + b2*u1) / denom.clamp_min(conflict_epsilon)
    uncertainty = (u1*u2) / denom.clamp_min(conflict_epsilon)
    certain = denom <= conflict_epsilon
    belief = torch.where(certain, torch.zeros_like(raw), raw.clamp_min(0))
    uncertainty = torch.where(certain, torch.ones_like(uncertainty), uncertainty.clamp(0, 1))
    mass = belief.sum(-1, keepdim=True) + uncertainty
    belief, uncertainty = belief / mass.clamp_min(conflict_epsilon), uncertainty / mass.clamp_min(conflict_epsilon)
    return belief, uncertainty, conflict

def opinion_to_alpha(belief: Tensor, uncertainty: Tensor, prior: float = 1.) -> Tensor:
    """Recover Dirichlet alpha from opinion, retaining vacuous prior at u=1."""
    if prior <= 0: raise ValueError("prior must be positive")
    k = belief.shape[-1]
    strength = k / uncertainty.clamp_min(torch.finfo(belief.dtype).tiny)
    evidence = belief * strength
    return (evidence + prior).clamp_min(prior)

def fuse_alpha(alpha1: Tensor, alpha2: Tensor, conflict_epsilon: float = 1e-7) -> tuple[Tensor, Tensor]:
    """Combine alpha opinions; return fused alpha and conflict coefficient."""
    b1,u1=alpha_to_opinion(alpha1); b2,u2=alpha_to_opinion(alpha2)
    belief,u,conflict=combine_opinions(b1,u1,b2,u2,conflict_epsilon)
    return opinion_to_alpha(belief,u), conflict
