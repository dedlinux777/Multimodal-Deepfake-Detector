"""Dirichlet evidential classification helpers."""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def alpha_from_evidence(evidence: Tensor, prior: float = 1.0) -> Tensor:
    """Convert finite, nonnegative evidence to valid Dirichlet parameters."""
    if evidence.ndim < 1 or not torch.isfinite(evidence).all():
        raise ValueError("Evidence must be a finite tensor with a class dimension")
    if (evidence < 0).any() or prior <= 0:
        raise ValueError("Evidence must be nonnegative and prior must be positive")
    return evidence + prior

def expected_probabilities(alpha: Tensor) -> Tensor:
    """Return E[p]=alpha/sum(alpha), with a safe denominator."""
    if alpha.ndim < 1 or (alpha <= 0).any() or not torch.isfinite(alpha).all():
        raise ValueError("Dirichlet alpha must contain finite positive values")
    return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(alpha.dtype).tiny)

class EvidentialHead(nn.Module):
    """Linear evidence head with Softplus activation and Dirichlet prior."""
    def __init__(self, input_dim: int, num_classes: int = 2, prior: float = 1.0):
        super().__init__()
        if input_dim <= 0 or num_classes < 2 or prior <= 0: raise ValueError("Invalid head dimensions/prior")
        self.linear, self.prior = nn.Linear(input_dim, num_classes), prior
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.linear.in_features:
            raise ValueError(f"Expected [batch,{self.linear.in_features}] features; got {tuple(x.shape)}")
        evidence = F.softplus(self.linear(x))
        alpha = alpha_from_evidence(evidence, self.prior)
        return evidence, alpha, expected_probabilities(alpha)
