import torch
from torch import nn
import pytest

from models.evidential import alpha_from_evidence, expected_probabilities
from models.fusion import alpha_to_opinion, combine_opinions, fuse_alpha
from models.etmc import ETMC


def test_dirichlet_parameters_valid():
    alpha=alpha_from_evidence(torch.tensor([[0., 2.]]))
    assert torch.equal(alpha,torch.tensor([[1.,3.]]))
    assert torch.allclose(expected_probabilities(alpha),torch.tensor([[.25,.75]]))

def test_equal_evidence_fusion():
    alpha=torch.tensor([[3.,3.]])
    result, conflict=fuse_alpha(alpha,alpha)
    assert torch.allclose(expected_probabilities(result),torch.tensor([[.5,.5]]),atol=1e-6)
    assert torch.isfinite(result).all() and conflict.item() >= 0

def test_zero_evidence_is_vacuous():
    alpha=torch.ones(2,2)
    belief,u=alpha_to_opinion(alpha)
    assert torch.equal(belief,torch.zeros_like(belief))
    assert torch.equal(u,torch.ones_like(u))

def test_high_conflict_is_finite_and_bounded():
    b,u,k=combine_opinions(torch.tensor([[.999,0.]]),torch.tensor([[.001]]),
                           torch.tensor([[0.,.999]]),torch.tensor([[.001]]))
    assert k.item()>.99
    assert torch.isfinite(b).all() and torch.isfinite(u).all()
    assert torch.allclose(b.sum(-1,keepdim=True)+u,torch.ones_like(u),atol=1e-6)

def test_numerical_stability_extreme_alpha():
    a=torch.tensor([[1e-6,1e20]])
    fused,_=fuse_alpha(a,a)
    assert torch.isfinite(fused).all()
    assert torch.allclose(expected_probabilities(fused).sum(-1),torch.ones(1))

class AudioStub(nn.Module):
    def forward(self,x): return x
class VisualStub(nn.Module):
    def forward(self,x): return x

@pytest.fixture
def model():
    return ETMC(audio_encoder=AudioStub(),visual_encoder=VisualStub(),pretrained_visual=False)

def test_full_and_modality_only_forward_shapes(model):
    audio=torch.randn(2,1024); video=torch.randn(2,2560)
    out=model(audio,video)
    for value in (out.audio_alpha,out.visual_alpha,out.pseudo_alpha,out.intermediate_alpha,out.fused_alpha,
                  out.audio_probabilities,out.visual_probabilities,out.pseudo_probabilities,out.fused_probabilities):
        assert value.shape==(2,2)
        assert torch.isfinite(value).all()
    assert model(audio=audio).fused_alpha.shape==(2,2)
    assert model(video=video).fused_alpha.shape==(2,2)
    with pytest.raises(ValueError): model()
