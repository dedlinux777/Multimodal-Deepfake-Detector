"""Checkpoint-backed end-to-end smoke test against a sample MP4.

Set ETMC_SMOKE_CHECKPOINT and ETMC_SMOKE_VIDEO to run with a trained artifact
and representative sample. The test skips cleanly when artifacts are absent.
"""
import os
from pathlib import Path
import pytest
from inference.predict import VideoPredictor

checkpoint=os.getenv("ETMC_SMOKE_CHECKPOINT")
video=os.getenv("ETMC_SMOKE_VIDEO")
@pytest.mark.skipif(not checkpoint or not video,reason="Set ETMC_SMOKE_CHECKPOINT and ETMC_SMOKE_VIDEO")
def test_sample_mp4_end_to_end():
    result=VideoPredictor(checkpoint).predict(video)
    assert result["prediction"] in {"REAL","FAKE"}
    assert 0<=result["fused_fake_probability"]<=1
    assert set(result["audio"]) >= {"prediction","fake_probability","evidence","uncertainty"}
    assert set(result["visual"]) >= {"prediction","fake_probability","evidence","uncertainty"}
    assert set(result["pseudo_view"]) >= {"prediction","fake_probability","evidence","uncertainty"}
