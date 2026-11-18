from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset.audio_extraction import AudioExtractionError, extract_audio
from dataset.preprocessing import VideoPreprocessor
from dataset.split import split_manifest
from scripts.validate_dataset import LABELS


@pytest.mark.parametrize("category,labels", LABELS.items())
def test_category_label_mapping(category, labels):
    assert LABELS[category] == labels
    assert labels[2] == int(labels[0] or labels[1])


def test_video_preprocessing_shape(tmp_path: Path):
    import cv2
    path = str(tmp_path / "sample.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    for i in range(5): writer.write(np.full((48, 64, 3), i * 30, np.uint8))
    writer.release()
    out = VideoPreprocessor(num_frames=3, image_size=32)(path)
    assert tuple(out.shape) == (3, 3, 32, 32)


def test_audio_resampling_and_duration(tmp_path: Path):
    import soundfile as sf
    import subprocess
    source = tmp_path / "tone.wav"
    t = np.arange(8000, dtype=np.float32) / 8000
    sf.write(source, np.sin(2 * np.pi * 440 * t), 8000)
    video = tmp_path / "tone.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1",
                    "-i", str(source), "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(video)], check=True)
    result = extract_audio(video, 16000, 1)
    assert result.shape == (16000,)
    assert np.isfinite(result).all()


def test_missing_audio_raises(tmp_path: Path):
    path = tmp_path / "silent.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1",
                    "-c:v", "mpeg4", str(path)], check=True)
    with pytest.raises(AudioExtractionError): extract_audio(path)


def test_group_split_has_no_leakage():
    rows = [{"group_id": f"g{i}", "overall_label": i % 2, "video_path": f"{i}.mp4"} for i in range(40)]
    splits = split_manifest(pd.DataFrame(rows), seed=7)
    groups = [set(x.group_id) for x in splits.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
