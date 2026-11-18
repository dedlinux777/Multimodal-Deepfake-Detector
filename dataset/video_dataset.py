"""PyTorch dataset for FakeAVCeleb manifest rows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio_extraction import extract_audio
from .preprocessing import VideoPreprocessor


class FakeAVCelebDataset(Dataset[dict[str, Any]]):
    """Load paired video/audio and three labels from a CSV/Parquet manifest."""
    def __init__(self, manifest: str | Path | pd.DataFrame, num_frames: int = 8,
                 image_size: int = 256, sample_rate: int = 16000,
                 duration_seconds: float = 4.0):
        if isinstance(manifest, pd.DataFrame):
            self.rows = manifest.reset_index(drop=True).copy()
        else:
            path = Path(manifest)
            self.rows = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        required = {"video_path", "visual_label", "audio_label", "overall_label"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        self.video = VideoPreprocessor(num_frames, image_size)
        self.sample_rate, self.duration = sample_rate, duration_seconds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        video_path = str(row.video_path)
        frames = self.video(video_path)
        # Failure propagates with path context; no synthetic silent fallback.
        waveform = extract_audio(video_path, self.sample_rate, self.duration)
        return {"frames": frames, "audio": torch.from_numpy(waveform),
                "visual_label": int(row.visual_label), "audio_label": int(row.audio_label),
                "overall_label": int(row.overall_label), "video_path": video_path,
                "group_id": str(row.get("group_id", ""))}
