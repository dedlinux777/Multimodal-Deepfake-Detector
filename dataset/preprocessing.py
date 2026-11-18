"""OpenCV video decoding and frame preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class VideoPreprocessor:
    """Decode evenly spaced RGB frames and normalize for ImageNet encoders."""
    num_frames: int = 8
    image_size: int = 256
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.num_frames < 1 or self.image_size < 1:
            raise ValueError("num_frames and image_size must be positive")

    def __call__(self, path: str | Path) -> Tensor:
        """Return frames [T,3,H,W]; raise ValueError for unreadable/invalid video."""
        path = str(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {path}")
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(f"Video has no decodable frames: {path}")
            indices = np.linspace(0, count - 1, self.num_frames).round().astype(int)
            frames: list[Tensor] = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, bgr = cap.read()
                if not ok or bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.size == 0:
                    raise ValueError(f"Invalid frame {index} in video: {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
                mean = torch.tensor(self.mean)[:, None, None]
                std = torch.tensor(self.std)[:, None, None]
                frames.append((tensor - mean) / std)
            return torch.stack(frames)
        finally:
            cap.release()
