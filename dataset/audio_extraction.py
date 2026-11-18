"""Extract and normalize audio tracks using FFmpeg and SoundFile."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioExtractionError(RuntimeError):
    """Audio extraction failed; the reason is suitable for validation reports."""


def extract_audio(video_path: str | Path, sample_rate: int = 16000,
                  duration_seconds: float = 4.0, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Extract mono float waveform, resample, peak normalize, then pad/truncate.

    Missing streams and decoder failures raise AudioExtractionError; silence is
    never substituted for a failed extraction.
    """
    path = Path(video_path)
    if sample_rate <= 0 or duration_seconds <= 0:
        raise ValueError("sample_rate and duration_seconds must be positive")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise AudioExtractionError("FFmpeg executable not found")
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        cmd = [executable, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
               "-ac", "1", "-c:a", "pcm_f32le", str(wav_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not wav_path.exists():
            reason = result.stderr.strip() or "No audio stream or decode failure"
            raise AudioExtractionError(f"Audio extraction failed for {path}: {reason}")
        try:
            audio, source_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise AudioExtractionError(f"Cannot read extracted audio for {path}: {exc}") from exc
    if audio.size == 0 or source_rate <= 0:
        raise AudioExtractionError(f"Empty or invalid audio stream: {path}")
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        from math import gcd
        divisor = gcd(int(source_rate), int(sample_rate))
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor).astype(np.float32)
    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak
    target = int(round(sample_rate * duration_seconds))
    result = np.zeros(target, dtype=np.float32)
    result[:min(target, len(mono))] = mono[:target]
    return result
