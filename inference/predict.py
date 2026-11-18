"""Inference for the trained ETMC model using checkpoint-saved configuration."""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
from typing import Any
import torch
from models.etmc import ETMC
from dataset.audio_extraction import AudioExtractionError, extract_audio
from dataset.preprocessing import VideoPreprocessor

class PredictionError(RuntimeError):
    """Invalid input or unavailable/incompatible prediction artifact."""

def _label(probabilities: torch.Tensor) -> str:
    return "FAKE" if int(probabilities.argmax(-1).item())==1 else "REAL"

class VideoPredictor:
    """Load the checkpoint's exact model config and apply training preprocessing."""
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.is_file(): raise PredictionError(f"Checkpoint not found: {self.checkpoint}")
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try: saved=torch.load(self.checkpoint,map_location=self.device,weights_only=False)
        except Exception as exc: raise PredictionError(f"Cannot read checkpoint: {exc}") from exc
        self.config=saved.get("config") if isinstance(saved,dict) else None
        if not self.config: raise PredictionError("Checkpoint does not contain its training config; exact preprocessing cannot be guaranteed")
        mc=self.config["model"]
        try:
            self.model=ETMC(pretrained_visual=False,freeze_visual=True,
              rawnet_checkpoint=None,visual_checkpoint=None,prior=float(mc["heads"].get("dirichlet_prior",1.)))
            self.model.load_state_dict(saved["model"],strict=True)
        except Exception as exc: raise PredictionError(f"Checkpoint architecture/weights are incompatible: {exc}") from exc
        self.model.to(self.device).eval()
        data=self.config["data"]
        rate=int(data.get("audio_sample_rate",16000))
        if rate!=16000: raise PredictionError(f"Checkpoint training sample rate is {rate}; this detector requires the configured 16000 Hz audio path")
        duration=float(data.get("audio_num_samples",64000))/rate
        self.audio_sample_rate=rate; self.audio_duration=duration
        self.video=VideoPreprocessor(num_frames=int(data.get("video_num_frames",16)),image_size=int(data.get("image_size",600)))
        self.max_bytes=int(self.config.get("inference",{}).get("max_upload_mb",500))*1024*1024

    @torch.inference_mode()
    def predict(self, video_path: str | Path) -> dict[str, Any]:
        """Predict from a local MP4 and return probabilities plus separate evidence."""
        path=Path(video_path)
        if not path.is_file(): raise PredictionError("Uploaded video does not exist")
        if path.suffix.lower()!=".mp4": raise PredictionError("Upload must be an MP4 file")
        if path.stat().st_size==0: raise PredictionError("Uploaded MP4 is empty")
        if path.stat().st_size>self.max_bytes: raise PredictionError(f"Video exceeds upload limit ({self.max_bytes//(1024*1024)} MB)")
        try:
            frames=self.video(path)
            waveform=extract_audio(path,self.audio_sample_rate,self.audio_duration)
        except (ValueError,AudioExtractionError) as exc: raise PredictionError(str(exc)) from exc
        audio=torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        visual=frames.unsqueeze(0).to(self.device)
        try: out=self.model(audio,visual)
        except Exception as exc: raise PredictionError(f"Model inference failed: {exc}") from exc
        if out.fused_alpha is None or out.fused_probabilities is None: raise PredictionError("Model did not produce a fused prediction")
        prior=float(self.config["model"]["heads"].get("dirichlet_prior",1.)); classes=out.fused_alpha.shape[-1]
        def branch(alpha, probs):
            if alpha is None or probs is None: return None
            evidence=(alpha-prior).clamp_min(0)
            return {"prediction":_label(probs[0]),"fake_probability":float(probs[0,1]),
                    "class_probabilities":{"REAL":float(probs[0,0]),"FAKE":float(probs[0,1])},
                    "evidence":{"REAL":float(evidence[0,0]),"FAKE":float(evidence[0,1])},
                    "uncertainty":float(classes/alpha.sum(-1)[0])}
        return {"prediction":_label(out.fused_probabilities[0]),
                "fused_fake_probability":float(out.fused_probabilities[0,1]),
                "fused_class_probabilities":{"REAL":float(out.fused_probabilities[0,0]),"FAKE":float(out.fused_probabilities[0,1])},
                "audio":branch(out.audio_alpha,out.audio_probabilities),
                "visual":branch(out.visual_alpha,out.visual_probabilities),
                "pseudo_view":branch(out.pseudo_alpha,out.pseudo_probabilities),
                "fused_evidence":{"REAL":float((out.fused_alpha-prior).clamp_min(0)[0,0]),"FAKE":float((out.fused_alpha-prior).clamp_min(0)[0,1])},
                "fused_uncertainty":float(classes/out.fused_alpha.sum(-1)[0]),
                "evidence_note":"Evidence is alpha minus the Dirichlet prior; uncertainty is K/sum(alpha). These are not calibrated class probabilities."}

def main():
    p=argparse.ArgumentParser();p.add_argument("video");p.add_argument("--checkpoint",default="artifacts/training/best.pt");p.add_argument("--device");a=p.parse_args()
    print(json.dumps(VideoPredictor(a.checkpoint,a.device).predict(a.video),indent=2))
if __name__=="__main__":main()
