"""Simple Gradio MP4 analyzer for a trained ETMC checkpoint."""
from __future__ import annotations
import os
from pathlib import Path
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inference.predict import PredictionError, VideoPredictor

CHECKPOINT=os.environ.get("ETMC_CHECKPOINT","artifacts/training/best.pt")
_predictor=None

def predictor():
    global _predictor
    if _predictor is None: _predictor=VideoPredictor(CHECKPOINT)
    return _predictor

def analyze(video_path,progress=gr.Progress()):
    if not video_path:
        return "Upload an MP4 video first.",None,None
    try:
        progress(0,desc="Validating video and loading audio")
        result=predictor().predict(video_path)
        progress(1,desc="Analysis complete")
        summary=f"## Final prediction: **{result['prediction']}**\n\nFused P(FAKE): **{result['fused_fake_probability']:.4f}**\n\n" \
          f"Audio: **{result['audio']['prediction']}** · P(FAKE) {result['audio']['fake_probability']:.4f}\n\n" \
          f"Visual: **{result['visual']['prediction']}** · P(FAKE) {result['visual']['fake_probability']:.4f}\n\n" \
          f"Pseudo-view: **{result['pseudo_view']['prediction']}** · P(FAKE) {result['pseudo_view']['fake_probability']:.4f}\n\n" \
          f"Fused uncertainty: {result['fused_uncertainty']:.4f}. Evidence values are separate from calibrated probabilities."
        fig,ax=plt.subplots(figsize=(7,2.6)); names=["Audio","Visual","Pseudo-view","Fused"]
        vals=[result[k]["fake_probability"] for k in ("audio","visual","pseudo_view")]+[result["fused_fake_probability"]]
        ax.barh(names,vals,color=["#5B8FF9","#61DDAA","#65789B","#F6BD16"]);ax.set_xlim(0,1);ax.set_xlabel("P(FAKE)");ax.invert_yaxis();fig.tight_layout()
        return summary,result,fig
    except PredictionError as exc:
        return f"**Unable to analyze video:** {exc}",None,None
    except Exception as exc:
        return f"**Unexpected analysis error:** {type(exc).__name__}: {exc}",None,None

with gr.Blocks(title="Multimodal Deepfake Detector") as demo:
    gr.Markdown("# Multimodal Deepfake Detector\nUpload an MP4 with embedded audio. The app uses the checkpoint's saved model and preprocessing configuration.")
    video=gr.Video(label="MP4 video",sources=["upload"],format="mp4")
    button=gr.Button("Analyze",variant="primary")
    status=gr.Markdown(label="Processing status")
    plot=gr.Plot(label="Branch fake-probability comparison")
    details=gr.JSON(label="Branch probabilities, evidence, and uncertainty")
    button.click(analyze,inputs=video,outputs=[status,details,plot])

if __name__=="__main__":
    if not Path(CHECKPOINT).is_file(): raise SystemExit(f"Checkpoint not found: {CHECKPOINT}. Set ETMC_CHECKPOINT.")
    demo.queue().launch()
