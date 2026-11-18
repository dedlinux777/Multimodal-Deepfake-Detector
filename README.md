# Multimodal Deepfake Detection using Evidential Learning and Dempster–Shafer Fusion

Research-oriented binary REAL/FAKE video classification using embedded audio and video. The system predicts from four views: audio, visual, pseudo-view (concatenated projected features), and Dempster–Shafer fused evidence. It reports branch predictions, class probabilities, evidence, and uncertainty separately.

## Architecture

- **Audio:** Adapted RawNet2 design: sinc-filter waveform frontend, residual 1-D blocks, bidirectional GRU; 1024-dimensional representation. Source: Jung et al., *Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Deep Residual Networks*, Interspeech 2020, [arXiv:2004.00578](https://arxiv.org/abs/2004.00578). This is not an official authors' implementation or checkpoint-compatible reproduction.
- **Visual:** Torchvision EfficientNet-B7 feature extractor; global pooled representation is 2560 dimensions. It supports pretrained torchvision weights, freezing, and staged unfreezing.
- Audio projection: 1024→1024. Visual projection: 2560→1024.
- Audio and visual evidential heads each output two Softplus evidence values. Pseudo-view receives the concatenated 2048 projected features. Evidence `e` maps to Dirichlet `alpha=e+1`; expected probabilities are `alpha/sum(alpha)`.
- Audio and visual opinions are combined first, then combined with pseudo-view evidence using normalized Dempster–Shafer fusion. Conflict is recorded by the model.

## Dataset preparation

1. Obtain FakeAVCeleb through its authorized distribution. Media is not included here.
2. Make a CSV manifest with `video_path,category,visual_label,audio_label,overall_label,group_id`. The four category mappings are `RealVideo-RealAudio=(0,0,0)`, `RealVideo-FakeAudio=(0,1,1)`, `FakeVideo-RealAudio=(1,0,1)`, and `FakeVideo-FakeAudio=(1,1,1)`. Labels are ordered visual, audio, overall; `group_id` must identify source identities/source videos reliably.
3. Validate the source manifest with `python scripts/validate_dataset.py <manifest.csv> --report artifacts/evaluation/validation_errors.csv`.
4. Create group-disjoint reproducible split manifests with `dataset.split.split_manifest` and save them as CSV. Group integrity depends on correct source group IDs; the software cannot infer identity relationships from pixels.
5. Set the train and validation manifest paths in `config.yaml`. The training entry point does not load `data.test_manifest`.

FFmpeg must be installed and on `PATH`; OpenCV must have a working MP4 decoder. Audio decode failures are reported rather than replaced with silence.

## Installation

Python 3.10–3.12 is intended. Create a virtual environment, install a matched PyTorch/torchvision build for your CPU or CUDA runtime using the [official selector](https://pytorch.org/get-started/locally/), then install this project's dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

The generic torch requirements cannot select the right CUDA wheel for every host. Use the official selector first and ensure torchvision matches torch. FFmpeg is a system executable and is not installed by pip.

## Training

Edit `config.yaml` with dataset manifests. The current config uses 600×600 frames and 16 frames per video; EfficientNet-B7 at this setting is memory intensive. Train/validation group overlap is rejected when both manifests contain `group_id`.

```powershell
python scripts/train.py --config config.yaml
```

Stage one freezes both encoders and learns projections/heads. `training.unfreeze_schedule` progressively unfreezes configured EfficientNet stages and RawNet. `training.class_weights` are branch-specific; auxiliary labels default to balanced weights while pseudo/fused overall-label heads use inverse-frequency weights. `balanced_sampler` is an alternative overall-label sampling strategy. Training writes `artifacts/training/best.pt`, `latest.pt`, `history.csv`, `config_snapshot.yaml`, and `training.log`.

## Evaluation and threshold calibration

Thresholds are selected independently for each view from validation predictions only, then fixed for test scoring. Supported methods: `max_balanced_accuracy`, `max_f1`, and `recall_at_least` (with `--operating-point`). Never pass test predictions into calibration.

```powershell
python -m evaluation.evaluate `
  --checkpoint artifacts/training/best.pt `
  --validation <validation.csv> `
  --test <test.csv> `
  --config config.yaml `
  --output artifacts/evaluation/test `
  --threshold max_balanced_accuracy
```

Outputs include `test_metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curves.png`, `probability_distributions.png`, `predictions.csv`, `branch_comparison.csv`, and excluded-file reports where needed. For robustness analysis, include a `generator`, `dataset`, or `domain` column. Results for listed unseen domains are descriptive and do not establish broad generalization from one dataset.

## Inference

```powershell
python -m inference.predict path\to\video.mp4 --checkpoint artifacts/training/best.pt
```

Inference loads the model state and saved training config from the checkpoint and reuses the training `VideoPreprocessor` and `extract_audio` implementations. It rejects non-MP4, oversized, unreadable, or audio-less inputs. Evidence and subjective-logic uncertainty are reported distinctly from expected class probabilities.

## Web application

```powershell
$env:ETMC_CHECKPOINT = "artifacts/training/best.pt"
python app/app.py
```

The Gradio page accepts an MP4 and shows fused and branch predictions, probabilities, evidence/uncertainty, and a fake-probability comparison plot.

## Hardware

CPU execution is supported for development and inference, but full B7 video training is slow. CUDA is recommended for training. A practical GPU memory amount depends on video frame count, resolution, and batch size; 600px×16-frame B7 can exceed common consumer GPU memory. Reduce batch size and/or frame count in config when necessary, recording the resulting experiment settings. Mixed precision activates only on CUDA.

## Known limitations

- No FakeAVCeleb media, trained checkpoint, or measured project result is checked in.
- RawNet is a documented adaptation, not official source-code reproduction; no pretrained RawNet weights are bundled.
- Group split safety is only as reliable as manifest `group_id` assignment. Same source content under different IDs can still leak.
- The frame aggregator currently averages frame-level EfficientNet features. Temporal ordering and synchronization are not modeled.
- Train/test distribution shift, generator generalization, and calibration quality require evaluation across additional held-out sources.
- The inference smoke test requires `ETMC_SMOKE_CHECKPOINT` and `ETMC_SMOKE_VIDEO`; it is skipped if either is absent.

## Implementation and audit checklist

- [x] FakeAVCeleb category/label mapping and validator.
- [x] Group-aware split utility and group-overlap checks.
- [x] Shared audio/video preprocessing for dataset and inference.
- [x] Encoder feature dimensions, evidential heads, and DS fusion tests.
- [x] Branch-specific training losses and imbalance weights.
- [x] Test-isolated training and validation-only threshold calibration.
- [x] Strict best-checkpoint state loading.
- [x] One-epoch CPU synthetic trainer and strict best-checkpoint round-trip test.
- [x] CPU inference smoke with generated MP4 and app-callback/standalone output parity.
- [x] CPU/CUDA device selection paths implemented; CUDA hardware not available for this audit.
- [ ] Real FakeAVCeleb training, held-out test evaluation, CUDA execution, and sample-video inference using an actual trained checkpoint remain unverified without data/checkpoint/hardware.
