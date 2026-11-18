"""Validate FakeAVCeleb media, labels, and manifest integrity."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from dataset.audio_extraction import AudioExtractionError, extract_audio

LABELS = {
    "RealVideo-RealAudio": (0, 0, 0),
    "RealVideo-FakeAudio": (0, 1, 1),
    "FakeVideo-RealAudio": (1, 0, 1),
    "FakeVideo-FakeAudio": (1, 1, 1),
}


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_manifest(manifest: str | Path, report_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate rows; report errors per path and label distributions."""
    path = Path(manifest)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"video_path", "category", "visual_label", "audio_label", "overall_label"}
    errors: list[dict[str, str]] = []
    for col in required - set(df.columns):
        errors.append({"row": "", "video_path": "", "error": f"missing column {col}"})
    if required - set(df.columns):
        report = {"valid": False, "errors": errors}
        if report_path: pd.DataFrame(errors).to_csv(report_path, index=False)
        return df, report
    missing = df[list(required)].isna().any(axis=1)
    for i in df.index[missing]: errors.append({"row": str(i), "video_path": str(df.at[i, "video_path"]), "error": "missing required value"})
    normalized = df.video_path.astype(str).map(lambda p: str(Path(p).resolve()))
    duplicates = normalized.duplicated(keep=False)
    for i in df.index[duplicates]: errors.append({"row": str(i), "video_path": str(df.at[i, "video_path"]), "error": "duplicate video path"})
    for i, row in df.iterrows():
        video = Path(str(row.video_path))
        if any(pd.isna(row.get(col)) for col in required):
            continue
        expected = LABELS.get(str(row.category))
        try: actual = tuple(int(row[col]) for col in ("visual_label", "audio_label", "overall_label"))
        except (TypeError, ValueError, OverflowError): actual = ()
        if expected is None or actual != expected:
            errors.append({"row": str(i), "video_path": str(video), "error": "invalid/inconsistent category labels"})
        if len(actual)!=3 or any(x not in (0, 1) for x in actual):
            errors.append({"row": str(i), "video_path": str(video), "error": "labels must be binary"})
        if not video.is_file():
            errors.append({"row": str(i), "video_path": str(video), "error": "video path missing"})
            continue
        cap = cv2.VideoCapture(str(video))
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame is None or frame.size == 0: break
            decoded += 1
        cap.release()
        if decoded == 0 or (reported > 0 and decoded < reported):
            errors.append({"row": str(i), "video_path": str(video), "error": "corrupt/unreadable video"})
            continue
        try:
            extract_audio(video)
        except AudioExtractionError as exc:
            errors.append({"row": str(i), "video_path": str(video), "error": str(exc)})
    distributions = {col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
                     for col in ("visual_label", "audio_label", "overall_label")}
    report = {"valid": not errors, "row_count": len(df), "errors": errors,
              "class_distributions": distributions}
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors, columns=["row", "video_path", "error"]).to_csv(report_path, index=False)
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="CSV or Parquet manifest")
    parser.add_argument("--report", default="artifacts/evaluation/dataset_validation_errors.csv")
    args = parser.parse_args()
    _, report = validate_manifest(args.manifest, args.report)
    print(f"Rows: {report.get('row_count', 0)}; valid: {report['valid']}; errors: {len(report['errors'])}")
    print(report.get("class_distributions", {}))
    if not report["valid"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
