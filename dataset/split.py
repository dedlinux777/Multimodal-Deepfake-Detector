"""Reproducible group-disjoint train/validation/test manifest splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def split_manifest(manifest: pd.DataFrame, group_column: str = "group_id",
                   label_column: str = "overall_label", test_size: float = .2,
                   validation_size: float = .1, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split by groups; approximate stratification on overall label.

    Ratios are approximate because indivisible groups take precedence. Raises
    rather than allowing group leakage or silently falling back to row splits.
    """
    if group_column not in manifest or label_column not in manifest:
        raise ValueError(f"Manifest requires {group_column!r} and {label_column!r}")
    if manifest[[group_column, label_column]].isna().any().any():
        raise ValueError("Group IDs and stratification labels must not be missing")
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("Invalid split proportions")
    groups = manifest[group_column].astype(str)
    y = manifest[label_column].astype(int)
    n_splits = max(2, round(1 / test_size))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(manifest, y, groups))
    trainval = manifest.iloc[trainval_idx].copy()
    relative_val = validation_size / (1 - test_size)
    inner_n = max(2, round(1 / relative_val))
    inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=seed + 1)
    local_train, local_val = next(inner.split(trainval, trainval[label_column].astype(int),
                                             trainval[group_column].astype(str)))
    result = {"train": trainval.iloc[local_train].copy(), "validation": trainval.iloc[local_val].copy(),
              "test": manifest.iloc[test_idx].copy()}
    group_sets = [set(frame[group_column].astype(str)) for frame in result.values()]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Group leakage detected")
    return result


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write split manifests as CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output / f"{name}.csv", index=False)
