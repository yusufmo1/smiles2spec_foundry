"""Metrics loading utilities."""

import csv
import json
from pathlib import Path
from typing import Optional


def load_all_metrics(metrics_dir: Path) -> dict:
    """Load all metrics files into single dict."""
    metrics = {}
    files = {
        "e2e": "e2e_evaluation.json",
        "oracle": "part_b_evaluation.json",
        "part_a_lgbm": "part_a_lgbm_metrics.json",
        "part_a_hybrid": "part_a_metrics.json",
        "part_b": "part_b_metrics.json",
    }
    for key, filename in files.items():
        path = metrics_dir / filename
        if path.exists():
            with open(path) as f:
                metrics[key] = json.load(f)
    return metrics


def find_latest_log(log_dir: Path, model: str) -> Optional[Path]:
    """Find most recent epoch log for a model."""
    logs = sorted(log_dir.glob(f"train_{model}_*_epochs.csv"), reverse=True)
    return logs[0] if logs else None


def read_epoch_log(log_path: Path) -> dict:
    """Read CSV epoch log into dict of lists."""
    data = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_r2": [], "val_r2": [], "lr": [],
        "recon_loss": [], "kl_loss": [], "beta": [],
    }
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                if key in row:
                    data[key].append(float(row[key]))
    return data
