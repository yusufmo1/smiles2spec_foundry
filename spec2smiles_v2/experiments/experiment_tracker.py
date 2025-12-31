#!/usr/bin/env python
"""Experiment tracker for Mamba-Diffusion hyperparameter search."""

import json
import os
from datetime import datetime
from pathlib import Path

EXPERIMENTS_FILE = Path(__file__).parent / "results.json"


def load_experiments():
    """Load existing experiments."""
    if EXPERIMENTS_FILE.exists():
        with open(EXPERIMENTS_FILE) as f:
            return json.load(f)
    return {"experiments": [], "best": None}


def save_experiments(data):
    """Save experiments."""
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def log_experiment(
    name: str,
    config: dict,
    results: dict,
    notes: str = "",
):
    """Log an experiment result."""
    data = load_experiments()

    exp = {
        "id": len(data["experiments"]) + 1,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "notes": notes,
    }

    data["experiments"].append(exp)

    # Update best if this is better
    tanimoto = results.get("mean_tanimoto", 0)
    if data["best"] is None or tanimoto > data["best"].get("results", {}).get("mean_tanimoto", 0):
        data["best"] = exp
        print(f"🏆 NEW BEST: {name} - Tanimoto: {tanimoto:.4f}")

    save_experiments(data)
    print(f"📊 Logged experiment #{exp['id']}: {name}")
    return exp


def print_summary():
    """Print experiment summary."""
    data = load_experiments()

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for exp in data["experiments"]:
        r = exp["results"]
        print(f"\n[{exp['id']:2d}] {exp['name']}")
        print(f"     Tanimoto: {r.get('mean_tanimoto', 0):.4f} | "
              f"Hit@10: {r.get('hit_at_10', 0):.2%} | "
              f"Validity: {r.get('validity', 0):.2%}")

    if data["best"]:
        print("\n" + "-" * 60)
        print(f"🏆 BEST: {data['best']['name']}")
        print(f"   Tanimoto: {data['best']['results'].get('mean_tanimoto', 0):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    print_summary()
