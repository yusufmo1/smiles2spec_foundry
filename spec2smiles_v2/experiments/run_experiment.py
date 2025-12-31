#!/usr/bin/env python
"""Run a single Mamba-Diffusion experiment with config overrides."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from experiment_tracker import log_experiment


def create_config(base_config: dict, overrides: dict, name: str) -> Path:
    """Create a temporary config with overrides."""
    config = base_config.copy()

    # Deep merge overrides
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(config, overrides)

    # Save to temp config
    config_path = Path(__file__).parent / f"config_{name}.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_training(config_path: Path, name: str) -> Path:
    """Run training and return log path."""
    log_path = Path(__file__).parent / f"logs/{name}_train.log"
    log_path.parent.mkdir(exist_ok=True)

    cmd = [
        "poetry", "run", "python", "-u",
        "scripts/train_mamba_diffusion.py",
        "--config", str(config_path),
    ]

    print(f"🚀 Training: {name}")
    start = time.time()

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={"PYTHONUNBUFFERED": "1", **dict(__import__("os").environ)},
        )

    elapsed = time.time() - start
    print(f"   Completed in {elapsed/60:.1f} min (exit code: {proc.returncode})")

    return log_path


def run_eval(config_path: Path, name: str, n_steps: int = 10) -> dict:
    """Run evaluation and return results."""
    log_path = Path(__file__).parent / f"logs/{name}_eval.log"

    cmd = [
        "poetry", "run", "python", "-u",
        "scripts/eval_mamba_diffusion.py",
        "--config", str(config_path),
        "--n-candidates", "50",
        "--n-steps", str(n_steps),
    ]

    print(f"📊 Evaluating: {name} (steps={n_steps})")
    start = time.time()

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={"PYTHONUNBUFFERED": "1", **dict(__import__("os").environ)},
        )

    elapsed = time.time() - start
    print(f"   Completed in {elapsed/60:.1f} min")

    # Parse results from JSON
    results_path = Path(__file__).parent.parent / "data/output/metrics/mamba_diffusion_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--overrides", type=json.loads, default={}, help="JSON config overrides")
    parser.add_argument("--eval-steps", type=int, default=10, help="Inference steps for eval")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only eval")
    parser.add_argument("--notes", default="", help="Experiment notes")
    args = parser.parse_args()

    # Load base config
    base_config_path = Path(__file__).parent.parent / "config_mamba_diffusion.yml"
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    # Create experiment config
    config_path = create_config(base_config, args.overrides, args.name)
    print(f"📝 Config: {config_path}")

    # Run training
    if not args.skip_train:
        run_training(config_path, args.name)

    # Run evaluation
    results = run_eval(config_path, args.name, args.eval_steps)

    # Log experiment
    log_experiment(
        name=args.name,
        config=args.overrides,
        results=results,
        notes=args.notes,
    )

    return results


if __name__ == "__main__":
    main()
