#!/usr/bin/env python
"""Bayesian Optimization for E2E generation parameters using Optuna.

Usage:
    python scripts/eval_e2e_bo.py --config config.yml --n-trials 50
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split

from src.config import settings, reload_config
from src.services.part_b import PartBService
from src.services.data_loader import DataLoaderService
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors


def load_lgbm_models(model_dir: Path):
    """Load LightGBM models for all descriptors."""
    model_file = model_dir / "models.pkl"
    with open(model_file, "rb") as f:
        return pickle.load(f)


def predict_descriptors_lgbm(models, spectra, descriptor_names):
    """Predict descriptors using LightGBM ensemble."""
    n_samples = len(spectra)
    predictions = np.zeros((n_samples, len(descriptor_names)))
    for i, name in enumerate(descriptor_names):
        if name in models:
            predictions[:, i] = models[name].predict(spectra)
    return predictions


def compute_hit_at_k(candidates_list, true_smiles_list, k_values=[1, 5, 10, 50]):
    """Compute Hit@K metrics."""
    hits = {k: 0 for k in k_values}
    total = 0

    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        mol = Chem.MolFromSmiles(true_smiles)
        if mol is None:
            continue
        true_canonical = Chem.MolToSmiles(mol, canonical=True)
        total += 1

        canonical_candidates = []
        for cand in candidates:
            cand_mol = Chem.MolFromSmiles(cand)
            if cand_mol is not None:
                canonical_candidates.append(Chem.MolToSmiles(cand_mol, canonical=True))

        for k in k_values:
            if true_canonical in canonical_candidates[:k]:
                hits[k] += 1

    return {k: hits[k] / total if total > 0 else 0.0 for k in k_values}


def compute_tanimoto(candidates_list, true_smiles_list):
    """Compute mean best Tanimoto similarity."""
    best_sims = []
    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        mol = Chem.MolFromSmiles(true_smiles)
        if mol is None:
            continue
        true_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)

        sims = []
        for cand in candidates:
            cand_mol = Chem.MolFromSmiles(cand)
            if cand_mol is not None:
                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                sims.append(DataStructs.TanimotoSimilarity(true_fp, cand_fp))

        if sims:
            best_sims.append(max(sims))

    return np.mean(best_sims) if best_sims else 0.0


class RichCallback:
    """Rich live logging callback for Optuna."""

    def __init__(self):
        self.best_value = 0.0
        self.start_time = time.time()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        elapsed = time.time() - self.start_time
        is_best = trial.value > self.best_value if trial.value else False
        if is_best and trial.value:
            self.best_value = trial.value

        # Build status line
        status = "NEW BEST!" if is_best else ""
        hit1 = trial.user_attrs.get("hit_at_1", 0)
        tan = trial.user_attrs.get("tanimoto", 0)

        print(
            f"[Trial {trial.number:3d}] "
            f"Hit@10={trial.value:.1%} | Hit@1={hit1:.1%} | Tan={tan:.3f} | "
            f"temp={trial.params.get('temperature', 0):.2f} | "
            f"top_p={trial.params.get('top_p', 0):.2f} | "
            f"n_cand={trial.params.get('n_candidates', 0):3d} | "
            f"{elapsed:.0f}s {status}",
            flush=True
        )


def create_objective(pred_scaled, smiles_list, part_b, batch_size):
    """Create Optuna objective function."""

    def objective(trial):
        # Sample parameters - nucleus sampling only
        temperature = trial.suggest_float("temperature", 0.1, 2.0)
        top_p = trial.suggest_float("top_p", 0.5, 1.0)
        n_candidates = trial.suggest_int("n_candidates", 10, 100, step=10)

        # Generate candidates
        all_candidates = []
        for batch_idx in range(0, len(pred_scaled), batch_size):
            batch = pred_scaled[batch_idx:batch_idx + batch_size]
            batch_cands = part_b.generate(
                batch,
                n_candidates=n_candidates,
                temperature=temperature,
                top_p=top_p
            )
            all_candidates.extend(batch_cands)

        # Compute metrics
        hit_at_k = compute_hit_at_k(all_candidates, smiles_list)
        tanimoto = compute_tanimoto(all_candidates, smiles_list)

        # Store secondary metrics
        trial.set_user_attr("hit_at_1", hit_at_k[1])
        trial.set_user_attr("hit_at_5", hit_at_k[5])
        trial.set_user_attr("hit_at_50", hit_at_k[50])
        trial.set_user_attr("tanimoto", tanimoto)

        return hit_at_k[10]  # Maximize Hit@10

    return objective


def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization for E2E generation params")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=50, help="Number of BO trials")
    parser.add_argument("--n-samples", type=int, default=500, help="Subsample size")
    parser.add_argument("--batch-size", type=int, default=128, help="Generation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    global settings
    settings = reload_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80, flush=True)
    print("BAYESIAN OPTIMIZATION FOR E2E GENERATION PARAMETERS", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset: {settings.dataset}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Trials: {args.n_trials}", flush=True)
    print(f"Samples: {args.n_samples}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print()

    # Load Part A (LightGBM)
    print("Loading Part A (LightGBM)...", flush=True)
    lgbm_dir = settings.models_path / "part_a_lgbm"
    lgbm_models = load_lgbm_models(lgbm_dir)
    print(f"  Loaded {len(lgbm_models)} models", flush=True)

    # Load Part B
    print("Loading Part B (DirectDecoder)...", flush=True)
    part_b = PartBService()
    part_b.load(settings.models_path / "part_b")
    print(f"  Vocab size: {part_b.encoder.vocab_size}", flush=True)
    print()

    # Load test data
    print("Loading test data...", flush=True)
    data_loader = DataLoaderService(
        data_dir=Path(settings.data_input_dir) / settings.dataset
    )
    raw_data, _ = data_loader.load_raw_data()

    # Split to get test set
    train_val, test_data = train_test_split(
        raw_data,
        test_size=settings.test_ratio,
        random_state=settings.random_seed
    )

    # Process test data
    spectra_list = []
    smiles_list = []

    for sample in test_data:
        spectrum = process_spectrum(
            sample["peaks"],
            n_bins=settings.n_bins,
            transform=settings.transform,
            normalize=settings.normalize,
        )
        desc = calculate_descriptors(sample["smiles"], settings.descriptor_names)
        if desc is not None:
            spectra_list.append(spectrum)
            smiles_list.append(sample["smiles"])

    spectra = np.array(spectra_list)
    print(f"  Total test samples: {len(smiles_list)}", flush=True)

    # Random subsample
    np.random.seed(args.seed)
    if args.n_samples < len(smiles_list):
        indices = np.random.choice(len(smiles_list), args.n_samples, replace=False)
        spectra = spectra[indices]
        smiles_list = [smiles_list[i] for i in indices]
    print(f"  Using subsample: {len(smiles_list)}", flush=True)
    print()

    # Predict descriptors once (Part A)
    print("Predicting descriptors (Part A)...", flush=True)
    pred_descriptors = predict_descriptors_lgbm(lgbm_models, spectra, settings.descriptor_names)
    pred_scaled = part_b.scaler.transform(pred_descriptors)
    print("  Done", flush=True)
    print()

    # Create Optuna study
    print("=" * 80, flush=True)
    print("STARTING OPTIMIZATION", flush=True)
    print("=" * 80, flush=True)
    print()
    print("Search space:", flush=True)
    print("  temperature: [0.1, 2.0]", flush=True)
    print("  top_p:       [0.5, 1.0]", flush=True)
    print("  n_candidates:[10, 100]", flush=True)
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        study_name="e2e_generation_params"
    )

    # Optimize with rich callback
    callback = RichCallback()
    objective = create_objective(pred_scaled, smiles_list, part_b, args.batch_size)

    start_time = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[callback],
        show_progress_bar=False
    )
    total_time = time.time() - start_time

    # Results
    print()
    print("=" * 80, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print()
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Trials: {len(study.trials)}", flush=True)
    print()
    print("BEST PARAMETERS:", flush=True)
    print(f"  temperature:  {study.best_params['temperature']:.3f}", flush=True)
    print(f"  top_p:        {study.best_params['top_p']:.3f}", flush=True)
    print(f"  n_candidates: {study.best_params['n_candidates']}", flush=True)
    print()
    print("BEST METRICS:", flush=True)
    print(f"  Hit@1:     {study.best_trial.user_attrs['hit_at_1']:.1%}", flush=True)
    print(f"  Hit@5:     {study.best_trial.user_attrs['hit_at_5']:.1%}", flush=True)
    print(f"  Hit@10:    {study.best_value:.1%}", flush=True)
    print(f"  Hit@50:    {study.best_trial.user_attrs['hit_at_50']:.1%}", flush=True)
    print(f"  Tanimoto:  {study.best_trial.user_attrs['tanimoto']:.3f}", flush=True)
    print()

    # Top 5 trials
    print("TOP 5 TRIALS:", flush=True)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(sorted_trials[:5]):
        print(
            f"  #{i+1}: Hit@10={t.value:.1%} | "
            f"temp={t.params['temperature']:.2f} | "
            f"top_p={t.params['top_p']:.2f} | "
            f"n_cand={t.params['n_candidates']}",
            flush=True
        )
    print()

    # Save results
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": {
            "hit_at_1": study.best_trial.user_attrs["hit_at_1"],
            "hit_at_5": study.best_trial.user_attrs["hit_at_5"],
            "hit_at_10": study.best_value,
            "hit_at_50": study.best_trial.user_attrs["hit_at_50"],
            "tanimoto": study.best_trial.user_attrs["tanimoto"],
        },
        "config": {
            "n_trials": args.n_trials,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "total_time_seconds": total_time,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ]
    }

    output_path = settings.metrics_path / "e2e_bo_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
