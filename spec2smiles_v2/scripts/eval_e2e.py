#!/usr/bin/env python
"""End-to-end evaluation: Spectrum -> Part A -> Part B -> SMILES.

Usage:
    python scripts/eval_e2e.py --config config_gnps_unique28.yml --part-a lgbm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt

from src.config import settings, reload_config
from src.services.part_b import PartBService
from src.domain.spectrum import process_spectrum
from src.domain.descriptors import calculate_descriptors
from src.models.hybrid import HybridCNNTransformer


def load_hybrid_model(model_dir: Path, device: str = "cuda"):
    """Load Hybrid CNN-Transformer model."""
    model_file = model_dir / "hybrid.pt"
    scaler_file = model_dir / "descriptor_scaler.pkl"

    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    # Create model with saved config
    config = checkpoint.get("config", {})
    model = HybridCNNTransformer(
        input_dim=config.get("input_dim", 500),
        output_dim=config.get("output_dim", 28),
        cnn_hidden=config.get("cnn_hidden", 128),
        transformer_dim=config.get("transformer_dim", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load scaler
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_descriptors_hybrid(model, scaler, spectra, device: str = "cuda"):
    """Predict descriptors using Hybrid CNN-Transformer."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(spectra, dtype=torch.float32).to(device)
        pred_scaled = model(X).cpu().numpy()
        # Inverse transform to get original scale
        predictions = scaler.inverse_transform(pred_scaled)
    return predictions


def load_lgbm_models(model_dir: Path):
    """Load LightGBM models for all descriptors."""
    model_file = model_dir / "models.pkl"
    with open(model_file, "rb") as f:
        models = pickle.load(f)
    return models


def predict_descriptors_lgbm(models, spectra, descriptor_names):
    """Predict descriptors using LightGBM ensemble (handles both classification and regression)."""
    n_samples = len(spectra)
    predictions = np.zeros((n_samples, len(descriptor_names)))

    for i, name in enumerate(descriptor_names):
        if name in models:
            model = models[name]
            # Check if classifier or regressor from objective
            params = model.dump_model()
            obj = params.get("objective", "regression")

            pred = model.predict(spectra)
            if "multiclass" in obj:
                # Classifier: pred is (n_samples, n_classes), take argmax
                predictions[:, i] = np.argmax(pred, axis=1)
            else:
                # Regressor: pred is (n_samples,)
                predictions[:, i] = pred

    return predictions


def compute_exact_match(candidates_list, true_smiles_list):
    """Compute exact match rate."""
    matches = 0
    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        mol = Chem.MolFromSmiles(true_smiles)
        if mol is None:
            continue
        true_canonical = Chem.MolToSmiles(mol, canonical=True)

        for cand in candidates:
            cand_mol = Chem.MolFromSmiles(cand)
            if cand_mol is not None:
                cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
                if cand_canonical == true_canonical:
                    matches += 1
                    break

    return matches / len(true_smiles_list) if true_smiles_list else 0.0


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
                sim = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
                sims.append(sim)

        if sims:
            best_sims.append(max(sims))

    return np.mean(best_sims) if best_sims else 0.0


def compute_hit_at_k(candidates_list, true_smiles_list, k_values=[1, 5, 10, 50]):
    """Compute Hit@K metrics for multiple K values."""
    hits = {k: 0 for k in k_values}
    total = 0

    for candidates, true_smiles in zip(candidates_list, true_smiles_list):
        mol = Chem.MolFromSmiles(true_smiles)
        if mol is None:
            continue
        true_canonical = Chem.MolToSmiles(mol, canonical=True)
        total += 1

        # Canonicalize candidates
        canonical_candidates = []
        for cand in candidates:
            cand_mol = Chem.MolFromSmiles(cand)
            if cand_mol is not None:
                canonical_candidates.append(Chem.MolToSmiles(cand_mol, canonical=True))

        # Check hit at each K
        for k in k_values:
            if true_canonical in canonical_candidates[:k]:
                hits[k] += 1

    return {k: hits[k] / total if total > 0 else 0.0 for k in k_values}


def generate_visualizations(results: dict, output_dir: Path, part_a_type: str):
    """Generate E2E evaluation visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Bar chart comparing metrics
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Exact Match', 'Tanimoto', 'Validity']
    values = [
        results['end_to_end']['exact_match'] * 100,
        results['end_to_end']['mean_best_tanimoto'] * 100,
        results['end_to_end']['validity'] * 100,
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'End-to-End Performance ({part_a_type.upper()} → Part B)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / f'e2e_{part_a_type}_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {output_dir}/e2e_{part_a_type}_metrics.png")

    # 2. Comparison with Oracle (if we have oracle data)
    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = ['Oracle\n(True Desc)', f'E2E\n({part_a_type.upper()})']
    exact_matches = [82.2, results['end_to_end']['exact_match'] * 100]
    colors = ['#27ae60', '#e74c3c']

    bars = ax.bar(conditions, exact_matches, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    ax.set_ylabel('Exact Match (%)', fontsize=12)
    ax.set_title('Oracle vs End-to-End Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, exact_matches):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add degradation annotation
    degradation = exact_matches[0] - exact_matches[1]
    ax.annotate(f'↓ {degradation:.1f} pts', xy=(0.5, 50), fontsize=14,
                ha='center', color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / f'e2e_{part_a_type}_vs_oracle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {output_dir}/e2e_{part_a_type}_vs_oracle.png")


def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part-a", type=str, choices=["lgbm", "hybrid"], default="lgbm")
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--sweep", action="store_true", help="Run temperature/top_p sweep")
    parser.add_argument("--sweep-samples", type=int, default=500, help="Random subsample size for sweep")
    args = parser.parse_args()

    # Load config
    global settings
    settings = reload_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("End-to-End Evaluation (Spectrum -> SMILES)")
    print("=" * 60)
    print(f"Dataset: {settings.dataset}")
    print(f"Part A: {args.part_a}")
    print(f"Part B: {settings.part_b_model}")
    print(f"Device: {device}")
    print()

    # Load Part A
    print("Loading Part A models...")
    if args.part_a == "lgbm":
        lgbm_dir = settings.models_path / "part_a_lgbm"
        lgbm_models = load_lgbm_models(lgbm_dir)
        print(f"  Loaded {len(lgbm_models)} LightGBM models")
        hybrid_model, hybrid_scaler = None, None
    else:
        hybrid_dir = settings.models_path / "part_a"
        hybrid_model, hybrid_scaler = load_hybrid_model(hybrid_dir, device)
        print(f"  Loaded Hybrid CNN-Transformer model")
        lgbm_models = None

    # Load Part B
    print("Loading Part B model...")
    part_b = PartBService()
    part_b.load(settings.models_path / "part_b")
    print(f"  Vocab size: {part_b.encoder.vocab_size}")
    print(f"  Device: {part_b.device}")
    print()

    # Load test data from saved split
    print("Loading test data...")
    data_dir = Path(settings.data_input_dir) / settings.dataset
    test_path = data_dir / "test_data.jsonl"

    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Run preprocessing first: python scripts/preprocess_splits.py")
        sys.exit(1)

    with open(test_path) as f:
        test_data = [json.loads(line) for line in f]

    # Process test data
    spectra_list = []
    smiles_list = []
    true_desc_list = []

    for sample in test_data:
        # Use pre-binned spectrum if available, otherwise process peaks
        if "spectrum" in sample:
            spectrum = np.array(sample["spectrum"])
        else:
            spectrum = process_spectrum(
                sample["peaks"],
                n_bins=settings.n_bins,
                transform=settings.transform,
                normalize=settings.normalize,
            )

        # Use pre-computed descriptors if available, otherwise calculate
        if "descriptors" in sample:
            desc = np.array(sample["descriptors"])
        else:
            desc = calculate_descriptors(sample["smiles"], settings.descriptor_names)

        if desc is not None:
            spectra_list.append(spectrum)
            smiles_list.append(sample["smiles"])
            true_desc_list.append(desc)

    spectra = np.array(spectra_list)
    true_descriptors = np.array(true_desc_list)

    print(f"Test samples: {len(smiles_list)}")

    # Limit samples if specified
    if args.n_samples:
        spectra = spectra[:args.n_samples]
        smiles_list = smiles_list[:args.n_samples]
        true_descriptors = true_descriptors[:args.n_samples]
        print(f"Using first {args.n_samples} samples")

    # Random subsample for sweep
    if args.sweep and args.sweep_samples < len(smiles_list):
        np.random.seed(42)
        indices = np.random.choice(len(smiles_list), args.sweep_samples, replace=False)
        spectra = spectra[indices]
        smiles_list = [smiles_list[i] for i in indices]
        true_descriptors = true_descriptors[indices]
        print(f"Random subsample: {len(smiles_list)} samples")

    print()

    # Sweep mode
    if args.sweep:
        temps = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
        top_ps = [0.85, 0.9, 0.95, 1.0]  # 1.0 = no nucleus filtering (pure temp)

        print("=" * 70)
        print("E2E PARAMETER SWEEP")
        print(f"Samples: {len(smiles_list)} | Candidates: {args.n_candidates} | Batch: {args.batch_size}")
        print("=" * 70)

        # Predict descriptors once (Part A)
        pred_descriptors = predict_descriptors_lgbm(lgbm_models, spectra, settings.descriptor_names)
        pred_scaled = part_b.scaler.transform(pred_descriptors)

        results = []
        for temp in temps:
            for top_p in top_ps:
                # Generate with current params
                all_candidates = []
                for batch_idx in range(0, len(pred_scaled), args.batch_size):
                    batch = pred_scaled[batch_idx:batch_idx + args.batch_size]
                    batch_cands = part_b.generate(batch, n_candidates=args.n_candidates,
                                                  temperature=temp, top_p=top_p)
                    all_candidates.extend(batch_cands)

                # Compute metrics
                hit_at_k = compute_hit_at_k(all_candidates, smiles_list)
                tanimoto = compute_tanimoto(all_candidates, smiles_list)

                results.append({
                    "temperature": temp, "top_p": top_p,
                    "hit_at_1": hit_at_k[1], "hit_at_10": hit_at_k[10],
                    "tanimoto": tanimoto
                })

                print(f"temp={temp:.1f}, top_p={top_p:.2f}: Hit@1={hit_at_k[1]:.1%}, Hit@10={hit_at_k[10]:.1%}, Tan={tanimoto:.3f}")

        # Find best
        best = max(results, key=lambda x: x["hit_at_10"])
        print("\n" + "=" * 70)
        print(f"BEST: temp={best['temperature']}, top_p={best['top_p']}")
        print(f"  Hit@1:  {best['hit_at_1']:.1%}")
        print(f"  Hit@10: {best['hit_at_10']:.1%}")
        print(f"  Tanimoto: {best['tanimoto']:.3f}")
        print("=" * 70)

        # Save results
        sweep_path = settings.metrics_path / "e2e_sweep.json"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump({"results": results, "best": best}, f, indent=2)
        print(f"Sweep results saved to {sweep_path}")
        return

    # Part A: Predict descriptors from spectra
    print("Part A: Predicting descriptors from spectra...")
    pred_descriptors = predict_descriptors_lgbm(
        lgbm_models, spectra, settings.descriptor_names
    )

    # Compute Part A metrics - separate classification vs regression
    from sklearn.metrics import r2_score, accuracy_score

    # Identify classification vs regression descriptors
    class_indices = []
    reg_indices = []
    for i, name in enumerate(settings.descriptor_names):
        if name in lgbm_models:
            params = lgbm_models[name].dump_model()
            obj = params.get("objective", "regression")
            if "multiclass" in obj:
                class_indices.append(i)
            else:
                reg_indices.append(i)

    # Compute metrics for each type
    class_accuracies = []
    for i in class_indices:
        acc = accuracy_score(true_descriptors[:, i].astype(int), pred_descriptors[:, i].astype(int))
        class_accuracies.append(acc)

    reg_r2_scores = []
    for i in reg_indices:
        r2 = r2_score(true_descriptors[:, i], pred_descriptors[:, i])
        reg_r2_scores.append(r2)

    mean_class_acc = np.mean(class_accuracies) if class_accuracies else 0.0
    mean_reg_r2 = np.mean(reg_r2_scores) if reg_r2_scores else 0.0

    print(f"  Classification ({len(class_indices)} desc): Mean Accuracy = {mean_class_acc:.4f}")
    print(f"  Regression ({len(reg_indices)} desc): Mean R² = {mean_reg_r2:.4f}")
    print()

    # Scale descriptors for Part B
    pred_scaled = part_b.scaler.transform(pred_descriptors)

    # Part B: Generate SMILES candidates
    print(f"Part B: Generating SMILES candidates (batch_size={args.batch_size})...")
    all_candidates = []
    n_batches = (len(pred_scaled) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="Generating"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(pred_scaled))
        batch_desc = pred_scaled[start:end]
        batch_candidates = part_b.generate(
            batch_desc,
            n_candidates=args.n_candidates,
            temperature=settings.temperature,
            top_p=args.top_p,
        )
        all_candidates.extend(batch_candidates)

    # Compute metrics
    print()
    print("Computing metrics...")

    exact_match = compute_exact_match(all_candidates, smiles_list)
    mean_tanimoto = compute_tanimoto(all_candidates, smiles_list)
    hit_at_k = compute_hit_at_k(all_candidates, smiles_list)

    # Validity
    valid_count = sum(
        1 for cands in all_candidates
        for c in cands
        if Chem.MolFromSmiles(c) is not None
    )
    total_count = sum(len(cands) for cands in all_candidates)
    validity = valid_count / total_count if total_count > 0 else 0.0

    print()
    print("=" * 60)
    print("END-TO-END RESULTS")
    print("=" * 60)
    print(f"  Part A Classification Acc: {mean_class_acc:.4f} ({len(class_indices)} desc)")
    print(f"  Part A Regression R²:      {mean_reg_r2:.4f} ({len(reg_indices)} desc)")
    print(f"  Hit@1:              {hit_at_k[1]:.4f} ({hit_at_k[1] * 100:.1f}%)")
    print(f"  Hit@5:              {hit_at_k[5]:.4f} ({hit_at_k[5] * 100:.1f}%)")
    print(f"  Hit@10:             {hit_at_k[10]:.4f} ({hit_at_k[10] * 100:.1f}%)")
    print(f"  Hit@50:             {hit_at_k[50]:.4f} ({hit_at_k[50] * 100:.1f}%)")
    print(f"  Exact Match:        {exact_match:.4f} ({exact_match * 100:.1f}%)")
    print(f"  Mean Best Tanimoto: {mean_tanimoto:.4f}")
    print(f"  Validity:           {validity:.4f} ({validity * 100:.1f}%)")
    print()
    print(f"  Candidates per sample: {args.n_candidates}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Test samples: {len(smiles_list)}")

    # Save results
    results = {
        "part_a": {
            "model": args.part_a,
            "classification_accuracy": float(mean_class_acc),
            "classification_n_descriptors": len(class_indices),
            "regression_r2": float(mean_reg_r2),
            "regression_n_descriptors": len(reg_indices),
        },
        "end_to_end": {
            "exact_match": float(exact_match),
            "mean_best_tanimoto": float(mean_tanimoto),
            "validity": float(validity),
            "hit_at_1": float(hit_at_k[1]),
            "hit_at_5": float(hit_at_k[5]),
            "hit_at_10": float(hit_at_k[10]),
            "hit_at_50": float(hit_at_k[50]),
        },
        "config": {
            "n_candidates": args.n_candidates,
            "n_samples": len(smiles_list),
            "temperature": settings.temperature,
            "top_p": args.top_p,
        }
    }

    output_path = settings.metrics_path / "e2e_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    # Save detailed predictions
    print("\nSaving detailed predictions...")
    detailed_results = []
    for i in range(len(smiles_list)):
        # Check if any candidate matches
        true_smiles = smiles_list[i]
        candidates = all_candidates[i]

        # Canonicalize true SMILES
        true_mol = Chem.MolFromSmiles(true_smiles)
        true_canonical = Chem.MolToSmiles(true_mol, canonical=True) if true_mol else true_smiles

        # Find best match and Tanimoto
        exact_matched = False
        best_tanimoto = 0.0
        best_candidate = candidates[0] if candidates else ""

        if true_mol:
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)
            for cand in candidates:
                cand_mol = Chem.MolFromSmiles(cand)
                if cand_mol:
                    cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)
                    if cand_canonical == true_canonical:
                        exact_matched = True
                        best_candidate = cand
                        best_tanimoto = 1.0
                        break
                    cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                    sim = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
                    if sim > best_tanimoto:
                        best_tanimoto = sim
                        best_candidate = cand

        detailed_results.append({
            "index": i,
            "true_smiles": true_smiles,
            "true_canonical": true_canonical,
            "best_candidate": best_candidate,
            "exact_match": exact_matched,
            "best_tanimoto": float(best_tanimoto),
            "n_candidates": len(candidates),
            "all_candidates": candidates[:10],  # Save top 10 candidates
        })

    # Save as JSONL for easy processing
    detailed_path = settings.metrics_path / f"e2e_predictions_{args.part_a}.jsonl"
    with open(detailed_path, "w") as f:
        for item in detailed_results:
            f.write(json.dumps(item) + "\n")

    print(f"Detailed predictions saved to {detailed_path}")

    # Print some examples
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Show some exact matches
    matches = [r for r in detailed_results if r['exact_match']][:3]
    print(f"\nExact Matches ({len([r for r in detailed_results if r['exact_match']])} total):")
    for r in matches:
        print(f"  [{r['index']}] {r['true_smiles'][:50]}...")
        print(f"       → {r['best_candidate'][:50]}...")

    # Show some near matches
    near_matches = sorted([r for r in detailed_results if not r['exact_match']],
                          key=lambda x: x['best_tanimoto'], reverse=True)[:3]
    print(f"\nBest Non-Exact Matches:")
    for r in near_matches:
        print(f"  [{r['index']}] True: {r['true_smiles'][:40]}...")
        print(f"       Pred: {r['best_candidate'][:40]}... (Tanimoto: {r['best_tanimoto']:.3f})")


if __name__ == "__main__":
    main()
