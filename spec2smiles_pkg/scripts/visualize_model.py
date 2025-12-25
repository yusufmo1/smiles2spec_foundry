#!/usr/bin/env python
"""Visualize model performance with detailed plots."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spec2smiles.data.datasets import MolecularDataset
from spec2smiles.models.part_b.encoder import SELFIESEncoder
from spec2smiles.models.part_b.direct_decoder import DirectDecoder


def evaluate_and_collect(
    model: DirectDecoder,
    encoder: SELFIESEncoder,
    test_loader: DataLoader,
    test_selfies: list,
    test_smiles: list,
    device: torch.device,
    beam_width: int = 40,
    n_return: int = 100,
) -> dict:
    """Evaluate model and collect detailed results for visualization."""
    model.eval()

    results = {
        'exact_matches': [],
        'formula_matches': [],
        'tanimoto_scores': [],
        'true_smiles': [],
        'pred_smiles': [],
        'true_mw': [],
        'pred_mw': [],
        'rank_of_correct': [],  # Rank where correct structure found (0 if not found)
    }

    selfies_idx = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        descriptors = batch["descriptors"].to(device)
        batch_size = descriptors.size(0)

        candidates = model.generate_beam(
            descriptors,
            beam_width=beam_width,
            n_return=n_return,
            length_penalty=0.6,
        )

        for i in range(batch_size):
            if selfies_idx >= len(test_selfies):
                break

            true_selfies = test_selfies[selfies_idx]
            true_smiles = encoder.selfies_to_smiles(true_selfies)

            if true_smiles is None:
                selfies_idx += 1
                continue

            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                selfies_idx += 1
                continue

            true_canonical = Chem.MolToSmiles(true_mol, canonical=True)
            true_formula = Chem.rdMolDescriptors.CalcMolFormula(true_mol)
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2)
            true_mw = Descriptors.MolWt(true_mol)

            best_tanimoto = 0.0
            best_pred_smiles = None
            best_pred_mw = None
            found_exact = False
            found_formula = False
            rank_correct = 0

            for rank, cand_tokens in enumerate(candidates):
                cand_smiles = encoder.decode(cand_tokens[i].cpu().numpy().tolist())
                if cand_smiles is None:
                    continue

                cand_mol = Chem.MolFromSmiles(cand_smiles)
                if cand_mol is None:
                    continue

                cand_canonical = Chem.MolToSmiles(cand_mol, canonical=True)

                # Track best prediction (first valid one)
                if best_pred_smiles is None:
                    best_pred_smiles = cand_canonical
                    best_pred_mw = Descriptors.MolWt(cand_mol)

                if cand_canonical == true_canonical:
                    found_exact = True
                    if rank_correct == 0:
                        rank_correct = rank + 1

                cand_formula = Chem.rdMolDescriptors.CalcMolFormula(cand_mol)
                if cand_formula == true_formula:
                    found_formula = True

                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2)
                tanimoto = DataStructs.TanimotoSimilarity(true_fp, cand_fp)
                best_tanimoto = max(best_tanimoto, tanimoto)

            results['exact_matches'].append(found_exact)
            results['formula_matches'].append(found_formula)
            results['tanimoto_scores'].append(best_tanimoto)
            results['true_smiles'].append(true_canonical)
            results['pred_smiles'].append(best_pred_smiles or "")
            results['true_mw'].append(true_mw)
            results['pred_mw'].append(best_pred_mw or 0)
            results['rank_of_correct'].append(rank_correct)

            selfies_idx += 1

    return results


def create_visualizations(results: dict, output_dir: Path, model_name: str):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_size = (10, 6)

    # 1. Tanimoto Distribution
    fig, ax = plt.subplots(figsize=fig_size)
    tanimoto = np.array(results['tanimoto_scores'])
    ax.hist(tanimoto, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(tanimoto), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(tanimoto):.3f}')
    ax.axvline(np.median(tanimoto), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(tanimoto):.3f}')
    ax.set_xlabel('Tanimoto Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{model_name}: Tanimoto Similarity Distribution', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'tanimoto_distribution.png', dpi=150)
    plt.close()

    # 2. Molecular Weight Correlation
    fig, ax = plt.subplots(figsize=fig_size)
    true_mw = np.array(results['true_mw'])
    pred_mw = np.array([mw for mw in results['pred_mw'] if mw > 0])
    true_mw_valid = np.array([results['true_mw'][i] for i in range(len(results['pred_mw']))
                              if results['pred_mw'][i] > 0])

    if len(pred_mw) > 0:
        ax.scatter(true_mw_valid, pred_mw, alpha=0.5, s=20, c='steelblue')
        max_mw = max(true_mw_valid.max(), pred_mw.max())
        ax.plot([0, max_mw], [0, max_mw], 'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('True Molecular Weight (Da)', fontsize=12)
        ax.set_ylabel('Predicted Molecular Weight (Da)', fontsize=12)
        ax.set_title(f'{model_name}: Molecular Weight Prediction', fontsize=14)
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'mw_correlation.png', dpi=150)
    plt.close()

    # 3. Performance by Molecular Weight
    fig, ax = plt.subplots(figsize=fig_size)
    mw_bins = [0, 150, 200, 250, 300, 350, 400, 500]
    mw_labels = ['<150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-500']

    exact = np.array(results['exact_matches'])
    mw = np.array(results['true_mw'])

    bin_exact_rates = []
    bin_counts = []
    for i in range(len(mw_bins) - 1):
        mask = (mw >= mw_bins[i]) & (mw < mw_bins[i+1])
        if mask.sum() > 0:
            bin_exact_rates.append(exact[mask].mean() * 100)
            bin_counts.append(mask.sum())
        else:
            bin_exact_rates.append(0)
            bin_counts.append(0)

    bars = ax.bar(mw_labels, bin_exact_rates, color='steelblue', edgecolor='black')
    ax.set_xlabel('Molecular Weight Range (Da)', fontsize=12)
    ax.set_ylabel('Exact Match Rate (%)', fontsize=12)
    ax.set_title(f'{model_name}: Performance by Molecular Weight', fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_mw.png', dpi=150)
    plt.close()

    # 4. Rank Distribution (where correct answer found)
    fig, ax = plt.subplots(figsize=fig_size)
    ranks = np.array(results['rank_of_correct'])
    found_ranks = ranks[ranks > 0]

    if len(found_ranks) > 0:
        ax.hist(found_ranks, bins=range(1, 52), edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Rank of Correct Structure', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{model_name}: Rank Distribution of Correct Structures\n'
                     f'(Found in {len(found_ranks)}/{len(ranks)} = {100*len(found_ranks)/len(ranks):.1f}%)',
                     fontsize=14)
        ax.set_xlim(0, 50)
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_distribution.png', dpi=150)
    plt.close()

    # 5. Cumulative Hit Rate (Hit@K curve)
    fig, ax = plt.subplots(figsize=fig_size)
    ranks = np.array(results['rank_of_correct'])
    n_total = len(ranks)

    k_values = list(range(1, 101))
    hit_rates = []
    for k in k_values:
        hits = ((ranks > 0) & (ranks <= k)).sum()
        hit_rates.append(100 * hits / n_total)

    ax.plot(k_values, hit_rates, linewidth=2, color='steelblue')
    ax.axhline(hit_rates[0], color='green', linestyle=':', alpha=0.7, label=f'Hit@1: {hit_rates[0]:.1f}%')
    ax.axhline(hit_rates[4], color='orange', linestyle=':', alpha=0.7, label=f'Hit@5: {hit_rates[4]:.1f}%')
    ax.axhline(hit_rates[9], color='red', linestyle=':', alpha=0.7, label=f'Hit@10: {hit_rates[9]:.1f}%')
    ax.set_xlabel('K (Top-K Candidates)', fontsize=12)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title(f'{model_name}: Cumulative Hit@K Performance', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / 'hit_at_k_curve.png', dpi=150)
    plt.close()

    # 6. Summary metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = {
        'Exact Match': np.mean(results['exact_matches']) * 100,
        'Formula Match': np.mean(results['formula_matches']) * 100,
        'Mean Tanimoto': np.mean(results['tanimoto_scores']) * 100,
        'Median Tanimoto': np.median(results['tanimoto_scores']) * 100,
    }

    bars = ax.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
                  edgecolor='black')
    ax.set_ylabel('Percentage / Score × 100', fontsize=12)
    ax.set_title(f'{model_name}: Summary Metrics', fontsize=14)
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_metrics.png', dpi=150)
    plt.close()

    print(f"\nVisualization saved to {output_dir}/")
    print(f"  - tanimoto_distribution.png")
    print(f"  - mw_correlation.png")
    print(f"  - performance_by_mw.png")
    print(f"  - rank_distribution.png")
    print(f"  - hit_at_k_curve.png")
    print(f"  - summary_metrics.png")


def main():
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Paths - use the 6x augmented model
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data_30desc"
    model_dir = script_dir / "models_30desc" / "direct_decoder_aug"
    output_dir = script_dir / "visualizations" / "direct_decoder_aug"

    model_name = "DirectDecoder (6x Augmented)"

    print(f"Loading from: {model_dir}")

    # Load test data
    print("Loading test data...")
    with open(data_dir / "test_data.jsonl") as f:
        test_data = [json.loads(line) for line in f]

    # Load encoder from the augmented model
    encoder_path = model_dir / "encoder.pkl"
    if not encoder_path.exists():
        # Try loading from 10x model
        encoder_path = script_dir / "models_30desc" / "direct_decoder_aug10x" / "encoder.pkl"
    if not encoder_path.exists():
        # Try data dir
        encoder_path = data_dir / "encoder.pkl"

    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            encoder_state = pickle.load(f)
        encoder = SELFIESEncoder(max_len=100)
        encoder.set_state(encoder_state)
    else:
        print("Building encoder from test data...")
        encoder = SELFIESEncoder(max_len=100)
        test_smiles = [d["smiles"] for d in test_data]
        encoder.build_vocab_from_smiles(test_smiles)

    # Get test SELFIES
    test_smiles = [d["smiles"] for d in test_data]
    test_selfies = []
    test_valid = []
    for i, smi in enumerate(test_smiles):
        sf = encoder.smiles_to_selfies(smi)
        if sf is not None:
            test_selfies.append(sf)
            test_valid.append(i)
    test_data_valid = [test_data[i] for i in test_valid]
    test_smiles_valid = [test_smiles[i] for i in test_valid]

    # Extract descriptors
    X_test = np.array([d["descriptors_scaled"] for d in test_data_valid])

    # Create dataset
    test_dataset = MolecularDataset(
        X_test, test_smiles_valid,
        encoder=encoder, max_len=100
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    print("Loading model...")
    model = DirectDecoder.load(model_dir / "model.pt", device=device)
    model.eval()

    print(f"\nTest set: {len(test_selfies)} molecules")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Evaluate and collect results
    print("\nEvaluating with beam search (beam=40, return=100)...")
    results = evaluate_and_collect(
        model, encoder, test_loader, test_selfies, test_smiles_valid,
        device=device, beam_width=40, n_return=100
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Exact match rate: {100*np.mean(results['exact_matches']):.1f}%")
    print(f"  Formula match rate: {100*np.mean(results['formula_matches']):.1f}%")
    print(f"  Mean Tanimoto: {np.mean(results['tanimoto_scores']):.3f}")
    print(f"  Median Tanimoto: {np.median(results['tanimoto_scores']):.3f}")
    print(f"{'='*60}")

    # Create visualizations
    create_visualizations(results, output_dir, model_name)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'exact_match_rate': float(np.mean(results['exact_matches'])),
            'formula_match_rate': float(np.mean(results['formula_matches'])),
            'mean_tanimoto': float(np.mean(results['tanimoto_scores'])),
            'median_tanimoto': float(np.median(results['tanimoto_scores'])),
            'total': len(results['exact_matches']),
        }, f, indent=2)


if __name__ == "__main__":
    main()
