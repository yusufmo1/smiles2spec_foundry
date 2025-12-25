"""
Evaluation script for trained RL agents.

Loads a trained model and evaluates on test data with detailed metrics.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"


def compute_tanimoto(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two molecules."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def is_exact_match(smiles1: str, smiles2: str) -> bool:
    """Check if two SMILES represent the same molecule."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return False
        canon1 = Chem.MolToSmiles(mol1, canonical=True)
        canon2 = Chem.MolToSmiles(mol2, canonical=True)
        return canon1 == canon2
    except Exception:
        return False


def evaluate_agent(
    agent,
    env,
    test_data: List[Dict],
    n_candidates: int = 1,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive evaluation of trained agent.

    Args:
        agent: Trained PPOAgent
        env: MoleculeGameEnv
        test_data: List of test data points
        n_candidates: Number of candidates to generate per spectrum
        deterministic: Use greedy decoding
        verbose: Show progress bar

    Returns:
        Dictionary with detailed metrics
    """
    results = []

    iterator = tqdm(test_data, desc="Evaluating") if verbose else test_data

    for data_point in iterator:
        target_smiles = data_point["smiles"]
        spectrum = data_point["spectrum"]
        descriptors = data_point["descriptors"]

        candidates = []
        for _ in range(n_candidates):
            # Generate molecule
            obs = env.reset(
                target_spectrum=spectrum,
                target_descriptors=descriptors,
                target_smiles=target_smiles,
            )

            done = False
            while not done:
                action, _, _ = agent.select_action(obs, deterministic=deterministic)
                obs, _, done, info = env.step(action)

            generated_smiles = info.get("smiles")
            if generated_smiles:
                candidates.append(generated_smiles)

        # Compute metrics for this sample
        sample_result = {
            "target_smiles": target_smiles,
            "candidates": candidates,
            "valid": len(candidates) > 0,
        }

        if candidates:
            # Tanimoto scores
            tanimoto_scores = [compute_tanimoto(c, target_smiles) for c in candidates]
            sample_result["best_tanimoto"] = max(tanimoto_scores)
            sample_result["mean_tanimoto"] = np.mean(tanimoto_scores)

            # Exact match
            sample_result["exact_match"] = any(
                is_exact_match(c, target_smiles) for c in candidates
            )

            # Uniqueness
            unique_canons = set()
            for c in candidates:
                mol = Chem.MolFromSmiles(c)
                if mol:
                    unique_canons.add(Chem.MolToSmiles(mol, canonical=True))
            sample_result["n_unique"] = len(unique_canons)
        else:
            sample_result["best_tanimoto"] = 0.0
            sample_result["mean_tanimoto"] = 0.0
            sample_result["exact_match"] = False
            sample_result["n_unique"] = 0

        results.append(sample_result)

    # Aggregate metrics
    n_samples = len(results)
    valid_results = [r for r in results if r["valid"]]

    metrics = {
        "n_samples": n_samples,
        "n_candidates": n_candidates,
        "validity_rate": len(valid_results) / n_samples if n_samples > 0 else 0.0,
        "exact_match_rate": np.mean([r["exact_match"] for r in results]),
        "mean_tanimoto": np.mean([r["mean_tanimoto"] for r in valid_results]) if valid_results else 0.0,
        "best_tanimoto_mean": np.mean([r["best_tanimoto"] for r in valid_results]) if valid_results else 0.0,
        "tanimoto_std": np.std([r["best_tanimoto"] for r in valid_results]) if valid_results else 0.0,
    }

    # Hit@K metrics (for multiple candidates)
    if n_candidates > 1:
        for k in [1, 5, 10, 20, 50]:
            if k <= n_candidates:
                # Hit@K: was the exact match in top K?
                hits = sum(
                    1 for r in results
                    if r["valid"] and any(
                        is_exact_match(c, r["target_smiles"])
                        for c in r["candidates"][:k]
                    )
                )
                metrics[f"hit_at_{k}"] = hits / n_samples if n_samples > 0 else 0.0

    # Tanimoto percentiles
    if valid_results:
        tanimotos = [r["best_tanimoto"] for r in valid_results]
        for p in [25, 50, 75, 90, 95]:
            metrics[f"tanimoto_p{p}"] = np.percentile(tanimotos, p)

    return metrics, results


def load_agent(checkpoint_path: str, device: str = "cpu"):
    """Load trained agent from checkpoint."""
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

    from .agent import MoleculePolicy, PPOAgent, AgentConfig
    from spec2smiles.models.part_b.encoder import SELFIESEncoder

    # Load encoder
    encoder_path = Path(checkpoint_path).parent / Path(checkpoint_path).name.replace("agent", "encoder").replace(".pt", ".pkl")
    if encoder_path.exists():
        with open(encoder_path, 'rb') as f:
            encoder_state = pickle.load(f)
        encoder = SELFIESEncoder.from_state(encoder_state)
    else:
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create policy
    policy = MoleculePolicy(
        vocab_size=encoder.vocab_size,
        spectrum_dim=500,
        descriptor_dim=30,
        hidden_dim=256,
        n_layers=2,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device)

    # Create agent
    agent = PPOAgent(policy, device=device)

    return agent, encoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint")
    parser.add_argument("--dataset", type=str, default="hpj", choices=["hpj", "GNPS"])
    parser.add_argument("--n_candidates", type=int, default=50, help="Candidates per spectrum")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Load agent
    print(f"Loading agent from {args.checkpoint}")
    agent, encoder = load_agent(args.checkpoint, args.device)

    # Load test data
    from .train import load_data
    data = load_data(args.dataset)
    test_data = data["test"]
    print(f"Loaded {len(test_data)} test samples")

    # Create environment
    from .environment import MoleculeGameEnv, EnvConfig
    env_config = EnvConfig(max_length=100)
    env = MoleculeGameEnv(encoder, config=env_config)

    # Evaluate
    print(f"Evaluating with {args.n_candidates} candidates per spectrum...")
    metrics, results = evaluate_agent(
        agent, env, test_data,
        n_candidates=args.n_candidates,
        deterministic=False,  # Use sampling for multiple candidates
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']}")
    print(f"Candidates per sample: {metrics['n_candidates']}")
    print(f"\nValidity Rate: {metrics['validity_rate']:.1%}")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.1%}")
    print(f"Mean Tanimoto: {metrics['mean_tanimoto']:.4f}")
    print(f"Best Tanimoto (mean): {metrics['best_tanimoto_mean']:.4f}")

    if 'hit_at_10' in metrics:
        print(f"\nHit@1: {metrics.get('hit_at_1', 0):.1%}")
        print(f"Hit@5: {metrics.get('hit_at_5', 0):.1%}")
        print(f"Hit@10: {metrics.get('hit_at_10', 0):.1%}")
        print(f"Hit@50: {metrics.get('hit_at_50', 0):.1%}")

    print(f"\nTanimoto Percentiles:")
    for p in [25, 50, 75, 90, 95]:
        key = f"tanimoto_p{p}"
        if key in metrics:
            print(f"  P{p}: {metrics[key]:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "results" / f"eval_{Path(args.checkpoint).stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "metrics": metrics,
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "n_candidates": args.n_candidates,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
