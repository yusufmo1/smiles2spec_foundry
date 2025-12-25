"""
Training script for RL molecule generation.

Trains a PPO agent to build molecules that match target spectra.
Outputs are written to outputs/ directory for live monitoring via Mutagen.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
for d in [CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_data(dataset: str = "hpj") -> Dict:
    """
    Load dataset for training.

    Args:
        dataset: Dataset name ("hpj" or "GNPS")

    Returns:
        Dictionary with train/val/test splits
    """
    # Add parent directory to path for imports
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

    from spec2smiles.data.loaders import DataLoader
    from spec2smiles.data.processors import DescriptorCalculator, SpectrumProcessor

    data_dir = SCRIPT_DIR.parent.parent / "smiles2spec" / "data" / "input" / dataset
    data_file = data_dir / "spectral_data.jsonl"

    logger.info(f"Loading data from {data_file}")

    # Load raw data
    raw_data = DataLoader.load_jsonl(str(data_file))
    logger.info(f"Loaded {len(raw_data)} records")

    # Initialize processors
    spectrum_processor = SpectrumProcessor()
    descriptor_calc = DescriptorCalculator()

    # Process data
    processed = []
    for record in tqdm(raw_data, desc="Processing data"):
        smiles = record.get("smiles")
        peaks = record.get("peaks", [])

        if not smiles or not peaks:
            continue

        # Process spectrum
        spectrum = spectrum_processor.process(peaks)
        if spectrum is None:
            continue

        # Calculate descriptors
        try:
            descriptors = descriptor_calc.calculate(smiles)
            if descriptors is None:
                continue
        except Exception:
            continue

        processed.append({
            "smiles": smiles,
            "spectrum": spectrum,
            "descriptors": descriptors,
        })

    logger.info(f"Processed {len(processed)} valid records")

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(processed))

    n_train = int(0.8 * len(processed))
    n_val = int(0.1 * len(processed))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [processed[i] for i in train_idx],
        "val": [processed[i] for i in val_idx],
        "test": [processed[i] for i in test_idx],
    }


def build_encoder(smiles_list: List[str]):
    """Build SELFIES encoder from SMILES list."""
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
    from spec2smiles.models.part_b.encoder import SELFIESEncoder

    encoder = SELFIESEncoder(max_len=100)
    encoder.build_vocab_from_smiles(smiles_list, verbose=True)

    logger.info(f"Built vocabulary with {encoder.vocab_size} tokens")
    return encoder


def sample_batch(data: List[Dict], batch_size: int) -> List[Dict]:
    """Sample a batch of data."""
    indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)
    return [data[i] for i in indices]


def run_episode(env, agent, data_point: Dict, deterministic: bool = False) -> Dict:
    """
    Run a single episode.

    Args:
        env: MoleculeGameEnv
        agent: PPOAgent
        data_point: Dictionary with spectrum, descriptors, smiles
        deterministic: Whether to use greedy actions

    Returns:
        Episode statistics
    """
    obs = env.reset(
        target_spectrum=data_point["spectrum"],
        target_descriptors=data_point["descriptors"],
        target_smiles=data_point["smiles"],
    )

    total_reward = 0
    steps = 0

    while True:
        action, log_prob, value = agent.select_action(obs, deterministic=deterministic)
        next_obs, reward, done, info = env.step(action)

        if not deterministic:
            agent.store_transition(obs, action, log_prob, reward, value, done)

        total_reward += reward
        steps += 1
        obs = next_obs

        if done:
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "valid": info.get("valid", False),
        "exact_match": info.get("exact_match", False),
        "tanimoto": info.get("tanimoto", 0.0),
        "smiles": info.get("smiles"),
        "target_smiles": data_point["smiles"],
    }


def evaluate(env, agent, data: List[Dict], n_samples: int = 100) -> Dict:
    """
    Evaluate agent on validation/test data.

    Args:
        env: Environment
        agent: Agent
        data: List of data points
        n_samples: Number of samples to evaluate

    Returns:
        Evaluation metrics
    """
    results = []
    sample_data = sample_batch(data, n_samples)

    for data_point in sample_data:
        episode_result = run_episode(env, agent, data_point, deterministic=True)
        results.append(episode_result)

    # Aggregate metrics
    metrics = {
        "mean_reward": np.mean([r["reward"] for r in results]),
        "validity_rate": np.mean([r["valid"] for r in results]),
        "exact_match_rate": np.mean([r["exact_match"] for r in results]),
        "mean_tanimoto": np.mean([r["tanimoto"] for r in results if r["valid"]]) if any(r["valid"] for r in results) else 0.0,
        "mean_steps": np.mean([r["steps"] for r in results]),
    }

    return metrics


def save_metrics(episode: int, metrics: Dict):
    """Save metrics to JSON file."""
    metrics_file = METRICS_DIR / f"episode_{episode:06d}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_checkpoint(agent, encoder, episode: int):
    """Save agent checkpoint."""
    ckpt_path = CHECKPOINT_DIR / f"agent_ep{episode:06d}.pt"
    agent.save(str(ckpt_path))

    # Also save encoder
    encoder_path = CHECKPOINT_DIR / f"encoder_ep{episode:06d}.pkl"
    import pickle
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder.get_state(), f)

    logger.info(f"Saved checkpoint at episode {episode}")


def train(
    dataset: str = "hpj",
    n_episodes: int = 10000,
    batch_size: int = 32,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 4,
    eval_interval: int = 100,
    save_interval: int = 500,
    device: str = "auto",
):
    """
    Main training loop.

    Args:
        dataset: Dataset to use ("hpj" or "GNPS")
        n_episodes: Number of training episodes
        batch_size: Batch size for PPO updates
        rollout_steps: Steps to collect before each update
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        n_epochs: PPO epochs per update
        eval_interval: Episodes between evaluations
        save_interval: Episodes between checkpoints
        device: Device to use ("auto", "cuda", "cpu", "mps")
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    data = load_data(dataset)
    logger.info(f"Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")

    # Build encoder
    all_smiles = [d["smiles"] for d in data["train"]]
    encoder = build_encoder(all_smiles)

    # Create environment and agent
    from .environment import MoleculeGameEnv, EnvConfig
    from .agent import MoleculePolicy, PPOAgent, AgentConfig

    env_config = EnvConfig(max_length=100)
    env = MoleculeGameEnv(encoder, config=env_config)

    policy = MoleculePolicy(
        vocab_size=encoder.vocab_size,
        spectrum_dim=500,
        descriptor_dim=30,
        hidden_dim=256,
        n_layers=2,
    )

    agent_config = AgentConfig()
    agent = PPOAgent(policy, config=agent_config, lr=lr, device=device)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"Episodes: {n_episodes}, Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}, Gamma: {gamma}")
    logger.info("=" * 60)

    best_val_reward = -float('inf')
    episode_rewards = []
    start_time = time.time()

    for episode in range(1, n_episodes + 1):
        # Collect rollouts
        episode_stats = []
        steps_collected = 0

        while steps_collected < rollout_steps:
            # Sample random data point
            data_point = sample_batch(data["train"], 1)[0]

            # Run episode
            stats = run_episode(env, agent, data_point, deterministic=False)
            episode_stats.append(stats)
            steps_collected += stats["steps"]

        # PPO update
        update_stats = agent.update(
            n_epochs=n_epochs,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Track episode statistics
        mean_reward = np.mean([s["reward"] for s in episode_stats])
        mean_tanimoto = np.mean([s["tanimoto"] for s in episode_stats if s["valid"]]) if any(s["valid"] for s in episode_stats) else 0.0
        valid_rate = np.mean([s["valid"] for s in episode_stats])
        exact_match_rate = np.mean([s["exact_match"] for s in episode_stats])

        episode_rewards.append(mean_reward)

        # Log progress
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Ep {episode:5d} | "
                f"Reward: {mean_reward:.3f} | "
                f"Tanimoto: {mean_tanimoto:.3f} | "
                f"Valid: {valid_rate:.1%} | "
                f"Exact: {exact_match_rate:.1%} | "
                f"PL: {update_stats['policy_loss']:.4f} | "
                f"VL: {update_stats['value_loss']:.4f} | "
                f"Ent: {update_stats['entropy']:.3f} | "
                f"Time: {elapsed/60:.1f}m"
            )

        # Periodic evaluation
        if episode % eval_interval == 0:
            val_metrics = evaluate(env, agent, data["val"], n_samples=100)
            logger.info(f"[EVAL] Episode {episode}:")
            logger.info(f"  Reward: {val_metrics['mean_reward']:.3f}")
            logger.info(f"  Tanimoto: {val_metrics['mean_tanimoto']:.3f}")
            logger.info(f"  Valid: {val_metrics['validity_rate']:.1%}")
            logger.info(f"  Exact Match: {val_metrics['exact_match_rate']:.1%}")

            # Save metrics
            metrics = {
                "episode": episode,
                "train": {
                    "mean_reward": mean_reward,
                    "mean_tanimoto": mean_tanimoto,
                    "validity_rate": valid_rate,
                    "exact_match_rate": exact_match_rate,
                },
                "val": val_metrics,
                "update": update_stats,
                "elapsed_minutes": elapsed / 60,
            }
            save_metrics(episode, metrics)

            # Save best model
            if val_metrics["mean_reward"] > best_val_reward:
                best_val_reward = val_metrics["mean_reward"]
                save_checkpoint(agent, encoder, episode)
                logger.info(f"New best model! Reward: {best_val_reward:.3f}")

        # Periodic checkpoint
        if episode % save_interval == 0:
            save_checkpoint(agent, encoder, episode)

    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("Final evaluation on test set")
    test_metrics = evaluate(env, agent, data["test"], n_samples=len(data["test"]))
    logger.info(f"Test Results:")
    logger.info(f"  Mean Reward: {test_metrics['mean_reward']:.3f}")
    logger.info(f"  Mean Tanimoto: {test_metrics['mean_tanimoto']:.3f}")
    logger.info(f"  Validity Rate: {test_metrics['validity_rate']:.1%}")
    logger.info(f"  Exact Match Rate: {test_metrics['exact_match_rate']:.1%}")

    # Save final results
    final_results = {
        "dataset": dataset,
        "n_episodes": n_episodes,
        "test_metrics": test_metrics,
        "training_time_minutes": (time.time() - start_time) / 60,
        "best_val_reward": best_val_reward,
    }
    results_file = RESULTS_DIR / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for molecule generation")
    parser.add_argument("--dataset", type=str, default="hpj", choices=["hpj", "GNPS"])
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    train(
        dataset=args.dataset,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
