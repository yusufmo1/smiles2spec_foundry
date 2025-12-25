"""
FAST Training script for RL molecule generation.

Key optimizations:
1. Parallel environments (N_ENVS simultaneous episodes)
2. Proper episode reward tracking
3. Stable hyperparameters
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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

for d in [CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "training_fast.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Overwrite log
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_data(dataset: str = "hpj") -> Dict:
    """Load dataset for training."""
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
    from spec2smiles.data.loaders import DataLoader
    from spec2smiles.data.processors import DescriptorCalculator, SpectrumProcessor

    data_dir = SCRIPT_DIR.parent.parent / "smiles2spec" / "data" / "input" / dataset
    data_file = data_dir / "spectral_data.jsonl"
    logger.info(f"Loading data from {data_file}")

    raw_data = DataLoader.load_jsonl(str(data_file))
    logger.info(f"Loaded {len(raw_data)} records")

    spectrum_processor = SpectrumProcessor()
    descriptor_calc = DescriptorCalculator()

    processed = []
    for record in tqdm(raw_data, desc="Processing data"):
        smiles = record.get("smiles")
        peaks = record.get("peaks", [])
        if not smiles or not peaks:
            continue
        spectrum = spectrum_processor.process(peaks)
        if spectrum is None:
            continue
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

    np.random.seed(42)
    indices = np.random.permutation(len(processed))
    n_train = int(0.8 * len(processed))
    n_val = int(0.1 * len(processed))

    return {
        "train": [processed[i] for i in indices[:n_train]],
        "val": [processed[i] for i in indices[n_train:n_train + n_val]],
        "test": [processed[i] for i in indices[n_train + n_val:]],
    }


def build_encoder(smiles_list: List[str]):
    """Build SELFIES encoder."""
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
    from spec2smiles.models.part_b.encoder import SELFIESEncoder

    encoder = SELFIESEncoder(max_len=100)
    encoder.build_vocab_from_smiles(smiles_list, verbose=True)
    logger.info(f"Built vocabulary with {encoder.vocab_size} tokens")
    return encoder


def run_episode(env, agent, data_point: Dict, deterministic: bool = False) -> Dict:
    """Run a single episode and return stats."""
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
    }


def train_fast(
    dataset: str = "hpj",
    n_episodes: int = 5000,
    n_envs: int = 16,  # Parallel environments (reduced for stability)
    batch_size: int = 128,
    rollout_episodes: int = 16,  # Episodes per rollout (not steps)
    lr: float = 5e-5,  # Lower LR for stability
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 3,  # Fewer epochs to prevent overfitting
    eval_interval: int = 100,
    save_interval: int = 500,
    device: str = "auto",
    entropy_coef: float = 0.05,  # Initial entropy coefficient
    entropy_target: float = 2.0,  # Target entropy (log of vocab_size * 0.1)
):
    """Fast training with parallel episode collection."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Parallel envs: {n_envs}")

    data = load_data(dataset)
    logger.info(f"Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")

    all_smiles = [d["smiles"] for d in data["train"]]
    encoder = build_encoder(all_smiles)

    from .environment import MoleculeGameEnv, EnvConfig
    from .agent import MoleculePolicy, PPOAgent, AgentConfig

    env_config = EnvConfig(max_length=100)

    # Create N parallel environments
    envs = [MoleculeGameEnv(encoder, config=env_config) for _ in range(n_envs)]
    eval_env = MoleculeGameEnv(encoder, config=env_config)

    policy = MoleculePolicy(
        vocab_size=encoder.vocab_size,
        spectrum_dim=500,
        descriptor_dim=30,
        hidden_dim=256,
        n_layers=2,
    )

    # Max entropy for vocab_size=44 is log(44)≈3.78
    # Set min_entropy higher to prevent collapse
    agent_config = AgentConfig(
        entropy_coef=entropy_coef,
        entropy_target=entropy_target,
        min_entropy=2.0,  # Higher threshold - about 50% of max entropy
        adaptive_entropy=True,
        clip_eps=0.1,  # Tighter clipping for stability
    )
    agent = PPOAgent(policy, config=agent_config, lr=lr, device=device)

    logger.info("=" * 60)
    logger.info("FAST Training with Adaptive Entropy")
    logger.info(f"Episodes: {n_episodes}, Parallel Envs: {n_envs}")
    logger.info(f"Rollout episodes: {rollout_episodes}, PPO epochs: {n_epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Entropy: coef={entropy_coef}, target={entropy_target}, min=0.5")
    logger.info(f"Adaptive entropy: ON (prevents policy collapse)")
    logger.info("=" * 60)

    best_val_reward = -float('inf')
    episode_count = 0
    all_episode_stats = []
    start_time = time.time()

    while episode_count < n_episodes:
        # Collect rollout_episodes episodes in parallel
        rollout_stats = []

        # Run episodes in parallel batches
        episodes_this_rollout = 0
        while episodes_this_rollout < rollout_episodes:
            # Sample data points for each env
            data_points = [
                data["train"][np.random.randint(len(data["train"]))]
                for _ in range(n_envs)
            ]

            # Run episodes in parallel (each env runs one episode)
            for i, (env, dp) in enumerate(zip(envs, data_points)):
                stats = run_episode(env, agent, dp, deterministic=False)
                rollout_stats.append(stats)
                all_episode_stats.append(stats)
                episodes_this_rollout += 1
                episode_count += 1

                if episode_count >= n_episodes:
                    break

            if episode_count >= n_episodes:
                break

        # PPO update after collecting episodes
        if len(agent.buffer['rewards']) > batch_size:
            update_stats = agent.update(
                n_epochs=n_epochs,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
        else:
            update_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Log progress every 10 episodes
        if episode_count % 10 == 0 and rollout_stats:
            recent = all_episode_stats[-min(100, len(all_episode_stats)):]
            mean_reward = np.mean([s['reward'] for s in recent])
            valid_rate = np.mean([s['valid'] for s in recent])
            tanimoto = np.mean([s['tanimoto'] for s in recent if s['valid']]) if any(s['valid'] for s in recent) else 0
            exact_rate = np.mean([s['exact_match'] for s in recent])
            elapsed = time.time() - start_time
            eps_per_min = episode_count / (elapsed / 60) if elapsed > 0 else 0

            # Get current entropy coefficient
            if agent.log_entropy_coef is not None:
                current_ent_coef = agent.log_entropy_coef.exp().item()
            else:
                current_ent_coef = agent.config.entropy_coef

            logger.info(
                f"Ep {episode_count:5d} | "
                f"Reward: {mean_reward:.3f} | "
                f"Tani: {tanimoto:.3f} | "
                f"Valid: {valid_rate:.1%} | "
                f"Exact: {exact_rate:.1%} | "
                f"Ent: {update_stats['entropy']:.2f} (α={current_ent_coef:.3f}) | "
                f"Speed: {eps_per_min:.1f} ep/min"
            )

        # Evaluation
        if episode_count % eval_interval == 0 and episode_count > 0:
            val_results = []
            for _ in range(50):
                data_point = data["val"][np.random.randint(len(data["val"]))]
                stats = run_episode(eval_env, agent, data_point, deterministic=True)
                val_results.append(stats)

            val_reward = np.mean([r['reward'] for r in val_results])
            val_valid = np.mean([r['valid'] for r in val_results])
            val_tani = np.mean([r['tanimoto'] for r in val_results if r['valid']]) if any(r['valid'] for r in val_results) else 0

            logger.info(f"[EVAL] Ep {episode_count}: Reward={val_reward:.3f}, Valid={val_valid:.1%}, Tani={val_tani:.3f}")

            # Save metrics
            metrics = {
                "episode": episode_count,
                "val_reward": val_reward,
                "val_valid": val_valid,
                "val_tanimoto": val_tani,
            }
            metrics_file = METRICS_DIR / f"episode_{episode_count:06d}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                agent.save(str(CHECKPOINT_DIR / "best_model_fast.pt"))
                logger.info(f"New best! Reward: {best_val_reward:.3f}")

        # Periodic checkpoint
        if episode_count % save_interval == 0 and episode_count > 0:
            agent.save(str(CHECKPOINT_DIR / f"agent_fast_ep{episode_count:06d}.pt"))
            logger.info(f"Saved checkpoint at episode {episode_count}")

    # Final evaluation
    logger.info("=" * 60)
    logger.info("Final evaluation on test set")
    test_results = []
    for dp in tqdm(data["test"][:100], desc="Testing"):
        stats = run_episode(eval_env, agent, dp, deterministic=True)
        test_results.append(stats)

    test_reward = np.mean([r['reward'] for r in test_results])
    test_valid = np.mean([r['valid'] for r in test_results])
    test_tani = np.mean([r['tanimoto'] for r in test_results if r['valid']]) if any(r['valid'] for r in test_results) else 0
    test_exact = np.mean([r['exact_match'] for r in test_results])

    logger.info(f"Test Results:")
    logger.info(f"  Reward: {test_reward:.3f}")
    logger.info(f"  Valid: {test_valid:.1%}")
    logger.info(f"  Tanimoto: {test_tani:.3f}")
    logger.info(f"  Exact Match: {test_exact:.1%}")

    # Save final results
    final_results = {
        "dataset": dataset,
        "n_episodes": n_episodes,
        "test_reward": test_reward,
        "test_valid": test_valid,
        "test_tanimoto": test_tani,
        "test_exact_match": test_exact,
        "best_val_reward": best_val_reward,
        "total_time_minutes": (time.time() - start_time) / 60,
    }
    results_file = RESULTS_DIR / f"fast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fast RL training with adaptive entropy")
    parser.add_argument("--dataset", type=str, default="hpj", choices=["hpj", "GNPS"])
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rollout_episodes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="Initial entropy coefficient")
    parser.add_argument("--entropy_target", type=float, default=2.0, help="Target entropy level")

    args = parser.parse_args()

    train_fast(
        dataset=args.dataset,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        rollout_episodes=args.rollout_episodes,
        lr=args.lr,
        n_epochs=args.n_epochs,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
        entropy_coef=args.entropy_coef,
        entropy_target=args.entropy_target,
    )


if __name__ == "__main__":
    main()
