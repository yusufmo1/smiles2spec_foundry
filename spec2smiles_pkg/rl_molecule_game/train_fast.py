"""
FAST Training script for RL molecule generation.

Key optimizations:
1. Parallel environments (N_ENVS simultaneous episodes)
2. Batched policy inference
3. Reduced rollout steps
4. Fewer PPO epochs
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

for d in [CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "training_fast.log"
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


class ParallelEnvs:
    """Manages N parallel environments for batched rollouts."""

    def __init__(self, env_class, encoder, config, n_envs: int, data: List[Dict]):
        self.n_envs = n_envs
        self.data = data
        self.envs = [env_class(encoder, config=config) for _ in range(n_envs)]
        self.encoder = encoder

        # Current state for each env
        self.obs = [None] * n_envs
        self.data_points = [None] * n_envs
        self.dones = [True] * n_envs  # Start as done to trigger reset

    def reset_env(self, idx: int):
        """Reset a single environment with random data point."""
        data_point = self.data[np.random.randint(len(self.data))]
        self.data_points[idx] = data_point
        self.obs[idx] = self.envs[idx].reset(
            target_spectrum=data_point["spectrum"],
            target_descriptors=data_point["descriptors"],
            target_smiles=data_point["smiles"],
        )
        self.dones[idx] = False
        return self.obs[idx]

    def step_all(self, actions: List[int]) -> tuple:
        """Step all environments, auto-resetting done ones."""
        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []

        for i in range(self.n_envs):
            if self.dones[i]:
                # Reset this env
                obs = self.reset_env(i)
                all_obs.append(obs)
                all_rewards.append(0.0)
                all_dones.append(False)
                all_infos.append({})
            else:
                obs, reward, done, info = self.envs[i].step(actions[i])
                self.obs[i] = obs
                self.dones[i] = done
                all_obs.append(obs)
                all_rewards.append(reward)
                all_dones.append(done)
                all_infos.append(info)

        return all_obs, all_rewards, all_dones, all_infos

    def get_batch_obs(self) -> Dict[str, torch.Tensor]:
        """Get batched observations for all envs."""
        # Stack observations
        tokens = torch.stack([torch.tensor(o['tokens']) for o in self.obs])
        spectrum = torch.stack([torch.tensor(o['spectrum']) for o in self.obs])
        descriptors = torch.stack([torch.tensor(o['descriptors']) for o in self.obs])

        return {
            'tokens': tokens,
            'spectrum': spectrum,
            'descriptors': descriptors,
        }


def train_fast(
    dataset: str = "hpj",
    n_episodes: int = 10000,
    n_envs: int = 32,  # Parallel environments
    batch_size: int = 256,
    rollout_steps: int = 512,  # Reduced from 2048
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 2,  # Reduced from 4
    eval_interval: int = 50,  # More frequent
    save_interval: int = 200,
    device: str = "auto",
):
    """Fast training with parallel envs."""
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

    # Create parallel environments
    parallel_envs = ParallelEnvs(
        MoleculeGameEnv, encoder, env_config,
        n_envs=n_envs, data=data["train"]
    )

    # Single env for evaluation
    eval_env = MoleculeGameEnv(encoder, config=env_config)

    policy = MoleculePolicy(
        vocab_size=encoder.vocab_size,
        spectrum_dim=500,
        descriptor_dim=30,
        hidden_dim=256,
        n_layers=2,
    )

    agent_config = AgentConfig()
    agent = PPOAgent(policy, config=agent_config, lr=lr, device=device)

    logger.info("=" * 60)
    logger.info("FAST Training with Parallel Environments")
    logger.info(f"Episodes: {n_episodes}, Parallel Envs: {n_envs}")
    logger.info(f"Rollout steps: {rollout_steps}, PPO epochs: {n_epochs}")
    logger.info("=" * 60)

    # Initialize all envs
    for i in range(n_envs):
        parallel_envs.reset_env(i)

    best_val_reward = -float('inf')
    episode_count = 0
    total_steps = 0
    start_time = time.time()

    episode_stats = []

    while episode_count < n_episodes:
        # Collect rollout from parallel envs
        rollout_rewards = [[] for _ in range(n_envs)]
        rollout_valids = [[] for _ in range(n_envs)]

        for step in range(rollout_steps // n_envs):
            # Get batched observations
            batch_obs = parallel_envs.get_batch_obs()

            # Batched action selection
            with torch.no_grad():
                tokens = batch_obs['tokens'].to(device)
                spectrum = batch_obs['spectrum'].float().to(device)
                descriptors = batch_obs['descriptors'].float().to(device)

                # Get actions for all envs at once
                actions = []
                log_probs = []
                values = []

                for i in range(n_envs):
                    obs = parallel_envs.obs[i]
                    action, log_prob, value = agent.select_action(obs, deterministic=False)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)

            # Step all environments
            all_obs, all_rewards, all_dones, all_infos = parallel_envs.step_all(actions)
            total_steps += n_envs

            # Store transitions
            for i in range(n_envs):
                if not all_dones[i] or all_rewards[i] != 0:  # Skip auto-reset steps
                    agent.store_transition(
                        parallel_envs.obs[i], actions[i], log_probs[i],
                        all_rewards[i], values[i], all_dones[i]
                    )

                if all_dones[i] and all_infos[i]:
                    rollout_rewards[i].append(sum(all_rewards))
                    rollout_valids[i].append(all_infos[i].get('valid', False))
                    episode_count += 1

                    episode_stats.append({
                        'reward': all_rewards[i],
                        'valid': all_infos[i].get('valid', False),
                        'tanimoto': all_infos[i].get('tanimoto', 0.0),
                        'exact_match': all_infos[i].get('exact_match', False),
                    })

        # PPO update
        if len(agent.buffer['rewards']) > batch_size:
            update_stats = agent.update(
                n_epochs=n_epochs,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
        else:
            update_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Log progress
        if episode_count % 10 == 0 and episode_stats:
            recent = episode_stats[-min(100, len(episode_stats)):]
            mean_reward = np.mean([s['reward'] for s in recent])
            valid_rate = np.mean([s['valid'] for s in recent])
            tanimoto = np.mean([s['tanimoto'] for s in recent if s['valid']]) if any(s['valid'] for s in recent) else 0
            exact_rate = np.mean([s['exact_match'] for s in recent])
            elapsed = time.time() - start_time
            eps_per_min = episode_count / (elapsed / 60) if elapsed > 0 else 0

            logger.info(
                f"Ep {episode_count:5d} | "
                f"Reward: {mean_reward:.3f} | "
                f"Tani: {tanimoto:.3f} | "
                f"Valid: {valid_rate:.1%} | "
                f"Exact: {exact_rate:.1%} | "
                f"Speed: {eps_per_min:.1f} ep/min | "
                f"Time: {elapsed/60:.1f}m"
            )

        # Evaluation
        if episode_count % eval_interval == 0 and episode_count > 0:
            val_results = []
            for _ in range(50):  # Quick eval
                data_point = data["val"][np.random.randint(len(data["val"]))]
                obs = eval_env.reset(
                    target_spectrum=data_point["spectrum"],
                    target_descriptors=data_point["descriptors"],
                    target_smiles=data_point["smiles"],
                )
                total_reward = 0
                while True:
                    action, _, _ = agent.select_action(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    total_reward += reward
                    if done:
                        break
                val_results.append({
                    'reward': total_reward,
                    'valid': info.get('valid', False),
                    'tanimoto': info.get('tanimoto', 0.0),
                })

            val_reward = np.mean([r['reward'] for r in val_results])
            val_valid = np.mean([r['valid'] for r in val_results])
            val_tani = np.mean([r['tanimoto'] for r in val_results if r['valid']]) if any(r['valid'] for r in val_results) else 0

            logger.info(f"[EVAL] Ep {episode_count}: Reward={val_reward:.3f}, Valid={val_valid:.1%}, Tani={val_tani:.3f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                agent.save(str(CHECKPOINT_DIR / "best_model.pt"))
                logger.info(f"New best! Reward: {best_val_reward:.3f}")

        # Save checkpoint
        if episode_count % save_interval == 0 and episode_count > 0:
            agent.save(str(CHECKPOINT_DIR / f"agent_ep{episode_count:06d}.pt"))
            logger.info(f"Saved checkpoint at episode {episode_count}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"Best validation reward: {best_val_reward:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Fast RL training with parallel envs")
    parser.add_argument("--dataset", type=str, default="hpj", choices=["hpj", "GNPS"])
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    train_fast(
        dataset=args.dataset,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        n_epochs=args.n_epochs,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
