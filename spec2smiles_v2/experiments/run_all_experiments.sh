#!/bin/bash
# Master experiment runner for Mamba-Diffusion optimization
# Budget: 32 training runs on RTX 4090

set -e
cd "$(dirname "$0")/.."

# Create logs directory
mkdir -p experiments/logs

# Helper function
run_exp() {
    name="$1"
    overrides="$2"
    notes="$3"
    eval_steps="${4:-20}"

    echo ""
    echo "============================================================"
    echo "EXPERIMENT: $name"
    echo "============================================================"

    PYTHONUNBUFFERED=1 poetry run python experiments/run_experiment.py \
        --name "$name" \
        --overrides "$overrides" \
        --eval-steps "$eval_steps" \
        --notes "$notes"
}

# Log baseline (already run)
echo "Logging baseline results..."
poetry run python -c "
from experiments.experiment_tracker import log_experiment
log_experiment(
    'baseline',
    {'encoder': {'type': 'mamba2'}, 'training': {'epochs': 100}},
    {'hit_at_1': 0.0, 'hit_at_10': 0.0, 'mean_tanimoto': 0.167, 'validity': 1.0},
    'Original Mamba-2 encoder, early stopped at 43 epochs'
)
"

# ============================================================
# PHASE 1: ARCHITECTURE (8 runs)
# ============================================================

# 1. Hybrid encoder (Mamba + attention)
run_exp "hybrid_encoder" \
    '{"encoder": {"type": "hybrid"}}' \
    "Samba-style hybrid: Mamba blocks with sparse attention"

# 2. Deeper encoder
run_exp "deep_encoder" \
    '{"encoder": {"n_layers": 12}}' \
    "12 Mamba layers instead of 8"

# 3. Wider model
run_exp "wide_model" \
    '{"encoder": {"d_model": 768, "d_state": 192}, "decoder": {"d_model": 768, "d_ff": 3072}}' \
    "Larger model: d_model=768"

# 4. Smaller model (faster iteration)
run_exp "small_model" \
    '{"encoder": {"d_model": 256, "n_layers": 4}, "decoder": {"d_model": 256, "n_layers": 4, "d_ff": 1024}}' \
    "Smaller model for faster training"

# 5. More decoder layers
run_exp "deep_decoder" \
    '{"decoder": {"n_layers": 12}}' \
    "Deeper decoder: 12 layers"

# 6. More cross-attention heads
run_exp "more_xattn" \
    '{"cross_attention": {"n_heads": 16}, "decoder": {"n_heads": 16}}' \
    "16 attention heads for better spectrum-token alignment"

# 7. Bigger state space
run_exp "big_state" \
    '{"encoder": {"d_state": 256, "expand": 4}}' \
    "Larger SSM state for longer-range dependencies"

# 8. CNN-style conv layers
run_exp "conv_encoder" \
    '{"encoder": {"d_conv": 8, "n_layers": 6}}' \
    "Larger conv kernels for local spectral patterns"

# ============================================================
# PHASE 2: TRAINING HYPERPARAMETERS (8 runs)
# ============================================================

# 9. Lower learning rate
run_exp "lr_1e5" \
    '{"training": {"learning_rate": 1e-5, "epochs": 150, "early_stopping": {"patience": 25}}}' \
    "Lower LR for more stable training"

# 10. Higher learning rate
run_exp "lr_3e4" \
    '{"training": {"learning_rate": 3e-4}}' \
    "Higher LR for faster convergence"

# 11. Long training, no early stop
run_exp "long_train" \
    '{"training": {"epochs": 200, "early_stopping": {"patience": 50}}}' \
    "Extended training: 200 epochs, patience=50"

# 12. Larger batch size
run_exp "batch_64" \
    '{"training": {"batch_size": 64, "learning_rate": 1.5e-4}}' \
    "Batch size 64 with scaled LR"

# 13. Full masking at end
run_exp "full_mask" \
    '{"training": {"mask_schedule": {"start_ratio": 0.15, "end_ratio": 1.0, "warmup_epochs": 30}}}' \
    "Train to 100% masking"

# 14. Slow warmup
run_exp "slow_warmup" \
    '{"training": {"warmup_steps": 3000, "mask_schedule": {"warmup_epochs": 40}}}' \
    "Slower warmup for better initial learning"

# 15. Higher weight decay
run_exp "weight_decay" \
    '{"training": {"weight_decay": 0.05}}' \
    "More regularization"

# 16. Gradient clip adjustment
run_exp "grad_clip" \
    '{"training": {"gradient_clip": 0.5}}' \
    "Tighter gradient clipping for stability"

# ============================================================
# PHASE 3: DIFFUSION PROCESS (8 runs)
# ============================================================

# 17. More inference steps
run_exp "steps_50" \
    '{"inference": {"sampling": {"steps": 50}}}' \
    "50 inference steps instead of 10" \
    50

# 18. More training timesteps
run_exp "timesteps_200" \
    '{"decoder": {"diffusion": {"num_timesteps": 200}}}' \
    "200 training timesteps for finer diffusion"

# 19. Linear noise schedule
run_exp "linear_noise" \
    '{"decoder": {"diffusion": {"noise_schedule": "linear"}}}' \
    "Linear noise schedule instead of cosine"

# 20. Sqrt noise schedule
run_exp "sqrt_noise" \
    '{"decoder": {"diffusion": {"noise_schedule": "sqrt"}}}' \
    "Sqrt noise schedule"

# 21. Lower temperature
run_exp "temp_low" \
    '{"inference": {"temperature": 0.5}}' \
    "Lower temperature (0.5) for more focused sampling"

# 22. Higher temperature
run_exp "temp_high" \
    '{"inference": {"temperature": 1.0}}' \
    "Temperature 1.0 for more diversity"

# 23. Absorbing mask schedule
run_exp "absorbing" \
    '{"decoder": {"diffusion": {"mask_schedule": "absorbing"}}}' \
    "Absorbing state masking (D3PM style)"

# 24. More candidates
run_exp "candidates_100" \
    '{"inference": {"n_candidates": 100}}' \
    "Generate 100 candidates per sample"

# ============================================================
# PHASE 4: DATA & REPRESENTATION (8 runs)
# ============================================================

# 25. Log transform
run_exp "log_transform" \
    '{"spectrum": {"transform": "log"}}' \
    "Log transform on spectrum intensities"

# 26. No transform
run_exp "no_transform" \
    '{"spectrum": {"transform": "none"}}' \
    "Raw spectrum intensities (no transform)"

# 27. Higher resolution spectrum
run_exp "bins_1000" \
    '{"spectrum": {"n_bins": 1000}}' \
    "1000 spectrum bins for higher resolution"

# 28. Lower resolution spectrum
run_exp "bins_250" \
    '{"spectrum": {"n_bins": 250}}' \
    "250 spectrum bins (compressed)"

# 29. Longer sequences
run_exp "seq_150" \
    '{"tokenizer": {"max_length": 150}}' \
    "Allow longer SELFIES sequences (150 tokens)"

# 30. Shorter sequences
run_exp "seq_64" \
    '{"tokenizer": {"max_length": 64}}' \
    "Shorter sequences (64) - may lose large molecules"

# 31. Combined best so far
run_exp "combined_v1" \
    '{"encoder": {"type": "hybrid"}, "training": {"learning_rate": 5e-5, "epochs": 150}, "inference": {"sampling": {"steps": 30}}}' \
    "Combination of promising settings"

# 32. Combined v2
run_exp "combined_v2" \
    '{"encoder": {"type": "hybrid", "d_model": 768}, "training": {"learning_rate": 5e-5, "epochs": 150, "mask_schedule": {"end_ratio": 1.0}}, "decoder": {"diffusion": {"num_timesteps": 200}}}' \
    "Full combined experiment"

# Print final summary
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
poetry run python experiments/experiment_tracker.py
