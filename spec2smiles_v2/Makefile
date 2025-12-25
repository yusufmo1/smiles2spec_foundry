# SPEC2SMILES Pipeline Makefile
.PHONY: install train-part-a train-part-a-lgbm train-part-a-transformer train-part-a-hybrid train-part-b train-part-b-vae train-full train-full-lgbm-vae train-full-transformer-vae train-full-hybrid-vae predict evaluate visualize visualize-part-a visualize-part-b visualize-pipeline visualize-hybrid clean clean-logs lint format help

# Log directory
LOG_DIR := logs
$(shell mkdir -p $(LOG_DIR))

# Default target
help:
	@echo "SPEC2SMILES Pipeline Commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install dependencies with Poetry"
	@echo ""
	@echo "  Training (Part A - Spectrum -> Descriptors):"
	@echo "    make train-part-a-lgbm        LightGBM ensemble (fast)"
	@echo "    make train-part-a-transformer Transformer (deep learning)"
	@echo "    make train-part-a-hybrid      CNN-Transformer hybrid (local+global)"
	@echo ""
	@echo "  Training (Part B - Descriptors -> SMILES):"
	@echo "    make train-part-b-vae         Conditional VAE with SELFIES"
	@echo ""
	@echo "  Training (Full Pipeline):"
	@echo "    make train-full-lgbm-vae         LightGBM -> VAE"
	@echo "    make train-full-transformer-vae  Transformer -> VAE"
	@echo "    make train-full-hybrid-vae       Hybrid CNN-Transformer -> VAE"
	@echo ""
	@echo "  Inference:"
	@echo "    make predict          Run predictions"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make evaluate         Compute metrics on test set"
	@echo ""
	@echo "  Visualization:"
	@echo "    make visualize           Generate all visualization figures"
	@echo "    make visualize-part-a    Generate Part A figures only"
	@echo "    make visualize-part-b    Generate Part B figures only"
	@echo "    make visualize-pipeline  Generate pipeline results figures"
	@echo "    make visualize-hybrid    Generate hybrid model visualizations"
	@echo ""
	@echo "  Development:"
	@echo "    make lint             Run linter (ruff)"
	@echo "    make format           Format code (ruff)"
	@echo "    make clean            Remove generated files"
	@echo "    make clean-logs       Remove log files"
	@echo ""
	@echo "  Note: All commands log output to logs/ with timestamps"
	@echo ""
	@echo "  Configuration:"
	@echo "    Edit config.yml to change settings (dataset, device, hyperparameters)"
	@echo "    Or use: make train-part-a-lgbm CONFIG=custom_config.yml"

# Installation
install:
	poetry install

# Optional config file override
CONFIG ?=
CONFIG_FLAG = $(if $(CONFIG),--config $(CONFIG),)

# Training Part A: Spectrum -> Descriptors
train-part-a: train-part-a-lgbm  # Default to LightGBM

train-part-a-lgbm:
	poetry run python scripts/train_part_a.py --model lgbm $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_a_lgbm_$$(date +%Y%m%d_%H%M%S).log

train-part-a-transformer:
	poetry run python scripts/train_part_a.py --model transformer $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_a_transformer_$$(date +%Y%m%d_%H%M%S).log

train-part-a-hybrid:
	poetry run python scripts/train_part_a.py --model hybrid $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_a_hybrid_$$(date +%Y%m%d_%H%M%S).log

# Training Part B: Descriptors -> SMILES
train-part-b: train-part-b-vae  # Default to VAE

train-part-b-vae:
	poetry run python scripts/train_part_b.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_b_vae_$$(date +%Y%m%d_%H%M%S).log

# Training Full Pipeline
train-full: train-full-lgbm-vae  # Default to LightGBM -> VAE

train-full-lgbm-vae:
	$(MAKE) train-part-a-lgbm CONFIG=$(CONFIG) && $(MAKE) train-part-b-vae CONFIG=$(CONFIG)

train-full-transformer-vae:
	$(MAKE) train-part-a-transformer CONFIG=$(CONFIG) && $(MAKE) train-part-b-vae CONFIG=$(CONFIG)

train-full-hybrid-vae:
	$(MAKE) train-part-a-hybrid CONFIG=$(CONFIG) && $(MAKE) train-part-b-vae CONFIG=$(CONFIG)

# Inference
predict:
	poetry run python scripts/predict.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/predict_$$(date +%Y%m%d_%H%M%S).log

# Evaluation
evaluate:
	poetry run python scripts/evaluate.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/evaluate_$$(date +%Y%m%d_%H%M%S).log

visualize:
	poetry run python scripts/visualize.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/visualize_$$(date +%Y%m%d_%H%M%S).log

visualize-part-a:
	poetry run python scripts/visualize.py --part-a-only $(CONFIG_FLAG)

visualize-part-b:
	poetry run python scripts/visualize.py --part-b-only $(CONFIG_FLAG)

visualize-pipeline:
	poetry run python scripts/visualize.py --pipeline-only $(CONFIG_FLAG)

visualize-hybrid:
	poetry run python scripts/visualize.py --hybrid-only $(CONFIG_FLAG)

# Development
lint:
	poetry run ruff check src/ scripts/

format:
	poetry run ruff format src/ scripts/
	poetry run ruff check --fix src/ scripts/

# Cleanup
clean:
	rm -rf data/output/models/* data/output/metrics/* data/output/figures/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-logs:
	rm -rf $(LOG_DIR)/*.log
