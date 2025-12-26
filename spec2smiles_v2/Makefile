# SPEC2SMILES Pipeline Makefile
.PHONY: install train-part-a train-part-b train-part-b-vae train-part-b-direct train-part-b-vae-augmented train-part-b-direct-augmented train-full predict evaluate eval-nucleus eval-nucleus-sweep viz viz-part-a viz-part-b viz-pipeline viz-benchmark visualize clean clean-output clean-logs init-dirs lint format help

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
	@echo "    make train-part-a             Hybrid CNN-Transformer"
	@echo ""
	@echo "  Training (Part B - Descriptors -> SMILES):"
	@echo "    make train-part-b-vae              Conditional VAE with SELFIES"
	@echo "    make train-part-b-direct           DirectDecoder (transformer, no VAE)"
	@echo "    make train-part-b-vae-augmented    VAE with 6x SMILES augmentation"
	@echo "    make train-part-b-direct-augmented DirectDecoder with 6x augmentation"
	@echo ""
	@echo "  Training (Full Pipeline):"
	@echo "    make train-full              Hybrid CNN-Transformer -> VAE"
	@echo ""
	@echo "  Inference:"
	@echo "    make predict          Run predictions"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make evaluate            Compute metrics on test set"
	@echo "    make eval-nucleus        Nucleus sampling eval (DirectDecoder)"
	@echo "    make eval-nucleus-sweep  Parameter sweep for nucleus sampling"
	@echo ""
	@echo "  Visualization:"
	@echo "    make viz                 Generate all figures"
	@echo "    make viz-part-a          Part A figures only"
	@echo "    make viz-part-b          Part B figures only"
	@echo "    make viz-pipeline        Pipeline comparison figures"
	@echo "    make viz-benchmark       MassSpecGym benchmark comparison"
	@echo ""
	@echo "  Development:"
	@echo "    make lint             Run linter (ruff)"
	@echo "    make format           Format code (ruff)"
	@echo "    make init-dirs        Create all output directories"
	@echo "    make clean            Remove all generated files"
	@echo "    make clean-output     Remove model/metrics/figures only"
	@echo "    make clean-logs       Remove log files only"
	@echo ""
	@echo "  Note: All commands log output to logs/ with timestamps"
	@echo ""
	@echo "  Configuration:"
	@echo "    Edit config.yml to change settings (dataset, device, hyperparameters)"
	@echo "    Or use: make train-part-a-hybrid CONFIG=custom_config.yml"

# Installation
install:
	poetry install

# Optional config file override
CONFIG ?=
CONFIG_FLAG = $(if $(CONFIG),--config $(CONFIG),)

# Training Part A: Spectrum -> Descriptors (Hybrid CNN-Transformer)
train-part-a:
	poetry run python scripts/train_part_a.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_a_$$(date +%Y%m%d_%H%M%S).log

# Training Part B: Descriptors -> SMILES
train-part-b: train-part-b-vae  # Default to VAE

train-part-b-vae:
	poetry run python scripts/train_part_b.py --model vae $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_b_vae_$$(date +%Y%m%d_%H%M%S).log

train-part-b-direct:
	poetry run python scripts/train_part_b.py --model direct $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_b_direct_$$(date +%Y%m%d_%H%M%S).log

train-part-b-vae-augmented:
	poetry run python scripts/train_part_b.py --model vae --augment --n-augment 5 $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_b_vae_aug_$$(date +%Y%m%d_%H%M%S).log

train-part-b-direct-augmented:
	poetry run python scripts/train_part_b.py --model direct --augment --n-augment 5 $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_b_direct_aug_$$(date +%Y%m%d_%H%M%S).log

# Training Full Pipeline (Hybrid CNN-Transformer -> VAE)
train-full:
	$(MAKE) train-part-a CONFIG=$(CONFIG) && $(MAKE) train-part-b-vae CONFIG=$(CONFIG)

# Inference
predict:
	poetry run python scripts/predict.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/predict_$$(date +%Y%m%d_%H%M%S).log

# Evaluation
evaluate:
	poetry run python scripts/evaluate.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/evaluate_$$(date +%Y%m%d_%H%M%S).log

eval-nucleus:
	poetry run python scripts/eval_nucleus.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/eval_nucleus_$$(date +%Y%m%d_%H%M%S).log

eval-nucleus-sweep:
	poetry run python scripts/eval_nucleus.py --sweep $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/eval_nucleus_sweep_$$(date +%Y%m%d_%H%M%S).log

viz: ## Regenerate all figures
	poetry run python scripts/visualize.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/visualize_$$(date +%Y%m%d_%H%M%S).log

viz-part-a: ## Part A figures only
	poetry run python scripts/visualize.py --part-a $(CONFIG_FLAG)

viz-part-b: ## Part B figures only
	poetry run python scripts/visualize.py --part-b $(CONFIG_FLAG)

viz-pipeline: ## Pipeline comparison figures
	poetry run python scripts/visualize.py --pipeline $(CONFIG_FLAG)

viz-benchmark: ## MassSpecGym benchmark comparison
	poetry run python scripts/visualize.py --benchmark $(CONFIG_FLAG)

visualize: viz  ## Backwards compatibility alias

# Development
lint:
	poetry run ruff check src/ scripts/

format:
	poetry run ruff format src/ scripts/
	poetry run ruff check --fix src/ scripts/

# Directory management
init-dirs:
	@mkdir -p data/output/models/part_a data/output/models/part_b
	@mkdir -p data/output/metrics
	@mkdir -p data/output/figures
	@mkdir -p $(LOG_DIR)
	@echo "Created output directories"

# Cleanup
clean: clean-output clean-logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-output:
	rm -rf data/output/models/* data/output/metrics/* data/output/figures/*
	@echo "Cleaned output directories"

clean-logs:
	rm -rf $(LOG_DIR)/*.log
	@echo "Cleaned log files"
