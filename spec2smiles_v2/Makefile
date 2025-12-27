# SPEC2SMILES Pipeline Makefile
.PHONY: install train-part-a train-part-a-lgbm train-part-b train-full evaluate viz viz-part-a viz-part-b viz-pipeline viz-benchmark viz-analysis clean clean-output clean-logs init-dirs lint format help

# Log directory
LOG_DIR := logs
$(shell mkdir -p $(LOG_DIR))

# Default target
help:
	@echo "SPEC2SMILES Pipeline Commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install              Install dependencies with Poetry"
	@echo ""
	@echo "  Training (Part A - Spectrum -> Descriptors):"
	@echo "    make train-part-a         Hybrid CNN-Transformer"
	@echo "    make train-part-a-lgbm    LightGBM ensemble (recommended)"
	@echo ""
	@echo "  Training (Part B - Descriptors -> SMILES):"
	@echo "    make train-part-b         DirectDecoder with SELFIES"
	@echo ""
	@echo "  Training (Full Pipeline):"
	@echo "    make train-full           LightGBM -> DirectDecoder (recommended)"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make evaluate             End-to-end evaluation on test set"
	@echo ""
	@echo "  Visualization:"
	@echo "    make viz                  Generate all figures"
	@echo "    make viz-part-a           Part A figures only"
	@echo "    make viz-part-b           Part B figures only"
	@echo "    make viz-pipeline         Pipeline comparison figures"
	@echo "    make viz-benchmark        MassSpecGym benchmark comparison"
	@echo "    make viz-analysis         Hit@K and Tanimoto analysis"
	@echo ""
	@echo "  Development:"
	@echo "    make lint                 Run linter (ruff)"
	@echo "    make format               Format code (ruff)"
	@echo "    make init-dirs            Create all output directories"
	@echo "    make clean                Remove all generated files"
	@echo ""
	@echo "  Configuration:"
	@echo "    Edit config.yml to change settings"
	@echo "    Or use: make train-part-a CONFIG=custom_config.yml"

# Installation
install:
	poetry install

# Optional config file override
CONFIG ?=
CONFIG_FLAG = $(if $(CONFIG),--config $(CONFIG),)

# Training Part A: Spectrum -> Descriptors
train-part-a: ## Hybrid CNN-Transformer
	poetry run python scripts/train_part_a.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_a_$$(date +%Y%m%d_%H%M%S).log

train-part-a-lgbm: ## LightGBM ensemble (recommended)
	poetry run python scripts/train_part_a_lgbm.py $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_a_lgbm_$$(date +%Y%m%d_%H%M%S).log

# Training Part B: Descriptors -> SMILES (DirectDecoder)
train-part-b: ## DirectDecoder with SELFIES
	poetry run python scripts/train_part_b.py --model direct $(CONFIG_FLAG) 2>&1 | tee $(LOG_DIR)/train_part_b_$$(date +%Y%m%d_%H%M%S).log

# Training Full Pipeline (LightGBM -> DirectDecoder)
train-full: ## Complete pipeline
	$(MAKE) train-part-a-lgbm CONFIG=$(CONFIG) && $(MAKE) train-part-b CONFIG=$(CONFIG)

# Evaluation
evaluate: ## End-to-end evaluation
	poetry run python scripts/eval_e2e.py --config config.yml --part-a lgbm 2>&1 | tee $(LOG_DIR)/evaluate_$$(date +%Y%m%d_%H%M%S).log

# Visualization
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

viz-analysis: ## Hit@K and Tanimoto analysis figures
	poetry run python scripts/visualize.py --analysis $(CONFIG_FLAG)

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
