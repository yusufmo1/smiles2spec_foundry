# SMILES2SPEC Pipeline Makefile
# Usage: make <target>

.PHONY: install convert featurize train-rf train-nn train-ensemble train-full evaluate viz lint format clean help

# Default target
help:
	@echo "SMILES2SPEC Pipeline Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install dependencies via Poetry"
	@echo ""
	@echo "Data Processing:"
	@echo "  convert          Convert MSP files to JSONL"
	@echo "  featurize        Extract molecular features"
	@echo ""
	@echo "Training:"
	@echo "  train-rf         Train Random Forest model"
	@echo "  train-nn         Train neural network (default: modular_net)"
	@echo "  train-nn-all     Train all neural network architectures"
	@echo "  train-ensemble   Train ensemble model"
	@echo "  train-full       Full pipeline: featurize + train all"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate         Evaluate models on test set"
	@echo "  benchmark        Compare with SOTA methods"
	@echo ""
	@echo "Visualization:"
	@echo "  viz              Generate all visualizations"
	@echo "  viz-diagnostics  Generate diagnostic plots"
	@echo "  viz-performance  Generate performance analysis"
	@echo ""
	@echo "Development:"
	@echo "  lint             Run ruff linter"
	@echo "  format           Format code with ruff"
	@echo "  clean            Remove generated files"
	@echo ""
	@echo "Options:"
	@echo "  CONFIG=path      Use custom config file"
	@echo "  MODEL=name       Specify neural network architecture"

# Configuration
CONFIG ?= config.yml
MODEL ?= modular_net

# Setup
install:
	poetry install

# Data Processing
convert:
	poetry run python scripts/convert_msp.py --config $(CONFIG)

featurize:
	poetry run python scripts/featurize.py --config $(CONFIG)

# Training
train-rf:
	poetry run python scripts/train_rf.py --config $(CONFIG)

train-nn:
	poetry run python scripts/train_nn.py --config $(CONFIG) --model $(MODEL)

train-nn-all:
	poetry run python scripts/train_nn.py --config $(CONFIG) --model modular_net
	poetry run python scripts/train_nn.py --config $(CONFIG) --model hierarchical_net
	poetry run python scripts/train_nn.py --config $(CONFIG) --model sparse_gated_net
	poetry run python scripts/train_nn.py --config $(CONFIG) --model regional_expert

train-ensemble:
	poetry run python scripts/train_ensemble.py --config $(CONFIG)

train-full: featurize train-rf train-nn-all train-ensemble

# Evaluation
evaluate:
	poetry run python scripts/evaluate.py --config $(CONFIG)

benchmark:
	poetry run python scripts/evaluate.py --config $(CONFIG) --benchmark

# Visualization
viz:
	poetry run python scripts/visualize.py --config $(CONFIG) --all

viz-diagnostics:
	poetry run python scripts/visualize.py --config $(CONFIG) --diagnostics

viz-performance:
	poetry run python scripts/visualize.py --config $(CONFIG) --performance

# Development
lint:
	poetry run ruff check src/ scripts/

format:
	poetry run ruff format src/ scripts/

# Cleanup
clean:
	rm -rf data/output/*
	rm -rf data/cache/*
	rm -rf logs/*
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
