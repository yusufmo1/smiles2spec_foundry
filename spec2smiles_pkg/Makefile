# SPEC2SMILES Pipeline Makefile
.PHONY: install train-part-a train-part-b train-full predict evaluate visualize clean test lint format help

# Default target
help:
	@echo "SPEC2SMILES Pipeline Commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install dependencies with Poetry"
	@echo ""
	@echo "  Training:"
	@echo "    make train-part-a     Train Part A (Spectrum -> Descriptors)"
	@echo "    make train-part-b     Train Part B (Descriptors -> SMILES)"
	@echo "    make train-full       Train complete pipeline"
	@echo ""
	@echo "  Inference:"
	@echo "    make predict          Run predictions"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make evaluate         Compute metrics on test set"
	@echo "    make visualize        Generate visualization figures"
	@echo ""
	@echo "  Development:"
	@echo "    make test             Run unit tests"
	@echo "    make lint             Run linter (ruff)"
	@echo "    make format           Format code (ruff)"
	@echo "    make clean            Remove generated files"
	@echo ""
	@echo "  Environment Variables:"
	@echo "    SPEC2SMILES_DATASET   Dataset name (default: hpj)"
	@echo "    SPEC2SMILES_DEVICE    Torch device (default: auto)"

# Installation
install:
	poetry install

# Training
train-part-a:
	poetry run python scripts/train_part_a.py

train-part-b:
	poetry run python scripts/train_part_b.py

train-full:
	poetry run python scripts/train_full.py

# Inference
predict:
	poetry run python scripts/predict.py

# Evaluation
evaluate:
	poetry run python scripts/evaluate.py

visualize:
	poetry run python scripts/visualize.py

# Development
test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check src/ scripts/ tests/

format:
	poetry run ruff format src/ scripts/ tests/
	poetry run ruff check --fix src/ scripts/ tests/

# Cleanup
clean:
	rm -rf data/output/models/* data/output/metrics/* data/output/figures/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
