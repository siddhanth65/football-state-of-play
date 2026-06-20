# Reproducibility entry points. Every result table is regenerable from a clean
# clone via these targets. See README for the Windows/PowerShell equivalents.
#
# Usage: `make <target>`. Requires `uv` (https://docs.astral.sh/uv/).

PY := uv run

.PHONY: help setup data match-ids baselines gnn transformer ablations \
        case-studies dashboard test lint format all clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install all dependencies (CUDA torch)
	uv venv --python 3.11
	uv sync --extra dev --extra logging

match-ids: ## Cache the 360-available match-ID list to data/processed/
	$(PY) python -m data.load

data: match-ids ## Build the processed possessions dataset (Week 2+)
	$(PY) python -m data.possessions
	$(PY) python -m data.labels

baselines: ## Train + evaluate baselines B0-B3 -> results/baselines.csv
	$(PY) python -m train.train_baselines

gnn: ## Train the GAT -> results/final_table.csv
	$(PY) python -m train.train_gnn

transformer: ## Train the transformer baseline
	$(PY) python -m train.train_transformer

ablations: ## Run ablation suite -> results/ablations.csv
	$(PY) python -m eval.ablations

figures: ## Evaluation figures + run analysis -> results/figures/, run_analysis.csv
	$(PY) python -m eval.figures

slices: ## Slice analyses (7.2) -> results/slices.csv
	$(PY) python -m eval.slices

multiseed: ## Multi-seed error bars -> results/final_table_seeds.csv
	$(PY) python -m train.multiseed

dxt: ## Dynamic-xT target ablation -> results/dxt.csv
	$(PY) python -m eval.dxt

position: ## Position-feature ablation -> results/position.csv
	$(PY) python -m eval.position

transfer: ## Cross-gender transfer -> results/transfer.csv
	$(PY) python -m eval.transfer

defense-features: ## Defensive-shape feature ablation -> results/defense_features.csv
	$(PY) python -m eval.defense_features

team-metrics: ## Per-team attacking/defensive fingerprints -> results/team_metrics.csv
	$(PY) python -m eval.team_metrics

tracking: ## Run the model on free tracking data (kloppy: Metrica/SkillCorner) [needs network]
	$(PY) python -m eval.tracking_bridge --provider metrica --match-ids 1

tracking-runs: ## Per-player run prediction on tracking (dense-data capability) [needs network]
	$(PY) python -m eval.tracking_runs --match-ids 1 2 3

case-studies: ## Select + render case studies -> results/case_studies/
	$(PY) python -m eval.case_studies

precompute: ## Build the dashboard prediction cache -> app/assets/predictions.parquet
	$(PY) python -m app.precompute

dashboard: ## Launch the Streamlit dashboard
	$(PY) streamlit run app/streamlit_app.py

test: ## Run the unit test suite
	$(PY) pytest

lint: ## Lint with ruff
	$(PY) ruff check .

format: ## Auto-format with ruff
	$(PY) ruff format .

all: data baselines gnn transformer ablations figures slices multiseed dxt position transfer defense-features team-metrics precompute case-studies ## Reproduce every result table + figure
	@echo "All result tables and figures regenerated under results/."

clean: ## Remove caches and processed data (keeps raw clone + checkpoints)
	rm -rf .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f data/processed/*.parquet data/processed/*.json data/processed/*.npy
