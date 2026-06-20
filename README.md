# Football State-of-Play

Modelling **state of play** and **attacker instinct** in attacking-third football
possessions, on [StatsBomb 360](https://github.com/statsbomb/open-data) open data.

Given a freeze-frame of all visible players during an attacking-third possession,
the model represents the moment as a **graph of players** and predicts three things:

1. **Success** — will this possession produce a shot within ~15 s, or enter the
   penalty area within ~10 s? (binary)
2. **xT progression** — how much expected threat will it generate? (continuous,
   Karun Singh's xT grid as ground truth)
3. **Attacker run** — where will the most relevant attacker be in ~1.5 s? (2D
   regression — the "instinct" head)

The primary model is a **Graph Attention Network** (`torch_geometric`); a
**transformer over player tokens** is the secondary model for comparison. See
[docs/REPORT.md](docs/REPORT.md) for the method, results, and the literature that
informs the design.

---

## Setup

Requires **Python 3.11** (pinned in [.python-version](.python-version)) and
[`uv`](https://docs.astral.sh/uv/). The training box is a single GPU
(this project was developed on an RTX 3050 Laptop, 4 GB VRAM — see
[Hardware notes](#hardware-notes)).

```bash
# Create the 3.11 venv and install everything (pulls CUDA 12.4 torch).
uv venv --python 3.11
uv sync --extra dev
```

> **CPU-only machine?** Delete the `[tool.uv.sources]` / `[[tool.uv.index]]`
> blocks in `pyproject.toml` before syncing to get the CPU torch build, or use
> the `pip` fallback in `requirements.txt`.

### Get the data

The StatsBomb open data is free and needs no credentials. Clone it into
`data/raw/` (gitignored):

```bash
git clone --depth 1 https://github.com/statsbomb/open-data.git data/raw/open-data
```

Then cache the list of matches that have 360 freeze-frame data:

```bash
uv run python -m data.load          # -> data/processed/match_ids.json
```

---

## Reproduce

Every result table is regenerable from a clean clone. With `make` available
(Git Bash on Windows, or any Unix shell):

```bash
make setup        # venv + install
make match-ids    # cache the 360 match list
make data         # build processed possessions dataset      (Week 2+)
make baselines    # train + eval B0-B3 -> results/baselines.csv
make gnn          # train the GAT     -> results/final_table.csv
make all          # everything above
make dashboard    # launch the Streamlit app
```

On Windows without `make`, run the underlying commands directly, e.g.
`uv run python -m data.load`, `uv run python -m train.train_gnn`,
`uv run streamlit run app/streamlit_app.py`. Each `make` target maps to one such
command — see the [Makefile](Makefile).

---

## Repository layout

```
data/        load / possession-parsing / labelling / graph construction
features/    engineered + pitch-control baselines (Spearman 2018, OBSO)
models/      GAT, transformer, shared heads, B0-B4 baselines
train/       training entry points, multi-task loss, configs
eval/        metrics, ablations, case-study helpers
app/         Streamlit dashboard
notebooks/   01_eda ... 06_case_studies
results/     csv tables, figures, checkpoints (gitignored)
docs/        REPORT — method, results, limitations
tests/       unit tests (pytest)
```

---

## Hardware notes

- Graphs are small (≤ 22 nodes / ≤ 231 edges per freeze-frame), so GNN training
  is light on VRAM. The 4 GB ceiling mostly constrains the **transformer** batch
  size — drop it below the default 32 if you hit OOM.
- All seeds are fixed (`seed=42`) for reproducibility.

## Development

```bash
uv run pytest          # unit tests (network tests deselected by default)
uv run ruff check .    # lint
uv run ruff format .   # format
```

## License

MIT — see [LICENSE](LICENSE).
