"""Team-identity metrics from the learned relations (REPORT §4.9).

The supervisor's ask: "understand the relations -> derive attacking and defensive team
metrics." The graph *is* the relations and every head is a relational read-out, so this needs
**no new model** — it aggregates the trained GAT's per-possession outputs into an **attacking
fingerprint** (computed over a team's own possessions) and a **defensive fingerprint** (over
possessions where the team is defending). Writes ``results/team_metrics.csv``.

Attacking identity (team in possession): chance creation (success), threat afforded
(Dynamic-xT), option richness (receiver-distribution entropy), counter share, realised success.
Defensive identity (team defending): solidity (1 − success conceded), recovery rate (def_stop),
press decisiveness (top presser prob), line height (offside line), lane suppression (1 − xPass).

Run ``python -m eval.team_metrics`` (needs ``graphs.pt`` + the trained GAT checkpoint).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from data.graphs import load_graph_cache
from models.gnn import GAT

logger = logging.getLogger(__name__)
TEAM_METRICS_PATH = Path("results/team_metrics.csv")
MIN_POSSESSIONS = 30  # don't report a fingerprint built on too few possessions


@torch.no_grad()
def possession_outputs(model: torch.nn.Module, graphs: list) -> pd.DataFrame:
    """Per-possession head read-outs, in graph order (one row per possession)."""
    model.eval()
    sp, dxt, ent, ptop, xpm = [], [], [], [], []
    for batch in DataLoader(graphs, batch_size=128, shuffle=False):
        out = model(batch)
        sp.append(torch.sigmoid(out["success"]).numpy())
        dxt.append(out["dxt"].numpy())
        rp = out["receiver_probs"].clamp_min(1e-9)
        ent.append(float_(-(rp.log() * rp).sum(dim=1)))  # receiver-distribution entropy
        ptop.append(float_(out["presser_probs"].max(dim=1).values))
        xp = out["xpass_dense"]
        xpm.append(float_(xp.sum(dim=1) / (xp > 0).sum(dim=1).clamp_min(1)))
    ids = [(int(g.match_id), int(g.possession)) for g in graphs]
    return pd.DataFrame(
        {
            "match_id": [i[0] for i in ids],
            "possession": [i[1] for i in ids],
            "m_success": np.concatenate(sp),
            "m_dxt": np.concatenate(dxt),
            "recv_entropy": np.concatenate(ent),
            "presser_top": np.concatenate(ptop),
            "xpass_mean": np.concatenate(xpm),
        }
    )


def float_(t: torch.Tensor) -> np.ndarray:
    """Detach a tensor to a numpy array (small helper for the read-out loop)."""
    return t.detach().cpu().numpy()


def _match_teams() -> dict[int, set[str]]:
    """match_id -> {home, away} from the cached match list."""
    recs = json.loads(Path("data/processed/match_ids.json").read_text(encoding="utf-8"))
    return {int(r["match_id"]): {r.get("home_team"), r.get("away_team")} for r in recs}


def build_team_metrics(model: torch.nn.Module, graphs: list, poss: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-possession outputs into per-team attacking + defensive fingerprints."""
    out = possession_outputs(model, graphs)
    keep = ["match_id", "possession", "possession_team", "def_stop", "success",
            "from_counter", "offside_trigger_x"]
    df = out.merge(poss[keep], on=["match_id", "possession"], how="left")
    teams = _match_teams()
    df["defender"] = [
        next(iter(teams.get(m, set()) - {a}), None)
        for m, a in zip(df["match_id"], df["possession_team"], strict=True)
    ]

    atk = df.groupby("possession_team")
    deff = df.groupby("defender")
    attacking = pd.DataFrame(
        {
            "n_attack": atk.size(),
            "attack_success": atk["m_success"].mean(),
            "attack_dxt": atk["m_dxt"].mean(),
            "option_richness": atk["recv_entropy"].mean(),
            "counter_share": atk["from_counter"].mean(),
            "realised_success": atk["success"].mean(),
        }
    )
    defending = pd.DataFrame(
        {
            "n_defend": deff.size(),
            "solidity": 1.0 - deff["m_success"].mean(),  # 1 − success conceded
            "recovery_rate": deff["def_stop"].mean(),
            "press_decisiveness": deff["presser_top"].mean(),
            "line_height": deff["offside_trigger_x"].mean(),
            "lane_suppression": 1.0 - deff["xpass_mean"].mean(),  # 1 − opponent lane openness
        }
    )
    table = attacking.join(defending, how="outer")
    table = table[(table["n_attack"].fillna(0) >= MIN_POSSESSIONS)]
    return table.reset_index(names="team").round(4).sort_values("attack_dxt", ascending=False)


def main() -> None:
    """Build the per-team fingerprints and write ``results/team_metrics.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    model = GAT()
    model.load_state_dict(torch.load("results/checkpoints/gnn.pt", weights_only=True))
    table = build_team_metrics(model, load_graph_cache(), poss)
    TEAM_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TEAM_METRICS_PATH, index=False)
    print(f"\n{len(table)} teams (>= {MIN_POSSESSIONS} possessions). Top attacking threat:")
    print(table.head(8).to_string(index=False))
    print(f"\nWrote {TEAM_METRICS_PATH}")


if __name__ == "__main__":
    main()
