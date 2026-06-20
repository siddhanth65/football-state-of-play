"""Case-study selection + rendering (CLAUDE.md §7.4, Week 6).

CLAUDE.md asks for famous attacking-third sequences. The StatsBomb 360 open data
carries no Premier League matches, so the §7.4 wishlist (Manchester derbies, Klopp
Liverpool) is adapted to the tournaments we do have: the men's and women's World Cup
and Euro **finals**, plus Bayer Leverkusen's 2023/24 title season. Within each famous
match the selection is data-driven: the highest-threat successful counter, the best
sustained build-up, and instructive *failed* counters.

Two phases:
- ``--select`` writes ``results/case_studies/case_studies.json`` (no torch needed).
- ``--render`` draws each case from the dashboard cache (run ``app.precompute``
  between the phases so non-test case-study matches enter the cache).

Default (no flags): select, then render if the cache exists.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
CASE_DIR = Path("results/case_studies")
CASES_PATH = CASE_DIR / "case_studies.json"
PREDICTIONS_PATH = Path("app/assets/predictions.parquet")

# Famous matches present in the 360 open data, by (home, away, season) in either
# orientation. Verified against data/processed/match_ids.json at selection time.
FAMOUS_MATCHES = [
    ("Argentina", "France", "2022", "World Cup 2022 final"),
    ("Spain", "England", "2024", "Euro 2024 final"),
    ("Italy", "England", "2020", "Euro 2020 final"),
    ("England Women's", "Germany Women's", "2022", "Women's Euro 2022 final"),
    ("Spain Women's", "England Women's", "2023", "Women's World Cup 2023 final"),
    ("Bayer Leverkusen", "Bayern Munich", "2023/2024", "Leverkusen vs Bayern 2023/24"),
]


def _find_match(matches: pd.DataFrame, home: str, away: str, season: str) -> dict | None:
    """Find a match by team pair (either orientation) and season substring."""
    m = matches[
        matches["season_name"].astype(str).str.contains(season.split("/")[0])
        & (
            ((matches["home_team"] == home) & (matches["away_team"] == away))
            | ((matches["home_team"] == away) & (matches["away_team"] == home))
        )
    ]
    return m.iloc[0].to_dict() if not m.empty else None


def _pick(poss: pd.DataFrame, mask: pd.Series, sort_col: str, ascending: bool = False):
    """Top possession row matching a mask, or None."""
    sub = poss[mask].sort_values(sort_col, ascending=ascending)
    return sub.iloc[0] if not sub.empty else None


def select_cases() -> list[dict]:
    """Pick 8-10 case-study possessions across the famous matches."""
    matches = pd.DataFrame(
        json.loads(Path("data/processed/match_ids.json").read_text(encoding="utf-8"))
    )
    poss = pd.read_parquet("data/processed/possessions.parquet")

    cases: list[dict] = []

    def add(row, title: str, note: str) -> None:
        if row is None:
            return
        cases.append(
            {
                "match_id": int(row["match_id"]),
                "possession": int(row["possession"]),
                "title": title,
                "note": note,
            }
        )

    for home, away, season, label in FAMOUS_MATCHES:
        rec = _find_match(matches, home, away, season)
        if rec is None:
            logger.warning("case studies: match not found: %s", label)
            continue
        mp = poss[poss["match_id"] == int(rec["match_id"])]
        if mp.empty:
            continue

        # The most threatening successful counter in the match.
        add(
            _pick(mp, mp["from_counter"] & mp["success"], "xt_progression"),
            f"{label}: the counter that came off",
            f"{rec['home_team']} vs {rec['away_team']}, {rec['match_date']}. The highest-"
            "threat successful counter-attack of the final. Watch the receiver rings: who "
            "the model expects the ball to find next.",
        )
        # The best sustained build-up (long possession that reached threat).
        add(
            _pick(
                mp, mp["success"] & (mp["n_events"] >= 12) & ~mp["from_counter"], "xt_progression"
            ),
            f"{label}: sustained build-up",
            f"{rec['home_team']} vs {rec['away_team']}. A long possession that worked the "
            "ball into the box. Toggle the pitch-control surface to see the space the "
            "attack created.",
        )
        # An instructive failure: a counter the defence recovered.
        add(
            _pick(mp, mp["from_counter"] & ~mp["success"], "xt_progression"),
            f"{label}: the counter that died",
            f"{rec['home_team']} vs {rec['away_team']}. A counter-attack the defence "
            "swallowed. Compare the model's success probability here against the "
            "successful counters.",
        )

    # Keep it at 8-10: prioritise one per (match, kind) in insertion order.
    return cases[:10]


def render_cases(cases: list[dict]) -> int:
    """Render each case from the prediction cache to results/case_studies/."""
    from app.components.pitch_viewer import close, frame_figure

    if not PREDICTIONS_PATH.exists():
        logger.warning("render: prediction cache missing; run `python -m app.precompute` first")
        return 0
    cache = pd.read_parquet(PREDICTIONS_PATH)
    n = 0
    for case in cases:
        sel = cache[
            (cache["match_id"] == case["match_id"]) & (cache["possession"] == case["possession"])
        ]
        if sel.empty:
            logger.warning("render: case not in cache: %s", case["title"])
            continue
        fig = frame_figure(sel.iloc[0], show_control=True)
        out = CASE_DIR / f"case_{case['match_id']}_{case['possession']}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        close(fig)
        case["figure"] = str(out)
        n += 1
    return n


def main() -> None:
    """Select case studies, then render them if the cache is available."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true", help="only select (write JSON)")
    parser.add_argument("--render", action="store_true", help="only render from existing JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.render and CASES_PATH.exists():
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    else:
        cases = select_cases()

    rendered = 0
    if not args.select:
        rendered = render_cases(cases)

    CASES_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"\n{len(cases)} case studies -> {CASES_PATH} ({rendered} rendered)")
    for c in cases:
        print(f"  - {c['title']}  (match {c['match_id']}, poss {c['possession']})")


if __name__ == "__main__":
    main()
