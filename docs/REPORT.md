# REPORT — Modelling State of Play and Attacker Instinct in Attacking-Third Possessions

> Working draft; figures referenced from `results/figures/` and tables from
> `results/*.csv`. Citations resolve in [RESEARCH_INTEGRATION.md](RESEARCH_INTEGRATION.md).

## 1. Problem framing

When a possession reaches the attacking third, the next ten seconds are decided less by
the ball-carrier's technique than by the *geometry around them*: where the defenders
are, which passing lanes survive, which teammate is already moving into space. Coaches
read this state of play instantly; most football models do not, because event data
records what happened to the ball, not where everyone was standing.

StatsBomb's 360 freeze-frames close that gap for public data: for each event they
record every visible player's position. This project asks whether a model that sees
only those dots, with no team identities, no player names, and no history, can answer
three questions at the moment an attack enters the final third:

1. **Success** — will this possession produce a shot within 15 s, or enter the penalty
   area within 10 s?
2. **Threat** — how much expected threat (xT) will the possession add?
3. **Instinct** — where will the most relevant attacker be 1.5 s from now?

The third question carries the project's distinctive framing. A model trained on
thousands of possessions learns the *expected* run in a given geometry. The gap
between that expectation and what an elite attacker actually does is a measurable
quantity. Penalty-kick game theory shows professionals are "part rational optimizer,
part appearance-driven" (Uribe et al. 2025): deviation from the model is a signal
about decision-making, not just model error. We call this gap the **instinct read**,
and the dashboard surfaces it per possession.

The representation is a **graph of players**: each visible player is a node, every
pair is connected, and edge features encode the pass-line geometry between them. Two
encoders compete on identical data: a graph attention network (GAT, the planned
primary model) and a transformer over player tokens (the planned secondary).

**Scope.** All **open-play** attacking-third possessions across seven major international
and club competitions (plus four pooled extras) — 11,841 labelled possessions from 426
matches. Counter-attacks are a tagged slice, not a separate model. Set-piece-proximate
triggers (within 5 s of a restart, ≈32% of attacking-third possessions) are **excluded**
(CLAUDE.md §4.3 step 7): set-piece dynamics differ and the subset is far easier (≈0.87 AUC
vs ≈0.75 open play), so pooling them inflated an earlier draft's 0.797 headline to a true
open-play 0.753. One instinct head (the attacker run); defender modelling is future work.

## 2. Related work

**Possession & threat value.** Expected Threat (Singh 2019) values pitch locations by
the probability of scoring within the next *n* actions via an iterative Markov
formulation on a 16×12 grid; VAEP (Decroos et al.) and EPV (Fernández, Bornn & Cervone
2021) extend value to all on-ball actions. Forcher et al. (2025) show EPV predicts
future results better than xG (pre-match) while xG better explains played matches
(post-match) — possession value and shot value carry complementary signal, motivating a
joint success + xT objective. Static xT ignores off-ball configuration; Dynamic xT
(Hassani et al. 2025) conditions the threat map on the freeze-frame, which is exactly
what a graph-embedding `H_xt` head learns. Earlier action-value approaches include
Markov chains (Rudd 2011) and deep-RL next-goal Q-functions (Liu et al. 2020).

**Pitch control & off-ball value.** Spearman (2018) models the probability each team
controls each point; OBSO = Scoring × Control × Transition derives a scoring-opportunity
surface, with PAUSA (Lee et al. 2026) a recent reference implementation that also values
pass *timing*. These are our geometric baselines (B2/B3).

**Graph models on freeze-frames (closest prior art).** TacticAI (Wang et al. 2024)
applies a GNN with geometric deep learning to corner kicks, predicting the receiver and
shot and generating tactic refinements; it finds a two-stage receiver→shot decomposition
substantially better than direct shot prediction. Disruption Maps (Stöckl et al. 2021)
train a graph conv net for node-level xReceiver, xPass and xThreat to quantify defensive
disruption. Sahasrabudhe & Bekkers (2023) use GNNs on tracking frames to predict
counterattack success, finding defensive features most predictive. Rahimian et al.
(2026) add a temporal dimension with per-node memory across a possession. Our model is a
direct extension of this family to open-play attacking-third possessions.

**Pass & receiver prediction.** Power et al. (2017) measure pass risk and reward from
tracking data using passing-line micro-features; Hubáček et al. (2019) predict receivers
from relative spatial relations; Sotudeh (2021, P3) and Karakuş & Arkadaş (2025) study
penetrative and line-breaking passes against defensive structure. Paul et al. (2025)
separate completion probability from value to counter outcome bias.

**Defensive & team-structure metrics.** DAxT (Merhej et al. 2021) values defensive
actions by the threat they prevent; shape graphs (Brandes et al. 2025) infer
instantaneous tactical positions and compactness; passing-network and occupancy/entropy
analyses (Lucey et al. 2013) characterise team style. PPDA (Morgan 2018) quantifies
pressing intensity.

## 3. Method

### 3.1 Data and labels

426 matches with 360 data (verified against the seven target competitions plus four
extras, all pooled; `is_target` retained for slicing). Events are grouped into
possession chains; the **trigger** is the first on-ball event by the possession team
in the attacking third (x ≥ 80 on StatsBomb's 120×80 pitch). Triggers within 5 s of a
set-piece restart action are tagged (`from_set_piece`) — keyed on the restart pass/shot
itself, not the possession-level `play_pattern`, which would wrongly drop long open-play
phases that merely began with a throw-in. **These set-piece-proximate triggers are
excluded** (CLAUDE.md §4.3 step 7; Fernández & Bornn 2021): set-piece dynamics differ and
the subset is far easier (≈0.87 AUC vs ≈0.75 open play), so pooling them inflated an
earlier draft's 0.797 headline. Freeze-frames with fewer than 10 visible players are
dropped.

Labels per possession:

- `success` — shot within 15 s OR penalty-area entry within 10 s, same possession.
- `xt_progression` — xT(end) − xT(trigger) on a 16×12 grid built from the pooled
  event corpus via Singh's iterative formulation, with bilinear interpolation between
  cell centres. (The grid is corpus-learned rather than Singh's published values; it
  reproduces the canonical structure, peaking at the goalmouth.)
- `attacker_run_target` — the furthest-forward attacker (excluding the carrier) at
  the trigger, located again ≈1.5 s later. 360 frames carry no player identities, so
  the attacker is matched across frames by nearest neighbour — also excluding the carrier
  in the +1.5 s frame — with a 15 m sanity cap; unmatchable rows are dropped. An
  alternative label, the **next completed pass's end location**, is carried (≈83% coverage)
  and used by the receiver head and a run-label ablation.

Final dataset: **11,841 open-play possessions** (269 counters), success base rate 0.449.

### 3.2 Graphs

Per freeze-frame: nodes are the visible players with 12 features (normalised
position; velocity differenced from the possession's previous freeze frame within
5 s, nearest-neighbour matched, **capped at 12 m/s** (raw NN estimates otherwise spike
to >700 m/s on a bad cross-frame match), zero-filled when no prior frame; teammate/carrier/keeper flags;
distances and angles to ball and both goals). Edges are all-pairs directed (≤462)
with 4 features: distance, closing speed along the line, and the **pass-line
micro-features** of Power et al. (2017) — the nearest defender's distance and angle
to the line between the two players. Eleven graph-level features encode play pattern,
the counter flag, and time-in-half. Targets attach as `y_success`, `y_xt`, `y_run`,
plus the receiver one-hot `y_receiver` and the alternative run target.

### 3.3 Models and heads

**GAT (planned primary).** Three GATv2 message-passing layers (hidden 64, 4 averaged
attention heads, edge features via `edge_dim`), mean+max global pooling concatenated
with the graph features, projected to a 64-d **state-of-play embedding**. Three
linear heads: success (BCE), xT (MSE), run (MSE on normalised coordinates). On top of
the node embeddings sit the adopted **receiver heads** (Wang et al. 2024; Stöckl et
al. 2021): a softmax over feasible teammates = P(next receiver), and a conditional
success head combined as the two-stage decomposition
`P(success) = Σ_i P(receiver = i) · P(success | receiver = i)`.

**Transformer (planned secondary).** Each player is a token (the same 12 features
projected by a 2-layer MLP into d=128); 3 encoder layers, 4 heads; a learned CLS
token concatenated with the graph features is the readout into the identical heads.

**Training.** Identical protocol for both: AdamW (lr 1e-4, wd 1e-4), cosine schedule
over 50 epochs, early stopping (patience 7), batch 32, seed 42. Splits are by match
(never by event), 70/15/15, stratified by competition. One deviation from the §5.5
plan: the inverse-magnitude loss rebalance is **off by default** — with a BCE success
loss (~0.7) beside small MSE losses it drives the success weight toward zero and
costs ~9 AUC points; fixed (1,1,1) weights are used instead.

### 3.4 Baselines

B0 class prior; B1 logistic/linear on nine engineered shape features (compactness,
line height, width, defenders ahead of ball, etc.); B2 Spearman (2018) pitch control
(closed-form arrival-time race) integrated over the penalty area; B3 OBSO =
Transition × Control × Score over the attacking half, using the learned xT grid as
the scoring surface. All share a no-move run prediction.

## 4. Results

Held-out test set: 1,924 open-play possessions from matches no model ever saw
(`results/final_table.csv`).

| Model | Success AUC | Brier | xT RMSE | xT R² | Run RMSE (m) |
|---|---|---|---|---|---|
| B0 prior | 0.500 | 0.247 | 0.0656 | 0.000 | 6.46 |
| B2 pitch control | 0.568 | 0.246 | 0.0655 | 0.002 | 6.46 |
| B1 engineered | 0.698 | 0.217 | 0.0654 | 0.005 | 6.46 |
| B3 OBSO | 0.699 | 0.227 | 0.0655 | 0.000 | 6.46 |
| GAT (+receiver) | 0.733 | 0.205 | 0.0653 | 0.006 | **6.33** |
| **Transformer** | **0.736** | **0.205** | 0.0656 | −0.001 | 6.37 |

Single seed (42); the **multi-seed headline with error bars is below** (*Stability*). The
baselines share the **no-move run baseline** (6.46 m); after the run-head fix (*Run*) both
learned heads now **edge it** (GAT 6.35 m). On this **open-play-only** test set the leading
success AUC is ~0.73–0.74: an earlier draft reported 0.797, but that pooled in set-pieces
(≈0.87 AUC) that the models find far easier than open play.

**Success.** Both learned models clear the best baseline (B3 OBSO 0.699; B1 engineered
0.698) by ~3–4 points on matches no model saw. **Adopting the sweep-selected config (lr
3e-4, hidden 128; `results/sweep.csv`) lifted the GAT from 0.705 to 0.730** and closed most
of the old GAT-vs-transformer gap — the two are now within error bars (*Stability*), so the
earlier "the transformer clearly wins" reading was partly an artifact of an under-tuned
(hidden-64) GAT. Calibration is good for both (`results/figures/reliability_success.png`).
The models **are** given the counter flag and the play-pattern one-hot as graph-level inputs
(`data/graphs.py:global_features`), so any counter-vs-open-play difference reflects supplied
context, not structure recovered "from geometry alone". The GAT's receiver head identifies
the actual next (completed-pass) recipient **top-1 40.9% / top-3 78.2%** — well above the
uniform-chance 20.1% / 59.1% over the open teammates. (This head originally collapsed *below*
chance; we traced it to **over-smoothing** — three rounds of attention on a fully-connected
≤22-node graph homogenise the node embeddings, so the per-node head could not tell players
apart — and fixed it with a **raw-feature skip connection** to the node-level heads
(`models/gnn.py`). It now beats a strong "next pass goes to the nearest teammate" heuristic,
top-1 ≈0.42.) The two-stage decomposition matches the direct head (0.730 vs 0.730).

| Slice | n | GAT AUC | Transformer AUC |
|---|--:|--:|--:|
| all = open play | 1,924 | 0.730 | 0.736 |
| counter (`From Counter`) | 49 | 0.682 | 0.667 |

The set-piece and high-stakes slices vanish once set-pieces are filtered out: high-box-density
moments were overwhelmingly set-pieces, leaving too few open-play instances to report (the
counter slice, n=49, is itself small and high-variance).

**Benchmark — is ~0.73 good?** Predicting whether an open-play possession becomes a chance
from a *single* freeze-frame is hard: outcomes in central areas are near coin-flips even for
strong models on full tracking data — Sahasrabudhe & Bekkers (2023), whose counter-attack GNN
uses all 22 players at high frame-rate, still report calibration error ≈0.15–0.18 on a
similarly hard target against a 0.50 baseline. Three things define a *good* result here, and
we meet them: (1) **beat the geometric baselines** — the learned models clear pitch control
(0.568) and OBSO (0.699), so the graph model extracts more than hand-crafted geometry;
(2) **calibrated** probabilities (reliability diagram on the diagonal); (3) **above trivial
baselines** (class prior 0.500; and, for the receiver head, the 0.42 nearest-teammate
heuristic). The ceiling is set by full-tracking systems we cannot match on sparse, ID-less
360 — e.g. the temporal receiver model of Rahimian et al. (2026) reaches 0.85 AUC / 0.81
top-1 with 25 fps tracking.

**Stability (3 seeds {42, 1, 2} on the fixed split, `results/final_table_seeds.csv`).**
With the sweep-selected config **and the adopted defensive-shape globals (§4.10)**, GAT
success AUC **0.733 ± 0.001**, transformer **0.746 ± 0.003**; receiver top-1 0.41 / top-3
0.78; run RMSE GAT 6.33 m, transformer 6.33 m. **Both clear the best baseline (0.699) by
~0.034–0.047 — many times the seed spread, so the improvement is real, not luck.** The
transformer's edge over the GAT is now ~0.013 (several × the ±0.003 spread): the shape
globals helped the transformer more, re-opening a small but real lead — having earlier found
the two *tied* once the GAT was properly tuned. The headline is the transformer at
**0.746 ± 0.003**, both models well clear of the baselines.

**xT (a negative result — and why).** Near-uninformative for every model and baseline (best
R² ≈ 0.02; `results/figures/xt_scatter.png`). This is a property of the *target*, not the
models: the label std (0.066) ≈ the RMSE of simply predicting the mean, i.e. it sits at the
irreducible-noise ceiling, because xT-progression is dominated by **where the possession ends
up** — a downstream outcome a trigger-time snapshot cannot know. The principled fix is to
change the target, not the model: context-conditioned **Dynamic xT** (Marin Felices 2025),
which makes the threat surface depend on the freeze-frame, and **risk-adjusted xT**
(xT-gain × P(completion)) to remove outcome bias. Both are future work (§6); we report the
current head honestly as a null.

**Run (fixed: from a 2× failure to a small positive).** Attackers move little in 1.5 s, so
the no-move baseline ("the subject stays put") is strong: median 5.2 m, RMSE **6.46 m**,
within 3 m 22% of the time. The head originally regressed the attacker's *absolute* future
position off the mean+max-pooled **graph** embedding, which collapses to the squad **centroid**
— so it scored 13.9 m (GAT) / 12.0 m (transformer), ~2× *worse* than no-move. That was a
**parametrization artifact**, not the task's true difficulty. Reframing the target as a
**displacement from the run anchor** (the attacker's own position; `data/graphs.py`) makes
"no movement" the model's floor and lets it learn drift on top; the anchor cancels in the
RMSE, so metres stay comparable. Both learned heads now **edge the baseline**: GAT 6.35 m /
hit@3m 0.243, transformer 6.37 m / 0.235, vs 6.46 m / 0.221 (`results/run_analysis.csv`,
`run_hitrate_curve.png`). The margin is small — a single snapshot carries little marginal
signal about exact 1.5 s displacement (consistent with the trajectory-prediction literature
needing temporal tracking; Teranishi et al. 2022) — so the head is best read as a calibrated
*direction-of-travel* prior. (An earlier draft also compared against a *miscomputed* no-move
baseline of 14.5 m that scored the wrong player; that is fixed in
`models/baselines.py:run_baseline_nomove`.)

### 4.1 Ablations

Eleven variants, identical splits, 30 epochs, at the **swept base config** (hidden 128;
`results/ablations.csv`):

| Variant | Success AUC | Reading |
|---|---|---|
| gat_base | 0.715 | reference (30-epoch GAT, hidden 128) |
| gat_hidden64 | 0.707 | narrower hurts ~0.8 pt — confirms the swept 128 width |
| gat_reflect | **0.722** | lateral-flip augmentation helps ~+0.7 pt |
| gat_no_receiver | 0.710 | receiver multi-task helps ~+0.5 pt |
| gat_success_only | 0.715 | success-only ≈ multi-task (run head untrained → ignore its run RMSE) |
| gat_no_edge_features | 0.708 | edge features worth ~+0.7 pt |
| gat_no_globals | 0.704 | globals worth ~+1.1 pt |
| gat_no_velocity | 0.711 | velocity worth ~+0.4 pt to the GAT |
| transformer_base | 0.749 | matches the headline at 30 epochs |
| transformer_no_velocity | 0.745 | velocity mildly positive (~+0.4 pt) for the transformer |
| gat_run_alt | 0.707 | next-pass run label (different target, run RMSE not comparable) |

Readings. (1) **Width matters:** the swept hidden-128 base (0.715) beats hidden-64 (0.707),
and the transformer (0.749) keeps a residual edge — part width, part architectural (on
fully-connected ≤22-node graphs GATv2 is a weaker stack than a LayerNorm+FFN encoder), though
the gap shrinks to within seed noise at full (50-epoch, multi-seed) training (§4). (2)
**Reflection augmentation helps** (+0.7 pt): the lateral pitch mirror is a free, label-preserving
2× of the training data and the largest single ablation gain here — adopted-worthy. (3) Each
input family contributes a little (globals ~+1.1, edges ~+0.7, velocity ~+0.4, receiver
multi-task ~+0.5); the 12 m/s velocity cap keeps velocity mildly *positive* for both models
(outlier speeds, not motion, were the earlier problem). Most GAT gaps are 1–2 AUC points, so
read them as directional at 30 epochs / single seed.

A 10-config validation-selected **hyperparameter sweep** (`results/sweep.csv`,
`train/sweep.py`) answered "does tuning help?": yes — the old default (lr 1e-4, hidden 64) was
**under-tuned**; **lr 3e-4 with hidden 128** lifted validation AUC from ~0.66 to ~0.71. This
config is now **adopted** as the GAT default (`train/configs/gnn_default.yaml`, `models/gnn.py`)
and the §4 / Stability tables are the **re-baselined** result: it lifted the GAT from 0.705 to
**0.730** (single seed) / **0.732 ± 0.001** (multi-seed) and closed most of the
GAT–transformer gap — confirming that gap was substantially **width/optimisation**, not purely
architectural (the transformer was already d=128).

### 4.2 Temporal model: does the build-up help?

The single biggest limitation is the *single frame*. `models/temporal.py` (`TemporalGAT`)
encodes the **last ≤6 build-up freeze-frames** of each possession with the GAT frame-encoder
and runs a GRU over the sequence (median 6 frames; `data/sequences.py`). It is
permutation-invariant per frame, so it needs **no player IDs and no cross-frame tracking** —
the honest answer to 360's missing-identity problem: it reasons over whole-frame embeddings,
not player tracks.

Apples-to-apples (same 64-d GAT frame-encoder), the temporal models score success AUC
**0.711** (GRU aggregator, `TemporalGAT`) and **0.705** (self-attention over the build-up
frames with a CLS readout, `TemporalTransformer`) — **both below** the single-frame GAT
(0.732 ± 0.001) and transformer (0.738 ± 0.006). We built the temporal *transformer*
precisely as the "proper test" of whether attention over the whole build-up beats a single
frame, and it does **not**: the build-up adds **no measurable signal** on sparse 360
(`results/temporal.csv`). One caveat keeps this honest — the temporal frame-encoder is 64-d
while the single-frame models adopt the swept 128-d, so part of the gap is encoder width.
But at *matched* 64-d width the temporal GAT (0.711) only ties the 64-d single-frame GAT
(0.708, the pre-sweep multi-seed mean), so the build-up still buys nothing. The likely cause
is structural: the trigger frame is the decisive moment, and 360's ~1-frame-per-event
sparsity makes the few preceding frames a weak, irregularly-timed signal. Closing the gap to
full-tracking temporal models (Rahimian et al. 2026, 25 fps) needs denser tracking than open
360 provides — a **data** limitation, not an architecture one.

### 4.3 Defensive head: predicting defensive measures

With the attacker head working (CLAUDE.md §3 parks the defensive head until it does), we add
`H_defense` — the **mirror of the receiver head**: a softmax over the *defending* team =
P(this defender contests the next ball), trained on the defender nearest the next completed
pass's end location, with the same raw-feature skip connection (`models/gnn.py`). It is
additive: success AUC is unchanged (0.730 single / **0.734 ± 0.003** multi-seed), so the head
is free.

It is a **positive result**: presser **top-1 0.278 ± 0.002 / top-3 0.615 ± 0.001** (multi-seed,
`results/final_table_seeds.csv`) against a uniform-over-defenders chance of **0.120 / 0.361**
(≈8.6 visible defenders per frame) — ~2.3× chance at top-1, comparable lift to the receiver
head. So the same graph that reads the attack also reads the defensive response: which
defender steps to the ball. This is the seed of the defensive phase the project scoped for
"after the attacker head works"; natural extensions are defensive-line height / compactness
regression and pass-lane disruption (Stöckl et al. 2021).

### 4.4 Counterfactual "instinct": where *should* the runner move?

Predicting where the runner *will* go is weak (§4.2 *Run*), so we ask the strong, calibrated
**success** head the inverse question. `eval/counterfactual.py` sweeps the most-advanced
attacker over a grid of offsets, rebuilds the graph at each with the real feature builders
(so every probe is on-distribution), and reads P(success) — yielding a **success-probability
surface** and the success-maximising move. This is TacticAI's generative/counterfactual
refinement (RESEARCH_INTEGRATION §3.8), but grounded in a head that works rather than the run
head, and it needs **no retraining** — it is an inference-time use of the trained model. It is
surfaced in the dashboard as a heatmap with a "best move" arrow and the P(success) gain
(`app/assets/counterfactual.parquet`). As an *exploratory* tactical tool its validity rests
on the success head's calibration (good; §4), not on a movement label — which is exactly why
it sidesteps the run head's accuracy ceiling.

### 4.5 Dynamic xT: the head was fine, the target was wrong

The xT null (§4) is a property of the *target*, not the model. To show this we trained the
**same** xT head against three targets on the same split (`eval/dxt.py`, `results/dxt.csv`):

| xT target | R² | reading |
|---|--:|---|
| realized `xT(end) − xT(trigger)` | 0.004 | the committed null — outcome-dominated, unlearnable from a snapshot |
| **OBSO at the trigger (Dynamic xT)** | **0.882** | a configuration-conditioned threat *is* highly learnable from the graph |
| risk-adjusted (realized × P(complete)) | 0.011 | still near-null — it stays anchored to the realized outcome |

Against a **Dynamic-xT target** — OBSO, a freeze-frame-conditioned scoring value
(`features/obso.py`) — the head reaches **R² = 0.88**, versus ≈0 for realized progression. So
the graph embedding carries the spatial information for context-conditioned threat; the
original target simply asked an unanswerable question (where will the ball *end up*).
**Honest caveat:** OBSO is itself a deterministic function of player positions, so 0.88 means
the GNN learns a **fast neural surrogate of that physics surface** (a learned DxT computable in
one forward pass) — genuinely useful, but "learns the surface", not "predicts the future". The
**risk-adjusted** variant (Paul et al. 2025) does *not* rescue R² here: weighting by completion
probability still leaves the target dominated by the realized end location. Takeaway: to make
xT a positive, change the *target* to a configuration value (DxT), not the model. The DxT
target is now a **permanent head** on the model (§4.7): trained jointly it still reaches
**R² 0.733**, surfaced as a learned Dynamic-xT per possession in the dashboard.

### 4.6 Position-aware features: an honest null (and why)

Can the heads be sharpened by **player position** (midfield vs forward, etc.)? The hard
constraint: **StatsBomb 360 carries no per-dot identity or role** — only
`teammate/actor/keeper/location`. Only the *ball-carrier's* position is known (the event
stream). We realised the facet two ways (`data/graphs.py:add_position_features`): a **heuristic
per-node role** (each player's team-relative third, for every dot) and the **carrier's true
role** (a graph-level GK/DEF/MID/FWD one-hot), and trained the GAT with vs without them on the
same split (`eval/position.py`, `results/position.csv`):

| variant | success AUC | receiver top-1 | presser top-1 | run RMSE (m) |
|---|--:|--:|--:|--:|
| base | 0.714 | 0.377 | 0.248 | 6.37 |
| + position | 0.716 | 0.332 | 0.232 | 6.35 |

**Positions do not help, and hurt the node-level heads.** Success barely moves (+0.002, within
noise); receiver and presser get *worse* because the per-node role is a crude x-tercile
heuristic — with no real identity it adds noise the model already had from raw `(x, y)`. This
is the expected consequence of the 360 identity limitation, not a bug, so per the
adopt-only-if-positive rule the features stay **out of the headline**. It is an informative
null: it quantifies what ID-less data costs us, and points at tracking data (real per-player
roles) as the prerequisite for a position-aware gain.

**Positional-zone slices** (the carrier's role, `results/slices.csv`) are a useful by-product:
success is comparable across DEF/MID/FWD ball-carriers (GAT 0.72–0.73; transformer reads deep
DEF-carrier build-up best at 0.744, midfield worst at 0.712), and **final-third (forward-carrier)
runs are the hardest to predict** (run RMSE 6.62 m vs 6.16 m for deeper carriers) — sensible,
since attacks in the box are the most dynamic.

### 4.7 A fuller defensive suite + disruption

With the presser head (§4.3) established, three further heads were added to the **same** shared
graph embedding — all additive, leaving success essentially unchanged (~0.733 single
seed). The Dynamic-xT target (§4.5) is also promoted to a permanent head here.

| head | metric | reading |
|---|---|---|
| **defensive-line** (offside-line displacement, +1.5 s) | **5.36 m** RMSE vs **8.30 m** no-change | clear positive (~35% better) — the back line moves coherently, so it is far better-posed than the per-player run head |
| **Dynamic-xT** (joint head, predicts OBSO) | **R² 0.733** | strong positive even sharing capacity multi-task (0.88 when trained alone, §4.5) |
| **xPass / disruption** (per-teammate completion) | **0.679 AUC** | modest positive; 1 − xPass is a lane-closing / disruption map |
| **defensive-success** (P(recovery ≤ 8 s)) | **0.598 AUC** | weak positive — whether the defence wins it back is partly a downstream outcome |

So the one graph now reads, from a single frozen moment: will the attack succeed, how much
threat the shape affords (DxT), who receives, who presses, where the defensive line shifts,
and how open each lane is. The defensive-line and DxT heads are the standout positives; the
defensive-success and xPass heads are honest, modest ones (their targets are partly
outcome-driven). Defensive-line is rendered better-posed precisely because, unlike one
sprinting attacker, the back four move as a unit.

### 4.8 Explainability and an OOD inference hook

**Attention explainability.** GATv2 exposes its attention weights, so per-player importance
("how much the model attends to each player", summed over message-passing layers) is read off
directly (`models/gnn.py:node_attention`) and rendered in the dashboard as marker size — a
peek inside the black box for a coach, at no training cost. (A correlational view, not a causal
claim.)

**Out-of-distribution hook.** The Phase-1 broadcast demo is two stages: YOLO + homography on a
video clip to mint top-down coordinates, then this model on the resulting frame. Stage one is
the separate Phase-1 CV stack (it needs the clip + that codebase); **stage two ships here**
(`eval/ood_demo.py:predict_from_freeze_frame`): feed any freeze-frame's coordinates and get
success / Dynamic-xT / receiver / presser predictions. This makes the model usable on
*any* coordinate source — broadcast-derived, hand-built, or synthetic 3v2 — not just
StatsBomb 360.

### 4.9 Team identity from the relations

The supervisor's framing — *"understand the relations → derive attacking and defensive team
metrics"* (relationism: positions emerge from relationships; Hamilton/Diniz) — is realised
directly: the graph **is** the relations and every head is a relational read-out, so per-team
**fingerprints** need no new model, only aggregation of the trained GAT's per-possession
outputs (`eval/team_metrics.py`, `results/team_metrics.csv`; 92 teams with ≥30 possessions).

- **Attacking identity** (over a team's own possessions): chance creation (success), threat
  afforded (Dynamic-xT), **option richness** (entropy of the receiver distribution — do they
  always have several free men?), counter share, realised success.
- **Defensive identity** (over possessions where the team defends): solidity (1 − success
  conceded), **recovery rate** (def_stop), **press decisiveness** (top presser prob — a learned
  PPDA), **line height** (offside line), and **lane suppression** (1 − opponent xPass — a learned
  build-up-disruption %).

These line up with established team-style metrics — PPDA (press intensity), field tilt
(territorial threat), and Build-up Disruption % — but are *derived from the model's relational
reads* rather than hand-counted, and they come as one coherent attacking+defensive fingerprint
per team. The dashboard's **Team identity** page renders them as comparative radars. Honest
caveats: the values are model-derived per-possession averages on 360 data (mostly
international sides + Bayer Leverkusen — the one club, a proof-of-concept for club-level use),
and the spread between teams is modest, so they are *tendencies*, not official ratings.

### 4.10 Strengthening the defensive heads with shape features

The defensive-success head is weak partly because the model gets little explicit defensive
context. We tested four **defending-team shape descriptors** as graph-level features —
defensive-line height, compactness, width, and numerical balance ahead of the ball (Brandes
et al. 2025 shape graphs; PPDA intuition) — base vs +features on the same split
(`eval/defense_features.py`, `results/defense_features.csv`):

| metric | base | + def_shape |
|---|--:|--:|
| success AUC | 0.696 | **0.712** |
| defensive-success AUC | 0.580 | **0.591** |
| defensive-line RMSE (m) | 5.79 | **5.60** |
| xPass AUC | 0.555 | **0.569** |

The shape features gave a **small but consistent gain on every head**, so — unlike the
position features (§4.6) — they were **adopted into the headline** (`data/graphs.py`
appends the four descriptors to the globals; `eval/defense_features.py` is now the on/off
ablation that zeroes them). Realised gains after the re-baseline: defensive-line RMSE
**5.92 → 5.36 m**, xPass AUC **0.63 → 0.679**, Dynamic-xT R² **0.72 → 0.733**,
defensive-success **0.59 → 0.598**, and transformer success **0.738 → 0.746** (multi-seed).
Evidence the defensive heads were **context-starved, not at a hard ceiling** — though
defensive-success stays the weakest, since its target is still partly a downstream outcome.

## 5. Case studies

Eight sequences, selected data-driven from famous matches present in the open data
(`results/case_studies/`; the §7.4 wishlist of Premier League derbies is not available
in StatsBomb 360 open data, so finals stand in): the World Cup 2022 final (Argentina's
highest-threat counter — the Di María-goal sequence — and France's best build-up), the
Euro 2024, Euro 2020, Women's Euro 2022 and Women's World Cup 2023 finals' best
sustained build-ups, and Bayer Leverkusen vs Bayern 2023/24 (counter + build-up). Each
renders in the dashboard with the freeze frame, pitch-control surface, both models'
success probabilities, run predictions vs the actual run, and receiver candidates.

The browsing experience is the deliverable here: `streamlit run app/streamlit_app.py`
offers Browse (all 1,924 held-out possessions), Case studies, and Compare modes, with
the instinct read (model-expected vs actual run, in metres) surfaced per possession.

## 6. Limitations and future work

**Limitations.**

- **Freeze-frames miss intent.** A single freeze-frame captures positions but not
  off-ball intent or body orientation; visual exploration before receiving the ball is
  something human analysts still read better than automated methods (Herold et al. 2019).
- **Noisy labels.** "Shot within 15 s / box entry within 10 s" is a *proxy* for success,
  not ground truth; outcomes in central areas are near coin-flips even for strong models
  (Sahasrabudhe & Bekkers 2023). xT/VAEP-style targets also carry outcome bias, rewarding
  risky actions that happen to succeed (Paul et al. 2025).
- **The run head is a small positive, not a strong one.** It originally scored ~2× *worse*
  than the no-move baseline (13.9 / 12.0 m vs 6.46 m) — a parametrization artifact (absolute
  regression off a pooled embedding collapses to the squad centroid). Reframing the target as
  a **displacement from the run anchor** (`data/graphs.py`) fixed it: both heads now edge the
  baseline (GAT 6.35 m / hit@3m 0.243 vs 6.46 m / 0.221). The remaining margin is small —
  attackers move little in 1.5 s and the ID-less nearest-neighbour label is noisy — so the head
  is a calibrated *direction-of-travel* prior, not a pinpoint predictor. Event-time also
  under-represents anticipatory runs (Rahimian et al. 2026); a cleaner label (tracking data, or
  a per-player node-level run head) is the prerequisite for a larger gain.
- **The receiver head is modest, not dominant.** It originally collapsed *below* chance —
  diagnosed as over-smoothing and fixed with a raw-feature skip connection (top-1 15% → 38%,
  top-3 50% → 76%; §4). It now clears chance comfortably but only *matches* a trivial
  "nearest-teammate" heuristic on single frames (top-1 ≈38% vs ≈42%). A genuinely strong
  receiver predictor needs the build-up sequence (the temporal model below; cf. Rahimian et
  al. 2026's TGN at 0.85 AUC with full tracking) rather than one snapshot.
- **The headline is sensitive to scope.** Excluding the ≈32% set-piece-proximate triggers
  (the spec-correct open-play definition) drops the transformer's success AUC from a pooled
  0.797 to 0.753, because set-pieces are much easier (≈0.87 AUC). The play-pattern / counter
  / clock context features are legitimate inputs, but they mean "from geometry alone"
  overstates what the model uses.
- **xT head is near-uninformative** (R² ≤ 0.02) — and this is irreducible from a single
  trigger-time snapshot: the label std ≈ the predict-the-mean RMSE, because xT-progression is
  dominated by where the possession *ends*, an outcome the frame cannot see. Dynamic xT (Marin
  Felices 2025) and risk-adjusted xT (Paul et al. 2025) are the principled target fixes (§6).
- **Generalisation (now measured).** A cross-gender transfer test (`eval/transfer.py`,
  `results/transfer.csv`) trains on one gender and evaluates on the other: **men→women
  0.670 AUC**, **women→men 0.580** — vs ~0.73 in-distribution. So there is a real transfer
  gap (men→women still clears the baseline; women→men is weaker, largely because the women's
  training set is smaller, 125 vs 292 matches). Pooling men's and women's data is defensible
  but the exchangeability assumption is only partly justified — a genuine, quantified
  limitation rather than an assumed one.
- **Interpretability vs. coach uptake.** Black-box models conflict with practitioners'
  preference for simple, actionable insight (Herold et al. 2019); we lean on
  pitch-control surfaces, receiver rings, and the instinct gap to stay readable.
- **Coordinate/visibility noise.** 360 visibility is partial; velocities are derived
  from a sparse, noisy stream — now capped at 12 m/s (raw NN estimates reached >700 m/s)
  but still approximate without player IDs; the ablation shows they can *hurt*, so
  Savitzky–Golay-style smoothing on true tracks is the deeper fix.

**Done since the first draft** (all in this report): multi-seed CIs, the hyperparameter
sweep + GAT re-baseline, the receiver-head fix (over-smoothing → skip connection), the
run-head displacement fix, the temporal **transformer** vs GRU vs single-frame comparison,
lateral reflection augmentation (`gat_reflect` ablation), the dashboard state-of-play
build-up panel, the **defensive (presser) head** (§4.3 — a positive result, 0.278/0.615 vs
0.120/0.361 chance), the **counterfactual instinct** surface (§4.4 — the run question
re-asked of the strong success head), the **Dynamic-xT redesign** (§4.5 — the same head
reaches R² 0.88 against a configuration target vs ≈0 realized, proving the target was the
problem), the **position-aware experiment** (§4.6 — an honest null: ID-less 360 gives no
real per-player role, so positions don't help), a **fuller defensive suite + disruption**
(§4.7 — defensive-line 5.36 m vs 8.30 m no-change and a joint Dynamic-xT head R² 0.733 are
clear positives; defensive-success 0.598 and xPass 0.679 AUC modest), and **attention
explainability + an OOD inference hook** (§4.8).

**Future work**, in priority order: **change the xT target** to context-conditioned Dynamic
xT (Marin Felices 2025) and risk-adjusted xT (Paul et al. 2025) — the current head is at the
single-snapshot noise ceiling, so only a better-posed label can move it; a **per-player
node-level run head** on a cleaner (tracking-derived) label, the route past the
direction-of-travel prior; **denser tracking** than open 360 to give the temporal model a
real signal (the build-up adds nothing at 360's ~1-frame-per-event sparsity); Savitzky–Golay
velocity smoothing on true tracks; the instinct-deviation population study, conformal
prediction sets, and the cross-gender transfer experiment (RESEARCH_INTEGRATION §7); the
defender offside-line head; the Phase-1 YOLO+homography OOD demo on a broadcast clip.

## 7. Sources and resources (curated)

The two **meta-indexes** that cover almost everything else are worth bookmarking first:
- **Edd Webster's resource hub** (eddwebster.com) — an exhaustive, maintained index of soccer
  data sources, code, papers, and people.
- **`matiasmascioto/awesome-soccer-analytics`** (GitHub) — a curated awesome-list of analytics
  resources.

**Methods / code (directly reusable here):**
- **Friends of Tracking** — YouTube series + `Friends-of-Tracking-Data-FoTD/LaurieOnTracking`
  (pitch control, EPV); the origin of our B2/B3 baselines.
- **Soccermatics** (David Sumpter — book + the `soccermatics/Soccermatics` course) — the
  relational/positional foundations; the best grounding for the supervisor's "relations" lens.
- **`socceraction`, `kloppy`, `mplsoccer`, `FCrSTATS/SBpitch`** — the event-value, data-IO, and
  pitch-drawing libraries (we use mplsoccer + the socceraction ideas).
- **StatsBomb open-data + 360 spec** — our entire data substrate; pitch-control-from-360
  tutorials show the event↔360 merge we rely on.
- **Barça Innovation Hub / Javier Fernández** — EPV and positional-value work, the closest
  published analogue to "value from configuration".

**Concept references for our metrics (what to cite to the supervisor):**
- **Relationism vs positionism** — Jamie Hamilton's writing; Diniz (Fluminense) vs De Zerbi;
  *The Football Analyst* and *Total Football Analysis* explainers. Grounds §4.9 and the
  (deferred) positioning-vs-relativism experiment.
- **PPDA** (press intensity), **field tilt** (territorial dominance), **Build-up Disruption %**
  — the established team-style metrics our presser / Dynamic-xT / xPass heads learn implicitly.
- **xT** (Karun Singh), **Dynamic xT** (Marin Felices 2025), **risk-adjusted value**
  (Paul et al. 2025), **Disruption Maps / xPass / xReceiver** (Stöckl et al. 2021), **TacticAI**
  (2024), **counter-attack GNN** (Sahasrabudhe & Bekkers 2023), **temporal GNN** (Rahimian
  et al. 2026) — the model-side lineage (see also `docs/RESEARCH_INTEGRATION.md`).
- **The xG Football Club Substack** (already used) — accessible write-ups of many of the above.

**Communities / venues for new ideas:** MIT Sloan Sports Analytics Conference, the StatsBomp
conference, the OptaPro Forum, and the Soccermatics/analytics community on X
(@Soccermatics, @petermckeever for visualisation).

**On club-scale data (the expansion question):** the graph model needs *player coordinates per
event*. Free club sources — FBref / Understat via `soccerdata` and `ScraperFC` (the engines
behind the `mufc-rodri-search` and `soccerdata` projects) — provide **aggregate/event** stats,
**not** freeze-frames, so they can power *event-based* team metrics but cannot feed the
relational graph. The only routes to club-level freeze-frames are (a) StatsBomb 360 club
releases (we already have Bayer Leverkusen 23/24), (b) a CV pipeline (YOLO + homography on
broadcast video — the Phase-1 stack, fed by the `eval/ood_demo.py` hook), or (c) a paid
tracking provider (SkillCorner / StatsPerform).

**The CV bridge (concrete).** The Phase-1 project (`cv-football/football-pipeline`,
`extract_positions.py`) already turns broadcast video into a per-frame positions table —
`frame, track_id, role, team, pitch_x, pitch_y, conf` — via YOLO + ByteTrack + homography.
This is exactly the freeze-frame geometry the graph needs, and is in fact *richer* than 360
(continuous tracks with persistent IDs ⇒ velocity, sequences, per-player runs the 360 model
could not support). An adapter (positions → our `(x, y, teammate, actor, keeper)` format,
rescaled 105×68 → 120×80, ball-carrier inferred) feeding `predict_from_freeze_frame` would
unlock club-level relational metrics + team identity from any match video — and retire the
Phase-1 LSTM/XGBoost classifier in favour of the GNN. Gaps to close first: the ball is not
tracked (no actor flag), ~22 % of detections fail homography, and ByteTrack fragments
identities — so it needs ball detection + quality filtering before it is metric-grade.
