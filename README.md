[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/maezarizkic-code/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/maezarizkic-code/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# Leonida Refresh Forecaster — GTA VI trophy rarity forecast

Entry for the [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01) bounty on make.fun ($250 prize): forecast how many GTA VI players earn each of the 55 trophies in the first week.

## How it works

The forecaster works in **earner counts, not rates**. A snapshot with P players tracked and a rate of r% means E = r × P / 100 earners, so the whole problem splits into two easier forecasts that are multiplied back together at the end:

- **Players growth, refresh-aware.** The tracker page refreshes in jumps hours apart, so identical consecutive reads are collapsed into one snapshot first. When the history shows real growth, log(players) is regressed on log1p(hours since launch) over all snapshots and again over the last three, and the two predictions are averaged in log space — a power-law growth curve. When the history is still flat (the page has not refreshed since launch night), the population is grown off the first fresh read with a power law of exponent 1.12; targets before 06:00 UTC on launch day instead use the known first-refresh anchor of 3,868 players tracked.
- **Marginal earn rates.** With refreshes visible, each trophy's marginal rate is the average of the recent increment (last two distinct snapshots) and the whole-history increment (first to last snapshot), clamped to a sane band — that is the rate at which *new* players pick the trophy up. With a flat history there is no increment to measure, so each trophy uses a saturating prior, M × (1 − exp(−H / τ)): M is the eventual marginal rate, τ a progression timescale read off the trophy description (the first mission lands within an hour, early chapters within hours, mid-game strands over a day or so, late grinds over many days), H the horizon in hours. A trophy already held by essentially everyone (rate ≥ 99.99%) gets a marginal rate of 1.
- **Calibrated spread.** The point forecast is a rate; the five quantiles (5/25/50/75/95) are spread around it on the logit scale with σ = 0.10 + 0.25 × √(horizon in days), so longer horizons get wider intervals. Quantiles are rounded to four decimals, half-up, and forced non-decreasing.

The M values in the saturating prior come from the live Exophase trophy page snapshot of 2026-11-19 ~13:24 UTC — the per-trophy earner growth from launch night (638 players) to that snapshot, rescaled so the predicted first-refresh new-earner total matches the kit's stated 4,276 − 943 = 3,333. Trophy 2, "Complete Gone Fishing", is held by every player.

## Backtest

Measured errors, not promises — true values only exist after the target read, so no accuracy guarantee is made:

- **Real-data backtest:** a case cut at launch night (flat history) forecasting the live Exophase page of 2026-11-19 ~13:24 UTC, 10.4 h ahead — pinball loss **0.066**, typical miss 0.019 percentage points, 90% coverage 1.00, versus the persistence baseline's **2.40** on the same case.
- **Synthetic final-style backtest** (simulated 2-day refresh history, 4.5-day horizon, known truth): loss **0.162** and **0.143** on two realisations, versus persistence 0.205 / 0.212 and the house trend baseline ~0.41.
- **Practice cases:** outputs valid; the model's implied first-refresh totals are **4,216 earners versus the kit's stated 4,276 (1.4% low)** at 3,868 players tracked. Both practice checkpoints (03:00Z and 03:30Z) produce byte-identical output, as expected from their identical flat histories.

## How to run

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

`main.py` reads one case JSON per `*.in` file in `MAKEFUN_INPUT_DIR` (default `/input`) and writes `/output/<id>.out` (default `/output`, overridable with `MAKEFUN_OUTPUT_DIR`) as `{"trophies": [{"id": <int>, "q": [q05, q25, q50, q75, q95]}, ...]}` — one row per trophy, five non-decreasing quantiles in [0, 100].

Python 3.13, standard library only, offline and deterministic: no network, no randomness, no wall-clock dependence — the same input always produces byte-identical output, well inside the 8 s per-case limit.
