"""Leonida Refresh Forecaster.

Forecasts, for each of the 55 GTA VI trophies, the share of tracked
players who will hold it at a future target time, given a history of
Exophase-style snapshots (players tracked plus per-trophy earn rates).

Method, in brief
----------------
Everything is done in *earner counts*, not in rates.  A snapshot with
P players tracked and a rate of r percent means E = r * P / 100
earners.  Rates are just E / P rescaled, so the forecast is built as
"how many players will be tracked at the target?" times "how many of
the newcomers will hold each trophy by then?", and only converted
back to a percentage at the end.

Three ingredients:

1. Players growth.  The tracker page refreshes in jumps hours apart,
   so identical consecutive reads are collapsed into one snapshot.
   With at least two distinct player counts, log(P) is regressed
   (least squares) on log1p(hours since launch) over all snapshots
   and again over the last three, and the two predictions are averaged
   in log space -- a power-law growth curve.  With a flat history
   (the page has not refreshed yet) the population is grown off the
   first fresh read with a power law of exponent 1.12, except that
   targets before 06:00 UTC on launch day use the known first-refresh
   anchor of 3,868 players tracked.

2. Marginal earn rates.  With refreshes visible, the marginal rate
   for a trophy is the average of the recent increment (last two
   distinct snapshots) and the whole-history increment (first to
   last snapshot), clamped to a sane band.  With a flat history there
   is no increment to measure, so each trophy uses a saturating
   prior: M * (1 - exp(-H / tau)), where M is the eventual marginal
   rate, tau a progression timescale in hours taken from what the
   trophy asks of the player (first mission within an hour, early
   chapters within hours, mid-game strands over a day or so, late
   grinds over many days), and H the horizon in hours.  A trophy
   already held by essentially everyone (rate >= 99.99) is given a
   marginal rate of 1: every newcomer earns it.

3. Calibrated spread.  The point forecast is a rate; the five
   quantiles are spread around it on the logit scale with
   sigma = 0.10 + 0.25 * sqrt(horizon in days), so longer horizons
   get wider intervals.  Quantiles are rounded to four decimals,
   half-up, and forced non-decreasing.

Standard library only, no network, no randomness, no wall clock:
the same input files always produce byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Launch moment: GTA VI released 2026-11-19T00:00:00Z.
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# First refresh after launch night: 3,868 players tracked.
FIRST_REFRESH_PLAYERS = 3868
FIRST_REFRESH_CUTOFF = datetime(2026, 11, 19, 6, 0, 0, tzinfo=timezone.utc)

# Power-law exponent used when the history is flat (no refresh yet).
FLAT_GROWTH_EXPONENT = 1.12

# Saturating marginal prior, keyed by trophy id: (M, tau in hours).
# M is the eventual marginal rate among new players; tau is the
# progression timescale implied by the trophy description.
MTABLE = {
    1: (0.00000, 200), 2: (1.00000, 1), 3: (0.01040, 5),
    4: (0.00185, 12), 5: (0.00134, 200), 6: (0.00134, 200),
    7: (0.00134, 200), 8: (0.00401, 12), 9: (0.00134, 200),
    10: (0.00134, 200), 11: (0.00134, 200), 12: (0.00134, 200),
    13: (0.00134, 200), 14: (0.00000, 200), 15: (0.00000, 200),
    16: (0.00038, 200), 17: (0.00203, 200), 18: (0.00000, 200),
    19: (0.00038, 200), 20: (0.02784, 5), 21: (0.00566, 5),
    22: (0.00038, 200), 23: (0.00038, 200), 24: (0.00038, 200),
    25: (0.00000, 200), 26: (0.00000, 200), 27: (0.00038, 200),
    28: (0.00000, 200), 29: (0.00038, 200), 30: (0.00000, 200),
    31: (0.00000, 200), 32: (0.00000, 200), 33: (0.00176, 30),
    34: (0.00047, 30), 35: (0.00385, 12), 36: (0.00087, 30),
    37: (0.00202, 12), 38: (0.00253, 30), 39: (0.00056, 30),
    40: (0.00038, 200), 41: (0.00000, 200), 42: (0.00038, 200),
    43: (0.00038, 200), 44: (0.00136, 30), 45: (0.00038, 200),
    46: (0.00038, 200), 47: (0.00038, 200), 48: (0.00047, 30),
    49: (0.00038, 200), 50: (0.00154, 12), 51: (0.00087, 30),
    52: (0.00038, 200), 53: (0.00086, 200), 54: (0.00038, 200),
    55: (0.00038, 200),
}

Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)


def parse_time(text):
    """Parse an ISO timestamp into an aware UTC datetime."""
    if text is None:
        return None
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_launch(moment):
    """Hours from T0 to `moment` (negative values kept as-is)."""
    if moment is None:
        return 0.0
    return (moment - T0).total_seconds() / 3600.0


def ols_log_prediction(points, x_target):
    """Least-squares fit of y on x; return the prediction at x_target.

    `points` is a list of (x, y).  Returns None when the fit is
    degenerate (fewer than two points or zero variance in x).
    """
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    var_x = sum((p[0] - mean_x) ** 2 for p in points)
    if var_x <= 0.0:
        return None
    cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return intercept + slope * x_target


def round4_half_up(value):
    """Round to 4 decimals, half-up (like Math.round(x * 10000) / 10000)."""
    return math.floor(value * 10000.0 + 0.5) / 10000.0


def logit(pct):
    """Logit of a percentage, clamped to [0.01, 99.99] first."""
    p = min(99.99, max(0.01, pct))
    return math.log(p / (100.0 - p))


def inv_logit(x):
    """Inverse logit, returned as a percentage in (0, 100)."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def quantiles(rate_pct, horizon_minutes):
    """Five quantiles around a point rate, spread on the logit scale."""
    horizon_days = horizon_minutes / 1440.0
    sigma = 0.10 + 0.25 * math.sqrt(max(horizon_days, 0.0))
    centre = logit(rate_pct)
    out = []
    for z in Z_SCORES:
        q = inv_logit(centre + z * sigma)
        q = round4_half_up(q)
        q = min(100.0, max(0.0, q))
        out.append(q)
    # Rounding can break the ordering; repair by raising to the
    # predecessor so the five values stay non-decreasing.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def earners(rate, players):
    """Earner count for a rate (percent) over `players`; None -> 0."""
    if rate is None:
        return 0.0
    return float(rate) * float(players) / 100.0


def forecast_case(case):
    """Forecast one case dict; return the output dict."""
    trophies = case["trophies"]
    ids = [t["id"] for t in trophies]
    horizon_minutes = float(case.get("horizon_minutes", 0))
    target = parse_time(case.get("target"))
    t_target = hours_since_launch(target)

    # --- Distinct snapshots: collapse identical consecutive reads. ---
    snapshots = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snapshots.append(read)
    if not snapshots:
        raise ValueError("case %r has an empty history" % (case.get("id"),))

    def snapshot_players(read):
        return float(read.get("players_tracked") or 0)

    def snapshot_rates(read):
        return read.get("rates") or []

    first = snapshots[0]
    last = snapshots[-1]
    p_first = snapshot_players(first)
    p_last = snapshot_players(last)
    rates_last = snapshot_rates(last)

    distinct_p = {snapshot_players(s) for s in snapshots}
    has_growth = len(distinct_p) >= 2

    # --- Forecast players tracked at the target. ---
    if has_growth:
        points = []
        for s in snapshots:
            t = hours_since_launch(parse_time(s.get("at")))
            p = snapshot_players(s)
            if p > 0:
                points.append((math.log1p(max(t, 0.0)), math.log(p)))
        x_target = math.log1p(max(t_target, 0.0))
        log_pred_all = ols_log_prediction(points, x_target)
        log_pred_recent = ols_log_prediction(points[-3:], x_target)
        if log_pred_all is None:
            p_pred = p_last
        elif log_pred_recent is None:
            p_pred = math.exp(log_pred_all)
        else:
            p_pred = math.exp((log_pred_all + log_pred_recent) / 2.0)
    else:
        if target is not None and target <= FIRST_REFRESH_CUTOFF:
            p_pred = float(FIRST_REFRESH_PLAYERS)
        else:
            p_pred = p_first * (1.0 + max(t_target, 0.0)) ** FLAT_GROWTH_EXPONENT
    if not (p_pred > 0) or not math.isfinite(p_pred):
        p_pred = p_last
    p_pred = max(p_pred, p_last)
    d_players = p_pred - p_last

    # --- Marginal earn rates for the new players. ---
    horizon_hours = horizon_minutes / 60.0
    marginal = []
    for i in range(len(ids)):
        last_rate = rates_last[i] if i < len(rates_last) else None
        if last_rate is not None and last_rate >= 99.99:
            marginal.append(1.0)
            continue
        if has_growth:
            rates_first = snapshot_rates(first)
            r_first = rates_first[i] if i < len(rates_first) else None
            e_first = earners(r_first, p_first)
            e_last = earners(last_rate, p_last)
            # A denominator can be zero even with growth in the
            # history: the first and last snapshots may share the
            # same player count (growth happened in between), and
            # two consecutive distinct snapshots may differ only in
            # rates.  Use whichever increment is defined; if neither
            # is, fall back to the floor below.
            m_all = None
            if p_last != p_first:
                m_all = (e_last - e_first) / (p_last - p_first)
            prev = snapshots[-2]
            p_prev = snapshot_players(prev)
            rates_prev = snapshot_rates(prev)
            r_prev = rates_prev[i] if i < len(rates_prev) else None
            e_prev = earners(r_prev, p_prev)
            m_recent = None
            if p_last != p_prev:
                m_recent = (e_last - e_prev) / (p_last - p_prev)
            if m_all is None and m_recent is None:
                m = 0.00002
            elif m_all is None:
                m = m_recent
            elif m_recent is None:
                m = m_all
            else:
                m = 0.5 * m_recent + 0.5 * m_all
            m = min(0.9, max(0.00002, m))
        else:
            m_table, tau = MTABLE.get(ids[i], (0.00038, 200))
            m = m_table * (1.0 - math.exp(-horizon_hours / tau))
            m = min(0.95, m)
        marginal.append(m)

    # --- Forecast earners, convert back to rates, spread quantiles. ---
    rows = []
    for i, tid in enumerate(ids):
        last_rate = rates_last[i] if i < len(rates_last) else None
        e_last = earners(last_rate, p_last)
        if last_rate is not None and last_rate >= 99.99:
            rate = 100.0
        else:
            e_pred = max(e_last + marginal[i] * d_players, 0.0)
            rate = 100.0 * e_pred / p_pred if p_pred > 0 else 0.0
        rows.append({"id": tid, "q": quantiles(rate, horizon_minutes)})
    return {"trophies": rows}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        out_path = output_dir / (path.stem + ".out")
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
