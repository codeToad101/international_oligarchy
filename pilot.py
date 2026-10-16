"""
pilot.py -- [C4] pilot runs, calibration targets and diagnostics for the M0 model.

Placeholders in Params are calibrated against TARGETS (structural plausibility, NOT fitted to data;
see project_overview.txt Sections 2 and 5.5). Windows scale with n_ticks:
    burn-in = first 20% of ticks, embedding window = 20%..50%, outcome window = 50%..100%
(100 ticks -> burn-in 0-20, embed 20-50, outcome 50-100). The same windows feed the analysis tables.

USAGE
    python pilot.py --report            # target table over the 4 slide-6 gamma levels
    python pilot.py --price-check       # static-price adequacy check (5.4)
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

import numpy as np

from model_core import NI, Params
from m0_model import M0World
from runlog import RunLog

GAMMAS = (0.04, 0.08, 0.12, 0.16)


def windows(T: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    e0, e1 = int(round(0.2 * T)), int(round(0.5 * T))
    return (e0, e1), (e1, T)


# ----------------------------------------------------------------------------
def summarize(rl: RunLog, p: Params) -> Dict[str, float]:
    """Scalar run summaries used by the calibration targets."""
    T = rl.T
    (e0, e1), (o0, o1) = windows(T)
    ol, st = rl.olig, rl.static
    act = ol["active"]
    W, W0 = ol["wealth"], st["initial_wealth"]
    growth = W[-1] / W0
    exited = ~np.isnan(st["exit_tick"])
    ruling = st["ruling"]
    posh, posf = ol["pos_home"], ol["pos_foreign"]
    tot = posh + posf

    a_o = act[o0:o1]
    m = a_o & (tot[o0:o1] > 1e-9)
    fshare = float((posf[o0:o1][m] / tot[o0:o1][m]).mean()) if m.any() else np.nan
    with_for = float(((posf[o0:o1] > 0.1) & a_o).sum() / max(a_o.sum(), 1))

    a_e = act[e0:e1]
    Ee = ol["E_foreign"][e0:e1][a_e]
    E_sd = float(np.nanstd(Ee)) if Ee.size else np.nan
    E_mean = float(np.nanmean(Ee)) if Ee.size else np.nan
    E_neg_frac = float((Ee < -1e-4).mean()) if Ee.size else np.nan
    E_pos_frac = float((Ee > 1e-4).mean()) if Ee.size else np.nan

    off_share = ol["offshore"][o0:o1] / np.maximum(W[o0:o1], 1e-9)
    off_share = float(off_share[a_o].mean()) if a_o.any() else np.nan

    c = rl.country
    out = dict(
        growth_med=float(np.median(growth)), growth_mean=float(np.mean(growth)),
        growth_med_active=float(np.median(growth[~exited])) if (~exited).any() else np.nan,
        exit_frac=float(exited.mean()),
        exit_frac_civil=float(exited[~ruling].mean()), exit_frac_ruling=float(exited[ruling].mean()),
        exit_tick_med=float(np.nanmedian(st["exit_tick"])) if exited.any() else np.nan,
        active_outcome=float(a_o[0].sum()) if a_o.size else 0.0,
        fshare=fshare, with_foreign=with_for,
        E_sd=E_sd, E_mean=E_mean, E_neg_frac=E_neg_frac, E_pos_frac=E_pos_frac,
        offshore_share=off_share,
        collapses_pc=float(c["collapse"].sum() / rl.C),
        health=float(np.nanmean(c["health_mean"])), health_end=float(np.nanmean(c["health_mean"][-1])),
        scarcity=float(np.nanmean(c["scarcity"])),
        abuse_frac=float(np.nanmean(c["abuse_frac"])),
        B_pre=float(np.nanmean(c["B_pre"])),
        G_mean_end=float(rl.world["G_mean"][-1]),
        S_foreign_end=float(rl.world["S_foreign_mean"][-1]), S_home_end=float(rl.world["S_home_mean"][-1]),
    )
    # embeddedness -> foreign capture share, pooled-correlation inputs (oligarchs active in both windows)
    Em = np.where(a_e, ol["E_foreign"][e0:e1], np.nan)
    E_o = np.nanmean(np.where(a_e, ol["E_foreign"][e0:e1], np.nan), axis=0) if a_e.any() else np.full(rl.O, np.nan)
    fc, dc = ol["foreign_capture"][o0:o1], ol["domestic_capture"][o0:o1]
    tc = fc + dc
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(a_o & (tc > 1e-9), fc / tc, np.nan)
    cap_o = np.nanmean(sh, axis=0) if a_o.any() else np.full(rl.O, np.nan)
    hv = np.where(a_o, ol["foreign_havoc"][o0:o1], np.nan)
    hav_o = np.nanmean(hv, axis=0) if a_o.any() else np.full(rl.O, np.nan)
    ok = ~np.isnan(E_o) & ~np.isnan(cap_o)
    out["corr_E_capshare"] = float(np.corrcoef(E_o[ok], cap_o[ok])[0, 1]) if ok.sum() > 4 and np.std(E_o[ok]) > 0 and np.std(cap_o[ok]) > 0 else np.nan
    ok2 = ~np.isnan(E_o) & ~np.isnan(hav_o)
    out["corr_E_havoc"] = float(np.corrcoef(E_o[ok2], hav_o[ok2])[0, 1]) if ok2.sum() > 4 and np.std(E_o[ok2]) > 0 and np.std(hav_o[ok2]) > 0 else np.nan
    return out


def run_many(p: Params, seeds: Iterable[int]) -> List[Dict[str, float]]:
    rows = []
    for s in seeds:
        w = M0World(p, s)
        rl = w.run()
        rows.append(summarize(rl, p))
    return rows


def mean_table(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


# ----------------------------------------------------------------------------
# Calibration targets (structural plausibility; documented in project_overview.txt 5.5)
# (low, high) applied per gamma level unless the key says otherwise.
# ----------------------------------------------------------------------------
TARGETS = {
    "growth_med":      (1.0, 3.0),       # median wealth multiple over the run (all oligarchs, incl. offshore)
    "exit_frac":       (0.05, 0.60),     # per gamma; middle levels also checked against (0.10, 0.40)
    "fshare":          (0.25, 0.65),     # foreign share of positions among active oligarchs, outcome window
    "with_foreign":    (0.60, 1.00),     # share of active oligarch-ticks with foreign presence
    "E_sd":            (0.015, 1.0),     # dispersion of embeddedness across active oligarch-ticks
    "offshore_share":  (0.0, 0.35),      # flight should not dominate wealth
    "abuse_frac":      (0.05, 0.50),     # abuse reachable in some pool-ticks, not all
    "health":          (0.40, 0.90),     # mean resource health
    "active_outcome":  (12.0, 25.0),     # oligarchs still on the stage when the outcome window opens (of 25)
}
MID_EXIT = (0.10, 0.40)                  # exit prevalence target for gamma in {0.08, 0.12}
COLLAPSES = {0.04: (0.2, 1.2), 0.08: (0.5, 2.0), 0.12: (1.0, 3.5), 0.16: (1.5, 5.0)}   # per country per run


def check(table: Dict[float, Dict[str, float]]) -> Tuple[float, List[str]]:
    """Return (penalty, messages). Penalty = summed relative distance outside target ranges, plus trend
    violations (collapses and exit must be non-decreasing in gamma)."""
    pen, msgs = 0.0, []
    def out_of(v, lo, hi):
        if np.isnan(v):
            return 10.0
        if v < lo:
            return (lo - v) / max(abs(lo), 1e-6) if lo != 0 else lo - v
        if v > hi:
            return (v - hi) / max(abs(hi), 1e-6)
        return 0.0
    for g, row in table.items():
        for k, (lo, hi) in TARGETS.items():
            d = out_of(row[k], lo, hi)
            if k == "exit_frac" and g in (0.08, 0.12):
                d = max(d, out_of(row[k], *MID_EXIT))
            if d > 0:
                pen += d; msgs.append(f"gamma={g}: {k}={row[k]:.3f} outside {(lo, hi)}")
        d = out_of(row["collapses_pc"], *COLLAPSES[g])
        if d > 0:
            pen += d; msgs.append(f"gamma={g}: collapses_pc={row['collapses_pc']:.2f} outside {COLLAPSES[g]}")
    gs = sorted(table)
    for k in ("collapses_pc", "exit_frac"):
        for a, b in zip(gs[:-1], gs[1:]):
            if table[b][k] < table[a][k] - 1e-9:
                pen += 1.0; msgs.append(f"trend: {k} falls from gamma={a} to {b}")
    return pen, msgs


def evaluate(p: Params, seeds=range(8), gammas=GAMMAS):
    table = {}
    for g in gammas:
        table[g] = mean_table(run_many(replace(p, gamma=g), seeds))
    return table


def print_table(table, cols=("growth_med", "exit_frac", "collapses_pc", "fshare", "with_foreign", "E_sd",
                             "offshore_share", "abuse_frac", "health", "scarcity", "active_outcome",
                             "corr_E_capshare", "corr_E_havoc")):
    print("gamma  " + "  ".join(f"{c[:12]:>12}" for c in cols))
    for g, row in table.items():
        print(f"{g:5.2f}  " + "  ".join(f"{row[c]:12.3f}" for c in cols))


# ----------------------------------------------------------------------------
def static_price_check(p: Params, seeds=range(6)):
    """[5.4] Are static prices enough for opportunities to vary? Measures, over the run:
    (a) time variation (CV) of the marginal extraction return q*G*exp(-qX) across (country, industry)
        pools -- driven by depletion, congestion and collapse, NOT price;
    (b) allocation shifts: for active oligarchs, the L1 change in normalized position shares per tick;
    (c) share of oligarch-ticks whose top (country, industry) target changes."""
    cvs, shifts, top_changes = [], [], []
    for s in seeds:
        w = M0World(p, s)
        prev_share, prev_top = None, None
        marg = []
        for _ in range(p.n_ticks):
            w.step()
            A = w.stack_positions()
            G = np.stack([c.industry_output for c in w.countries])
            marg.append(p.q * G * np.exp(-p.q * A.sum(0)))
            tot = A.sum((1, 2), keepdims=True)
            share = np.divide(A, tot, out=np.zeros_like(A), where=tot > 1e-9).reshape(w.O, -1)
            top = share.argmax(1)
            act = np.array([o.active for o in w.oligarchs]) & (tot.reshape(-1) > 1e-9)
            if prev_share is not None:
                both = act & prev_act
                if both.any():
                    shifts.append(np.abs(share - prev_share).sum(1)[both].mean())
                    top_changes.append((top != prev_top)[both].mean())
            prev_share, prev_top, prev_act = share, top, act
        marg = np.array(marg)                                         # (T,C,NI)
        mu, sd = marg.mean(0), marg.std(0)
        cvs.append(float(np.nanmean(sd[mu > 1e-6] / mu[mu > 1e-6])))
    print(f"static-price check ({len(list(seeds))} seeds): CV over time of marginal return per pool = {np.mean(cvs):.2f}; "
          f"mean L1 allocation shift per tick = {np.mean(shifts):.3f}; top-target change rate = {np.mean(top_changes):.3f}")
    return dict(cv=float(np.mean(cvs)), shift=float(np.mean(shifts)), top_change=float(np.mean(top_changes)))


def report(p: Params = None, seeds=range(8)):
    p = p or Params()
    table = evaluate(p, seeds)
    print_table(table)
    pen, msgs = check(table)
    print(f"\ncalibration penalty = {pen:.3f}")
    for m in msgs:
        print("  -", m)
    return table, pen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--price-check", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    if a.report or not a.price_check:
        report(seeds=range(a.seeds))
    if a.price_check:
        static_price_check(Params())