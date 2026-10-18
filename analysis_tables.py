#!/usr/bin/env python3
"""
analysis_tables.py -- [E1] analysis tables from the per-run .npz logs (runlog.RunLog).

Run LOCALLY after run_design.py (needs pandas; nothing else beyond numpy and the project files).
    python analysis_tables.py --out out                          # main + typeblind, interior gammas only
    python analysis_tables.py --out out --blocks main --period 5 # coarser person-period (smaller S table)
    python analysis_tables.py --out out --include-edges          # also gamma 0.04 / 0.20
Writes <out>/tables/P_table.csv and <out>/tables/S_person_period.csv.gz, then prints a short summary.

P TABLE: one row per oligarch per run (ALL 25, so dropout can be reported by type).
  Windows come from each run's meta (burn-in 20%, embedding 20-50%, outcome 50-100% of ticks).
  in_primary = 1 if the oligarch is active at the START of the outcome window. PRIMARY population = in_primary
    (analysis A: outcomes averaged over each oligarch's ACTIVE outcome ticks; n_active_out gives the length).
  survivor   = 1 if the oligarch never exits. COMPARISON population B = in_primary & survivor (full-window).
    A is the reference for hypothesis P; B shows survivors' tendencies and doubles as a robustness check.
  Predictors (embedding window, active ticks): E_foreign (PRIMARY), E_all, E_pos, E_neg, fp_emb (mean foreign
    position = foreign-presence control), with_foreign_emb (share of ticks with foreign position > 0.1).
  Outcomes (outcome window, active ticks only; NaN if none):
    cap_share_mean / cap_share_median : per-tick foreign / (foreign + domestic) capture gained   [DV, robustness]
    cap_rate   : foreign capture gained per unit foreign position (sum over active ticks / sum of position)
    cap_abs    : mean foreign capture gained per active tick
    hav_share_mean / hav_share_median, hav_rate, hav_abs : same treatments for foreign havoc
  IDs: run_id, block, gamma, overlap_sign, rep, seed, oligarch, home, country_run (random-intercept group).
  Static: ruling, initial_wealth, initial_active, divestment_propensity, home_resilience, exit_tick, n_active_out.

S TABLE: person-period, one row per oligarch per period START while active (period = --period ticks, default 1).
  event = 1 if the oligarch exits during the period. All covariates are LAGGED: measured at the end of the tick
  before the period starts. Rows start at tick k+1 (exits cannot occur before exit_window anyway).
  cum_seized_rel_lag        (collapse seizure + resident liquid hit) cumulative / initial_active   [headline]
  cum_seized_forced_rel_lag same + forced-liquidation loss                                         [robustness]
  E_foreign_lag, E_all_lag, E_pos_lag, E_neg_lag, pos_foreign_lag, S_home_lag, S_foreign_mean_lag
  Static: ruling, initial_active, log_initial_active, divestment_propensity, gamma, overlap_sign, block, ids,
    t (tick number at period start), post_burn (period starts after the burn-in).
  Voluntary-divestment loss is EXCLUDED (circular); current active capital is NOT a covariate (it IS the exit rule).
  A built-in check asserts that the number of event rows equals the number of exits the period grid can see.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("analysis_tables.py needs pandas (pip install pandas)")


def _nan(fn, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=0)


def _ids(rl, n_rows_idx):
    m = rl.meta
    return dict(run_id=m["run_id"], block=m["block"], gamma=float(m["gamma"]),
                overlap_sign=float(m["overlap_sign"]), rep=int(m["rep"]), seed=int(m["seed"]))


# ---------------------------------------------------------------------------------------------- P
def p_table_run(rl) -> "pd.DataFrame":
    m, ol, st, cs = rl.meta, rl.olig, rl.static, rl.country_static
    (e0, e1), (o0, o1) = m["windows"]["embed"], m["windows"]["outcome"]
    act = ol["active"]
    a_e, a_o = act[e0:e1], act[o0:o1]

    def g(k, a, b):
        return ol[k][a:b].astype(float)

    def emb(k):
        return _nan(np.nanmean, np.where(a_e, g(k, e0, e1), np.nan))

    fc, dc, fh, dh = (g(k, o0, o1) for k in ("foreign_capture", "domestic_capture", "foreign_havoc", "domestic_havoc"))
    pf = g("pos_foreign", o0, o1)

    def share(num, den):
        ok = a_o & (den > 1e-9)
        return np.where(ok, num / np.where(den > 1e-9, den, 1.0), np.nan)

    def rate(num):
        d = np.where(a_o, pf, 0.0).sum(0)
        s = np.where(a_o, num, 0.0).sum(0)
        return np.where(d > 1e-9, s / np.where(d > 1e-9, d, 1.0), np.nan)

    def mean_act(x):
        return _nan(np.nanmean, np.where(a_o, x, np.nan))

    cs_, hs_ = share(fc, fc + dc), share(fh, fh + dh)
    et = st["exit_tick"].astype(float)
    home = st["home"].astype(int)
    O = rl.O
    df = pd.DataFrame(dict(
        oligarch=np.arange(O), home=home,
        country_run=[f"{m['run_id']}_c{h}" for h in home],
        ruling=st["ruling"].astype(int),
        initial_wealth=st["initial_wealth"].astype(float), initial_active=st["initial_active"].astype(float),
        divestment_propensity=st["divestment_propensity"].astype(float),
        home_resilience=cs["resilience"].astype(float)[home],
        exit_tick=et, survivor=np.isnan(et).astype(int), in_primary=act[o0].astype(int),
        n_active_out=a_o.sum(0),
        E_foreign=emb("E_foreign"), E_all=emb("E_all"), E_pos=emb("E_pos"), E_neg=emb("E_neg"),
        fp_emb=emb("pos_foreign"),
        with_foreign_emb=_nan(np.nanmean, np.where(a_e, (g("pos_foreign", e0, e1) > 0.1).astype(float), np.nan)),
        cap_share_mean=_nan(np.nanmean, cs_), cap_share_median=_nan(np.nanmedian, cs_),
        cap_rate=rate(fc), cap_abs=mean_act(fc),
        hav_share_mean=_nan(np.nanmean, hs_), hav_share_median=_nan(np.nanmedian, hs_),
        hav_rate=rate(fh), hav_abs=mean_act(fh),
    ))
    for k, v in _ids(rl, O).items():
        df.insert(0, k, v)
    return df


# ---------------------------------------------------------------------------------------------- S
def s_table_run(rl, k: int = 1) -> "pd.DataFrame":
    m, ol, st = rl.meta, rl.olig, rl.static
    T, O = rl.T, rl.O
    e0 = m["windows"]["embed"][0]
    act, ex = ol["active"], ol["exited_now"]
    ia = np.maximum(st["initial_active"].astype(float), 1e-12)
    f = lambda key: ol[key].astype(float)
    cum_c = np.cumsum(f("loss_seized") + f("loss_home_hit"), axis=0) / ia
    cum_f = np.cumsum(f("loss_seized") + f("loss_home_hit") + f("loss_forced_liq"), axis=0) / ia
    lag = {"E_foreign_lag": f("E_foreign"), "E_all_lag": f("E_all"), "E_pos_lag": f("E_pos"), "E_neg_lag": f("E_neg"),
           "pos_foreign_lag": f("pos_foreign"), "S_home_lag": f("S_home"), "S_foreign_mean_lag": f("S_foreign_mean")}
    parts, last_s = [], None
    for s in range(k, T - k + 1, k):
        idx = np.flatnonzero(act[s])
        last_s = s
        if idx.size == 0:
            continue
        d = dict(oligarch=idx, t=np.full(idx.size, s + 1), post_burn=np.full(idx.size, int(s >= e0)),
                 event=ex[s:s + k][:, idx].any(0).astype(int),
                 cum_seized_rel_lag=cum_c[s - 1, idx], cum_seized_forced_rel_lag=cum_f[s - 1, idx])
        for key, arr in lag.items():
            d[key] = arr[s - 1, idx]
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    o = cat["oligarch"]
    home = st["home"].astype(int)[o]
    df = pd.DataFrame(cat)
    df["home"] = home
    df["country_run"] = [f"{m['run_id']}_c{h}" for h in home]
    df["ruling"] = st["ruling"].astype(int)[o]
    df["initial_active"] = ia[o]
    df["log_initial_active"] = np.log(ia[o])
    df["divestment_propensity"] = st["divestment_propensity"].astype(float)[o]
    for key, v in _ids(rl, 0).items():
        df.insert(0, key, v)
    et = st["exit_tick"].astype(float)
    cover_end = (last_s + k) if last_s is not None else 0
    expected = int(((et >= k + 1) & (et <= cover_end)).sum())
    assert int(df["event"].sum()) == expected, \
        f"{m['run_id']}: {int(df['event'].sum())} event rows vs {expected} visible exits"
    return df


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--blocks", default="main,typeblind")
    ap.add_argument("--include-edges", action="store_true", help="also gamma 0.04 / 0.20 (default: P/S suite = interior)")
    ap.add_argument("--period", type=int, default=1, help="ticks per person-period row (default 1)")
    ap.add_argument("--tables-dir", default=None)
    a = ap.parse_args()
    from runlog import RunLog
    blocks = [b for b in a.blocks.split(",") if b]
    files = sorted(glob.glob(os.path.join(a.out, "runs", "*.npz")))
    if not files:
        sys.exit(f"no runs found in {os.path.join(a.out, 'runs')}")
    P, S, n = [], [], 0
    for fp in files:
        rl = RunLog.load(fp)
        m = rl.meta
        if m.get("block") not in blocks or (m.get("edge") and not a.include_edges):
            continue
        P.append(p_table_run(rl))
        S.append(s_table_run(rl, a.period))
        n += 1
        if n % 200 == 0:
            print(f"  {n} runs processed")
    if not P:
        sys.exit("no runs matched --blocks / edge filter")
    Pt, St = pd.concat(P, ignore_index=True), pd.concat(S, ignore_index=True)
    for c in St.columns:
        if St[c].dtype == np.float64 and c not in ("gamma", "overlap_sign"):
            St[c] = St[c].astype(np.float32)
    td = a.tables_dir or os.path.join(a.out, "tables")
    os.makedirs(td, exist_ok=True)
    Pt.to_csv(os.path.join(td, "P_table.csv"), index=False)
    St.to_csv(os.path.join(td, "S_person_period.csv.gz"), index=False)
    print(f"{n} runs | P rows {len(Pt)} | S rows {len(St)} (period={a.period}) -> {td}")
    prim = Pt[Pt.in_primary == 1]
    print("P: in_primary by block/type (A) and full-window survivors (B):")
    for (b, r), d in Pt.groupby(["block", "ruling"]):
        pr = d[d.in_primary == 1]
        print(f"  {b:9s} {'ruling' if r else 'civil ':6s} n={len(d):5d}  in_primary={len(pr):5d}  "
              f"exit_before_outcome={1 - len(pr) / len(d):.3f}  survivors_B={int(pr.survivor.sum()):5d}  "
              f"exit_during_outcome={1 - pr.survivor.mean():.3f}")
    print("S: events by block:", St.groupby("block").event.sum().to_dict(),
          "| NaN in covariates:", int(St.filter(like="_lag").isna().sum().sum()))
    print(f"P outcome NaN in primary: cap_share_mean {int(prim.cap_share_mean.isna().sum())}, "
          f"hav_share_mean {int(prim.hav_share_mean.isna().sum())}")


if __name__ == "__main__":
    main()