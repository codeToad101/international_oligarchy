#!/usr/bin/env python3
"""
analysis_S.py -- [E3] hypothesis S: what removes an oligarch from the stage -- asset loss, lack of network ties, or type?
Run LOCALLY on the E1 person-period table (numpy + pandas only; the logit is implemented here, no statsmodels needed):
    python analysis_S.py --out out                      # main + typeblind, pooled over overlap signs (as in the plan)
    python analysis_S.py --out out --by-sign            # also one fit per overlap sign (sensitivity)
    python analysis_S.py --out out --period 5           # must match the --period used in analysis_tables.py

MODEL: discrete-time hazard. One row per oligarch-period while active; event = exits in the period; covariates lagged.
  Logit, cluster-robust (CR1) standard errors clustered on run (Liang & Zeger 1986). Continuous predictors are z-scored, so
  odds ratios (OR) are per +1 SD; type is 0/1 (ruling vs civil). Nested sequence:
    M0 = baseline hazard (t, t^2) + log initial active capital + gamma + overlap      [t terms: standard for discrete-time hazards]
    M1 = M0 + seized fraction (collapse seizure + resident liquid hit, cumulative / initial active capital, lagged)
    M2 = M1 + E_foreign (lagged)
    M3 = M2 + type        <- FULL MODEL
    M4 = M3 + foreign position (lagged)    [robustness: E = 0 conflates no ties with offsetting ties]
    M5 = M3 + recent seizure / prior capital (rec10_pre_lag, z)   [EXPLORATORY, not pre-specified; increments vs M3. Purpose: check
    whether the headline seized fraction's protective sign is a size/deployment artifact; overview 6.3, 13.7]
  Reported per model: BIC (n = rows), in-sample AUC (rank-based), McFadden pseudo-R2 (1 - LL/LL0), and increments vs the previous
  model (dBIC positive = lower BIC = the added term helps). Then drop-one-from-full: each term removed from M3 (DeltaBIC > 0 and DeltaAUC > 0 mean the term helps).
  Rows whose period cannot contain a possible exit tick (< --min-exit-tick, default 10 = exit_window) are dropped.
Output: <out>/tables/S_models.csv, S_terms.csv (ORs of M3 and M4), S_dropone.csv, and a printed summary.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("analysis_S.py needs pandas (pip install pandas)")

Z95 = 1.96


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def logit(X, y, g, tol=1e-8, maxit=60):
    """Newton/IRLS logistic regression with CR1 cluster-robust covariance. Returns dict(b, V, ll, p, conv)."""
    n, k = X.shape
    pb = float(y.mean())
    b = np.zeros(k)
    b[0] = math.log(pb / (1.0 - pb))
    conv = False
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-12)
        XtW = X.T * w
        bn = np.linalg.solve(XtW @ X, XtW @ (eta + (y - p) / w))
        done = np.max(np.abs(bn - b)) < tol
        b = bn
        if done:
            conv = True
            break
    eta = np.clip(X @ b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    w = p * (1 - p)
    Hi = np.linalg.inv((X.T * w) @ X)
    sc = X * (y - p)[:, None]
    ug, gi = np.unique(g, return_inverse=True)
    Sg = np.column_stack([np.bincount(gi, weights=sc[:, j], minlength=len(ug)) for j in range(k)])
    G = len(ug)
    return dict(b=b, V=Hi @ (Sg.T @ Sg) @ Hi * G / max(G - 1, 1), ll=ll, p=p, conv=conv)


def auc(p, y):
    r = pd.Series(p).rank().to_numpy()
    n1 = float(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_cols(d, cols):
    cols = [c for c in cols if d[c].std() > 1e-12]                 # drop constants (e.g. overlap inside a by-sign fit)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    f = logit(X, d["event"].to_numpy(float), d["run_id"].to_numpy())
    n, y = len(d), d["event"].to_numpy()
    k = X.shape[1]
    p0 = y.mean()
    ll0 = float(y.sum() * math.log(p0) + (n - y.sum()) * math.log(1 - p0))
    f.update(cols=cols, k=k, n=n, bic=-2 * f["ll"] + k * math.log(n), auc=auc(f["p"], y), mcf=1 - f["ll"] / ll0)
    return f


def prepare(df, seized_var):
    d = df.copy()
    d["t_z"], d["t2_z"] = z(d["t"]), z(d["t"].astype(float) ** 2)
    d["lia_z"], d["gamma_z"] = z(d["log_initial_active"]), z(d["gamma"])
    d["overlap"] = d["overlap_sign"].astype(float)
    d["seized_z"], d["E_z"], d["fp_z"] = z(d[seized_var]), z(d["E_foreign_lag"]), z(d["pos_foreign_lag"])
    if "rec10_pre_lag" in d.columns:
        d["rec_z"] = z(d["rec10_pre_lag"])
    return d


BASE = ["t_z", "t2_z", "lia_z", "gamma_z", "overlap"]
MODELS = {"M0": BASE, "M1": BASE + ["seized_z"], "M2": BASE + ["seized_z", "E_z"],
          "M3": BASE + ["seized_z", "E_z", "ruling"], "M4": BASE + ["seized_z", "E_z", "ruling", "fp_z"],
          "M5": BASE + ["seized_z", "E_z", "ruling", "rec_z"]}
DROP_TERMS = ["lia_z", "gamma_z", "overlap", "seized_z", "E_z", "ruling"]
LABEL = {"lia_z": "initial capital", "gamma_z": "gamma", "overlap": "overlap sign", "seized_z": "seized fraction",
         "E_z": "embeddedness E", "ruling": "type (ruling)", "rec_z": "recent seizure / prior capital"}
PARENT = {"M1": "M0", "M2": "M1", "M3": "M2", "M4": "M3", "M5": "M3"}   # increments are reported against the parent


def run_group(d, block, group):
    models, terms, drops, fits = [], [], [], {}
    for name, cols in MODELS.items():
        if "rec_z" in cols and "rec_z" not in d.columns:
            continue                                    # exploratory M5 needs rec10_pre_lag in the S table
        f = fit_cols(d, cols)
        fits[name] = f
        prev = fits.get(PARENT.get(name))
        models.append(dict(block=block, group=group, model=name, n=f["n"], events=int(d["event"].sum()), k=f["k"], ll=f["ll"],
                           bic=f["bic"], auc=f["auc"], mcfadden=f["mcf"], converged=f["conv"],
                           d_bic_prev=(prev["bic"] - f["bic"]) if prev else np.nan,
                           d_auc_prev=(f["auc"] - prev["auc"]) if prev else np.nan,
                           d_mcf_prev=(f["mcf"] - prev["mcf"]) if prev else np.nan))
    for name in ("M3", "M4", "M5"):
        if name not in fits:
            continue
        f = fits[name]
        se = np.sqrt(np.diag(f["V"]))
        for i, c in enumerate(f["cols"], start=1):
            b = f["b"][i]
            terms.append(dict(block=block, group=group, model=name, term=c, OR=math.exp(b), OR_lo=math.exp(b - Z95 * se[i]),
                              OR_hi=math.exp(b + Z95 * se[i]), coef=b, se=se[i],
                              p=math.erfc(abs(b / se[i]) / math.sqrt(2.0))))
    full = fits["M3"]
    for t in DROP_TERMS:
        if t not in full["cols"]:
            continue
        f = fit_cols(d, [c for c in MODELS["M3"] if c != t])
        drops.append(dict(block=block, group=group, dropped=t, d_bic=f["bic"] - full["bic"], d_auc=full["auc"] - f["auc"],
                          d_mcf=full["mcf"] - f["mcf"]))
    return models, terms, drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--tables-dir", default=None)
    ap.add_argument("--blocks", default="main,typeblind")
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--min-exit-tick", type=int, default=10)
    ap.add_argument("--seized-var", default="cum_seized_rel_lag", choices=["cum_seized_rel_lag", "cum_seized_forced_rel_lag"])
    ap.add_argument("--by-sign", action="store_true")
    a = ap.parse_args()
    td = a.tables_dir or os.path.join(a.out, "tables")
    path = os.path.join(td, "S_person_period.csv.gz")
    have = set(pd.read_csv(path, nrows=0).columns)
    cols = ["run_id", "block", "gamma", "overlap_sign", "t", "event", "ruling", "log_initial_active", "cum_seized_rel_lag",
            "cum_seized_forced_rel_lag", "E_foreign_lag", "pos_foreign_lag"] + (["rec10_pre_lag"] if "rec10_pre_lag" in have else [])
    if "rec10_pre_lag" not in have:
        print("note: rec10_pre_lag not in the S table; rebuild it with analysis_tables.py to get exploratory M5")
    S = pd.read_csv(path, usecols=cols)
    S = S[S["t"] + a.period - 1 >= a.min_exit_tick]
    M, T, D = [], [], []
    for blk in [b for b in a.blocks.split(",") if b]:
        base = S[S["block"] == blk]
        if base.empty:
            continue
        groups = [("pooled", base)]
        if a.by_sign:
            groups += [(f"s={s:+.0f}", base[base["overlap_sign"] == s]) for s in sorted(base["overlap_sign"].unique())]
        for gname, sub in groups:
            if int(sub["event"].sum()) < 20:
                print(f"skip {blk} {gname}: only {int(sub['event'].sum())} events")
                continue
            m, t, dr = run_group(prepare(sub, a.seized_var), blk, gname)
            M += m; T += t; D += dr
            print(f"\n=== {blk} / {gname}: {m[0]['n']} rows, {m[0]['events']} events (OR per +1 SD; type 0/1) ===")
            print("model   k      BIC      AUC  McFadden   dBIC_prev  dAUC_prev  dMcF_prev")
            for r in m:
                print(f"{r['model']:5s} {r['k']:3d} {r['bic']:10.1f} {r['auc']:8.4f} {r['mcfadden']:9.4f} "
                f"{r['d_bic_prev']:10.1f} {r['d_auc_prev']:10.4f} {r['d_mcf_prev']:10.4f}"
                + ("" if r["converged"] else "  NOT CONVERGED") + ("   <- EXPLORATORY, increments vs M3" if r["model"] == "M5" else ""))
            print("full model (M3) odds ratios [95% CI]:")
            for r in t:
                if r["model"] == "M3":
                    print(f"  {LABEL.get(r['term'], r['term']):18s} OR {r['OR']:.3f} [{r['OR_lo']:.3f}, {r['OR_hi']:.3f}]")
            if any(r["model"] == "M5" for r in t):
                print("EXPLORATORY M5 (= M3 + recent seizure / prior capital; not pre-specified) odds ratios [95% CI]:")
                for r in t:
                    if r["model"] == "M5" and r["term"] in ("seized_z", "rec_z"):
                        print(f"  {LABEL[r['term']]:32s} OR {r['OR']:.3f} [{r['OR_lo']:.3f}, {r['OR_hi']:.3f}]")
            print("drop-one-from-full (positive = term helps):  dBIC   dAUC    dMcF")
            for r in dr:
                print(f"  {LABEL[r['dropped']]:18s} {r['d_bic']:10.1f} {r['d_auc']:7.4f} {r['d_mcf']:8.4f}")
    if not M:
        sys.exit("nothing fitted")
    pd.DataFrame(M).to_csv(os.path.join(td, "S_models.csv"), index=False)
    pd.DataFrame(T).to_csv(os.path.join(td, "S_terms.csv"), index=False)
    pd.DataFrame(D).to_csv(os.path.join(td, "S_dropone.csv"), index=False)
    print(f"\nwritten: S_models.csv, S_terms.csv, S_dropone.csv in {td}")


if __name__ == "__main__":
    main()