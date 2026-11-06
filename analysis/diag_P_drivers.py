#!/usr/bin/env python3
"""
diag_P_drivers.py -- READ-ONLY diagnostic (temporary file). EDA for hypothesis P before production: what drives the E-slope?
    python analysis/diag_P_drivers.py --out pilot30            # needs only pandas + numpy; reads <out>/tables/P_table.csv
Population A (in_primary), interior gammas, one block x overlap-sign group at a time (signs are never pooled, as in analysis_P).
 1  E and its mass at zero (by type); E_pos / E_neg; correlation of E with foreign presence and wealth.
 2  DV distributions (raw skew / kurtosis; share of zeros), by type.
 3  Binned E (deciles within group) vs foreign presence and DVs, by type.
 4  Slope ladder: the E slope as controls enter one at a time (type, fp_emb, log wealth, n_active_out, outcome-window foreign
    position pf_out = cap_abs / cap_rate), plus E_pos and E_neg entered separately. Explains the NEGATIVE rate / level slopes.
 5  Residual checks of the analysis_P model (pooled OLS here): skew, kurtosis, residual-vs-fitted, spread vs E, run / country ICC.
 6  Per-run jackknife of the E slope (run FE, controls partialled out; leave-one-run-out with the controls held at full-sample
    coefficients, i.e. approximate): SE, largest influence, sign changes.
All DVs follow analysis_P: share DVs as is, rate / level DVs as log1p(x / mean); everything z-scored in the group sample.
TRANSFORM RULE: judge transforms by distribution / residual shape ONLY, never by which slope sign they give (circular).
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)   # constant columns (e.g. E_neg = 0 under collusion) give NaN corr, shown as nan
import numpy as np
import pandas as pd

EDGES = (0.04, 0.20)
PRIM = ("cap_share_mean", "hav_share_mean")
ROB = ("cap_rate", "hav_rate", "cap_abs", "hav_abs")
LOG = set(ROB)


def z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def skew_kurt(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; s = x.std()
    if len(x) < 8 or s < 1e-12: return np.nan, np.nan
    u = (x - x.mean()) / s
    return float((u ** 3).mean()), float((u ** 4).mean() - 3)


def ols(d, y, xs, cl="run_id"):
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in xs]); Y = d[y].to_numpy(float)
    Ai = np.linalg.pinv(X.T @ X); b = Ai @ X.T @ Y; u = Y - X @ b
    ug, gi = np.unique(d[cl].to_numpy(), return_inverse=True)
    S = np.zeros((len(ug), X.shape[1])); np.add.at(S, gi, X * u[:, None]); G = len(ug)
    V = Ai @ (S.T @ S) @ Ai * G / max(G - 1, 1)
    return dict(b=dict(zip(["const"] + xs, b)), se=dict(zip(["const"] + xs, np.sqrt(np.diag(V)))), u=u, fit=X @ b)


def prep(sub, dv):
    d = sub.dropna(subset=[dv, "E_foreign", "fp_emb", "initial_wealth"]).copy()
    if len(d) < 50: return None
    y = d[dv].astype(float)
    if dv in LOG: y = np.log1p(y / max(float(y.mean()), 1e-12))
    d["y"], d["Ez"] = z(y), z(d["E_foreign"])
    d["Epz"], d["Enz"] = z(d["E_pos"]), z(d["E_neg"])
    d["fpz"], d["lwz"], d["nz"] = z(d["fp_emb"]), z(np.log(d["initial_wealth"])), z(d["n_active_out"])
    if {"cap_abs", "cap_rate"} <= set(d.columns):
        ok = d["cap_rate"] > 1e-9
        d["pfo"] = np.where(ok, d["cap_abs"] / np.where(ok, d["cap_rate"], 1.0), np.nan)   # mean foreign position over outcome ticks
        d["pfz"] = z(d["pfo"].fillna(d["pfo"].mean()))
    for g in sorted(d["gamma"].unique())[1:]:
        d[f"gm{g:.2f}"] = (d["gamma"] == g).astype(float)
    return d


def fwl_slope(d, ctrl):
    """run-FE slope of y on Ez after partialling controls; returns slope, per-run sxy/sxx, residual vectors."""
    cols = ["y", "Ez"] + ctrl
    M = d[cols].to_numpy(float); gi = pd.factorize(d["run_id"])[0]
    cnt = np.bincount(gi).astype(float)
    mu = np.zeros((cnt.size, M.shape[1])); np.add.at(mu, gi, M); mu /= cnt[:, None]
    D = M - mu[gi]
    if ctrl:
        C = D[:, 2:]; keep = C.std(0) > 1e-10; C = C[:, keep]
        B = np.linalg.lstsq(C, D[:, :2], rcond=None)[0] if C.shape[1] else np.zeros((0, 2))
        R = D[:, :2] - (C @ B if C.shape[1] else 0)
    else:
        R = D[:, :2]
    ey, ex = R[:, 0], R[:, 1]
    sxy = np.bincount(gi, weights=ex * ey); sxx = np.bincount(gi, weights=ex * ex)
    return sxy.sum() / sxx.sum(), sxy, sxx, gi


def group_report(sub, name, blocks_dvs, show_bins):
    A = sub
    print(f"\n{'=' * 100}\n{name}: n={len(A)} oligarchs, runs={A['run_id'].nunique()}")
    # 1
    print("-- 1. E distribution (embedding window; E_foreign primary) --")
    for lab, g in [("all", A), ("civil", A[A.ruling == 0]), ("ruling", A[A.ruling == 1])]:
        e = g["E_foreign"]
        print(f"  {lab:6s} n={len(g):5d} E==0: {np.mean(np.abs(e) < 1e-9):.3f}  E<0: {np.mean(e < -1e-9):.3f}  mean {e.mean():+.4f} sd {e.std():.4f} "
              f"q05/25/50/75/95 " + "/".join(f"{q:+.3f}" for q in e.quantile([.05, .25, .5, .75, .95])))
    print(f"  E_pos mean {A.E_pos.mean():+.4f} sd {A.E_pos.std():.4f} | E_neg mean {A.E_neg.mean():+.4f} sd {A.E_neg.std():.4f} | "
          f"corr(E, fp_emb) {A.E_foreign.corr(A.fp_emb):+.2f}  corr(E, with_foreign_emb) {A.E_foreign.corr(A.with_foreign_emb):+.2f}  "
          f"corr(E, log wealth) {A.E_foreign.corr(np.log(A.initial_wealth)):+.2f}  corr(E_pos, E_neg) {A.E_pos.corr(A.E_neg):+.2f}")
    # 2
    print("-- 2. DV distributions (raw; log1p(x/mean) for rate / level DVs) --")
    print("  dv               mean     sd   skew  exkurt  zero%   | transformed skew/exkurt | civil mean  ruling mean")
    for dv in blocks_dvs:
        x = A[dv].dropna(); sk, ku = skew_kurt(x)
        t = np.log1p(x / max(x.mean(), 1e-12)) if dv in LOG else x
        sk2, ku2 = skew_kurt(t)
        print(f"  {dv:15s} {x.mean():8.4f} {x.std():7.4f} {sk:6.2f} {ku:7.2f} {np.mean(x.abs() < 1e-12):6.1%}   | {sk2:6.2f} {ku2:7.2f}            | "
              f"{A.loc[A.ruling == 0, dv].mean():9.4f} {A.loc[A.ruling == 1, dv].mean():9.4f}")
    # 3
    d0 = prep(A, "cap_share_mean")
    if d0 is not None and show_bins:
        d0["bin"] = pd.qcut(d0["E_foreign"].rank(method="first"), 10, labels=False)
        print("-- 3. E deciles (within group): E mean, foreign presence, outcome-window foreign position, DV means by type --")
        print("  bin   E_mean    fp_emb  pf_out  | cap_share civil/ruling | hav_share civil/ruling | cap_rate | cap_abs")
        for b, g in d0.groupby("bin"):
            c, r = g[g.ruling == 0], g[g.ruling == 1]
            print(f"  {b:3d} {g.E_foreign.mean():+8.4f} {g.fp_emb.mean():7.3f} {g.pfo.mean():7.3f}  | {c.cap_share_mean.mean():.3f}/{r.cap_share_mean.mean():.3f}"
                  f"          | {c.hav_share_mean.mean():.3f}/{r.hav_share_mean.mean():.3f}          | {g.cap_rate.mean():8.3f} | {g.cap_abs.mean():8.3f}")
    # 4 ladder
    print("-- 4. E-slope ladder (OLS, gamma dummies, CR1 by run; slope = SD of DV per SD of E; 95% CI) --")
    print("  dv               E alone      +ruling      +fp_emb      +log wealth  +n_active    +pf_out      | E_pos alone   E_neg alone   (jointly: E_pos, E_neg)")
    for dv in blocks_dvs:
        d = prep(A, dv)
        if d is None: continue
        gm = [c for c in d.columns if c.startswith("gm")]
        steps, cur, cells = [("Ez",), ("ruling",), ("fpz",), ("lwz",), ("nz",), ("pfz",)], [], []
        for s in steps:
            cur = cur + list(s)
            if "pfz" in cur and "pfz" not in d.columns: cells.append("      n/a    "); continue
            r = ols(d, "y", ["Ez"] + [c for c in cur if c != "Ez"] + gm)
            b, se = r["b"]["Ez"], r["se"]["Ez"]; cells.append(f"{b:+.3f}±{1.96 * se:.3f}")
        rp = ols(d, "y", ["Epz", "ruling"] + gm); rn = ols(d, "y", ["Enz", "ruling"] + gm); rj = ols(d, "y", ["Epz", "Enz", "ruling"] + gm)
        print(f"  {dv:15s} " + " ".join(f"{c:12s}" for c in cells) +
              f" | {rp['b']['Epz']:+.3f}      {rn['b']['Enz']:+.3f}      ({rj['b']['Epz']:+.3f}, {rj['b']['Enz']:+.3f})")
    # 5 + 6
    print("-- 5. Residuals of the analysis_P-style model (pooled OLS: E, ruling, E x ruling, fp, wealth, n_active, gamma) --")
    print("  dv               resid skew exkurt  corr(|res|,fit)  corr(|res|,E)  ICC run  ICC country-run")
    jk = []
    for dv in blocks_dvs:
        d = prep(A, dv)
        if d is None: continue
        d["Er"] = d["Ez"] * d["ruling"]
        gm = [c for c in d.columns if c.startswith("gm")]
        r = ols(d, "y", ["Ez", "ruling", "Er", "fpz", "lwz", "nz"] + gm); u = r["u"]
        sk, ku = skew_kurt(u)
        def icc(col):
            g = pd.Series(u).groupby(d[col].to_numpy()); m = g.transform("mean").to_numpy(); n = g.transform("size").to_numpy()
            within = np.sum((u - m) ** 2) / max(len(u) - g.ngroups, 1); between = max(np.var(m[n > 0]) - within / max(n.mean(), 1), 0)
            return between / (between + within) if between + within > 0 else np.nan
        print(f"  {dv:15s} {sk:10.2f} {ku:6.2f}  {np.corrcoef(np.abs(u), r['fit'])[0, 1]:+14.2f}  {np.corrcoef(np.abs(u), d['Ez'])[0, 1]:+13.2f}  {icc('run_id'):7.3f}  {icc('country_run'):9.3f}")
        jk.append((dv, d))
    print("-- 6. Per-run jackknife of the E slope (run FE; controls: ruling, fp, wealth, n_active partialled out) --")
    print("  dv               slope   jackknife SE  max |change|  (run)           sign changes   top-3 runs' share of influence")
    for dv, d in jk:
        sl, sxy, sxx, gi = fwl_slope(d, ["ruling", "fpz", "lwz", "nz"])
        G = len(sxy); loo = (sxy.sum() - sxy) / (sxx.sum() - sxx)
        se = np.sqrt((G - 1) / G * np.sum((loo - loo.mean()) ** 2)); ch = np.abs(loo - sl)
        rid = pd.factorize(d["run_id"])[1]; top = np.argsort(-ch)[:3]
        print(f"  {dv:15s} {sl:+.3f}  {se:12.4f}  {ch.max():12.4f}  {rid[top[0]]:15s} {int(np.sum(np.sign(loo) != np.sign(sl))):8d}   {ch[top].sum() / max(ch.sum(), 1e-12):6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--tables-dir", default=None)
    ap.add_argument("--blocks", default="main")
    ap.add_argument("--no-bins", action="store_true")
    a = ap.parse_args()
    P = pd.read_csv(os.path.join(a.tables_dir or os.path.join(a.out, "tables"), "P_table.csv"))
    P = P[~P["gamma"].round(4).isin(EDGES) & (P["in_primary"] == 1)]
    for blk in [b for b in a.blocks.split(",") if b]:
        for s in sorted(P["overlap_sign"].unique()):
            sub = P[(P.block == blk) & (P.overlap_sign == s)]
            if len(sub): group_report(sub, f"block={blk} overlap_sign={s:+.0f} (population A)", PRIM + ROB, not a.no_bins)
    print("\nReminder: choose DV transforms from sections 2 and 5 only; do not pick by slope sign. Everything here is descriptive.")


if __name__ == "__main__":
    main()