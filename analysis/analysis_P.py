#!/usr/bin/env python3
"""
analysis_P.py -- [E2] hypothesis P: does foreign capture / havoc rise with embeddedness E, similarly for civil and ruling?
Run LOCALLY on the E1 table:   python analysis_P.py --out out            (needs pandas; statsmodels for --engine mixed)
                               python analysis_P.py --out out --robust   (adds the robustness DVs)
                               python analysis_P.py --out pilot_out --engine fe    (numpy-only fallback)
                               python analysis_P.py --out out --diagnostics   (descriptive: what drives the E slope; --bins adds E deciles)

DESIGN (fixed before real runs; see project_overview.txt 6.4)
  Population A (primary) = in_primary (active at outcome-window start; outcomes over active ticks, n_active_out controlled).
  Population B (comparison / robustness) = full-window survivors. Blocks: main, typeblind (= positive control).
  One fit per (block, overlap_sign, population, DV): the E construction differs by sign, so signs are never pooled.
  DVs: cap_share_mean, hav_share_mean (primary; used as is). Robustness (--robust): median shares as is; cap_rate, hav_rate,
    cap_abs, hav_abs as log1p(x / mean(x)). Every DV is z-scored within the fit sample; E, foreign presence (fp_emb),
    log initial wealth and n_active_out likewise. Interior gamma enters as dummies.
  Models (engine mixed): y ~ Ez + ruling + [Ez:ruling] + fp + log wealth + n_active + gamma dummies,
    random intercepts for run and country-within-run (statsmodels MixedLM, REML).  Engine fe: run fixed effects + CR1
    run-clustered errors (within-run slopes only; also useful as a sensitivity check on the mixed model).
  Equivalence (TOST): diff = E-slope(ruling) - E-slope(civil) = coefficient of Ez:ruling. Margin Delta = m x |pooled slope|,
    pooled slope = E-slope of the SAME model without the interaction. m = 0.15 primary; 0.10 and 0.20 sensitivity.
    Equivalent iff the 90% CI of diff lies inside (-Delta, +Delta). If the pooled slope's 95% CI includes 0: "n/a".
    The typeblind block is the positive control: failure at 0.15 means the test is precision-limited (the margin is never widened).
  HEAD-TO-HEAD (the project's question): E effect per SD vs the type level effect (same DV units), and each one's UNIQUE R2 =
    R2(full) - R2(full without it) (marginal R2, Nakagawa & Schielzeth, for mixed; within-run R2 for fe).
  POWER (planning): power_m = TOST power if the true difference were 0 at the observed SE; need_x_m = factor by which the
    sample must grow for 80% power (SE <= Delta / 2.927). Use on a pilot with seeds disjoint from production.
  Also reported: per-type slopes with 95% CI (positivity), and E^2 / E^2 x type coefficients (curvature; CI only, no verdict).
Output: <out>/tables/P_results.csv and a printed summary.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("analysis_P.py needs pandas (pip install pandas)")
try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

MARGINS = (0.15, 0.10, 0.20)                       # primary first
PRIMARY_DVS = ("cap_share_mean", "hav_share_mean")
ROBUST_DVS = ("cap_share_median", "hav_share_median", "cap_rate", "hav_rate", "cap_abs", "hav_abs")
LOG_DVS = {"cap_rate", "hav_rate", "cap_abs", "hav_abs"}
EDGES = (0.04, 0.20)
Z90, Z95 = 1.6449, 1.96


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


class Fit:
    def __init__(self, names, b, V, n, converged=True, r2=float("nan")):
        self.names, self.b, self.V, self.n, self.converged, self.r2 = list(names), np.asarray(b, float), np.asarray(V, float), n, converged, r2

    def lin(self, w):
        v = np.array([w.get(nm, 0.0) for nm in self.names])
        for nm in w:
            if nm not in self.names:
                raise KeyError(f"term {nm} not in model (dropped as constant?)")
        return float(v @ self.b), float(np.sqrt(max(v @ self.V @ v, 0.0)))


def fit_mixed(d, cols):
    f = "y ~ " + " + ".join(cols)
    md = smf.mixedlm(f, d, groups=d["run_id"], vc_formula={"cty": "0 + C(country_run)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = md.fit(reml=True)
    k = len(res.fe_params)
    vf = float(np.var(np.asarray(res.model.exog, float) @ np.asarray(res.fe_params, float)))
    vr = float(np.asarray(res.cov_re).ravel()[0]) if getattr(res, "cov_re", None) is not None else 0.0
    vc = float(np.sum(res.vcomp)) if getattr(res, "vcomp", None) is not None and len(res.vcomp) else 0.0
    return Fit(res.fe_params.index, res.fe_params.values, np.asarray(res.cov_params())[:k, :k], len(d),
               bool(getattr(res, "converged", True)), vf / (vf + vr + vc + float(res.scale)))


def fit_fe(d, cols):
    X, y = d[cols].to_numpy(float), d["y"].to_numpy(float)
    ug, gi = np.unique(d["run_id"].to_numpy(), return_inverse=True)
    cnt = np.bincount(gi).astype(float)

    def dm(a):
        m = np.zeros((len(ug),) + a.shape[1:])
        np.add.at(m, gi, a)
        m /= cnt.reshape((-1,) + (1,) * (a.ndim - 1))
        return a - m[gi]
    Xd, yd = dm(X), dm(y)
    keep = Xd.std(0) > 1e-10                       # run-constant columns (gamma dummies) are absorbed
    Xd, names = Xd[:, keep], [c for c, k in zip(cols, keep) if k]
    Ai = np.linalg.inv(Xd.T @ Xd)
    b = Ai @ (Xd.T @ yd)
    u = yd - Xd @ b
    S = np.zeros((len(ug), Xd.shape[1]))
    np.add.at(S, gi, Xd * u[:, None])
    G = len(ug)
    r2 = 1.0 - float((u ** 2).sum() / max((yd ** 2).sum(), 1e-12))          # WITHIN-run R2
    return Fit(names, b, Ai @ (S.T @ S) @ Ai * G / max(G - 1, 1), len(d), True, r2)


def prepare(sub, dv):
    d = sub.dropna(subset=[dv, "E_foreign", "fp_emb", "initial_wealth"]).copy()
    if len(d) < 50 or d["ruling"].nunique() < 2:
        return None
    y = d[dv].astype(float)
    if dv in LOG_DVS:
        y = np.log1p(y / max(float(y.mean()), 1e-12))
    d["y"], d["Ez"] = z(y), z(d["E_foreign"])
    d["Ez2"] = d["Ez"] ** 2
    d["fpz"], d["lwz"], d["nactz"] = z(d["fp_emb"]), z(np.log(d["initial_wealth"])), z(d["n_active_out"])
    d["Ez_r"], d["Ez2_r"] = d["Ez"] * d["ruling"], d["Ez2"] * d["ruling"]
    gl = sorted(d["gamma"].unique())[1:]
    for i, g in enumerate(gl):
        d[f"gm{i}"] = (d["gamma"] == g).astype(float)
    return d, [f"gm{i}" for i in range(len(gl))]


def analyze(sub, dv, engine, block, sign, pop):
    row = dict(block=block, overlap_sign=sign, pop=pop, dv=dv, engine=engine)
    prep = prepare(sub, dv)
    if prep is None:
        return dict(row, note="too few rows / one type only")
    d, gm = prep
    ctrl = [c for c in ("fpz", "lwz", "nactz") if d[c].std() > 1e-12] + gm
    fit = fit_mixed if engine == "mixed" else fit_fe
    try:
        f0 = fit(d, ["Ez", "ruling"] + ctrl)
        f1 = fit(d, ["Ez", "ruling", "Ez_r"] + ctrl)
        f2 = fit(d, ["Ez", "Ez2", "ruling", "Ez_r", "Ez2_r"] + ctrl)
        pooled, pse = f0.lin({"Ez": 1})
        civ, cse = f1.lin({"Ez": 1})
        rul, rse = f1.lin({"Ez": 1, "Ez_r": 1})
        diff, dse = f1.lin({"Ez_r": 1})
        c2, c2se = f2.lin({"Ez2": 1})
        c2r, c2rse = f2.lin({"Ez2_r": 1})
        typ, tse = f0.lin({"ruling": 1})                       # type level effect at mean E (no interaction)
        f_dE, f_dT = fit(d, ["ruling"] + ctrl), fit(d, ["Ez"] + ctrl)   # full model f0 without E / without type
    except Exception as e:                           # singular fits etc.: record, keep going
        return dict(row, n=len(d), note=f"fit error: {type(e).__name__}: {str(e)[:80]}")
    row.update(n=len(d), n_runs=int(d["run_id"].nunique()), converged=bool(f0.converged and f1.converged and f2.converged),
               pooled=pooled, pooled_se=pse, civil=civ, civil_se=cse, ruling=rul, ruling_se=rse,
               diff=diff, diff_se=dse, ci90_lo=diff - Z90 * dse, ci90_hi=diff + Z90 * dse,
               pos_civil=civ - Z95 * cse > 0, pos_ruling=rul - Z95 * rse > 0,
               type_effect=typ, type_se=tse, r2_kind="marginal" if engine == "mixed" else "within", r2_full=f0.r2,
               r2_unique_E=f0.r2 - f_dE.r2, r2_unique_type=f0.r2 - f_dT.r2, r2_interaction=f1.r2 - f0.r2,
               curv_E2=c2, curv_E2_se=c2se, curv_E2xruling=c2r, curv_E2xruling_se=c2rse)
    weak = pooled - Z95 * pse <= 0 <= pooled + Z95 * pse
    for m in MARGINS:
        D = m * abs(pooled)
        p = max(1 - Phi((diff + D) / dse), 1 - Phi((D - diff) / dse))
        row[f"delta_{m}"], row[f"p_{m}"] = D, p
        row[f"equiv_{m}"] = "n/a" if weak else ("yes" if p < 0.05 else "no")
        # planning numbers: TOST power IF the true difference were 0, at this SE; and the sample-size multiplier for 80% power
        row[f"power_{m}"] = max(0.0, 2 * Phi(D / dse - 1.6449) - 1) if D > 0 and dse > 0 else 0.0
        row[f"need_x_{m}"] = (dse / (D / 2.927)) ** 2 if D > 0 else float("inf")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--tables-dir", default=None)
    ap.add_argument("--blocks", default="main,typeblind")
    ap.add_argument("--engine", default="mixed", choices=["mixed", "fe"])
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--diagnostics", action="store_true", help="descriptive EDA of what drives the E slope (overview 6.4, 13.8)")
    ap.add_argument("--bins", action="store_true", help="with --diagnostics: also print E deciles")
    a = ap.parse_args()
    engine = a.engine
    if engine == "mixed" and not HAVE_SM:
        print("statsmodels not installed -> falling back to --engine fe (run fixed effects, clustered SEs). "
              "Install statsmodels for the planned mixed models.")
        engine = "fe"
    td = a.tables_dir or os.path.join(a.out, "tables")
    P = pd.read_csv(os.path.join(td, "P_table.csv"))
    P = P[~P["gamma"].round(4).isin(EDGES) & (P["in_primary"] == 1)]
    dvs = PRIMARY_DVS + (ROBUST_DVS if a.robust else ())
    rows = []
    for blk in [b for b in a.blocks.split(",") if b]:
        for s in sorted(P["overlap_sign"].unique()):
            base = P[(P["block"] == blk) & (P["overlap_sign"] == s)]
            for pop, sub in (("A", base), ("B", base[base["survivor"] == 1])):
                for dv in dvs:
                    rows.append(analyze(sub, dv, engine, blk, s, pop))
    R = pd.DataFrame(rows)
    path = os.path.join(td, "P_results.csv")
    R.to_csv(path, index=False)
    ok = R[R.get("note").isna()] if "note" in R else R
    print(f"engine={engine}; {len(R)} fits ({len(R) - len(ok)} failed) -> {path}")
    print("slopes are per SD of E in SD of the DV; diff = ruling - civil; equiv at margins " + "/".join(str(m) for m in MARGINS))
    for _, r in ok.iterrows():
        eq = " ".join(f"{m}:{r[f'equiv_{m}']}" for m in MARGINS)
        print(f"{r.block:9s} s={r.overlap_sign:+.0f} {r['pop']} {r.dv:16s} n={int(r.n):5d} pooled {r.pooled:+.3f} "
              f"| civil {r.civil:+.3f} ruling {r.ruling:+.3f} | diff {r['diff']:+.3f} 90%CI [{r.ci90_lo:+.3f},{r.ci90_hi:+.3f}] | {eq} "
              f"| power@.15 {r['power_0.15']:.0%} need x{r['need_x_0.15']:.1f} samples for 80%")
    for _, r in R[R.get("note").notna()].iterrows() if "note" in R else []:
        print(f"FAILED {r.block} s={r.overlap_sign:+.0f} {r['pop']} {r.dv}: {r.note}")
    h = ok[(ok["pop"] == "A") & ok.dv.isin(PRIMARY_DVS)]
    if len(h):
        print(f"\nHEAD-TO-HEAD (population A, primary DVs; effects in SD of the DV; unique R2 = {h.r2_kind.iloc[0]}): "
              "E = per +1 SD of E; type = ruling - civil at mean E")
        for _, r in h.iterrows():
            print(f"{r.block:9s} s={r.overlap_sign:+.0f} {r.dv:16s} E {r.pooled:+.3f} [{r.pooled - Z95 * r.pooled_se:+.3f},{r.pooled + Z95 * r.pooled_se:+.3f}] | "
                  f"type {r.type_effect:+.3f} [{r.type_effect - Z95 * r.type_se:+.3f},{r.type_effect + Z95 * r.type_se:+.3f}] | "
                  f"unique R2: E {r.r2_unique_E:.4f}  type {r.r2_unique_type:.4f}  E x type {r.r2_interaction:.4f}")
    ctl = ok[(ok.block == "typeblind") & (ok["pop"] == "A") & ok.dv.isin(PRIMARY_DVS)]
    if len(ctl):
        n_ok = int((ctl["equiv_0.15"] == "yes").sum())
        print(f"POSITIVE CONTROL (typeblind, A, primary DVs, margin 0.15): {n_ok}/{len(ctl)} equivalent -> "
              + ("OK" if n_ok == len(ctl) else "PRECISION-LIMITED: report as such; do NOT widen the margin"))
    bad = ok[~ok.converged] if "converged" in ok else ok.iloc[0:0]
    if len(bad):
        print(f"WARNING: {len(bad)} fits flagged non-converged (see P_results.csv)")
    if a.diagnostics:
        run_diagnostics(P, [b for b in a.blocks.split(",") if b], a.bins)


# ---------------------------------------------------------------------------------------------- diagnostics (--diagnostics)
# DESCRIPTIVE only (no verdicts, no p-values). Purpose: see what drives the E slope (overview 13.8). Population A, interior gammas,
# one block x overlap-sign group at a time. Sections: 1 E distribution; 2 DV distributions; 3 E deciles (--bins); 4 slope ladder
# (E alone, then +ruling, +fp_emb, +log wealth, +n_active, +pf_out; E_pos / E_neg separately); 5 residual checks; 6 per-run jackknife.
# pf_out = cap_abs / cap_rate (mean foreign position in the outcome window) is derived FROM the DV, so the "+pf_out" step is
# partly an identity: read the +fp_emb step (embedding window) as the clean evidence. dom_abs (if in the table) is shown as one more
# level DV so a share's numerator (cap_abs) and denominator (dom_abs) can be told apart. TRANSFORMS are judged by distribution /
# residual shape ONLY, never by slope sign (circular). All fits here are pooled OLS with CR1 run-clustered SEs (not the planned mixed model).
DIAG_LOG = LOG_DVS | {"dom_abs"}


def _skew_kurt(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = x.std()
    if len(x) < 8 or s < 1e-12:
        return float("nan"), float("nan")
    u = (x - x.mean()) / s
    return float((u ** 3).mean()), float((u ** 4).mean() - 3)


def _ols_cr(d, y, xs):
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in xs])
    Y = d[y].to_numpy(float)
    Ai = np.linalg.pinv(X.T @ X)
    b = Ai @ X.T @ Y
    u = Y - X @ b
    ug, gi = np.unique(d["run_id"].to_numpy(), return_inverse=True)
    S = np.zeros((len(ug), X.shape[1]))
    np.add.at(S, gi, X * u[:, None])
    G = len(ug)
    V = Ai @ (S.T @ S) @ Ai * G / max(G - 1, 1)
    return dict(b=dict(zip(["const"] + xs, b)), se=dict(zip(["const"] + xs, np.sqrt(np.diag(V)))), u=u, fit=X @ b)


def _diag_prep(sub, dv):
    d = sub.dropna(subset=[dv, "E_foreign", "fp_emb", "initial_wealth"]).copy()
    if len(d) < 50:
        return None
    y = d[dv].astype(float)
    if dv in DIAG_LOG:
        y = np.log1p(y / max(float(y.mean()), 1e-12))
    d["y"], d["Ez"] = z(y), z(d["E_foreign"])
    d["Epz"], d["Enz"] = z(d["E_pos"]), z(d["E_neg"])
    d["fpz"], d["lwz"], d["nz"] = z(d["fp_emb"]), z(np.log(d["initial_wealth"])), z(d["n_active_out"])
    if {"cap_abs", "cap_rate"} <= set(d.columns):
        ok = d["cap_rate"] > 1e-9
        d["pfo"] = np.where(ok, d["cap_abs"] / np.where(ok, d["cap_rate"], 1.0), np.nan)
        d["pfz"] = z(d["pfo"].fillna(d["pfo"].mean()))
    for g in sorted(d["gamma"].unique())[1:]:
        d[f"gm{g:.2f}"] = (d["gamma"] == g).astype(float)
    return d


def _fwl(d, ctrl):
    """Run-FE slope of y on Ez with controls partialled out (Frisch-Waugh); per-run sums allow an approximate leave-one-run-out."""
    M = d[["y", "Ez"] + ctrl].to_numpy(float)
    gi = pd.factorize(d["run_id"])[0]
    cnt = np.bincount(gi).astype(float)
    mu = np.zeros((cnt.size, M.shape[1]))
    np.add.at(mu, gi, M)
    mu /= cnt[:, None]
    D = M - mu[gi]
    C = D[:, 2:]
    C = C[:, C.std(0) > 1e-10]
    R = D[:, :2] - (C @ np.linalg.lstsq(C, D[:, :2], rcond=None)[0] if C.shape[1] else 0)
    sxy, sxx = np.bincount(gi, weights=R[:, 1] * R[:, 0]), np.bincount(gi, weights=R[:, 1] ** 2)
    return sxy.sum() / sxx.sum(), sxy, sxx


def _diag_group(A, name, dvs, bins):
    print(f"\n{'=' * 100}\n{name}: n={len(A)} oligarchs, runs={A['run_id'].nunique()}")
    print("-- 1. E distribution (embedding window; E_foreign primary) --")
    for lab, g in [("all", A), ("civil", A[A.ruling == 0]), ("ruling", A[A.ruling == 1])]:
        e = g["E_foreign"]
        print(f"  {lab:6s} n={len(g):5d} E==0: {np.mean(np.abs(e) < 1e-9):.3f}  E<0: {np.mean(e < -1e-9):.3f}  mean {e.mean():+.4f} sd {e.std():.4f} "
              f"q05/25/50/75/95 " + "/".join(f"{q:+.3f}" for q in e.quantile([.05, .25, .5, .75, .95])))
    print(f"  E_pos mean {A.E_pos.mean():+.4f} sd {A.E_pos.std():.4f} | E_neg mean {A.E_neg.mean():+.4f} sd {A.E_neg.std():.4f} | "
          f"corr(E, fp_emb) {A.E_foreign.corr(A.fp_emb):+.2f}  corr(E, with_foreign_emb) {A.E_foreign.corr(A.with_foreign_emb):+.2f}  "
          f"corr(E, log wealth) {A.E_foreign.corr(np.log(A.initial_wealth)):+.2f}  corr(E_pos, E_neg) {A.E_pos.corr(A.E_neg):+.2f}")
    print("-- 2. DV distributions (raw; log1p(x/mean) for rate / level DVs) --")
    print("  dv               mean     sd   skew  exkurt  zero%   | transformed skew/exkurt | civil mean  ruling mean")
    for dv in dvs:
        x = A[dv].dropna()
        sk, ku = _skew_kurt(x)
        sk2, ku2 = _skew_kurt(np.log1p(x / max(x.mean(), 1e-12)) if dv in DIAG_LOG else x)
        print(f"  {dv:15s} {x.mean():8.4f} {x.std():7.4f} {sk:6.2f} {ku:7.2f} {np.mean(x.abs() < 1e-12):6.1%}   | {sk2:6.2f} {ku2:7.2f}            | "
              f"{A.loc[A.ruling == 0, dv].mean():9.4f} {A.loc[A.ruling == 1, dv].mean():9.4f}")
    d0 = _diag_prep(A, "cap_share_mean")
    if d0 is not None and bins:
        d0["bin"] = pd.qcut(d0["E_foreign"].rank(method="first"), 10, labels=False)
        print("-- 3. E deciles (within group): E mean, foreign presence, outcome-window foreign position, DV means by type --")
        print("  bin   E_mean    fp_emb  pf_out  | cap_share civil/ruling | hav_share civil/ruling | cap_rate | cap_abs")
        for b, g in d0.groupby("bin"):
            c, r = g[g.ruling == 0], g[g.ruling == 1]
            print(f"  {b:3d} {g.E_foreign.mean():+8.4f} {g.fp_emb.mean():7.3f} {g.pfo.mean():7.3f}  | {c.cap_share_mean.mean():.3f}/{r.cap_share_mean.mean():.3f}"
                  f"          | {c.hav_share_mean.mean():.3f}/{r.hav_share_mean.mean():.3f}          | {g.cap_rate.mean():8.3f} | {g.cap_abs.mean():8.3f}")
    print("-- 4. E-slope ladder (OLS, gamma dummies, CR1 by run; SD of DV per SD of E; +-1.96 SE) --")
    print("  dv               E alone      +ruling      +fp_emb      +log wealth  +n_active    +pf_out      | E_pos alone   E_neg alone   (jointly: E_pos, E_neg)")
    prepped = []
    for dv in dvs:
        d = _diag_prep(A, dv)
        if d is None:
            continue
        prepped.append((dv, d))
        gm = [c for c in d.columns if c.startswith("gm")]
        rE = _ols_cr(d, "y", ["Ez"] + gm)
        cells, cur = [f"{rE['b']['Ez']:+.3f}±{1.96 * rE['se']['Ez']:.3f}"], []
        for s_ in ("ruling", "fpz", "lwz", "nz", "pfz"):
            cur.append(s_)
            if s_ == "pfz" and "pfz" not in d.columns:
                cells.append("      n/a    ")
                continue
            r = _ols_cr(d, "y", ["Ez"] + cur + gm)
            cells.append(f"{r['b']['Ez']:+.3f}±{1.96 * r['se']['Ez']:.3f}")
        rp, rn, rj = (_ols_cr(d, "y", ["Epz", "ruling"] + gm), _ols_cr(d, "y", ["Enz", "ruling"] + gm),
                      _ols_cr(d, "y", ["Epz", "Enz", "ruling"] + gm))
        print(f"  {dv:15s} " + " ".join(f"{c:12s}" for c in cells) +
              f" | {rp['b']['Epz']:+.3f}      {rn['b']['Enz']:+.3f}      ({rj['b']['Epz']:+.3f}, {rj['b']['Enz']:+.3f})")
    print("-- 5. Residuals of the analysis_P-style model (pooled OLS: E, ruling, E x ruling, fp, wealth, n_active, gamma) --")
    print("  dv               resid skew exkurt  corr(|res|,fit)  corr(|res|,E)  ICC run  ICC country-run")
    for dv, d in prepped:
        d["Er"] = d["Ez"] * d["ruling"]
        gm = [c for c in d.columns if c.startswith("gm")]
        r = _ols_cr(d, "y", ["Ez", "ruling", "Er", "fpz", "lwz", "nz"] + gm)
        u = r["u"]
        sk, ku = _skew_kurt(u)

        def icc(col):
            g = pd.Series(u).groupby(d[col].to_numpy())
            m, n = g.transform("mean").to_numpy(), g.transform("size").to_numpy()
            within = np.sum((u - m) ** 2) / max(len(u) - g.ngroups, 1)
            between = max(np.var(m) - within / max(n.mean(), 1), 0)
            return between / (between + within) if between + within > 0 else float("nan")
        print(f"  {dv:15s} {sk:10.2f} {ku:6.2f}  {np.corrcoef(np.abs(u), r['fit'])[0, 1]:+14.2f}  {np.corrcoef(np.abs(u), d['Ez'])[0, 1]:+13.2f}  "
              f"{icc('run_id'):7.3f}  {icc('country_run'):9.3f}")
    print("-- 6. Per-run jackknife of the E slope (run FE; ruling, fp, wealth, n_active partialled out; controls held at full-sample values = approximate) --")
    print("  dv               slope   jackknife SE  max |change|  (run)           sign changes   top-3 runs' share of influence")
    for dv, d in prepped:
        sl, sxy, sxx = _fwl(d, ["ruling", "fpz", "lwz", "nz"])
        G = len(sxy)
        loo = (sxy.sum() - sxy) / (sxx.sum() - sxx)
        se, ch = np.sqrt((G - 1) / G * np.sum((loo - loo.mean()) ** 2)), np.abs(loo - sl)
        rid, top = pd.factorize(d["run_id"])[1], np.argsort(-np.abs(loo - sl))[:3]
        print(f"  {dv:15s} {sl:+.3f}  {se:12.4f}  {ch.max():12.4f}  {rid[top[0]]:15s} {int(np.sum(np.sign(loo) != np.sign(sl))):8d}   {ch[top].sum() / max(ch.sum(), 1e-12):6.1%}")


def run_diagnostics(P, blocks, bins=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant columns (e.g. E_neg = 0) give NaN correlations
        for blk in blocks:
            for s in sorted(P["overlap_sign"].unique()):
                sub = P[(P["block"] == blk) & (P["overlap_sign"] == s)]
                if len(sub):
                    dvs = PRIMARY_DVS + ROBUST_DVS + (("dom_abs",) if "dom_abs" in sub.columns else ())
                    _diag_group(sub, f"block={blk} overlap_sign={s:+.0f} (population A)", dvs, bins)
    print("\nDiagnostics are descriptive. Choose DV transforms from sections 2 and 5 only, never by slope sign.")

if __name__ == "__main__":
    main()