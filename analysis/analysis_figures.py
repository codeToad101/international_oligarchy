#!/usr/bin/env python3
"""
analysis_figures.py -- [E5] every figure and summary number for the write-up, from the tables the pipeline already wrote.
    python analysis/analysis_figures.py --out out --ofat ofat --ablation ablation_eta0
Needs numpy, pandas, matplotlib. Reads <out>/manifest.csv and <out>/tables/{P_table.csv, S_person_period.csv.gz, P_results.csv,
S_models.csv, S_terms.csv, S_dropone.csv} (run analysis_tables / analysis_P / analysis_S first), plus the OFAT and eta = 0 folders if present.
Writes <out>/figures/*.png, <out>/figures/INDEX.md (what each figure shows, which hypothesis, how to read it, what was skipped),
<out>/figures/summary.txt (headline numbers), <out>/tables/ofat_summary.csv and ablation_summary.csv.
One figure failing or missing an input never stops the rest (listed under SKIPPED / FAILED in INDEX.md).
DESCRIPTIVE ONLY: figures show estimates and intervals from the pre-specified analyses; nothing here changes a method.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("analysis_figures.py needs matplotlib (pip install matplotlib)")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_P as AP
try:
    import analysis_S as AS
except Exception:
    AS = None

EDGES = (0.04, 0.20)
SIGNS = (-1.0, 1.0)
SGN = {-1.0: "rivalry (-1)", 1.0: "collusion (+1)"}
TYPE = {0: "civil", 1: "ruling"}
COL = {0: "#2b6cb0", 1: "#dd6b20"}
SCOL = {-1.0: "#2b6cb0", 1.0: "#c53030"}
PRIM = AP.PRIMARY_DVS
SHORT = {"cap_share_mean": "capture share", "hav_share_mean": "havoc share"}
REG, OK, SKIPPED, FAILED = [], [], [], []


class Skip(Exception):
    pass


def need(x, what):
    if x is None or (hasattr(x, "empty") and x.empty):
        raise Skip(f"missing input: {what}")
    return x


def fig(fid, title, tag, note):
    def deco(fn):
        REG.append((fid, title, tag, note, fn))
        return fn
    return deco


def mci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (np.nan, np.nan)


def suite_band(ax):
    ax.axvspan(0.06, 0.18, color="grey", alpha=0.08, lw=0)


def by_gamma(ax, d, col, ls="-", tag=""):
    for s in SIGNS:
        g = d[d.overlap_sign == s].groupby("gamma")[col]
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.errorbar(m.index, m.values, yerr=1.96 * se.values, color=SCOL[s], ls=ls, marker="o", ms=4, capsize=2, label=f"{tag}{SGN[s]}")
    suite_band(ax)
    ax.set_xlabel("backlash sensitivity gamma (shaded = P/S suite; 0.04, 0.20 = edges)")


def binned(ax, g, xcol, ycol, color, label, q=10):
    b = pd.qcut(g[xcol].rank(method="first"), q, labels=False)
    t = g.groupby(b).agg(x=(xcol, "mean"), y=(ycol, "mean"), s=(ycol, "std"), n=(ycol, "size"))
    ax.errorbar(t.x, t.y, yerr=1.96 * t.s / np.sqrt(t.n), fmt="o", color=color, ms=4, capsize=2, label=label)
    if len(g) > 10 and g[xcol].std() > 0:
        k = np.polyfit(g[xcol], g[ycol], 1)
        xs = np.linspace(g[xcol].quantile(.02), g[xcol].quantile(.98), 20)
        ax.plot(xs, np.polyval(k, xs), color=color, lw=1.5)
        return k[0]
    return np.nan


# ================================================================================ manipulation checks
@fig("F01_collapses_vs_gamma", "Collapses per country vs backlash sensitivity", "design / manipulation check",
     "gamma should raise collapse frequency monotonically. Points = mean over replications, bars = 95% CI. Shaded band = the P/S suite; 0.04 and 0.20 are edge levels.")
def f01(C):
    d = need(C.man, "manifest.csv")
    d = d[d.block == "main"].copy()
    d["cpc"] = d.collapses / 5.0
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    by_gamma(ax, d, "cpc")
    ax.set_ylabel("collapses per country per run"); ax.legend(); ax.set_title("Manipulation check: gamma -> collapses")
    return f


@fig("F02_exit_vs_gamma", "Exit fraction vs backlash sensitivity (main vs type-blind)", "design / manipulation check; S context",
     "Share of the 25 oligarchs that exit by the end of the run. Solid = main block, dashed = type-blind. Should rise with gamma.")
def f02(C):
    d = need(C.man, "manifest.csv")
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    by_gamma(ax, d[d.block == "main"], "exit_frac")
    if (d.block == "typeblind").any():
        for s in SIGNS:
            g = d[(d.block == "typeblind") & (d.overlap_sign == s)].groupby("gamma").exit_frac.mean()
            ax.plot(g.index, g.values, color=SCOL[s], ls="--", marker="s", ms=3, alpha=.7)
    ax.set_ylabel("fraction of oligarchs exited"); ax.legend(); ax.set_title("Exit fraction by gamma")
    return f


@fig("F03_overlap_manipulation", "Overlap sign changes E", "design / manipulation check",
     "Overlap sign should change the sign of same-industry goodwill and hence E. Left: mean E by gamma and sign. Right: share of oligarchs with negative E.")
def f03(C):
    A = need(C.A, "P_table").query("block == 'main'").copy()
    A["neg"] = (A.E_foreign < 0).astype(float)
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    by_gamma(ax[0], A, "E_foreign"); ax[0].set_ylabel("mean E_foreign"); ax[0].legend()
    by_gamma(ax[1], A, "neg"); ax[1].set_ylabel("share with E < 0")
    return f


# ================================================================================ hypothesis P
def interaction(C, blk, fid_note):
    A = need(C.A, "P_table")
    d = A[A.block == blk]
    need(d, f"block {blk}")
    f, axes = plt.subplots(2, 2, figsize=(11, 8))
    for r, s in enumerate(SIGNS):
        for c, dv in enumerate(PRIM):
            ax, g = axes[r, c], d[d.overlap_sign == s]
            for t in (0, 1):
                sub = g[g.ruling == t]
                if len(sub) < 30:
                    continue
                k = binned(ax, sub, "E_foreign", dv, COL[t], None)
                ax.lines[-1].set_label(f"{TYPE[t]} (slope {k:+.3f} per unit E)")
            ax.set_title(f"{SHORT[dv]} - {SGN[s]}"); ax.set_xlabel("embeddedness E (foreign)"); ax.set_ylabel(f"mean {SHORT[dv]}"); ax.legend(fontsize=8)
    f.suptitle(f"E x type interaction, block '{blk}' (population A; deciles of E, 95% CI; raw units)")
    f.tight_layout()
    return f


@fig("F04_interaction_main", "E x type interaction on capture and havoc (main)", "P (primary figure; slide 8)",
     "Binned means of the foreign share against E, civil vs ruling, with OLS lines. Similar slopes for the two types are what P predicts.")
def f04(C):
    return interaction(C, "main", "")


@fig("F05_interaction_typeblind", "E x type interaction (type-blind positive control)", "P (null calibration)",
     "Same plot with ruling_home_mult = 1.0. With no built-in type asymmetry the two lines should coincide.")
def f05(C):
    return interaction(C, "typeblind", "")


def pr_rows(C):
    PR = need(C.PR, "P_results.csv")
    PR = PR[(PR["pop"] == "A") & PR.dv.isin(PRIM)]
    if "note" in PR:
        PR = PR[PR.note.isna()]
    return need(PR.sort_values(["block", "dv", "overlap_sign"]), "primary P fits")


@fig("F06_P_slopes_forest", "E-slopes by type and pooled (primary DVs)", "P",
     "Slope = SD of the DV per SD of E (95% CI). Blue civil, orange ruling, black pooled. Positive and similar slopes support P.")
def f06(C):
    R = pr_rows(C)
    f, ax = plt.subplots(figsize=(8, 0.9 * len(R) + 1.5))
    for i, (_, r) in enumerate(R.iterrows()):
        for off, val, se, col, lab in ((-.22, r.civil, r.civil_se, COL[0], "civil"), (0, r.ruling, r.ruling_se, COL[1], "ruling"), (.22, r.pooled, r.pooled_se, "k", "pooled")):
            ax.errorbar(val, i + off, xerr=1.96 * se, fmt="o", color=col, ms=4, capsize=2, label=lab if i == 0 else None)
    ax.set_yticks(range(len(R))); ax.set_yticklabels([f"{r.block} {SHORT[r.dv]} {SGN[r.overlap_sign]}" for _, r in R.iterrows()], fontsize=8)
    ax.axvline(0, color="grey", lw=.8); ax.invert_yaxis(); ax.legend(); ax.set_xlabel("E-slope (SD of DV per SD of E)")
    return f


@fig("F07_P_tost", "Equivalence test: E-slope difference (ruling - civil) vs margins", "P (similarity of civil and ruling)",
     "Point = difference, bar = 90% CI. Bands = +-margin (15% primary, 10% and 20% sensitivity) of the pooled slope. Equivalent iff the 90% CI lies inside the band. Type-blind rows are the positive control.")
def f07(C):
    R = pr_rows(C)
    f, ax = plt.subplots(figsize=(8, 0.8 * len(R) + 1.5))
    for i, (_, r) in enumerate(R.iterrows()):
        for m, lw, al in ((0.20, 14, .15), (0.15, 9, .3), (0.10, 4, .5)):
            D = r.get(f"delta_{m}", np.nan)
            ax.plot([-D, D], [i, i], color="#38a169", lw=lw, alpha=al, solid_capstyle="butt")
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r.ci90_lo], [r.ci90_hi - r["diff"]]], fmt="o", color="k", ms=4, capsize=3)
        ax.text(1.0, i, f" 15%: {r['equiv_0.15']}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_yticks(range(len(R))); ax.set_yticklabels([f"{r.block} {SHORT[r.dv]} {SGN[r.overlap_sign]}" for _, r in R.iterrows()], fontsize=8)
    ax.axvline(0, color="grey", lw=.8); ax.invert_yaxis(); ax.set_xlabel("difference in E-slope (ruling - civil), SD units")
    return f


@fig("F08_P_head_to_head", "Embeddedness vs type: effect sizes and unique R2", "main question (position vs relational strategy)",
     "Left: effect of +1 SD of E vs the ruling-civil level effect, same DV units (95% CI). Right: unique R2 of E, type and the E x type interaction.")
def f08(C):
    R = pr_rows(C)
    f, ax = plt.subplots(1, 2, figsize=(13, 0.7 * len(R) + 2.5))
    for i, (_, r) in enumerate(R.iterrows()):
        ax[0].errorbar(r.pooled, i - .15, xerr=1.96 * r.pooled_se, fmt="o", color="k", ms=4, capsize=2, label="E (per +1 SD)" if i == 0 else None)
        ax[0].errorbar(r.type_effect, i + .15, xerr=1.96 * r.type_se, fmt="s", color=COL[1], ms=4, capsize=2, label="type (ruling - civil)" if i == 0 else None)
    lab = [f"{r.block} {SHORT[r.dv]} {SGN[r.overlap_sign]}" for _, r in R.iterrows()]
    ax[0].set_yticks(range(len(R))); ax[0].set_yticklabels(lab, fontsize=8); ax[0].axvline(0, color="grey", lw=.8); ax[0].invert_yaxis(); ax[0].legend(); ax[0].set_xlabel("effect (SD of DV)")
    w = .27; x = np.arange(len(R))
    for k, (c, nm, col) in enumerate((("r2_unique_E", "E", "k"), ("r2_unique_type", "type", COL[1]), ("r2_interaction", "E x type", "#718096"))):
        ax[1].barh(x + (k - 1) * w, R[c].values, height=w, color=col, label=nm)
    ax[1].set_yticks(x); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].legend(); ax[1].set_xlabel(f"unique R2 ({R.r2_kind.iloc[0]})")
    f.tight_layout()
    return f


@fig("F09_P_slope_ladder", "What drives the E slope: slope as controls enter", "P (descriptive diagnostic)",
     "E-slope on capture/havoc share and on rate/level DVs as ruling, foreign presence, wealth, active ticks and outcome-window position enter (pooled OLS, run-clustered 95% CI). The pf_out step is partly an identity; read the fp_emb step.")
def f09(C):
    if not hasattr(AP, "_diag_prep"):
        raise Skip("analysis_P.py has no diagnostics block (_diag_prep)")
    A = need(C.A, "P_table").query("block == 'main'")
    steps = ["E alone", "+ruling", "+fp_emb", "+wealth", "+n_active", "+pf_out"]
    f, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, s in zip(axes, SIGNS):
        for dv, col in (("cap_share_mean", "#2b6cb0"), ("hav_share_mean", "#38a169"), ("cap_rate", "#c53030"), ("cap_abs", "#805ad5")):
            d = AP._diag_prep(A[A.overlap_sign == s], dv)
            if d is None:
                continue
            gm = [c for c in d.columns if c.startswith("gm")]
            vals, errs, cur = [], [], []
            r = AP._ols_cr(d, "y", ["Ez"] + gm); vals.append(r["b"]["Ez"]); errs.append(1.96 * r["se"]["Ez"])
            for s_ in ("ruling", "fpz", "lwz", "nz", "pfz"):
                cur.append(s_)
                if s_ == "pfz" and "pfz" not in d.columns:
                    vals.append(np.nan); errs.append(np.nan); continue
                r = AP._ols_cr(d, "y", ["Ez"] + cur + gm); vals.append(r["b"]["Ez"]); errs.append(1.96 * r["se"]["Ez"])
            ax.errorbar(range(len(steps)), vals, yerr=errs, marker="o", color=col, capsize=2, label=dv)
        ax.axhline(0, color="grey", lw=.8); ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, rotation=30, fontsize=8); ax.set_title(SGN[s])
    axes[0].set_ylabel("E-slope (SD of DV per SD of E)"); axes[0].legend(fontsize=8)
    return f


@fig("F10_E_distribution", "Distribution of embeddedness E", "P (descriptive)",
     "Top: E_foreign by type. Bottom: positive (partner) and negative (rival) parts of E. Main block, population A.")
def f10(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, axes = plt.subplots(2, 2, figsize=(11, 7))
    for c, s in enumerate(SIGNS):
        g = A[A.overlap_sign == s]
        for t in (0, 1):
            axes[0, c].hist(g[g.ruling == t].E_foreign, bins=40, alpha=.55, color=COL[t], label=TYPE[t], density=True)
        axes[0, c].set_title(f"E_foreign - {SGN[s]}"); axes[0, c].legend()
        axes[1, c].hist(g.E_pos, bins=40, alpha=.55, color="#38a169", label="E_pos (partners)", density=True)
        axes[1, c].hist(g.E_neg, bins=40, alpha=.55, color="#c53030", label="E_neg (rivals)", density=True); axes[1, c].legend()
    f.tight_layout()
    return f


@fig("F11_DV_distributions", "Distributions of the dependent variables", "P (descriptive; transform choice)",
     "Share DVs as is; rate and level DVs after log1p(x / mean), the pre-specified transform. Used to judge shape, never to pick by slope sign.")
def f11(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, dv in zip(axes.ravel(), ("cap_share_mean", "hav_share_mean", "cap_rate", "hav_rate", "cap_abs", "hav_abs")):
        x = A[dv].dropna()
        if dv in AP.LOG_DVS:
            x = np.log1p(x / x.mean())
        ax.hist(x, bins=50, color="#4a5568"); ax.set_title(dv + (" (log1p(x/mean))" if dv in AP.LOG_DVS else ""), fontsize=9)
    f.tight_layout()
    return f


@fig("F12_E_foreign_position", "E tracks foreign position size; returns diminish", "P (explains negative rate / level slopes)",
     "Left: E vs foreign presence in the embedding window. Middle: foreign capture per unit position vs presence deciles. Right: capture share vs presence deciles.")
def f12(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for s in SIGNS:
        g = A[A.overlap_sign == s]
        sm = g.sample(min(len(g), 3000), random_state=0)
        ax[0].scatter(sm.fp_emb, sm.E_foreign, s=4, alpha=.3, color=SCOL[s], label=SGN[s])
        b = pd.qcut(g.fp_emb.rank(method="first"), 10, labels=False); t = g.groupby(b).agg(x=("fp_emb", "mean"), r=("cap_rate", "mean"), c=("cap_share_mean", "mean"))
        ax[1].plot(t.x, t.r, "o-", color=SCOL[s]); ax[2].plot(t.x, t.c, "o-", color=SCOL[s], label=SGN[s])
    ax[0].set_xlabel("foreign presence (fp_emb)"); ax[0].set_ylabel("E_foreign"); ax[0].legend(markerscale=3)
    ax[1].set_xlabel("foreign presence decile mean"); ax[1].set_ylabel("capture per unit foreign position"); ax[2].set_xlabel("foreign presence decile mean"); ax[2].set_ylabel("capture share")
    return f


@fig("F13_P_slope_by_gamma", "E-slope by gamma and type", "P (heterogeneity across the suite)",
     "E-slope fitted separately at each interior gamma (run FE, clustered CI). Tests whether P holds across backlash sensitivity.")
def f13(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for r, dv in enumerate(PRIM):
        for c, s in enumerate(SIGNS):
            for t, key in ((0, "civil"), (1, "ruling")):
                xs, ys, es = [], [], []
                for g in sorted(A.gamma.unique()):
                    res = AP.analyze(A[(A.overlap_sign == s) & (A.gamma == g)], dv, "fe", "main", s, "A")
                    if isinstance(res.get("note"), str):
                        continue
                    xs.append(g); ys.append(res[key]); es.append(1.96 * res[key + "_se"])
                axes[r, c].errorbar(xs, ys, yerr=es, marker="o", color=COL[t], capsize=2, label=key)
            axes[r, c].axhline(0, color="grey", lw=.8); axes[r, c].set_title(f"{SHORT[dv]} - {SGN[s]}"); axes[r, c].set_xlabel("gamma"); axes[r, c].set_ylabel("E-slope"); axes[r, c].legend()
    return f


@fig("F14_survivorship", "Who drops out: exit before and during the outcome window", "P (survivorship)",
     "Top: share of oligarchs already exited before the outcome window (not in population A). Bottom: share of population A that exits during the window. By gamma, type and block; right-censoring caveat applies.")
def f14(C):
    P = need(C.P, "P_table")
    f, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for c, blk in enumerate(("main", "typeblind")):
        d = P[P.block == blk]
        if d.empty:
            continue
        for s, ls in zip(SIGNS, ("-", "--")):
            for t in (0, 1):
                g = d[(d.overlap_sign == s) & (d.ruling == t)]
                axes[0, c].plot(*zip(*sorted(g.groupby("gamma").in_primary.apply(lambda v: 1 - v.mean()).items())), ls=ls, marker="o", color=COL[t], label=f"{TYPE[t]} {SGN[s]}")
                a = g[g.in_primary == 1]
                axes[1, c].plot(*zip(*sorted(a.groupby("gamma").survivor.apply(lambda v: 1 - v.mean()).items())), ls=ls, marker="o", color=COL[t])
        axes[0, c].set_title(f"{blk}: exited before outcome window"); axes[1, c].set_title(f"{blk}: exits during window (among A)"); axes[1, c].set_xlabel("gamma")
    axes[0, 0].legend(fontsize=7)
    return f


# ================================================================================ hypothesis S
LAB = {"lia_z": "initial capital", "gamma_z": "gamma", "overlap": "overlap sign", "seized_z": "seized fraction", "E_z": "embeddedness E", "ruling": "type (ruling)", "rec_z": "recent seizure / prior capital", "fp_z": "foreign position"}


@fig("F16_S_forest", "Forest plot of exit odds ratios", "S (primary figure; slide 7)",
     "Odds ratio per +1 SD (type 0/1), 95% CI, log scale; full model M3. Time terms t, t^2 omitted (collinear; not interpreted). Right: exploratory M5 (seized fraction with recent seizure / prior capital). OR < 1 = lowers exit hazard.")
def f16(C):
    T = need(C.ST, "S_terms.csv")
    f, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    terms = ["lia_z", "gamma_z", "overlap", "seized_z", "E_z", "ruling"]
    groups = [(b, g) for b in ("main", "typeblind") for g in sorted(T[T.block == b].group.unique())]
    for k, (b, g) in enumerate(groups):
        d = T[(T.block == b) & (T.group == g) & (T.model == "M3")].set_index("term")
        for i, t in enumerate(terms):
            if t in d.index:
                r = d.loc[t]
                ax[0].errorbar(r.OR, i + (k - len(groups) / 2) * 0.1, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", ms=4, capsize=2,
                               color=plt.cm.tab10(k), label=f"{b} {g}" if i == 0 else None)
    ax[0].set_xscale("log"); ax[0].axvline(1, color="grey"); ax[0].set_yticks(range(len(terms))); ax[0].set_yticklabels([LAB[t] for t in terms]); ax[0].invert_yaxis(); ax[0].legend(fontsize=7); ax[0].set_xlabel("odds ratio (log scale)")
    for k, b in enumerate(("main", "typeblind")):
        d = T[(T.block == b) & (T.group == "pooled") & (T.model == "M5")].set_index("term")
        for i, t in enumerate(("seized_z", "rec_z")):
            if t in d.index:
                r = d.loc[t]; ax[1].errorbar(r.OR, i + k * .15, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", capsize=2, color=plt.cm.tab10(k), label=b if i == 0 else None)
    ax[1].set_xscale("log"); ax[1].axvline(1, color="grey"); ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["seized fraction", "recent seizure / prior capital"]); ax[1].invert_yaxis(); ax[1].legend(); ax[1].set_title("EXPLORATORY M5")
    f.tight_layout()
    return f


def hazard_bins(ax, d, xcol, q=10, by="gamma", label_fmt="gamma {:.2f}"):
    for k, (g, sub) in enumerate(sorted(d.groupby(by))):
        b = pd.qcut(sub[xcol].rank(method="first"), q, labels=False)
        t = sub.groupby(b).agg(x=(xcol, "mean"), h=("event", "mean"))
        ax.plot(t.x, 100 * t.h, "o-", ms=3, color=plt.cm.viridis(k / max(1, d[by].nunique() - 1)), label=label_fmt.format(g))
    ax.set_ylabel("exit probability per tick (%)")


@fig("F17_exit_vs_asset_loss", "Exit probability vs asset loss across gamma", "S (primary figure; slide 7)",
     "Per-tick exit probability by decile of lagged asset loss, one line per gamma. Left: pre-specified seized fraction (cumulative seizure / initial active capital). Right: exploratory recent seizure / prior capital. The headline measure scales with size, so a flat or falling curve there is not evidence that loss is harmless.")
def f17(C):
    S = need(C.S, "S_person_period.csv.gz")
    f, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    hazard_bins(ax[0], S, "cum_seized_rel_lag"); ax[0].set_xlabel("cumulative seizure / initial active capital (lagged)"); ax[0].legend(fontsize=7)
    if "rec10_pre_lag" in S:
        hazard_bins(ax[1], S, "rec10_pre_lag"); ax[1].set_xlabel("recent (10-tick) seizure / prior active capital, log1p (lagged)")
    return f


@fig("F18_S_model_fit", "Nested exit models: discrimination and drop-one", "S",
     "Left: in-sample AUC for M0-M5 (M5 exploratory). Right: loss in fit (delta BIC) when each term is removed from the full model; positive = the term helps.")
def f18(C):
    M, D = need(C.SM, "S_models.csv"), need(C.SD, "S_dropone.csv")
    f, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, b in enumerate(("main", "typeblind")):
        m = M[(M.block == b) & (M.group == "pooled")]
        ax[0].plot(m.model, m.auc, "o-", color=plt.cm.tab10(k), label=b)
        d = D[(D.block == b) & (D.group == "pooled")]
        ax[1].barh(np.arange(len(d)) + k * .35, d.d_bic, height=.35, color=plt.cm.tab10(k), label=b)
        ax[1].set_yticks(np.arange(len(d)) + .17); ax[1].set_yticklabels([LAB.get(t, t) for t in d.dropped])
    ax[0].set_ylabel("AUC (in-sample)"); ax[0].legend(); ax[1].invert_yaxis(); ax[1].set_xlabel("delta BIC when dropped"); ax[1].legend()
    return f


@fig("F19_exit_timing", "When exits happen", "S (right-censoring)",
     "Left: per-tick exit probability by tick (5-tick bins), per gamma. Right: exit-tick histogram by type (all exits, main block). Exits cluster late, so the outcome window is right-censored.")
def f19(C):
    S, P = need(C.S, "S_person_period.csv.gz"), need(C.P, "P_table")
    f, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    S = S.assign(tb=(S.t // 5) * 5)
    for k, (g, sub) in enumerate(sorted(S.groupby("gamma"))):
        h = sub.groupby("tb").event.mean() * 100
        ax[0].plot(h.index, h.values, "o-", ms=3, color=plt.cm.viridis(k / max(1, S.gamma.nunique() - 1)), label=f"gamma {g:.2f}")
    ax[0].set_xlabel("tick"); ax[0].set_ylabel("exit probability per tick (%)"); ax[0].legend(fontsize=7)
    x = P[(P.block == "main") & P.exit_tick.notna()]
    for t in (0, 1):
        ax[1].hist(x[x.ruling == t].exit_tick, bins=30, alpha=.55, color=COL[t], label=TYPE[t])
    ax[1].set_xlabel("exit tick"); ax[1].legend()
    return f


@fig("F20_exit_by_type", "Exit fraction by type, gamma and block", "S (type vs loss)",
     "Share of oligarchs that exit over the whole run, civil vs ruling. In the type-blind block the two should match.")
def f20(C):
    P = need(C.P, "P_table")
    f, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, blk in zip(axes, ("main", "typeblind")):
        d = P[P.block == blk]
        if d.empty:
            continue
        gs = sorted(d.gamma.unique()); x = np.arange(len(gs))
        for k, t in enumerate((0, 1)):
            v = [1 - d[(d.gamma == g) & (d.ruling == t)].survivor.mean() for g in gs]
            ax.bar(x + (k - .5) * .38, v, width=.38, color=COL[t], label=TYPE[t])
        ax.set_xticks(x); ax.set_xticklabels([f"{g:.2f}" for g in gs]); ax.set_title(blk); ax.set_xlabel("gamma")
    axes[0].set_ylabel("fraction exited"); axes[0].legend()
    return f


@fig("F21_exit_vs_E", "Exit hazard vs embeddedness and foreign position", "S (E protective?)",
     "Per-tick exit probability by decile of lagged E (left) and lagged foreign position (right). E depends on foreign positions, which are part of active capital, so part of any protective association is mechanical.")
def f21(C):
    S = need(C.S, "S_person_period.csv.gz")
    f, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for s in SIGNS:
        sub = S[S.overlap_sign == s]
        for a, col in zip(ax, ("E_foreign_lag", "pos_foreign_lag")):
            b = pd.qcut(sub[col].rank(method="first"), 10, labels=False); t = sub.groupby(b).agg(x=(col, "mean"), h=("event", "mean"))
            a.plot(t.x, 100 * t.h, "o-", color=SCOL[s], label=SGN[s]); a.set_xlabel(col); a.set_ylabel("exit probability per tick (%)")
    ax[0].legend()
    return f


# ================================================================================ robustness
DVS2 = {"cap_share_mean": "capture", "hav_share_mean": "havoc"}


def arm_metrics(root):
    td = os.path.join(root, "tables")
    P = pd.read_csv(os.path.join(td, "P_table.csv"))
    A = P[(P.in_primary == 1) & ~P.gamma.round(4).isin(EDGES)]
    out = []
    for s in SIGNS:
        for dv in PRIM:
            r = AP.analyze(A[A.overlap_sign == s], dv, "fe", "main", s, "A")
            if isinstance(r.get("note"), str):
                continue
            out.append(dict(metric=f"E-slope {DVS2[dv]} {SGN[s]}", value=r["pooled"], lo=r["pooled"] - 1.96 * r["pooled_se"], hi=r["pooled"] + 1.96 * r["pooled_se"], kind="slope"))
            if dv == "cap_share_mean":
                out.append(dict(metric=f"type diff in E-slope, capture {SGN[s]}", value=r["diff"], lo=r["diff"] - 1.96 * r["diff_se"], hi=r["diff"] + 1.96 * r["diff_se"], kind="slope"))
    sp = os.path.join(td, "S_person_period.csv.gz")
    if AS is not None and os.path.exists(sp):
        S = pd.read_csv(sp)
        S = S[S.t >= 10]
        if int(S.event.sum()) >= 20:
            _, t, _ = AS.run_group(AS.prepare(S, "cum_seized_rel_lag"), "main", "pooled")
            for term, nm in (("seized_z", "exit OR: seized fraction"), ("E_z", "exit OR: embeddedness E"), ("ruling", "exit OR: type (ruling)")):
                for x in t:
                    if x["model"] == "M3" and x["term"] == term:
                        out.append(dict(metric=nm, value=x["OR"], lo=x["OR_lo"], hi=x["OR_hi"], kind="or"))
    return out


def ofat_table(C):
    if C._ofat is not None:
        return C._ofat
    root = need(C.ofat if C.ofat and os.path.isdir(C.ofat) else None, "OFAT folder (run_pipeline.py --steps ofat)")
    if not os.path.isdir(os.path.join(root, "baseline", "tables")):
        raise Skip("OFAT baseline arm missing (ofat/baseline)")
    rows = []
    for arm in sorted(os.listdir(root)):
        if os.path.exists(os.path.join(root, arm, "tables", "P_table.csv")):
            for m in arm_metrics(os.path.join(root, arm)):
                rows.append(dict(arm=arm, **m))
    T = pd.DataFrame(rows)
    base = T[T.arm == "baseline"].set_index("metric")
    T["baseline"] = T.metric.map(base.value)
    T["delta"] = T.value - T.baseline
    null = np.where(T.kind == "or", 1.0, 0.0)
    blo, bhi = T.metric.map(base.lo), T.metric.map(base.hi)
    T["baseline_informative"] = (blo > null) | (bhi < null)          # baseline CI excludes the null: only then is a sign change meaningful
    same = np.where(T.kind == "or", (T.value > 1) == (T.baseline > 1), np.sign(T.value) == np.sign(T.baseline))
    T["sign_preserved"] = np.where(T.baseline_informative, same, True)
    T.to_csv(os.path.join(C.out, "tables", "ofat_summary.csv"), index=False)
    C._ofat = T
    return T


@fig("F22_ofat", "One-factor-at-a-time sensitivity", "validity (sensitivity analysis, slide 9)",
     "Each panel is one headline estimate; each row one arm (low / high level of one constant, all else at default, gamma 0.12, shared seeds). Dashed line and band = baseline estimate with 95% CI. Red = the arm flips the sign of an estimate whose baseline CI excludes the null (or crosses OR = 1); grey = baseline not distinguishable from the null, so a sign change is uninformative.")
def f22(C):
    T = ofat_table(C)
    arms = [a for a in sorted(T.arm.unique()) if a != "baseline"]
    metrics = list(dict.fromkeys(T.metric))
    n = len(metrics); nc = 3; nr = int(np.ceil(n / nc))
    f, axes = plt.subplots(nr, nc, figsize=(15, 3.2 * nr + .6), squeeze=False)
    for ax, mt in zip(axes.ravel(), metrics):
        d = T[T.metric == mt].set_index("arm"); b = d.loc["baseline"] if "baseline" in d.index else None
        for i, a in enumerate(arms):
            if a in d.index:
                r = d.loc[a]; ax.errorbar(r.value, i, xerr=[[r.value - r.lo], [r.hi - r.value]], fmt="o", ms=4, capsize=2, color=("#a0aec0" if not r.baseline_informative else ("k" if r.sign_preserved else "#c53030")))
        if b is not None:
            ax.axvline(b.value, color="#2b6cb0", ls="--"); ax.axvspan(b.lo, b.hi, color="#2b6cb0", alpha=.12)
        if d.kind.iloc[0] == "or":
            ax.set_xscale("log"); ax.axvline(1, color="grey", lw=.8)
        else:
            ax.axvline(0, color="grey", lw=.8)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms if ax is axes[0, 0] or ax in axes[:, 0] else [], fontsize=7); ax.invert_yaxis(); ax.set_title(mt, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    f.tight_layout()
    return f


@fig("F23_ablation_eta0", "eta = 0 ablation: how much of E -> capture is built in", "validity (disclosure: E -> capture partly built in via eta)",
     "E-slope on capture/havoc share with eta = 0.15 (production, same seeds, reps < 100, interior gamma) vs eta = 0 (partner presence no longer helps capture). A large drop shows the share of the slope that is mechanical.")
def f23(C):
    root = need(C.abl if C.abl and os.path.isdir(C.abl) else None, "eta = 0 ablation folder")
    A0 = pd.read_csv(os.path.join(root, "tables", "P_table.csv")); A0 = A0[A0.in_primary == 1]
    A1 = need(C.A, "P_table").query("block == 'main' and rep < 100 and gamma in [0.08, 0.12, 0.16]")
    rows = []
    for lab, A in (("eta = 0.15 (default)", A1), ("eta = 0", A0)):
        for s in SIGNS:
            for dv in PRIM:
                r = AP.analyze(A[A.overlap_sign == s], dv, "fe", "main", s, "A")
                if not isinstance(r.get("note"), str):
                    rows.append(dict(arm=lab, sign=s, dv=dv, slope=r["pooled"], se=r["pooled_se"]))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(C.out, "tables", "ablation_summary.csv"), index=False)
    f, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, s in zip(axes, SIGNS):
        for k, lab in enumerate(("eta = 0.15 (default)", "eta = 0")):
            d = T[(T.arm == lab) & (T.sign == s)]
            ax.bar(np.arange(len(d)) + (k - .5) * .38, d.slope, width=.38, yerr=1.96 * d.se, capsize=3, label=lab, color=("#2b6cb0", "#a0aec0")[k])
            ax.set_xticks(range(len(d))); ax.set_xticklabels([SHORT[x] for x in d.dv])
        ax.axhline(0, color="grey", lw=.8); ax.set_title(SGN[s])
    axes[0].set_ylabel("pooled E-slope (SD of DV per SD of E)"); axes[0].legend()
    return f


@fig("F24_replication_precision", "Precision vs number of replications", "validity (replication count)",
     "Left: pooled E-slope (95% CI) on the capture share using the first n replications. Right: SE of the civil-ruling slope difference vs n; the dotted line is the SE needed for 80% TOST power at the 15% margin. Shows the estimate stabilizing as replications accumulate.")
def f24(C):
    A = need(C.A, "P_table").query("block == 'main'")
    mx = int(A.rep.max()) + 1
    ns = [n for n in (10, 25, 50, 100, 200, 300, 500, 750, 1000) if n <= mx]
    if len(ns) < 2:
        raise Skip("fewer than two replication counts available")
    f, ax = plt.subplots(1, 2, figsize=(12, 4.3)); rows = []
    for s in SIGNS:
        sl, ci, dse, req = [], [], [], []
        for n in ns:
            r = AP.analyze(A[(A.overlap_sign == s) & (A.rep < n)], "cap_share_mean", "fe", "main", s, "A")
            sl.append(r["pooled"]); ci.append(1.96 * r["pooled_se"]); dse.append(r["diff_se"]); req.append(0.15 * abs(r["pooled"]) / 2.927)
        ax[0].errorbar(ns, sl, yerr=ci, marker="o", color=SCOL[s], capsize=2, label=SGN[s])
        ax[1].plot(ns, dse, "o-", color=SCOL[s], label=SGN[s]); ax[1].axhline(req[-1], color=SCOL[s], ls=":")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("replications used (per condition)")
    ax[0].set_ylabel("pooled E-slope, capture share"); ax[0].legend(); ax[1].set_ylabel("SE of slope difference (ruling - civil)")
    return f


@fig("F25_planning_power", "TOST planning power and required sample multiple", "P (precision of the equivalence test)",
     "TOST power if the true difference were 0, at the observed SE, for the 15% margin, and the factor by which the sample would need to grow for 80% power. Type-blind rows show whether the test is precision-limited.")
def f25(C):
    R = pr_rows(C)
    f, ax = plt.subplots(1, 2, figsize=(12, 0.55 * len(R) + 2.2))
    lab = [f"{r.block} {SHORT[r.dv]} {SGN[r.overlap_sign]}" for _, r in R.iterrows()]
    ax[0].barh(range(len(R)), R["power_0.15"], color="#38a169"); ax[0].axvline(.8, color="k", ls=":"); ax[0].set_yticks(range(len(R))); ax[0].set_yticklabels(lab, fontsize=8); ax[0].invert_yaxis(); ax[0].set_xlabel("planning power at 15% margin")
    ax[1].barh(range(len(R)), R["need_x_0.15"].clip(upper=20), color="#dd6b20"); ax[1].axvline(1, color="k", ls=":"); ax[1].set_yticks([]); ax[1].invert_yaxis(); ax[1].set_xlabel("sample multiple needed for 80% power (capped at 20)")
    f.tight_layout()
    return f


# ================================================================================ summary + index
def write_summary(C, path):
    L = []
    add = L.append
    add("SUMMARY NUMBERS (descriptive; generated by analysis_figures.py)\n")
    if C.man is not None:
        m = C.man
        add(f"runs in manifest: {len(m)} | blocks: {m.groupby('block').size().to_dict()} | mean runtime {m.runtime_s.mean():.2f}s | git: {sorted(m.git_commit.unique())[:3] if 'git_commit' in m else 'n/a'}")
        g = m[m.block == "main"].assign(cpc=lambda d: d.collapses / 5).groupby(["gamma", "overlap_sign"]).agg(collapses_per_country=("cpc", "mean"), exit_frac=("exit_frac", "mean")).round(3)
        add("\nmanipulation check (main block): collapses per country, exit fraction by gamma x sign\n" + g.to_string())
    if C.P is not None:
        P = C.P
        add("\nP table: oligarchs by block/type, in_primary share, exited-during-window share")
        for (b, t), d in P.groupby(["block", "ruling"]):
            a = d[d.in_primary == 1]
            add(f"  {b:9s} {TYPE[t]:6s} n={len(d)} in_primary={a.shape[0]} exit_before={1 - len(a) / len(d):.3f} exit_during={1 - a.survivor.mean():.3f}")
    if C.PR is not None:
        R = C.PR[(C.PR['pop'] == 'A') & C.PR.dv.isin(PRIM)]
        add("\nP (population A, primary DVs): slopes per SD; diff = ruling - civil; equivalence at 15/10/20%")
        for _, r in R.sort_values(["block", "dv", "overlap_sign"]).iterrows():
            if isinstance(r.get("note"), str):
                continue
            add(f"  {r.block:9s} {SHORT[r.dv]:14s} {SGN[r.overlap_sign]:15s} pooled {r.pooled:+.3f} civil {r.civil:+.3f} ruling {r.ruling:+.3f} diff {r['diff']:+.3f} "
                f"[90%CI {r.ci90_lo:+.3f},{r.ci90_hi:+.3f}] equiv {r['equiv_0.15']}/{r['equiv_0.1']}/{r['equiv_0.2']} | type {r.type_effect:+.3f} | uniqueR2 E {r.r2_unique_E:.4f} type {r.r2_unique_type:.4f}")
    if C.ST is not None:
        add("\nS: M3 odds ratios per +1 SD (type 0/1), pooled")
        for b in ("main", "typeblind"):
            d = C.ST[(C.ST.block == b) & (C.ST.group == "pooled") & (C.ST.model == "M3")]
            add(f"  {b}: " + "; ".join(f"{LAB.get(r.term, r.term)} {r.OR:.2f} [{r.OR_lo:.2f},{r.OR_hi:.2f}]" for _, r in d.iterrows() if r.term not in ("t_z", "t2_z")))
        d = C.ST[(C.ST.model == "M5") & (C.ST.group == "pooled") & C.ST.term.isin(["seized_z", "rec_z"])]
        if len(d):
            add("S EXPLORATORY M5: " + "; ".join(f"{r.block} {LAB[r.term]} {r.OR:.2f} [{r.OR_lo:.2f},{r.OR_hi:.2f}]" for _, r in d.iterrows()))
    if C.SM is not None:
        add("\nS model AUC (pooled): " + "; ".join(f"{r.block} {r.model} {r.auc:.3f}" for _, r in C.SM[C.SM.group == "pooled"].iterrows()))
    if C._ofat is not None:
        T = C._ofat
        bad = T[(T.arm != "baseline") & (~T.sign_preserved)]
        add(f"\nOFAT: {T.arm.nunique() - 1} arms; informative estimates (baseline CI excludes the null) whose sign / OR side changed vs baseline: {len(bad)}")
        for _, r in bad.iterrows():
            add(f"  {r.arm}: {r.metric} baseline {r.baseline:+.3f} -> {r.value:+.3f}")
    open(path, "w").write("\n".join(L) + "\n")


def write_index(path, only):
    L = ["# Figure index\n", "Generated by analysis_figures.py. Every figure is descriptive: it shows estimates and intervals from the pre-specified analyses.\n"]
    for fid, title, tag, note, _ in REG:
        if fid in OK:
            L.append(f"## {fid}\n**{title}**  \nHypothesis / purpose: {tag}  \n{note}\n")
    if SKIPPED:
        L.append("## SKIPPED (missing input)\n" + "\n".join(f"- {a}: {b}" for a, b in SKIPPED) + "\n")
    if FAILED:
        L.append("## FAILED (error)\n" + "\n".join(f"- {a}: {b}" for a, b in FAILED) + "\n")
    open(path, "w").write("\n".join(L))


class Ctx:
    pass


def load(a):
    C = Ctx()
    C.out, C.ofat, C.abl, C._ofat = a.out, a.ofat, a.ablation, None
    td = os.path.join(a.out, "tables")
    rd = lambda n: pd.read_csv(os.path.join(td, n)) if os.path.exists(os.path.join(td, n)) else None
    P = rd("P_table.csv")
    if P is not None:
        P = P[~P.gamma.round(4).isin(EDGES)]
    C.P, C.A = P, (P[P.in_primary == 1] if P is not None else None)
    C.PR, C.SM, C.ST, C.SD = rd("P_results.csv"), rd("S_models.csv"), rd("S_terms.csv"), rd("S_dropone.csv")
    mp = os.path.join(a.out, "manifest.csv")
    C.man = pd.read_csv(mp) if os.path.exists(mp) else None
    sp = os.path.join(td, "S_person_period.csv.gz")
    C.S = None
    if os.path.exists(sp):
        want = {"block", "gamma", "overlap_sign", "t", "event", "cum_seized_rel_lag", "E_foreign_lag", "pos_foreign_lag", "rec10_pre_lag"}
        parts = []
        for ch in pd.read_csv(sp, usecols=lambda c: c in want, chunksize=500_000):
            ch = ch[(ch.block == "main") & (ch.t >= 10)].drop(columns="block")
            parts.append(ch.astype({c: "float32" for c in ch.columns if c not in ("event",)}))
        C.S = pd.concat(parts, ignore_index=True) if parts else None
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--ofat", default="ofat")
    ap.add_argument("--ablation", default="ablation_eta0")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", default="", help="comma list of figure id prefixes, e.g. F04,F16")
    a = ap.parse_args()
    C = load(a)
    fd = os.path.join(a.out, "figures")
    os.makedirs(fd, exist_ok=True)
    only = [x for x in a.only.split(",") if x]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fid, title, tag, note, fn in REG:
            if only and not any(fid.startswith(o) for o in only):
                continue
            try:
                f = fn(C)
                f.savefig(os.path.join(fd, fid + ".png"), dpi=a.dpi, bbox_inches="tight")
                plt.close(f)
                OK.append(fid)
                print(f"  ok      {fid}")
            except Skip as e:
                SKIPPED.append((fid, str(e)))
                print(f"  skipped {fid}: {e}")
            except Exception as e:
                FAILED.append((fid, f"{type(e).__name__}: {e}"))
                print(f"  FAILED  {fid}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
            plt.close("all")
        try:
            write_summary(C, os.path.join(fd, "summary.txt"))
        except Exception as e:
            print("summary failed:", e)
    write_index(os.path.join(fd, "INDEX.md"), only)
    print(f"\n{len(OK)} figures, {len(SKIPPED)} skipped, {len(FAILED)} failed -> {fd} (INDEX.md, summary.txt)")


if __name__ == "__main__":
    main()