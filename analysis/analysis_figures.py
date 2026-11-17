#!/usr/bin/env python3
"""
analysis_figures.py -- [E5] every figure and summary number for the write-up, from the tables the pipeline already wrote.
    python analysis/analysis_figures.py --out out --ofat ofat --ablation ablation_eta0 --typeblind1000 typeblind1000
Needs numpy, pandas, matplotlib. Reads <out>/manifest.csv and <out>/tables/{P_table.csv, S_person_period.csv.gz, P_results.csv,
S_models.csv, S_terms.csv, S_dropone.csv} (run analysis_tables / analysis_P / analysis_S first), plus the OFAT, eta = 0 and
1000-rep type-blind folders if present (type-blind P figures fall back to the 100-rep control if the rerun is missing).
Writes <out>/figures/*.png (key results), <out>/figures/typeblind_exploratory/*.png (type-blind control and exploratory panels),
<out>/figures/INDEX.md, <out>/figures/summary.txt, <out>/tables/ofat_summary.csv and ablation_summary.csv.
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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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
SGN = {-1.0: "Rivalry (-1)", 1.0: "Collusion (+1)"}
TYPE = {0: "Civil", 1: "Ruling"}
BL = {"main": "Main", "typeblind": "Type-Blind"}
COL = {0: "#2b6cb0", 1: "#dd6b20"}
SCOL = {-1.0: "#2b6cb0", 1.0: "#c53030"}
PRIM = AP.PRIMARY_DVS
SHORT = {"cap_share_mean": "Capture Share", "hav_share_mean": "Havoc Share"}
DVLAB = {"cap_share_mean": "Capture Share", "hav_share_mean": "Havoc Share", "cap_rate": "Capture Rate",
         "hav_rate": "Havoc Rate", "cap_abs": "Capture (Absolute)", "hav_abs": "Havoc (Absolute)"}
TBDIR = "typeblind_exploratory"
REG, OK, SKIPPED, FAILED = [], [], [], []


class Skip(Exception):
    pass


def need(x, what):
    if x is None or (hasattr(x, "empty") and x.empty):
        raise Skip(f"missing input: {what}")
    return x


# ================================================================================ title-case helpers
MINOR = {"an", "and", "as", "at", "but", "by", "for", "in", "is", "nor", "of", "on", "or", "per", "the", "to", "vs"}


def _cap(w):
    for i, ch in enumerate(w):
        if ch.isalpha():
            return (w[:i] + ch.upper() + w[i + 1:]) if (i <= 1 and ch.islower()) else w
    return w


def tc(s):
    """Title Case: minor words lowercase, first/last/after-colon word capitalized, acronyms and digits left alone."""
    if not s:
        return s
    words = s.split(" ")
    out, start = [], True
    for i, w in enumerate(words):
        core = w.strip("()[],.;:")
        if i > 0 and not start and i < len(words) - 1 and core.lower() in MINOR and not (core.isupper() and len(core) > 1):
            out.append(w.lower())
        else:
            out.append("-".join(_cap(p) for p in w.split("-")))
        start = w.endswith(":")
    return " ".join(out)


def polish(f):
    """Safety net: Title Case on every title, axis label and legend entry of a figure."""
    for ax in f.axes:
        for get, setter in ((ax.get_title, ax.set_title), (ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)):
            s = get()
            if s:
                setter(tc(s))
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                t.set_text(tc(t.get_text()))
    for lg in f.legends:
        for t in lg.get_texts():
            t.set_text(tc(t.get_text()))
    st = getattr(f, "_suptitle", None)
    if st is not None:
        st.set_text(tc(st.get_text()))


def arm_label(a):
    s = a.replace("--", "").replace("_", " ").strip()
    for suf, rep in ((" lo", " (Low)"), (" hi", " (High)")):
        if s.endswith(suf):
            s = s[:-len(suf)] + rep
    return tc(s)


def fig(fid, title, tag, note, sub=""):
    def deco(fn):
        REG.append((fid, title, tag, note, fn, sub))
        return fn
    return deco


def pair(fid, title, tag, note, fn):
    """Register fn(C, blk) twice: key (main) figure in figures/, type-blind control copy in figures/typeblind_exploratory/."""
    fig(fid, title, tag, note)(lambda C: fn(C, "main"))
    a, b = fid.split("_", 1)

    def tb(C):
        f = fn(C, "typeblind")
        f.suptitle(f"{title}: Type-Blind Control ({C.TB_src})", y=1.0, va="bottom")
        return f
    fig(f"{a}_TB_{b}", f"{title} (Type-Blind Control)", f"{tag}; null calibration",
        note + " Type-blind control (ruling_home_mult = 1.0): with no built-in type asymmetry, the two types should match.", sub=TBDIR)(tb)


def mci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (np.nan, np.nan)


def by_gamma(ax, d, col, ls="-", tag=""):
    for s in SIGNS:
        g = d[d.overlap_sign == s].groupby("gamma")[col]
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.errorbar(m.index, m.values, yerr=1.96 * se.values, color=SCOL[s], ls=ls, marker="o", ms=4, capsize=2, label=f"{tag}{SGN[s]}")
    ax.axvspan(0.06, 0.18, color="grey", alpha=0.08, lw=0, label="Shaded: P/S Suite")
    ax.set_xlabel("Backlash Sensitivity (Gamma)")


def binned(ax, g, xcol, ycol, color, q=10):
    """Decile dots (95% CI), solid quadratic fit, dashed linear fit. Returns the average linear slope."""
    g = g.dropna(subset=[xcol, ycol])
    if len(g) < 30 or g[xcol].std() == 0:
        return np.nan
    b = pd.qcut(g[xcol].rank(method="first"), q, labels=False)
    t = g.groupby(b).agg(x=(xcol, "mean"), y=(ycol, "mean"), s=(ycol, "std"), n=(ycol, "size"))
    ax.errorbar(t.x, t.y, yerr=1.96 * t.s / np.sqrt(t.n), fmt="o", color=color, ms=4, capsize=2)
    xs = np.linspace(g[xcol].quantile(.02), g[xcol].quantile(.98), 50)
    k1 = np.polyfit(g[xcol], g[ycol], 1)
    k2 = np.polyfit(g[xcol], g[ycol], 2)
    ax.plot(xs, np.polyval(k1, xs), color=color, lw=1.2, ls="--")
    ax.plot(xs, np.polyval(k2, xs), color=color, lw=2, ls="-")
    return k1[0]


# ================================================================================ manipulation checks
@fig("F01_collapses_vs_gamma", "Collapses per Country vs. Backlash Sensitivity", "design / manipulation check",
     "Gamma should raise collapse frequency monotonically. Points = mean over replications, bars = 95% CI. Shaded band = the P/S suite; 0.04 and 0.20 are edge levels.")
def f01(C):
    d = need(C.man, "manifest.csv")
    d = d[d.block == "main"].copy()
    d["cpc"] = d.collapses / 5.0
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    by_gamma(ax, d, "cpc")
    ax.set_ylabel("Collapses per Country per Run"); ax.legend(); ax.set_title("Manipulation Check: Collapses vs. Gamma")
    return f


@fig("F02_exit_vs_gamma", "Exit Fraction vs. Backlash Sensitivity (Main)", "design / manipulation check; S context",
     "Share of the 25 oligarchs that exit by the end of the run, main block. Should rise with gamma. The main vs. type-blind comparison is in the type-blind folder.")
def f02(C):
    d = need(C.man, "manifest.csv")
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    by_gamma(ax, d[d.block == "main"], "exit_frac")
    ax.set_ylabel("Fraction of Oligarchs Exited"); ax.legend(); ax.set_title("Exit Fraction by Gamma")
    return f


@fig("F02_TB_exit_vs_gamma", "Exit Fraction vs. Backlash Sensitivity: Main vs. Type-Blind", "design / manipulation check (control)",
     "Solid lines with circles = main block; dashed lines with squares = type-blind block. The two should be similar if type asymmetry is not driving exit.", sub=TBDIR)
def f02tb(C):
    d = need(C.man, "manifest.csv")
    if not (d.block == "typeblind").any():
        raise Skip("no type-blind runs in manifest")
    f, ax = plt.subplots(figsize=(7.5, 4.6))
    by_gamma(ax, d[d.block == "main"], "exit_frac", tag="Main: ")
    for s in SIGNS:
        g = d[(d.block == "typeblind") & (d.overlap_sign == s)].groupby("gamma").exit_frac
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.errorbar(m.index, m.values, yerr=1.96 * se.values, color=SCOL[s], ls="--", marker="s", ms=4, capsize=2, alpha=.8, label=f"Type-Blind: {SGN[s]}")
    ax.set_ylabel("Fraction of Oligarchs Exited"); ax.legend(fontsize=8); ax.set_title("Exit Fraction by Gamma: Main (Solid) vs. Type-Blind (Dashed)")
    return f


@fig("F03_overlap_manipulation", "Overlap Sign Changes E", "design / manipulation check",
     "Overlap sign should change the sign of same-industry goodwill and hence E. Left: mean E by gamma and sign. Right: share of oligarchs with negative E.")
def f03(C):
    A = need(C.A, "P_table").query("block == 'main'").copy()
    A["neg"] = (A.E_foreign < 0).astype(float)
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    by_gamma(ax[0], A, "E_foreign"); ax[0].set_ylabel("Mean Foreign Embeddedness E"); ax[0].legend(fontsize=8); ax[0].set_title("Mean E by Gamma and Overlap Sign")
    by_gamma(ax[1], A, "neg"); ax[1].set_ylabel("Share of Oligarchs with E Below 0"); ax[1].set_title("Share with Negative E by Gamma")
    return f


# ================================================================================ hypothesis P
def ptab(C, blk):
    return (C.TP, C.TA) if blk == "typeblind" else (C.P, C.A)


def interaction(C, blk):
    _, A = ptab(C, blk)
    d = need(A, "P_table")
    d = need(d[d.block == blk], f"block {blk}")
    f, axes = plt.subplots(2, 2, figsize=(11, 8))
    for r, s in enumerate(SIGNS):
        for c, dv in enumerate(PRIM):
            ax, g = axes[r, c], d[d.overlap_sign == s]
            handles = []
            for t in (0, 1):
                sub = g[g.ruling == t]
                if len(sub) < 30:
                    continue
                k = binned(ax, sub, "E_foreign", dv, COL[t])
                handles.append(Line2D([0], [0], marker="o", color=COL[t], ls="", label=f"{TYPE[t]}: Average Linear Slope {k:+.3f} per Unit E"))
            handles += [Line2D([0], [0], color="k", lw=2, ls="-", label="Quadratic Fit"),
                        Line2D([0], [0], color="k", lw=1.2, ls="--", label="Linear Fit")]
            ax.legend(handles=handles, fontsize=8)
            ax.set_title(f"{SHORT[dv]}: {SGN[s]}"); ax.set_xlabel("Embeddedness E (Foreign)"); ax.set_ylabel(f"Mean {SHORT[dv]}")
    extra = f", {C.TB_src}" if blk == "typeblind" else ""
    f.suptitle("E × Type Interaction") #main block OR typeblind block, population A
                                        #deciles of E, o5% CI, raw units
    f.tight_layout()
    return f


fig("F04_interaction_main", "E × Type Interaction on Capture and Havoc (Main)", "P (primary figure; slide 8)",
    "Decile means of the foreign share against E, civil vs. ruling, with a solid quadratic curve and a dashed linear fit. The relationship is concave (steep at low E, flat above), so the linear slope is an average. Similar curves for the two types are what P predicts.")(lambda C: interaction(C, "main"))
fig("F05_interaction_typeblind", "E × Type Interaction (Type-Blind Positive Control)", "P (null calibration)",
    "Same plot with ruling_home_mult = 1.0. With no built-in type asymmetry the two curves should coincide.", sub=TBDIR)(lambda C: interaction(C, "typeblind"))


def pr_rows(C, blk):
    for PR, src in (((C.TPR, "1000 Reps"), (C.PR, "Original Run")) if blk == "typeblind" else ((C.PR, "Main"),)):
        if PR is None:
            continue
        R = PR[(PR["pop"] == "A") & PR.dv.isin(PRIM) & (PR.block == blk)]
        if "note" in R:
            R = R[R.note.isna()]
        if len(R):
            return R.sort_values(["dv", "overlap_sign"]), src
    raise Skip(f"missing input: primary P fits for block {blk}")


def rowlab(r):
    return f"{SHORT[r.dv]}: {SGN[r.overlap_sign]}"


def f06(C, blk):
    R, _ = pr_rows(C, blk)
    f, ax = plt.subplots(figsize=(8, 0.9 * len(R) + 1.5))
    for i, (_, r) in enumerate(R.iterrows()):
        for off, val, se, col, lab in ((-.22, r.civil, r.civil_se, COL[0], "Civil"), (0, r.ruling, r.ruling_se, COL[1], "Ruling"), (.22, r.pooled, r.pooled_se, "k", "Pooled")):
            ax.errorbar(val, i + off, xerr=1.96 * se, fmt="o", color=col, ms=4, capsize=2, label=lab if i == 0 else None)
    ax.set_yticks(range(len(R))); ax.set_yticklabels([rowlab(r) for _, r in R.iterrows()], fontsize=8)
    ax.axvline(0, color="grey", lw=.8); ax.invert_yaxis(); ax.legend(); ax.set_xlabel("E-Slope (SD of DV per SD of E)")
    return f


pair("F06_P_slopes_forest", "E-Slopes by Type", "P",
     "Slope = SD of the DV per SD of E (95% CI). Blue civil, orange ruling, black pooled. Positive and similar slopes support P.", f06)


def f07(C, blk):
    R, _ = pr_rows(C, blk)
    f, ax = plt.subplots(figsize=(9, 0.8 * len(R) + 2))
    for i, (_, r) in enumerate(R.iterrows()):
        for m, lw, al in ((0.20, 14, .15), (0.15, 9, .3), (0.10, 4, .5)):
            D = r.get(f"delta_{m}", np.nan)
            ax.plot([-D, D], [i, i], color="#38a169", lw=lw, alpha=al, solid_capstyle="butt")
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r.ci90_lo], [r.ci90_hi - r["diff"]]], fmt="o", color="k", ms=4, capsize=3)
        ax.text(1.0, i, f" Equivalent at 15%: {r['equiv_0.15']}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_yticks(range(len(R))); ax.set_yticklabels([rowlab(r) for _, r in R.iterrows()], fontsize=8)
    ax.axvline(0, color="grey", lw=.8); ax.invert_yaxis(); ax.set_xlabel("Difference in E-Slope (Ruling - Civil), SD Units")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="k", ls="-", label="Difference with 90% CI"),
                       Line2D([0], [0], color="#38a169", lw=14, alpha=.15, label="±20% Margin (Sensitivity)"),
                       Line2D([0], [0], color="#38a169", lw=9, alpha=.3, label="±15% Margin (Primary)"),
                       Line2D([0], [0], color="#38a169", lw=4, alpha=.5, label="±10% Margin (Sensitivity)")],
              fontsize=8, loc="upper left", bbox_to_anchor=(1.28, 1.0))
    return f


pair("F07_P_tost", "Equivalence Test: E-Slope Difference (Ruling - Civil) vs. Margins", "P (similarity of civil and ruling)",
     "Point = difference, bar = 90% CI. Green bands = margins (15% primary, 10% and 20% sensitivity) of the pooled slope. Equivalent only if the 90% CI lies inside the band.", f07)


def f08(C, blk):
    R, _ = pr_rows(C, blk)
    f, ax = plt.subplots(1, 2, figsize=(13, 0.7 * len(R) + 2.5))
    for i, (_, r) in enumerate(R.iterrows()):
        ax[0].errorbar(r.pooled, i - .15, xerr=1.96 * r.pooled_se, fmt="o", color="k", ms=4, capsize=2, label="E (per +1 SD)" if i == 0 else None)
        ax[0].errorbar(r.type_effect, i + .15, xerr=1.96 * r.type_se, fmt="s", color=COL[1], ms=4, capsize=2, label="Type (Ruling - Civil)" if i == 0 else None)
    lab = [rowlab(r) for _, r in R.iterrows()]
    ax[0].set_yticks(range(len(R))); ax[0].set_yticklabels(lab, fontsize=8); ax[0].axvline(0, color="grey", lw=.8); ax[0].invert_yaxis(); ax[0].legend()
    ax[0].set_xlabel("Effect (SD of DV)"); ax[0].set_title("Effect Sizes (95% CI)")
    w = .27; x = np.arange(len(R))
    for k, (c, nm, col) in enumerate((("r2_unique_E", "E", "k"), ("r2_unique_type", "Type", COL[1]), ("r2_interaction", "E × Type", "#718096"))):
        ax[1].barh(x + (k - 1) * w, R[c].values, height=w, color=col, label=nm)
    ax[1].set_yticks(x); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].legend(); ax[1].set_xlabel(f"Unique R2 ({R.r2_kind.iloc[0]})"); ax[1].set_title("Unique Explained Variance")
    f.tight_layout()
    return f


pair("F08_P_head_to_head", "Embeddedness vs. Type: Effect Sizes and Unique R2", "main question (position vs. relational strategy)",
     "Left: effect of +1 SD of E vs. the ruling-civil level effect, same DV units (95% CI). Right: unique R2 of E, type and the E × type interaction.", f08)


@fig("F09_P_slope_ladder", "What Drives the E Slope: Slope as Controls Enter", "P (descriptive diagnostic)",
     "E-slope on capture/havoc share and on rate/level DVs as ruling, foreign presence, wealth, active ticks and outcome-window position enter (pooled OLS, run-clustered 95% CI). The last step is partly an identity; read the foreign-presence step.")
def f09(C):
    if not hasattr(AP, "_diag_prep"):
        raise Skip("analysis_P.py has no diagnostics block (_diag_prep)")
    A = need(C.A, "P_table").query("block == 'main'")
    steps = ["E Alone", "+ Ruling", "+ Foreign Presence", "+ Wealth", "+ Active Ticks", "+ Outcome Position"]
    f, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
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
            ax.errorbar(range(len(steps)), vals, yerr=errs, marker="o", color=col, capsize=2, label=DVLAB[dv])
        ax.axhline(0, color="grey", lw=.8); ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
        ax.set_title(SGN[s]); ax.set_xlabel("Controls Added")
    axes[0].set_ylabel("E-Slope (SD of DV per SD of E)"); axes[0].legend(fontsize=8)
    return f


@fig("F10_E_distribution", "Distribution of Embeddedness E", "P (descriptive)",
     "Top: E_foreign by type. Bottom: positive (partner) and negative (rival) parts of E. Main block, population A.")
def f10(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, axes = plt.subplots(2, 2, figsize=(11, 7))
    for c, s in enumerate(SIGNS):
        g = A[A.overlap_sign == s]
        for t in (0, 1):
            axes[0, c].hist(g[g.ruling == t].E_foreign, bins=40, alpha=.55, color=COL[t], label=TYPE[t], density=True)
        axes[0, c].set_title(f"Foreign Embeddedness E: {SGN[s]}"); axes[0, c].legend(); axes[0, c].set_xlabel("Embeddedness E (Foreign)"); axes[0, c].set_ylabel("Density")
        axes[1, c].hist(g.E_pos, bins=40, alpha=.55, color="#38a169", label="Partner Presence", density=True)
        axes[1, c].hist(g.E_neg, bins=40, alpha=.55, color="#c53030", label="Rival Presence", density=True)
        axes[1, c].legend(fontsize=8); axes[1, c].set_title(f"Partner and Rival Parts of E: {SGN[s]}"); axes[1, c].set_xlabel("Embeddedness E (Component)"); axes[1, c].set_ylabel("Density")
    f.tight_layout()
    return f


@fig("F11_DV_distributions", "Distributions of the Dependent Variables", "P (descriptive; transform choice)",
     "Share DVs as is; rate and level DVs after the pre-specified log1p(x / mean) transform. Used to judge shape, never to pick by slope sign.")
def f11(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, dv in zip(axes.ravel(), ("cap_share_mean", "hav_share_mean", "cap_rate", "hav_rate", "cap_abs", "hav_abs")):
        x = A[dv].dropna()
        if dv in AP.LOG_DVS:
            x = np.log1p(x / x.mean())
        ax.hist(x, bins=50, color="#4a5568")
        ax.set_title(DVLAB[dv] + (": Log-Transformed, Relative to Mean" if dv in AP.LOG_DVS else ""), fontsize=9)
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
    f.tight_layout()
    return f


@fig("F12_E_foreign_position", "E Tracks Foreign Position Size; Returns Diminish", "P (explains negative rate / level slopes)",
     "Left: E vs. foreign presence in the embedding window. Middle: foreign capture per unit position vs. presence deciles. Right: capture share vs. presence deciles.")
def f12(C):
    A = need(C.A, "P_table").query("block == 'main'")
    f, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for s in SIGNS:
        g = A[A.overlap_sign == s]
        sm = g.sample(min(len(g), 3000), random_state=0)
        ax[0].scatter(sm.fp_emb, sm.E_foreign, s=4, alpha=.3, color=SCOL[s], label=SGN[s])
        b = pd.qcut(g.fp_emb.rank(method="first"), 10, labels=False); t = g.groupby(b).agg(x=("fp_emb", "mean"), r=("cap_rate", "mean"), c=("cap_share_mean", "mean"))
        ax[1].plot(t.x, t.r, "o-", color=SCOL[s], label=SGN[s]); ax[2].plot(t.x, t.c, "o-", color=SCOL[s], label=SGN[s])
    ax[0].set_xlabel("Foreign Presence"); ax[0].set_ylabel("Embeddedness E (Foreign)"); ax[0].legend(markerscale=3); ax[0].set_title("E vs. Foreign Presence")
    ax[1].set_xlabel("Foreign Presence (Decile Mean)"); ax[1].set_ylabel("Capture per Unit Foreign Position"); ax[1].set_title("Capture Rate vs. Presence"); ax[1].legend()
    ax[2].set_xlabel("Foreign Presence (Decile Mean)"); ax[2].set_ylabel("Capture Share"); ax[2].set_title("Capture Share vs. Presence"); ax[2].legend()
    f.tight_layout()
    return f


@fig("F13_P_slope_by_gamma", "E-Slope by Gamma and Type", "P (heterogeneity across the suite)",
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
                axes[r, c].errorbar(xs, ys, yerr=es, marker="o", color=COL[t], capsize=2, label=TYPE[t])
            axes[r, c].axhline(0, color="grey", lw=.8); axes[r, c].set_title(f"{SHORT[dv]}: {SGN[s]}")
            axes[r, c].set_xlabel("Gamma"); axes[r, c].set_ylabel("E-Slope"); axes[r, c].legend()
    return f


def f14(C, blk):
    allp, _ = ptab(C, blk)
    d = need(allp, "P_table")
    d = need(d[d.block == blk], f"block {blk}")
    f, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for s, ls in zip(SIGNS, ("-", "--")):
        for t in (0, 1):
            g = d[(d.overlap_sign == s) & (d.ruling == t)]
            lab = f"{TYPE[t]}, {SGN[s]}"
            x0 = sorted(g.groupby("gamma").in_primary.apply(lambda v: 1 - v.mean()).items())
            a = g[g.in_primary == 1]
            x1 = sorted(a.groupby("gamma").survivor.apply(lambda v: 1 - v.mean()).items())
            if x0:
                axes[0].plot(*zip(*x0), ls=ls, marker="o", color=COL[t], label=lab)
            if x1:
                axes[1].plot(*zip(*x1), ls=ls, marker="o", color=COL[t], label=lab)
    axes[0].set_title("Exited Before Outcome Window"); axes[1].set_title("Exited During Window (Among Population A)")
    for a in axes:
        a.set_xlabel("Gamma"); a.legend(fontsize=7)
    axes[0].set_ylabel("Share of Oligarchs")
    return f


pair("F14_survivorship", "Who Drops Out: Exit Before and During the Outcome Window", "P (survivorship)",
     "Left: share of oligarchs already exited before the outcome window (not in population A). Right: share of population A that exits during the window. By gamma and type; solid = rivalry, dashed = collusion; right-censoring caveat applies.", f14)


# ================================================================================ hypothesis S
LAB = {"lia_z": "Initial Capital", "gamma_z": "Gamma", "overlap": "Overlap Sign", "seized_z": "Seized Fraction", "E_z": "Embeddedness E",
       "ruling": "Type (Ruling)", "rec_z": "Recent Seizure / Prior Capital", "fp_z": "Foreign Position"}


def _or_pt(ax, r, y, color, label, fmt="o"):
    ax.errorbar(r.OR, y, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt=fmt, ms=4, capsize=2, color=color, label=label)

def glab(g):
    g = str(g).strip().lower()
    return {"pooled": "Pooled", "s=+1": "Collusion (+1)", "s=1": "Collusion (+1)", "s=-1": "Rivalry (-1)"}.get(g, g.title())

def f16(C, full):
    T = need(C.ST, "S_terms.csv")
    terms = ["lia_z", "gamma_z", "overlap", "seized_z", "E_z", "ruling"]
    if full:
        f, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.5, 1])); blocks = ("typeblind",)
    else:
        f, ax0 = plt.subplots(figsize=(8, 5)); axs = [ax0]; blocks = ("main",)
    ax0 = axs[0]
    groups = [(b, g) for b in blocks for g in sorted(T[T.block == b].group.unique())]
    for k, (b, g) in enumerate(groups):
        d = T[(T.block == b) & (T.group == g) & (T.model == "M3")].set_index("term")
        for i, t in enumerate(terms):
            if t in d.index:
                _or_pt(ax0, d.loc[t], i + (k - len(groups) / 2) * 0.1, plt.cm.tab10(k), f"{BL[b]}: {glab(g)}" if i == 0 else None)
    ax0.set_xscale("log"); ax0.axvline(1, color="grey"); ax0.set_yticks(range(len(terms))); ax0.set_yticklabels([LAB[t] for t in terms]); ax0.invert_yaxis()
    ax0.legend(fontsize=7); ax0.set_xlabel("Odds Ratio per +1 SD (Log Scale)"); ax0.set_title(f"Exit Odds Ratios, Full Model ({BL[blocks[0]]})")
    if full:
        for k, b in enumerate(("main", "typeblind")):
            d = T[(T.block == b) & (T.group == "pooled") & (T.model == "M5")].set_index("term")
            for i, t in enumerate(("seized_z", "rec_z")):
                if t in d.index:
                    _or_pt(axs[1], d.loc[t], i + k * .15, plt.cm.tab10(k), BL[b] if i == 0 else None)
        axs[1].set_xscale("log"); axs[1].axvline(1, color="grey"); axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(["Seized Fraction", "Recent Seizure / Prior Capital"])
        axs[1].invert_yaxis(); axs[1].legend(); axs[1].set_title("Exploratory Model M5"); axs[1].set_xlabel("Odds Ratio per +1 SD (Log Scale)")
    f.tight_layout()
    return f


fig("F16_S_forest", "Forest Plot of Exit Odds Ratios (Main)", "S (primary figure; slide 7)",
    "Odds ratio per +1 SD (type 0/1), 95% CI, log scale; full model M3, main block, pooled and subgroups. Time terms omitted (collinear; not interpreted). OR below 1 = lowers exit hazard.")(lambda C: f16(C, False))
fig("F16_TB_S_forest_typeblind_exploratory", "Exit Odds Ratios: Type-Blind M3 and Exploratory M5", "S (control and exploratory)",
    "Left: type-blind M3 (type should have no effect). Right: EXPLORATORY M5 (seized fraction with recent seizure / prior capital), main and type-blind. OR below 1 = lowers exit hazard.", sub=TBDIR)(lambda C: f16(C, True))


def hazard_bins(ax, d, xcol, q=10, by="gamma", label_fmt="Gamma {:.2f}"):
    for k, (g, sub) in enumerate(sorted(d.groupby(by))):
        b = pd.qcut(sub[xcol].rank(method="first"), q, labels=False)
        t = sub.groupby(b).agg(x=(xcol, "mean"), h=("event", "mean"))
        ax.plot(t.x, 100 * t.h, "o-", ms=3, color=plt.cm.viridis(k / max(1, d[by].nunique() - 1)), label=label_fmt.format(g))
    ax.set_ylabel("Exit Probability per Tick (%)")


@fig("F17_exit_vs_asset_loss", "Exit Probability vs. Asset Loss Across Gamma", "S (primary figure; slide 7)",
     "Per-tick exit probability by decile of lagged seized fraction (cumulative seizure / initial active capital), one line per gamma. This measure scales with size, so a flat or falling curve is not evidence that loss is harmless. The exploratory recent-seizure version is in the type-blind folder.")
def f17(C):
    S = need(C.S, "S_person_period.csv.gz")
    f, ax = plt.subplots(figsize=(7, 4.5))
    hazard_bins(ax, S, "cum_seized_rel_lag"); ax.set_xlabel("Seized Fraction: Cumulative Seizure / Initial Active Capital (Lagged)"); ax.legend(fontsize=7)
    ax.set_title("Exit Probability by Seized Fraction")
    return f


@fig("F17X_exit_vs_recent_loss", "Exit Probability vs. Recent Seizure (Exploratory)", "S (exploratory)",
     "EXPLORATORY. Per-tick exit probability by decile of lagged recent (10-tick) seizure / prior active capital, one line per gamma.", sub=TBDIR)
def f17x(C):
    S = need(C.S, "S_person_period.csv.gz")
    if "rec10_pre_lag" not in S:
        raise Skip("rec10_pre_lag not in S table")
    f, ax = plt.subplots(figsize=(7, 4.5))
    hazard_bins(ax, S, "rec10_pre_lag"); ax.set_xlabel("Recent 10-Tick Seizure / Prior Active Capital, Log-Scaled (Lagged)"); ax.legend(fontsize=7)
    ax.set_title("Exit Probability by Recent Seizure (Exploratory)")
    return f


def f18(C, full):
    M, D = need(C.SM, "S_models.csv"), need(C.SD, "S_dropone.csv")
    blocks = ("main", "typeblind") if full else ("main",)
    f, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, b in enumerate(blocks):
        m = M[(M.block == b) & (M.group == "pooled")]
        if not full:
            m = m[m.model.isin(["M0", "M1", "M2", "M3"])]
        ax[0].plot(m.model, m.auc, "o-", color=plt.cm.tab10(k), label=BL[b])
        d = D[(D.block == b) & (D.group == "pooled")]
        ax[1].barh(np.arange(len(d)) + k * .35, d.d_bic, height=.35, color=plt.cm.tab10(k), label=BL[b])
        ax[1].set_yticks(np.arange(len(d)) + .175 * (len(blocks) - 1)); ax[1].set_yticklabels([LAB.get(t, t) for t in d.dropped])
    ax[0].set_ylabel("AUC (In-Sample)"); ax[0].set_xlabel("Nested Model"); ax[0].legend()
    ax[0].set_title("Discrimination by Model (M4 Near-Proxy, M5 Exploratory)" if full else "Discrimination by Nested Model")
    ax[1].invert_yaxis(); ax[1].set_xlabel("Increase in BIC When Term Is Dropped"); ax[1].legend(); ax[1].set_title("Drop-One Loss in Fit")
    f.tight_layout()
    return f


fig("F18_S_model_fit", "Nested Exit Models: Discrimination and Drop-One (Main)", "S",
    "Left: in-sample AUC for M0 to M3, main block. Right: loss in fit (BIC increase) when each term is removed from the full model; positive = the term helps.")(lambda C: f18(C, False))
fig("F18_TB_S_model_fit_all", "Nested Exit Models: All Models, Main and Type-Blind", "S (control and exploratory)",
    "Left: in-sample AUC for M0 to M5 (M4 is a near-proxy for the exit rule; M5 exploratory), main and type-blind. Right: drop-one BIC loss, both blocks.", sub=TBDIR)(lambda C: f18(C, True))


@fig("F19_exit_timing", "When Exits Happen", "S (right-censoring)",
     "Left: per-tick exit probability by tick (5-tick bins), per gamma. Right: exit-tick histogram by type (all exits, main block). Exits cluster late, so the outcome window is right-censored.")
def f19(C):
    S, P = need(C.S, "S_person_period.csv.gz"), need(C.P, "P_table")
    f, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    S = S.assign(tb=(S.t // 5) * 5)
    for k, (g, sub) in enumerate(sorted(S.groupby("gamma"))):
        h = sub.groupby("tb").event.mean() * 100
        ax[0].plot(h.index, h.values, "o-", ms=3, color=plt.cm.viridis(k / max(1, S.gamma.nunique() - 1)), label=f"Gamma {g:.2f}")
    ax[0].set_xlabel("Tick"); ax[0].set_ylabel("Exit Probability per Tick (%)"); ax[0].legend(fontsize=7); ax[0].set_title("Exit Probability by Tick")
    x = P[(P.block == "main") & P.exit_tick.notna()]
    for t in (0, 1):
        ax[1].hist(x[x.ruling == t].exit_tick, bins=30, alpha=.55, color=COL[t], label=TYPE[t])
    ax[1].set_xlabel("Exit Tick"); ax[1].set_ylabel("Count of Exits"); ax[1].legend(); ax[1].set_title("Exit Ticks by Type (Main Block)")
    return f


def f20(C, blk):
    allp, _ = ptab(C, blk)
    d = need(allp, "P_table")
    d = need(d[d.block == blk], f"block {blk}")
    f, ax = plt.subplots(figsize=(7, 4.3))
    gs = sorted(d.gamma.unique()); x = np.arange(len(gs))
    for k, t in enumerate((0, 1)):
        v = [1 - d[(d.gamma == g) & (d.ruling == t)].survivor.mean() for g in gs]
        ax.bar(x + (k - .5) * .38, v, width=.38, color=COL[t], label=TYPE[t])
    ax.set_xticks(x); ax.set_xticklabels([f"{g:.2f}" for g in gs]); ax.set_xlabel("Gamma"); ax.set_ylabel("Fraction Exited"); ax.legend()
    ax.set_title(f"Exit Fraction by Type and Gamma ({BL[blk]} Block)")
    return f


pair("F20_exit_by_type", "Exit Fraction by Type and Gamma", "S (type vs. loss)",
     "Share of oligarchs that exit over the whole run, civil vs. ruling, by gamma.", f20)


@fig("F21_exit_vs_E", "Exit Hazard vs. Embeddedness and Foreign Position", "S (E protective?)",
     "Per-tick exit probability by decile of lagged E (left) and lagged foreign position (right). E depends on foreign positions, which are part of active capital, so part of any protective association is mechanical.")
def f21(C):
    S = need(C.S, "S_person_period.csv.gz")
    f, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    xl = {"E_foreign_lag": "Lagged Embeddedness E (Foreign)", "pos_foreign_lag": "Lagged Foreign Position"}
    for s in SIGNS:
        sub = S[S.overlap_sign == s]
        for a, col in zip(ax, ("E_foreign_lag", "pos_foreign_lag")):
            b = pd.qcut(sub[col].rank(method="first"), 10, labels=False); t = sub.groupby(b).agg(x=(col, "mean"), h=("event", "mean"))
            a.plot(t.x, 100 * t.h, "o-", color=SCOL[s], label=SGN[s]); a.set_xlabel(xl[col]); a.set_ylabel("Exit Probability per Tick (%)")
    ax[0].set_title("Exit Probability by Embeddedness"); ax[1].set_title("Exit Probability by Foreign Position")
    ax[0].legend(); ax[1].legend()
    return f


# ================================================================================ robustness
DVS2 = {"cap_share_mean": "Capture Share", "hav_share_mean": "Havoc Share"}


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
            out.append(dict(metric=f"E-Slope on {DVS2[dv]}: {SGN[s]}", value=r["pooled"], lo=r["pooled"] - 1.96 * r["pooled_se"], hi=r["pooled"] + 1.96 * r["pooled_se"], kind="slope"))
            if dv == "cap_share_mean":
                out.append(dict(metric=f"Type Gap in E-Slope on Capture Share: {SGN[s]}", value=r["diff"], lo=r["diff"] - 1.96 * r["diff_se"], hi=r["diff"] + 1.96 * r["diff_se"], kind="slope"))
    sp = os.path.join(td, "S_person_period.csv.gz")
    if AS is not None and os.path.exists(sp):
        S = pd.read_csv(sp)
        S = S[S.t >= 10]
        if int(S.event.sum()) >= 20:
            _, t, _ = AS.run_group(AS.prepare(S, "cum_seized_rel_lag"), "main", "pooled")
            for term, nm in (("seized_z", "Exit Odds Ratio: Seized Fraction"), ("E_z", "Exit Odds Ratio: Embeddedness E"), ("ruling", "Exit Odds Ratio: Type (Ruling)")):
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


@fig("F22_ofat", "One-Factor-at-a-Time Sensitivity", "validity (sensitivity analysis, slide 9)",
     "Each panel is one headline estimate; each row one arm (low / high level of one constant, all else at default, gamma 0.12, shared seeds). Dashed line and band = baseline estimate with 95% CI. Red = the arm flips the sign of an estimate whose baseline CI excludes the null (or crosses OR = 1); grey = baseline not distinguishable from the null, so a sign change is uninformative.")
def f22(C):
    T = ofat_table(C)
    arms = [a for a in sorted(T.arm.unique()) if a != "baseline"]
    metrics = list(dict.fromkeys(T.metric))
    n = len(metrics); nc = 3; nr = int(np.ceil(n / nc))
    f, axes = plt.subplots(nr, nc, figsize=(15, 3.2 * nr + .9), squeeze=False)
    for j, (ax, mt) in enumerate(zip(axes.ravel(), metrics)):
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
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([arm_label(a) for a in arms] if j % nc == 0 else [], fontsize=7); ax.invert_yaxis()
        ax.set_title(mt, fontsize=9); ax.set_xlabel("Odds Ratio (Log Scale)" if d.kind.iloc[0] == "or" else "Estimate")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    f.tight_layout()
    f.legend(handles=[Line2D([0], [0], color="#2b6cb0", ls="--", label="Baseline Estimate (Dashed Line)"),
                      Patch(color="#2b6cb0", alpha=.12, label="Baseline 95% CI (Shaded Band)"),
                      Line2D([0], [0], marker="o", color="k", ls="", label="Arm Estimate: Sign Preserved"),
                      Line2D([0], [0], marker="o", color="#c53030", ls="", label="Arm Estimate: Sign Flipped"),
                      Line2D([0], [0], marker="o", color="#a0aec0", ls="", label="Baseline Not Distinguishable from Null")],
             loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, -0.03))
    return f


@fig("F23_ablation_eta0", "Eta = 0 Ablation: How Much of the E-Capture Link Is Built In", "validity (disclosure: E-capture link partly built in via eta)",
     "E-slope on capture/havoc share with eta = 0.15 (production, same seeds, reps < 100, interior gamma) vs. eta = 0 (partner presence no longer helps capture). A large drop shows the share of the slope that is mechanical.")
def f23(C):
    root = need(C.abl if C.abl and os.path.isdir(C.abl) else None, "eta = 0 ablation folder")
    A0 = pd.read_csv(os.path.join(root, "tables", "P_table.csv")); A0 = A0[A0.in_primary == 1]
    A1 = need(C.A, "P_table").query("block == 'main' and rep < 100 and gamma in [0.08, 0.12, 0.16]")
    rows = []
    for lab, A in (("Eta = 0.15 (Default)", A1), ("Eta = 0", A0)):
        for s in SIGNS:
            for dv in PRIM:
                r = AP.analyze(A[A.overlap_sign == s], dv, "fe", "main", s, "A")
                if not isinstance(r.get("note"), str):
                    rows.append(dict(arm=lab, sign=s, dv=dv, slope=r["pooled"], se=r["pooled_se"]))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(C.out, "tables", "ablation_summary.csv"), index=False)
    f, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, s in zip(axes, SIGNS):
        for k, lab in enumerate(("Eta = 0.15 (Default)", "Eta = 0")):
            d = T[(T.arm == lab) & (T.sign == s)]
            ax.bar(np.arange(len(d)) + (k - .5) * .38, d.slope, width=.38, yerr=1.96 * d.se, capsize=3, label=lab, color=("#2b6cb0", "#a0aec0")[k])
            ax.set_xticks(range(len(d))); ax.set_xticklabels([SHORT[x] for x in d.dv])
        ax.axhline(0, color="grey", lw=.8); ax.set_title(SGN[s]); ax.set_xlabel("Dependent Variable")
    axes[0].set_ylabel("Pooled E-Slope (SD of DV per SD of E)"); axes[0].legend()
    return f


@fig("F24_replication_precision", "Precision vs. Number of Replications", "validity (replication count)",
     "Left: pooled E-slope (95% CI) on the capture share using the first n replications. Right: observed SE of the civil-ruling slope difference vs. n (solid) and the SE needed for 80% TOST power at the 15% margin (dotted). Shows the estimate stabilizing as replications accumulate.")
def f24(C):
    A = need(C.A, "P_table").query("block == 'main'")
    mx = int(A.rep.max()) + 1
    ns = [n for n in (10, 25, 50, 100, 200, 300, 500, 750, 1000) if n <= mx]
    if len(ns) < 2:
        raise Skip("fewer than two replication counts available")
    f, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for s in SIGNS:
        sl, ci, dse, req = [], [], [], []
        for n in ns:
            r = AP.analyze(A[(A.overlap_sign == s) & (A.rep < n)], "cap_share_mean", "fe", "main", s, "A")
            sl.append(r["pooled"]); ci.append(1.96 * r["pooled_se"]); dse.append(r["diff_se"]); req.append(0.15 * abs(r["pooled"]) / 2.927)
        ax[0].errorbar(ns, sl, yerr=ci, marker="o", color=SCOL[s], capsize=2, label=SGN[s])
        ax[1].plot(ns, dse, "o-", color=SCOL[s], label=f"Observed SE: {SGN[s]}")
        ax[1].axhline(req[-1], color=SCOL[s], ls=":", label=f"SE Needed for 80% Power: {SGN[s]}")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("Replications Used (per Condition)")
    ax[0].set_ylabel("Pooled E-Slope, Capture Share"); ax[0].legend(); ax[0].set_title("E-Slope Stabilizing with Replications")
    ax[1].set_ylabel("SE of Slope Difference (Ruling - Civil)"); ax[1].legend(fontsize=7); ax[1].set_title("Precision of the Type Comparison")
    return f


def f25(C, blk):
    R, _ = pr_rows(C, blk)
    f, ax = plt.subplots(1, 2, figsize=(12, 0.55 * len(R) + 2.4))
    lab = [rowlab(r) for _, r in R.iterrows()]
    ax[0].barh(range(len(R)), R["power_0.15"], color="#38a169"); ax[0].axvline(.8, color="k", ls=":", label="80% Power Target (Dotted)")
    ax[0].set_yticks(range(len(R))); ax[0].set_yticklabels(lab, fontsize=8); ax[0].invert_yaxis(); ax[0].set_xlabel("Planning Power at 15% Margin"); ax[0].legend(fontsize=8)
    ax[1].barh(range(len(R)), R["need_x_0.15"].clip(upper=20), color="#dd6b20"); ax[1].axvline(1, color="k", ls=":", label="Current Sample (1×, Dotted)")
    ax[1].set_yticks([]); ax[1].invert_yaxis(); ax[1].set_xlabel("Sample Multiple Needed for 80% Power (Capped at 20)"); ax[1].legend(fontsize=8)
    f.tight_layout()
    return f


pair("F25_planning_power", "TOST Planning Power and Required Sample Multiple", "P (precision of the equivalence test)",
     "TOST power if the true difference were 0, at the observed SE, for the 15% margin, and the factor by which the sample would need to grow for 80% power.", f25)


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
    if C.TPR is not None:
        R = C.TPR[(C.TPR['pop'] == 'A') & C.TPR.dv.isin(PRIM)]
        add("\nP type-blind 1000-rep rerun (population A, primary DVs)")
        for _, r in R.sort_values(["dv", "overlap_sign"]).iterrows():
            if isinstance(r.get("note"), str):
                continue
            add(f"  {SHORT[r.dv]:14s} {SGN[r.overlap_sign]:15s} pooled {r.pooled:+.3f} diff {r['diff']:+.3f} [90%CI {r.ci90_lo:+.3f},{r.ci90_hi:+.3f}] equiv {r['equiv_0.15']}/{r['equiv_0.1']}/{r['equiv_0.2']}")
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
    L = ["# Figure index\n", "Generated by analysis_figures.py. Every figure is descriptive: it shows estimates and intervals from the pre-specified analyses.\n",
         "Key results are in `figures/`. Type-blind control and exploratory versions are in `figures/typeblind_exploratory/`.\n"]
    for head, sub in (("Key Results", ""), ("Type-Blind Control and Exploratory", TBDIR)):
        L.append(f"# {head}\n")
        for fid, title, tag, note, _, s in REG:
            if fid in OK and s == sub:
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
    # type-blind P sources: prefer the 1000-rep rerun, fall back to the 100-rep control inside the main out folder
    C.TP = P[P.block == "typeblind"] if P is not None else None
    C.TPR, C.TB_src = None, "Original Run, 100 Reps"
    tb = os.path.join(a.typeblind1000, "tables") if a.typeblind1000 else ""
    if tb and os.path.exists(os.path.join(tb, "P_table.csv")):
        t = pd.read_csv(os.path.join(tb, "P_table.csv"))
        if "block" not in t:
            t["block"] = "typeblind"
        t = t[~t.gamma.round(4).isin(EDGES)]
        t = t[t.block == "typeblind"]
        if len(t):
            C.TP, C.TB_src = t, "1000-Rep Rerun"
            if os.path.exists(os.path.join(tb, "P_results.csv")):
                C.TPR = pd.read_csv(os.path.join(tb, "P_results.csv"))
    C.TA = C.TP[C.TP.in_primary == 1] if C.TP is not None else None
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
    ap.add_argument("--typeblind1000", default="typeblind1000", help="folder of the 1000-rep type-blind rerun (optional)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", default="", help="comma list of figure id prefixes, e.g. F04,F16")
    a = ap.parse_args()
    C = load(a)
    fd = os.path.join(a.out, "figures")
    os.makedirs(os.path.join(fd, TBDIR), exist_ok=True)
    only = [x for x in a.only.split(",") if x]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fid, title, tag, note, fn, sub in REG:
            if only and not any(fid.startswith(o) for o in only):
                continue
            try:
                f = fn(C)
                if getattr(f, "_suptitle", None) is None and not any(ax.get_title() for ax in f.axes):
                    f.suptitle(title, y=1.0, va="bottom")
                polish(f)
                f.savefig(os.path.join(fd, sub, fid + ".png"), dpi=a.dpi, bbox_inches="tight")
                plt.close(f)
                OK.append(fid)
                print(f"  ok      {os.path.join(sub, fid)}")
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