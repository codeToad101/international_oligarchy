"""
validate_c.py -- calibration validation + sanity checks (needs pilot.EDGES and 5-level GAMMAS from patch_c6).

  python validate_c.py --seeds 20 --workers 4                      # blocks x overlaps x 5 gammas, resumable, then report
  python validate_c.py --seeds 20 --workers 4 --seed0 5000 --csv heldout.csv   # [D] HELD-OUT seeds (seed = seed0 + rep)
  python validate_c.py --report [--csv FILE]                       # tables only
  python validate_c.py --typetest [--csv FILE]                     # [D] civil-vs-ruling exit permutation test
  python validate_c.py --compare OLD.csv NEW.csv                   # rename/patch-neutrality check
  python validate_c.py --time 1 | --price-check
  options: --blocks main,typeblind,civil13  --overlaps -1,1  --gammas 0.04,...  --set civil_cost_mult=1.3

Default seed0 = run_design.BASE_SEED (1000), i.e. the production seeds. Use a different --seed0 AND a different
--csv for held-out checks. Gammas in pilot.EDGES are EDGES ('*'): 'economy not destroyed' targets only, and
excluded from the exit-count, slope and type-test lines (those describe the interior suite).
E->capture / E->havoc numbers are REPORTED, never targets. Pooled slopes are unadjusted diagnostics, not the P test.
"""
import argparse, csv, os, time, warnings
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from model_core import Params
from m0_model import M0World
import pilot
from pilot import EDGES
from run_design import make_params, GAMMAS, BASE_SEED, _parse_sets

warnings.simplefilter("ignore", RuntimeWarning)
CSV = "validate_rows.csv"
TEXT = {"block", "exit_ticks"}
TYPES = ("civil", "ruling")
DVS = ("cap", "hav")


def one(args):
    base, block, g, s, r, seed0 = args
    p = make_params(base, block, g, s)
    t0 = time.time()
    rl = M0World(p, seed0 + r).run()
    rt = time.time() - t0
    row = pilot.summarize(rl, p)
    (e0, e1), (o0, o1) = pilot.windows(rl.T)
    ol, st = rl.olig, rl.static
    act, ruling = ol["active"], st["ruling"]
    ae, ao = act[e0:e1], act[o0:o1]
    E_o = np.nanmean(np.where(ae, ol["E_foreign"][e0:e1], np.nan), axis=0)
    fc, dc = ol["foreign_capture"][o0:o1], ol["domestic_capture"][o0:o1]
    tc = fc + dc
    sh = np.where(ao & (tc > 1e-9), fc / np.where(tc > 1e-9, tc, 1.0), np.nan)
    dv = {"cap": np.nanmean(sh, axis=0), "hav": np.nanmean(np.where(ao, ol["foreign_havoc"][o0:o1], np.nan), axis=0)}
    for ty in TYPES:
        sel = ruling if ty == "ruling" else ~ruling
        for k in DVS:
            ok = sel & ~np.isnan(E_o) & ~np.isnan(dv[k])
            x, y = E_o[ok], dv[k][ok]
            row.update({f"{k}_{ty}_n": float(ok.sum()), f"{k}_{ty}_sx": float(x.sum()), f"{k}_{ty}_sy": float(y.sum()),
                        f"{k}_{ty}_sxx": float((x * x).sum()), f"{k}_{ty}_sxy": float((x * y).sum())})
    et = st["exit_tick"]
    ex = ~np.isnan(et)
    row.update(n_exit=float(ex.sum()), n_exit_civil=float(ex[~ruling].sum()), n_exit_ruling=float(ex[ruling].sum()),
               n_ruling=float(ruling.sum()), exit_ticks=" ".join(str(int(x)) for x in et[ex]),
               block=block, gamma=g, overlap=s, rep=r, seed=float(seed0 + r), runtime=rt, T=float(rl.T))
    return row


def done_keys():
    if not os.path.exists(CSV):
        return set()
    with open(CSV) as f:
        return {(d["block"], float(d["gamma"]), float(d["overlap"]), int(float(d["rep"]))) for d in csv.DictReader(f)}


def append(rows):
    new = not os.path.exists(CSV)
    fields = list(rows[0].keys()) if new else list(next(csv.reader(open(CSV))))
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def run_all(base, blocks, overlaps, gammas, seeds, workers, seed0):
    have = done_keys()
    jobs = [(base, b, g, s, r, seed0) for b in blocks for s in overlaps for g in gammas for r in range(seeds)
            if (b, g, s, r) not in have]
    print(f"{len(jobs)} runs to do ({len(have)} already in {CSV}); seeds {seed0}..{seed0 + seeds - 1}")
    if not jobs:
        return
    t0, buf = time.time(), []
    it = (Pool(workers).imap_unordered(one, jobs) if workers > 1 else map(one, jobs))
    for k, row in enumerate(it, 1):
        buf.append(row)
        if len(buf) >= 20 or k == len(jobs):
            append(buf)
            buf = []
            print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ report
def load(path=None):
    rows = []
    with open(path or CSV) as f:
        for d in csv.DictReader(f):
            rows.append({k: (v if k in TEXT else float(v)) for k, v in d.items()})
    return rows


def mean(rows, k):
    return float(np.nanmean([r[k] for r in rows])) if rows else np.nan


def slope(rows, k, ty):
    n, sx, sy, sxx, sxy = (sum(r[f"{k}_{ty}_{a}"] for r in rows) for a in ("n", "sx", "sy", "sxx", "sxy"))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if n > 5 and den > 1e-12 else np.nan


def report():
    rows = load()
    COLS = ["growth_med", "exit_frac", "exit_frac_civil", "exit_frac_ruling", "collapses_pc", "fshare",
            "with_foreign", "E_sd", "E_neg_frac", "offshore_share", "abuse_frac", "health",
            "scarcity", "active_outcome", "corr_E_capshare", "corr_E_havoc"]
    T = int(rows[0]["T"])
    (e0, e1), (o0, o1) = pilot.windows(T)
    for blk in sorted({r["block"] for r in rows}):
        for ov in sorted({r["overlap"] for r in rows if r["block"] == blk}):
            sub = [r for r in rows if r["block"] == blk and r["overlap"] == ov]
            gs = sorted({r["gamma"] for r in sub})
            print(f"\n=== block={blk}  overlap_sign={ov:+.0f}  ({len(sub)} runs, {len({r['rep'] for r in sub})} seeds/gamma) ===")
            print("gamma  " + " ".join(f"{c[:9]:>9}" for c in COLS))
            table = {}
            for g in gs:
                gr = [r for r in sub if r["gamma"] == g]
                table[g] = {k: mean(gr, k) for k in gr[0] if k not in TEXT}
                print(f"{g:5.2f}{'*' if g in EDGES else ' '} " + " ".join(f"{table[g][c]:9.3f}" for c in COLS))
            pen, msgs = pilot.check(table)
            print(f"target penalty = {pen:.3f}  (* = edge; edges carry 'economy not destroyed' targets only)")
            for m in msgs:
                print("   -", m)
            suite = [r for r in sub if r["gamma"] not in EDGES]
            if not suite:
                continue
            ticks = [int(x) for r in suite for x in r["exit_ticks"].split()]
            print(f"interior-suite exits: {len(ticks)} events in {25 * len(suite)} oligarch-runs; "
                  f"civil {int(sum(r['n_exit_civil'] for r in suite))}, ruling {int(sum(r['n_exit_ruling'] for r in suite))}; "
                  f"runs with >=1 exit {np.mean([r['n_exit'] >= 1 for r in suite]):.2f}, >=3 exits {np.mean([r['n_exit'] >= 3 for r in suite]):.2f}")
            if ticks:
                bins = list(range(0, T + 1, 10))
                h, _ = np.histogram(ticks, bins=bins)
                print("exit-tick histogram: " + " ".join(f"{b}-{b + 10}:{c}" for b, c in zip(bins[:-1], h)))
                print(f"exit ticks: min {min(ticks)}, median {int(np.median(ticks))}; share at tick<={o0} (before outcome window): "
                      f"{np.mean(np.array(ticks) <= o0):.2f}")
            sl = {k: (slope(suite, k, "civil"), slope(suite, k, "ruling")) for k in DVS}
            print("pooled E-slope, interior (civil, ruling, ratio): " + "; ".join(
                f"{k}: {a:.3g}, {b:.3g}, {a / b if b and not np.isnan(b) else np.nan:.2f}" for k, (a, b) in sl.items()))
    rt = mean(rows, "runtime")
    print(f"\nruntime {rt:.2f}s/run mean -> 2000 runs (5 gammas x 2 overlaps x 100 reps x 2 blocks) ~ {2000 * rt / 60:.0f} min single core")
    print("windows: embed [%d,%d), outcome [%d,%d)" % (e0, e1, o0, o1))


def typetest(nperm=20000):
    """[D] Is civil-vs-ruling exit different from chance? Per seed: d = civil exit rate - ruling exit rate, averaged
    over the interior gammas (same seed => same type placement across gammas, so gammas are NOT independent;
    one d per seed). Sign-flip randomization test on the per-seed d. d > 0: civil exit more."""
    rows = load()
    rng = np.random.default_rng(0)
    O = Params().n_countries * Params().oligs_per_country
    print("civil-minus-ruling exit rate, interior gammas (sign-flip test over seeds)")
    for blk, ov in sorted({(r["block"], r["overlap"]) for r in rows}):
        sub = [r for r in rows if r["block"] == blk and r["overlap"] == ov and r["gamma"] not in EDGES]
        d = np.array([np.mean([r["n_exit_civil"] / (O - r["n_ruling"]) - r["n_exit_ruling"] / r["n_ruling"]
                               for r in sub if r["rep"] == rep]) for rep in sorted({r["rep"] for r in sub})])
        if len(d) < 3:
            continue
        obs = d.mean()
        null = (rng.choice([-1.0, 1.0], size=(nperm, len(d))) * d).mean(1)
        p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
        print(f"  {blk:9s} overlap {ov:+.0f}: mean d = {obs:+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f}, {len(d)} seeds), p = {p:.4f}")


def compare(a, b):
    def L(path):
        with open(path) as f:
            return {(d["block"], float(d["gamma"]), float(d["overlap"]), int(float(d["rep"]))): d for d in csv.DictReader(f)}
    A, B = L(a), L(b)
    keys = sorted(set(A) & set(B))
    print(f"{len(keys)} matching runs ({a} vs {b})")
    if not keys:
        print("!! no matching (block, gamma, overlap, rep) keys")
        return
    for c in ("growth_med", "exit_frac", "collapses_pc", "fshare", "E_mean", "offshore_share", "abuse_frac", "n_exit"):
        d = np.array([abs(float(A[k][c]) - float(B[k][c])) for k in keys])
        print(f"  {c:15s} max|diff| {np.nanmax(d):.3g}   mean|diff| {np.nanmean(d):.3g}")
    mism = sum(A[k]["exit_ticks"] != B[k]["exit_ticks"] for k in keys)
    print(f"  runs with different exit-tick lists: {mism} of {len(keys)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=BASE_SEED, help="seed = seed0 + rep (default = production seeds)")
    ap.add_argument("--blocks", default="main,typeblind")
    ap.add_argument("--overlaps", default="-1,1")
    ap.add_argument("--gammas", default=",".join(str(g) for g in GAMMAS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--typetest", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--price-check", action="store_true")
    ap.add_argument("--time", type=int, default=0, help="time N runs and exit")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    CSV = a.csv
    base = replace(Params(), **_parse_sets(a.set))
    if a.compare:
        compare(*a.compare)
    if a.time:
        for _ in range(a.time):
            r = one((base, "main", 0.08, -1.0, 0, a.seed0))
            print(f"1 run: {r['runtime']:.2f}s")
    if a.seeds:
        run_all(base, [b for b in a.blocks.split(",") if b], [float(x) for x in a.overlaps.split(",")],
                [float(x) for x in a.gammas.split(",")], a.seeds, a.workers, a.seed0)
    if a.report or (a.seeds and not a.no_report):
        report()
    if a.typetest:
        typetest()
    if a.price_check:
        pilot.static_price_check(base)