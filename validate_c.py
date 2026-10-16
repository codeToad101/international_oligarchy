"""
validate_c.py -- [C(a)-(c)] calibration validation. Run from the folder with the model files.

  python validate_c.py --seeds 10                         # everything (16 x seeds runs), resumable
  python validate_c.py --seeds 10 --blocks main --overlaps -1     # chunk it
  python validate_c.py --report                           # tables only, from validate_rows.csv
  python validate_c.py --price-check                      # static-price adequacy (5.4)
  python validate_c.py --time 1                           # time 1 run, project 1600-run cost

Per-run rows are appended to validate_rows.csv (skips finished runs, so you can stop/restart).
Seeds = 1000 + rep, same as run_design.py (common random numbers).
The E->capture / E->havoc numbers are REPORTED, never targets. Slopes below are pooled and unadjusted
(no run/country effects): a quick diagnostic only, NOT the P test (that is Batch E).
"""
import argparse, csv, os, time, warnings
from multiprocessing import Pool
import numpy as np

from model_core import Params
from m0_model import M0World
import pilot
from run_design import make_params, GAMMAS, BASE_SEED, _parse_sets

warnings.simplefilter("ignore", RuntimeWarning)
CSV = "validate_rows.csv"
TEXT = {"block", "exit_ticks"}
TYPES = ("civil", "ruling")
DVS = ("cap", "hav")


def one(args):
    base, block, g, s, r = args
    p = make_params(base, block, g, s)
    t0 = time.time()
    rl = M0World(p, BASE_SEED + r).run()
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
               block=block, gamma=g, overlap=s, rep=r, runtime=rt, T=float(rl.T))
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


def run_all(base, blocks, overlaps, seeds, workers):
    have = done_keys()
    jobs = [(base, b, g, s, r) for b in blocks for s in overlaps for g in GAMMAS for r in range(seeds)
            if (b, g, s, r) not in have]
    print(f"{len(jobs)} runs to do ({len(have)} already in {CSV})")
    t0, buf = time.time(), []
    it = (Pool(workers).imap_unordered(one, jobs) if workers > 1 else map(one, jobs))
    for k, row in enumerate(it, 1):
        buf.append(row)
        if len(buf) >= 10 or k == len(jobs):
            append(buf); buf = []
            print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ report
def load():
    rows = []
    with open(CSV) as f:
        for d in csv.DictReader(f):
            rows.append({k: (v if k in TEXT else float(v)) for k, v in d.items()})
    return rows


def mean(rows, k):
    return float(np.nanmean([r[k] for r in rows])) if rows else np.nan


def slope(rows, k, ty):
    n, sx, sy, sxx, sxy = (sum(r[f"{k}_{ty}_{a}"] for r in rows) for a in ("n", "sx", "sy", "sxx", "sxy"))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if n > 5 and den > 1e-12 else np.nan


def report(base):
    rows = load()
    COLS = ["growth_med", "exit_frac", "exit_frac_civil", "exit_frac_ruling", "collapses_pc", "fshare",
            "with_foreign", "E_sd", "E_neg_frac", "E_pos_frac", "offshore_share", "abuse_frac", "health",
            "scarcity", "active_outcome", "corr_E_capshare", "corr_E_havoc", "runtime"]
    T = int(rows[0]["T"])
    (e0, e1), (o0, o1) = pilot.windows(T)
    for blk in sorted({r["block"] for r in rows}):
        for ov in sorted({r["overlap"] for r in rows if r["block"] == blk}):
            sub = [r for r in rows if r["block"] == blk and r["overlap"] == ov]
            nseed = len({r["rep"] for r in sub})
            print(f"\n=== block={blk}  overlap_sign={ov:+.0f}  ({len(sub)} runs, {nseed} seeds/gamma) ===")
            print("gamma " + " ".join(f"{c[:9]:>9}" for c in COLS))
            table = {}
            for g in GAMMAS:
                gs = [r for r in sub if r["gamma"] == g]
                if not gs:
                    continue
                table[g] = {k: mean(gs, k) for k in gs[0] if k not in TEXT}
                print(f"{g:5.2f} " + " ".join(f"{table[g][c]:9.3f}" for c in COLS))
            if len(table) == len(GAMMAS):
                pen, msgs = pilot.check(table)
                print(f"target penalty = {pen:.3f}" + ("" if blk == "main" and ov < 0 else "  (targets were calibrated on main, overlap -1)"))
                for m in msgs:
                    print("   -", m)
            # exits: counts and timing
            ticks = [int(x) for r in sub for x in r["exit_ticks"].split()]
            print(f"exits: {len(ticks)} events / {int(sum(r['n_exit'] for r in sub))} of {25 * len(sub)} oligarch-runs; "
                  f"civil {int(sum(r['n_exit_civil'] for r in sub))}, ruling {int(sum(r['n_exit_ruling'] for r in sub))}; "
                  f"runs with >=1 exit {np.mean([r['n_exit'] >= 1 for r in sub]):.2f}, >=3 exits {np.mean([r['n_exit'] >= 3 for r in sub]):.2f}")
            if ticks:
                bins = list(range(0, T + 1, 10))
                h, _ = np.histogram(ticks, bins=bins)
                print("exit-tick histogram (bins of 10): " + " ".join(f"{b}-{b + 10}:{c}" for b, c in zip(bins[:-1], h)))
                print(f"exit ticks: min {min(ticks)}, median {int(np.median(ticks))}; before outcome window (tick<={o0}): "
                      f"{np.mean(np.array(ticks) <= o0):.2f}")
            # pooled slopes by type (diagnostic)
            sl = {k: (slope(sub, k, "civil"), slope(sub, k, "ruling")) for k in DVS}
            print("pooled E-slope (civil, ruling, ratio): " + "; ".join(
                f"{k}: {a:.3g}, {b:.3g}, {a / b if b and not np.isnan(b) else np.nan:.2f}" for k, (a, b) in sl.items()))
    rt = mean(rows, "runtime")
    print(f"\nruntime: {rt:.2f}s/run mean -> 1600 runs (800 main + 800 typeblind) ~ {1600 * rt / 60:.0f} min on one core")
    print("windows: embed [%d,%d), outcome [%d,%d)" % (e0, e1, o0, o1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0)
    ap.add_argument("--blocks", default="main,typeblind")
    ap.add_argument("--overlaps", default="-1,1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--price-check", action="store_true")
    ap.add_argument("--time", type=int, default=0, help="time N runs and exit")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    from dataclasses import replace
    base = replace(Params(), **_parse_sets(a.set))
    if a.time:
        for _ in range(a.time):
            r = one((base, "main", 0.08, -1.0, 0))
            print(f"1 run: {r['runtime']:.2f}s -> 1600 runs ~ {1600 * r['runtime'] / 60:.0f} min single core")
    if a.seeds:
        run_all(base, [b for b in a.blocks.split(",") if b], [float(x) for x in a.overlaps.split(",")], a.seeds, a.workers)
    if a.seeds or a.report:
        report(base)
    if a.price_check:
        pilot.static_price_check(base)