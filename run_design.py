"""
run_design.py -- [C3] the virtual experiment of slide 6.

DESIGN
  main block:       gamma in {0.04, 0.08, 0.12, 0.16}  x  overlap_sign in {-1 (rivalry), +1 (collusion)}
                    x  100 replications  = 800 runs at the slide grid (C6: 5 gamma levels incl. edges 0.04/0.20 = 1000 per block); everything else at Params defaults
                    (capture_rent 0.5, civil_cost_mult 1.0 [C6; 1.3 = block civil13], ruling_home_mult 1.5, 5 oligarchs x 5 countries).
  typeblind block:  identical grid with civil_cost_mult = ruling_home_mult = 1.0 (type-blind null
                    calibration; project_overview.txt 5.2). Size is [OPEN]; default = full grid,
                    reduce with --reps-typeblind.
SEEDS
  Replication r uses seed = base_seed + r in EVERY condition and block (common random numbers; Law 2015),
  so paired comparisons across gamma / overlap sign / block see the same initial draws and shocks.
OUTPUT
  <out>/runs/<run_id>.npz   one RunLog per run (see runlog.py); load with RunLog.load(path)
  <out>/manifest.csv        run_id, block, gamma, overlap_sign, rep, seed, config_hash, git_commit,
                            n_ticks, path, runtime_s, exit_frac, collapses  (+ windows in each run's meta)
USAGE
    python run_design.py --out out --dry-run
    python run_design.py --out out --reps 100 --blocks main,typeblind --workers 4
    python run_design.py --out pilot_out --reps 3 --blocks main          # quick smoke run
    python run_design.py --out out --set n_ticks=150                      # override any Params field
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

from model_core import Params
from m0_model import M0World
from pilot import windows, EDGES

GAMMAS = (0.04, 0.08, 0.12, 0.16, 0.20)   # [C6] 0.04 and 0.20 are EDGES (pilot.EDGES)
OVERLAPS = (-1.0, 1.0)
REPS = 100
BASE_SEED = 1000
BLOCKS = {"main": {}, "typeblind": dict(civil_cost_mult=1.0, ruling_home_mult=1.0),
          "civil13": dict(civil_cost_mult=1.3)}      # [C6] OFAT arm: the slides' civil cost multiplier


def make_params(base: Params, block: str, gamma: float, overlap: float) -> Params:
    return replace(base, gamma=gamma, overlap_sign=overlap, **BLOCKS[block])


def config_hash(p: Params) -> str:
    return hashlib.sha1(json.dumps(asdict(p), sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_commit() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"


def run_id(block: str, gamma: float, overlap: float, rep: int) -> str:
    return f"{block}_g{gamma:.2f}_s{'m' if overlap < 0 else 'p'}_r{rep:03d}"


def make_jobs(base: Params, out: str, blocks: List[str], reps: Dict[str, int], base_seed: int,
              gammas=GAMMAS, overlaps=OVERLAPS) -> List[dict]:
    commit = git_commit()
    jobs = []
    for b in blocks:
        for g in gammas:
            for s in overlaps:
                p = make_params(base, b, g, s)
                h = config_hash(p)
                for r in range(reps[b]):
                    rid = run_id(b, g, s, r)
                    jobs.append(dict(run_id=rid, block=b, gamma=g, overlap_sign=s, rep=r, seed=base_seed + r,
                                     params=p, config_hash=h, git_commit=commit,
                                     path=os.path.join(out, "runs", rid + ".npz")))
    return jobs


def run_one(job: dict) -> dict:
    t0 = time.time()
    p = job["params"]
    w = M0World(p, job["seed"])
    rl = w.run()
    (e0, e1), (o0, o1) = windows(p.n_ticks)
    meta = dict(edge=bool(job["gamma"] in EDGES), run_id=job["run_id"], block=job["block"], gamma=job["gamma"], overlap_sign=job["overlap_sign"],
                rep=job["rep"], seed=job["seed"], config_hash=job["config_hash"], git_commit=job["git_commit"],
                windows=dict(burn_in=e0, embed=[e0, e1], outcome=[o0, o1]), params=asdict(p))
    rl.save(job["path"], meta=meta)
    return dict(edge=int(job["gamma"] in EDGES), run_id=job["run_id"], block=job["block"], gamma=job["gamma"], overlap_sign=job["overlap_sign"],
                rep=job["rep"], seed=job["seed"], config_hash=job["config_hash"], git_commit=job["git_commit"],
                n_ticks=p.n_ticks, path=job["path"], runtime_s=round(time.time() - t0, 3),
                exit_frac=float(w.exited_array().mean()),
                collapses=int(sum(c.collapse_count for c in w.countries)))


def _parse_sets(items: List[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        default = getattr(Params(), k)
        out[k] = type(default)(float(v)) if isinstance(default, (int, float)) and not isinstance(default, bool) \
            else (v.lower() in ("1", "true", "yes") if isinstance(default, bool) else v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--reps-typeblind", type=int, default=None, help="default: same as --reps")
    ap.add_argument("--blocks", default="main,typeblind", help="also available: civil13 (OFAT arm)")
    ap.add_argument("--gammas", default=",".join(str(g) for g in GAMMAS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--set", nargs="*", default=[], help="override Params fields, e.g. n_ticks=150")
    ap.add_argument("--resume", action="store_true", help="skip runs whose .npz already exists")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = replace(Params(), **_parse_sets(a.set))
    blocks = [b for b in a.blocks.split(",") if b]
    reps = {b: (a.reps_typeblind if (b == "typeblind" and a.reps_typeblind is not None) else a.reps) for b in BLOCKS}
    jobs = make_jobs(base, a.out, blocks, reps, a.base_seed, gammas=tuple(float(x) for x in a.gammas.split(',')))
    print(f"{len(jobs)} runs: " + ", ".join(f"{b}={sum(j['block'] == b for j in jobs)}" for b in blocks)
          + f"  | commit={jobs[0]['git_commit'] if jobs else 'n/a'}")
    if a.dry_run:
        return
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    if a.resume:
        jobs = [j for j in jobs if not os.path.exists(j["path"])]
        print(f"resuming: {len(jobs)} runs remaining")
    t0 = time.time()
    rows = []
    if a.workers <= 1:
        for k, j in enumerate(jobs):
            rows.append(run_one(j))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s")
    else:
        with Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4)):
                rows.append(r)
                if (k + 1) % 50 == 0:
                    print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s")
    mpath = os.path.join(a.out, "manifest.csv")
    exists = os.path.exists(mpath)
    with open(mpath, "a", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if not exists and rows:
            wtr.writeheader()
        wtr.writerows(sorted(rows, key=lambda r: r["run_id"]))
    print(f"done: {len(rows)} runs in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()