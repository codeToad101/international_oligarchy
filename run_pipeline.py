#!/usr/bin/env python3
"""
run_pipeline.py -- the whole study in one command, from simulation to figures. Run from the project root.

    python run_pipeline.py --dry-run                 # print every command and run counts; changes nothing
    python run_pipeline.py --workers 4               # EVERYTHING: production, tables, P, S, eta=0 ablation, OFAT, figures
    python run_pipeline.py --steps ofat,figures      # any subset, in the canonical order
    python run_pipeline.py --workers 4 --resume      # continue an interrupted run (skips runs whose .npz exists)

STEPS (canonical order)
  production  run_design.py            main 1000 reps, type-blind 100 (BLOCK_REPS in run_design.py)  -> out/
  tables      analysis_tables.py       P table (all reps) and S table (reps < --s-max-rep)           -> out/tables/
  P           analysis_P.py            TOST, head-to-head, power (+ --robust, + --diagnostics if present)
  S           analysis_S.py            nested exit models, pooled and --by-sign
  ablation    run_design + tables      eta = 0 ablation (E -> capture is partly built in via eta)     -> ablation_eta0/
  ofat        run_design + tables      one-factor-at-a-time sensitivity (arms below + baseline)       -> ofat/<arm>/
  figures     analysis_figures.py      all figures, INDEX.md, summary.txt                             -> out/figures/
Every step's console output is saved to out/logs/<step>.txt (use these when writing up).

OFAT DESIGN (fixed here BEFORE any result is seen; levels are my own +-50% style choices, NOT calibrated and NOT picked to move a result)
  Each arm changes ONE constant to a low or high level, everything else at default; block main, gamma 0.12, both overlap signs,
  --ofat-reps replications (default 30; seeds 1000+rep, shared with production = common random numbers). A 'baseline' arm with
  the same seeds is run alongside so the comparison does not depend on the production run. Conclusions are judged on whether the
  headline estimates keep their sign and rough size, not on p-values.
  Not run (disclose as limitations): zeta_cap, zeta_bl, lam_cap, seize_frac, h_width, foreign_exposed, gamma 0.08 / 0.16 arms.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STEPS = ("production", "tables", "P", "S", "ablation", "ofat", "figures")

# param: (default in model_core.py, low, high)
OFAT_ARMS = {
    "capture_rent":     (0.5,   0.25,  0.75),
    "civil_cost_mult":  (1.0,   0.7,   1.3),     # 1.3 = the slides' value
    "ruling_home_mult": (1.5,   1.0,   2.0),
    "zeta_seize":       (0.5,   0.25,  0.75),
    "eta":              (0.15,  0.05,  0.30),
    "h_star":           (0.607, 0.5,   0.7),
    "abuse_coef":       (0.385, 0.20,  0.60),
    "w_scar":           (0.514, 0.25,  0.75),
    "exit_floor_frac":  (0.40,  0.35,  0.45),
}
OFAT_GAMMA = "0.12"
ABLATION = dict(param="eta", value="0", gammas="0.08,0.12,0.16", reps=100)


def script(rel):
    return os.path.join(ROOT, rel)


def supports(rel, flag):
    try:
        return flag in open(script(rel)).read()
    except OSError:
        return False


def arm_list(a):
    arms = [("baseline", None, None)]
    for p, (_, lo, hi) in OFAT_ARMS.items():
        arms += [(f"{p}_lo", p, lo), (f"{p}_hi", p, hi)]
    return arms


def commands(step, a):
    out, ofat, abl, W = a.out, a.ofat, a.ablation, str(a.workers)
    res = ["--resume"] if a.resume else []
    if step == "production":
        return [("production run", [PY, script("run_design.py"), "--out", out, "--workers", W] + res)]
    if step == "tables":
        return [("analysis tables", [PY, script("analysis/analysis_tables.py"), "--out", out, "--s-max-rep", str(a.s_max_rep)])]
    if step == "P":
        c = [PY, script("analysis/analysis_P.py"), "--out", out, "--robust"]
        if supports("analysis/analysis_P.py", "--diagnostics"):
            c += ["--diagnostics"]
        return [("P analysis", c)]
    if step == "S":
        return [("S analysis", [PY, script("analysis/analysis_S.py"), "--out", out, "--by-sign"])]
    if step == "ablation":
        return [("eta = 0 ablation: runs", [PY, script("run_design.py"), "--out", abl, "--blocks", "main", "--gammas", ABLATION["gammas"],
                                           "--reps", str(ABLATION["reps"]), "--workers", W, "--set", f"{ABLATION['param']}={ABLATION['value']}"] + res),
                ("eta = 0 ablation: tables", [PY, script("analysis/analysis_tables.py"), "--out", abl, "--blocks", "main"])]
    if step == "ofat":
        cmds = []
        for name, p, v in arm_list(a):
            d = os.path.join(ofat, name)
            run = [PY, script("run_design.py"), "--out", d, "--blocks", "main", "--gammas", OFAT_GAMMA, "--reps", str(a.ofat_reps), "--workers", W] + res
            if p:
                run += ["--set", f"{p}={v}"]
            cmds += [(f"OFAT {name}: runs", run), (f"OFAT {name}: tables", [PY, script("analysis/analysis_tables.py"), "--out", d, "--blocks", "main"])]
        return cmds
    if step == "figures":
        return [("figures", [PY, script("analysis/analysis_figures.py"), "--out", out, "--ofat", ofat, "--ablation", abl])]
    raise ValueError(step)


def run_cmd(label, argv, logf):
    head = f"\n=== {label}\n$ {' '.join(argv)}\n"
    print(head, end="")
    logf.write(head)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT, env=env)
    last = []
    for line in p.stdout:
        logf.write(line)
        last.append(line)
        last = last[-6:]
        if label.startswith("OFAT") and "runs" not in label:      # keep the OFAT console readable: log everything, print little
            continue
        print(line, end="")
    p.wait()
    if p.returncode and label.startswith("OFAT"):
        print("".join(last), end="")
    return p.returncode


def git_state():
    try:
        d = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return "dirty" if d else "clean"
    except Exception:
        return "nogit"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--out", default="out")
    ap.add_argument("--ofat", default="ofat")
    ap.add_argument("--ablation", default="ablation_eta0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--ofat-reps", type=int, default=30)
    ap.add_argument("--s-max-rep", type=int, default=200, help="S table only from reps < N (keeps the person-period table manageable)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue to later steps if one fails")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    want = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in want if s not in STEPS]
    if bad:
        sys.exit(f"unknown step(s) {bad}; choose from {STEPS}")
    want = [s for s in STEPS if s in want]
    need = {"production": ["run_design.py"], "tables": ["analysis/analysis_tables.py"], "P": ["analysis/analysis_P.py"], "S": ["analysis/analysis_S.py"],
            "ablation": ["run_design.py", "analysis/analysis_tables.py"], "ofat": ["run_design.py", "analysis/analysis_tables.py"],
            "figures": ["analysis/analysis_figures.py"]}
    miss = sorted({f for s in want for f in need[s] if not os.path.exists(script(f))})
    if miss:
        sys.exit(f"missing files: {miss} (run from the project root)")
    n_arm = len(arm_list(a))
    print(f"steps: {want} | out={a.out} ofat={a.ofat} ablation={a.ablation} | workers={a.workers}")
    print(f"expected runs: production 11000 (main 10000 + type-blind 1000) | ablation {3 * 2 * ABLATION['reps']} | OFAT {n_arm} arms x {2 * a.ofat_reps} = {n_arm * 2 * a.ofat_reps}")
    gs = git_state()
    if gs != "clean" and any(s in want for s in ("production", "ablation", "ofat")):
        print(f"WARNING: git state is '{gs}': manifests will say '+dirty' or 'nogit'. Commit the analysis plan and code BEFORE the production run.")
    if "production" in want and not a.resume and os.path.exists(os.path.join(a.out, "manifest.csv")) and not a.dry_run:
        sys.exit(f"{a.out}/manifest.csv already exists. Use --resume to continue that run, or move {a.out} aside; refusing to append duplicate rows.")
    if "ofat" in want and not a.dry_run:
        os.makedirs(a.ofat, exist_ok=True)
        with open(os.path.join(a.ofat, "arms.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "param", "value", "default", "gamma", "reps", "block"])
            for name, p, v in arm_list(a):
                w.writerow([name, p or "", "" if v is None else v, "" if p is None else OFAT_ARMS[p][0], OFAT_GAMMA, a.ofat_reps, "main"])
    t00 = time.time()
    for step in want:
        cmds = commands(step, a)
        if a.dry_run:
            print(f"\n[{step}] {len(cmds)} command(s)")
            for label, argv in cmds[:4] if step == "ofat" else cmds:
                print(f"  {label}: {' '.join(argv)}")
            if step == "ofat":
                print(f"  ... ({len(cmds)} commands: {n_arm} arms x runs + tables; arms listed in ofat/arms.csv)")
            continue
        os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
        t0, rc = time.time(), 0
        with open(os.path.join(a.out, "logs", f"{step}.txt"), "a") as logf:
            for label, argv in cmds:
                rc = run_cmd(label, argv, logf)
                if rc:
                    print(f"\n!! step '{step}' failed at '{label}' (exit {rc}); see {a.out}/logs/{step}.txt")
                    break
        print(f"[{step}] {'FAILED' if rc else 'done'} in {time.time() - t0:.0f}s")
        if rc and not a.keep_going:
            sys.exit(rc)
    if not a.dry_run:
        print(f"\nall requested steps finished in {time.time() - t00:.0f}s. Next: open {a.out}/figures/INDEX.md and {a.out}/figures/summary.txt")


if __name__ == "__main__":
    main()