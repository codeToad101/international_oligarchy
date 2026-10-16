"""
test_batchD.py -- Batch D1 (engineering tests) + D2 (manipulation checks).
Run from the project folder (or from tests/):   python test_batchD.py      (also works with: pytest -q)
Seeds >= 9000 are held out from production (1000+) and calibration. Roughly 30-60 s in total.
Tags: [D1] engineering, [D2] manipulation checks.
"""
import os, sys, tempfile, time, traceback
from dataclasses import replace
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
for d in (here, os.path.dirname(here)):
    if d not in sys.path:
        sys.path.insert(0, d)
from model_core import Params
from m0_model import M0World
import pilot

INFO = {}


def P(**kw):
    return replace(Params(), **kw)


def _same(a, b):
    for grp in ("olig", "country", "world", "static", "country_static"):
        da, db = getattr(a, grp), getattr(b, grp)
        assert da.keys() == db.keys(), f"{grp}: key mismatch"
        for k in da:
            assert np.array_equal(da[k], db[k], equal_nan=True), f"{grp}.{k} differs"


def _check_state(w, p):
    """[D1] bounds, symmetry, finiteness, exited agents hold nothing. Called after every tick."""
    A, Cp = w.stack_positions(), w.stack_capture()
    assert np.isfinite(A).all() and A.min() >= -1e-9, "positions negative/non-finite"
    assert np.isfinite(Cp).all() and Cp.min() >= -1e-9 and Cp.max() <= p.kmax + 1e-9, "capture outside [0,kmax]"
    for o in w.oligarchs:
        assert np.isfinite(o.capital) and np.isfinite(o.offshore_capital), "non-finite capital"
        assert o.capital >= -1e-9 and o.offshore_capital >= -1e-9, "negative capital"
        if not o.active:
            assert abs(o.capital) < 1e-12 and np.abs(o.positions).sum() < 1e-12 and np.abs(o.capture).sum() < 1e-12, \
                f"exited agent {o.idx} holds assets"
    for c in w.countries:
        assert (c.resource_health >= 0).all() and (c.resource_health <= 1).all(), "health outside [0,1]"
        assert np.isfinite(c.backlash_pressure) and c.backlash_pressure >= 0, "bad backlash"
    R = w.relations
    assert np.allclose(R.G, R.G.T, atol=1e-12), "G not symmetric"
    assert np.allclose(np.diag(R.G), 0.0), "G diagonal not zero"
    assert np.abs(R.G).max() <= 1 + 1e-12 and np.abs(R.S).max() <= 1 + 1e-12, "G or S outside [-1,1]"
    tgt = w.capture_terms(A)[3]
    assert tgt.min() >= -1e-12 and tgt.sum(0).max() <= p.kmax + 1e-9, "capture target outside [0,kmax]"
    tk = w.tick
    for k in ("wealth", "E_foreign", "E_all", "active_capital"):
        assert np.isfinite(tk[k]).all(), f"{k} non-finite"


# ----------------------------------------------------------------------------------- D1
def test_D1_determinism():
    a, b = M0World(P(), 9001).run(), M0World(P(), 9001).run()
    _same(a, b)
    c = M0World(P(), 9002).run()
    assert not np.array_equal(a.olig["wealth"], c.olig["wealth"], equal_nan=True), "different seeds gave identical runs"


def test_D1_bounds_every_tick():
    n = 0
    for g in (0.04, 0.20):
        for s in (-1.0, 1.0):
            for seed in (9001, 9002):
                p = P(gamma=g, overlap_sign=s)
                w = M0World(p, seed)
                for _ in range(p.n_ticks):
                    w.step()
                    _check_state(w, p)
                n += 1
    INFO["test_D1_bounds_every_tick"] = f"{n} runs x {Params().n_ticks} ticks checked"


def _identity(p, seed):
    w = M0World(p, seed)
    Os = w.oligarchs
    W = np.array([o.wealth for o in Os])
    pos = np.array([o.positions.sum() for o in Os])
    off = np.array([o.offshore_capital for o in Os])
    xl = np.zeros(w.O)
    cm = np.where(w.ruling, 1.0, p.civil_cost_mult)
    worst, n_inactive = 0.0, 0
    for _ in range(p.n_ticks):
        w.step()
        tk = w.tick
        act = tk["active"]
        loss = tk["loss_seized"] + tk["loss_home_hit"] + tk["loss_forced_liq"] + tk["loss_divest"]
        exp_ = tk["extraction"] + tk["rent"] - cm * p.op_cost * pos + p.offshore_return * off - loss
        resid = np.where(act, tk["wealth"] - W - exp_, tk["wealth"] - (W - xl))   # inactive: only the exit write-down
        worst = max(worst, float(np.abs(resid).max()))
        n_inactive += int((~act).sum())
        W, pos, off, xl = tk["wealth"], tk["pos_home"] + tk["pos_foreign"], tk["offshore"], tk["exit_liq_loss"]
    return worst, n_inactive


def test_D1_wealth_identity():
    """dW = extraction + rent - running cost + offshore return - (4 loss categories); exited: W flat after the
    one-off exit_liq_loss (offshore stops earning). Flight/deploy/exit transfer are wealth-neutral."""
    out = []
    for p, seed in ((P(gamma=0.12), 9003), (P(gamma=0.20, civil_cost_mult=1.3), 9004)):
        worst, n_in = _identity(p, seed)
        assert worst < 1e-8, f"identity residual {worst:.3e} (gamma={p.gamma}, civil_cost_mult={p.civil_cost_mult})"
        out.append(f"{worst:.1e}/{n_in} inactive obs")
    INFO["test_D1_wealth_identity"] = "max residual " + ", ".join(out)


def test_D1_exit_logic_and_zero_flows():
    FLOW = ("extraction", "rent", "foreign_capture", "domestic_capture", "foreign_havoc", "domestic_havoc",
            "loss_seized", "loss_home_hit", "loss_forced_liq", "loss_divest", "flight", "exit_liq_loss",
            "E_foreign", "E_all", "E_pos", "E_neg", "pos_home", "pos_foreign", "active_capital", "capital")
    n_exit = 0
    for seed in (9001, 9002, 9003):
        p = P(gamma=0.20)
        rl = M0World(p, seed).run()
        act, en = rl.olig["active"], rl.olig["exited_now"]
        assert np.all(np.diff(act.astype(int), axis=0) <= 0), "exit not absorbing"
        for o, et in enumerate(rl.static["exit_tick"]):
            if np.isnan(et):
                assert en[:, o].sum() == 0 and act[:, o].all(), "non-exiter flagged"
            else:
                t = int(et)
                n_exit += 1
                assert en[:, o].sum() == 1 and en[t - 1, o] and act[t - 1, o], "exit tick inconsistent"
                assert t == rl.T or not act[t, o], "still active after exit"
        for k in FLOW:
            assert np.all(rl.olig[k][~act] == 0), f"{k} nonzero for inactive agents"
    assert n_exit > 0, "no exits in 3 gamma=0.20 runs: test not informative"
    INFO["test_D1_exit_logic_and_zero_flows"] = f"{n_exit} exits checked"


def test_D1_loss_accounting_matches_log():
    rl = M0World(P(gamma=0.20), 9005).run()
    pairs = (("loss_seized", "cum_seized"), ("loss_home_hit", "cum_home_hit"), ("loss_forced_liq", "cum_forced_liq"),
             ("loss_divest", "cum_divest"), ("flight", "cum_flight"))
    for f, s in pairs:
        assert np.allclose(rl.olig[f].sum(0), rl.static[s], rtol=1e-3, atol=1e-2), f"{f} sum != {s}"


def test_D1_type_split_and_homes():
    for seed in (9001, 9002, 9003, 9004, 9005):
        w = M0World(P(), seed)
        assert int(w.ruling.sum()) == 12, f"ruling count {int(w.ruling.sum())}"
        assert (np.bincount(w.home) == 5).all(), "5 oligarchs per country"


def test_D1_runlog_roundtrip():
    rl = M0World(P(n_ticks=30), 9006).run()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.npz")
        rl.save(path, meta=dict(run_id="t"))
        _same(rl, type(rl).load(path))


# ----------------------------------------------------------------------------------- D2
def _collapses(p, seed):
    w = M0World(p, seed)
    w.run(log=False)
    return sum(c.collapse_count for c in w.countries) / w.C


def test_D2_gamma_raises_collapses():
    gs, seeds = (0.04, 0.08, 0.12, 0.16, 0.20), range(9100, 9112)
    m = {g: np.array([_collapses(P(gamma=g), s) for s in seeds]) for g in gs}
    means = [m[g].mean() for g in gs]
    assert all(b > a for a, b in zip(means, means[1:])), f"collapses not increasing in gamma: {np.round(means, 2)}"
    assert np.mean(m[0.20] > m[0.04]) >= 0.9, "gamma 0.20 vs 0.04 not higher in >=90% of paired seeds"
    INFO["test_D2_gamma_raises_collapses"] = "collapses/country " + " / ".join(f"{x:.2f}" for x in means)


def _probe(sign, same_industry):
    """One goodwill update with ONLY oligarchs a,b holding positions, decay and noise switched off."""
    w = M0World(P(overlap_sign=sign), 9200)
    R, p, O, C = w.relations, w.p, w.O, w.C
    G0 = R.G0.copy()
    idx = int(np.argmin(np.where(~np.eye(O, dtype=bool), np.abs(G0), 9.0)))
    a, b = divmod(idx, O)
    R.G = G0.copy()
    A = np.zeros((O, C, w.endow.shape[1]))
    A[a, 0, 0] = 1.0
    A[b, 0, 0 if same_industry else 1] = 1.0
    w._update_goodwill(A, np.zeros_like(A), np.ones((C, A.shape[2])) / A.shape[2],
                       np.zeros((O, O)), np.zeros((O, C)))
    assert abs(R.G[a, b] - R.G[b, a]) < 1e-15, "asymmetric update"
    return R.G[a, b] - G0[a, b], p.lam_oo


def test_D2_overlap_sign_flips_G_contribution():
    for sign in (-1.0, 1.0):
        d, lam = _probe(sign, same_industry=True)
        assert abs(d - sign * lam) < 1e-12, f"same-industry dG={d:.4g}, expected {sign * lam:.4g}"
        d2, _ = _probe(sign, same_industry=False)
        assert abs(d2 - lam) < 1e-12, f"complementary (same country, other industry) dG={d2:.4g}, expected {lam:.4g}"
    INFO["test_D2_overlap_sign_flips_G_contribution"] = "same-industry dG = sign*lam_oo; complementary = +lam_oo for both signs"


def test_D2_overlap_sign_shifts_E_negative_share():
    seeds = range(9300, 9310)
    f = {}
    for s in (-1.0, 1.0):
        p = P(gamma=0.12, overlap_sign=s)
        f[s] = np.mean([pilot.summarize(M0World(p, sd).run(), p)["E_neg_frac"] for sd in seeds])
    assert f[-1.0] - f[1.0] > 0.05, f"E_neg_frac rivalry {f[-1.0]:.3f} vs collusion {f[1.0]:.3f}"
    INFO["test_D2_overlap_sign_shifts_E_negative_share"] = f"E<0 share: rivalry {f[-1.0]:.3f}, collusion {f[1.0]:.3f}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fails = 0
    for n, f in tests:
        t0 = time.time()
        try:
            f()
            print(f"PASS {n} ({time.time() - t0:.1f}s)" + (f"  {INFO[n]}" if n in INFO else ""))
        except Exception as e:
            fails += 1
            ln = traceback.extract_tb(e.__traceback__)[-1].lineno
            print(f"FAIL {n}: {type(e).__name__}: {str(e)[:220]}  (line {ln})")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)