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


# ============================================================================ D3-D5 (appended)
from model_core import NI

# ---------------------------------------------------------------------------- D3: helpers
_TAB = {}


def _run_table(kw, seeds, signs=(-1.0, 1.0)):
    """In-memory multi-seed runs -> one row per oligarch ACTIVE at outcome-window start (survivors so far).
    cols: run, ruling, E (mean E_foreign, embedding window 20-50%), fp (mean foreign position, same window),
    cap (mean foreign capture share, outcome window 50-100%), hav (mean foreign havoc, outcome window)."""
    key = (tuple(sorted(kw.items())), tuple(seeds), tuple(signs))
    if key in _TAB:
        return _TAB[key]
    ks = ("active", "E_foreign", "foreign_capture", "domestic_capture", "foreign_havoc", "pos_foreign")
    rows, run = [], 0
    for s in signs:
        p = P(overlap_sign=s, **kw)
        T = p.n_ticks
        e0, e1, o0 = int(0.2 * T), int(0.5 * T), int(0.5 * T)
        for sd in seeds:
            w = M0World(p, sd)
            buf = {k: [] for k in ks}
            for _ in range(T):
                w.step()
                for k in ks:
                    buf[k].append(np.asarray(w.tick[k], dtype=float))
            B = {k: np.array(v) for k, v in buf.items()}
            act = B["active"] > 0.5
            for o in range(w.O):
                if not act[o0, o]:
                    continue
                a = act[o0:, o]
                fc, dc = B["foreign_capture"][o0:, o][a], B["domestic_capture"][o0:, o][a]
                tot = fc + dc
                ok = tot > 1e-12
                cap = float((fc[ok] / tot[ok]).mean()) if ok.any() else np.nan
                rows.append((run, float(w.ruling[o]), float(B["E_foreign"][e0:e1, o].mean()),
                             float(B["pos_foreign"][e0:e1, o].mean()), cap,
                             float(B["foreign_havoc"][o0:, o][a].mean())))
            run += 1
    arr = np.array(rows, dtype=float)
    tab = {k: arr[:, i] for i, k in enumerate(("run", "ruling", "E", "fp", "cap", "hav"))}
    tab["n_runs"] = run
    _TAB[key] = tab
    return tab


def _fe_cluster(y, X, g):
    """OLS with group (run) fixed effects via demeaning; cluster-robust (CR0) SEs clustered on g."""
    ug, gi = np.unique(g, return_inverse=True)
    cnt = np.bincount(gi).astype(float)

    def dm(a):
        m = np.zeros((len(ug),) + a.shape[1:])
        np.add.at(m, gi, a)
        m /= cnt.reshape((-1,) + (1,) * (a.ndim - 1))
        return a - m[gi]
    yd, Xd = dm(y), dm(X)
    Ai = np.linalg.inv(Xd.T @ Xd)
    b = Ai @ (Xd.T @ yd)
    u = yd - Xd @ b
    S = np.zeros((len(ug), X.shape[1]))
    np.add.at(S, gi, Xd * u[:, None])
    return b, np.sqrt(np.diag(Ai @ (S.T @ S) @ Ai))


def _z(x):
    return (x - x.mean()) / max(float(x.std()), 1e-12)


def _fit(tab, dv):
    """Within-run slopes of rank-standardized DV on [z(E), ruling, z(E)*ruling, z(foreign presence)].
    Returns (b, se); index 0 = E slope, 2 = E x type. Ranks make the havoc DV robust to its right skew."""
    m = np.isfinite(tab["cap"]) & np.isfinite(tab["hav"]) & np.isfinite(tab["E"])
    assert m.sum() >= 500, f"only {int(m.sum())} usable rows"
    y = _z(np.argsort(np.argsort(tab[dv][m])).astype(float))
    z, rr = _z(tab["E"][m]), tab["ruling"][m]
    X = np.column_stack([z, rr, z * rr, _z(tab["fp"][m])])
    return _fe_cluster(y, X, tab["run"][m])


# ---------------------------------------------------------------------------- D3
def test_D3_type_blind_slopes_equal():
    """[D3] With ruling_home_mult = 1.0 (and civil_cost_mult = 1.0) the two types follow identical rules, so
    labels are exchangeable and the E x type coefficient must be noise (|t| < 3.5, run-clustered). The main
    block's coefficient is reported for contrast, not asserted."""
    seeds, out = range(9400, 9450), []
    for dv in ("cap", "hav"):
        b, se = _fit(_run_table(dict(gamma=0.12, ruling_home_mult=1.0), seeds), dv)
        bm, sm = _fit(_run_table(dict(gamma=0.12), seeds), dv)
        t = b[2] / se[2]
        assert abs(t) < 3.5, f"{dv}: E x type in TYPE-BLIND block t={t:.2f} (b={b[2]:.3f}, se={se[2]:.3f})"
        out.append(f"{dv}: E slope {b[0]:+.3f}; ExType blind {b[2]:+.3f} (t={t:+.1f}), main {bm[2]:+.3f} (t={bm[2] / sm[2]:+.1f})")
    INFO["test_D3_type_blind_slopes_equal"] = " | ".join(out)


def test_D3_eta0_ablation():
    """[D3] eta = 0 removes the partner-presence term of capture effectiveness (the built-in E -> capture channel).
    HARD assertion: |E slope| is smaller than at eta = 0.15 (same seeds, CRN). 'Collapse' (ratio < 0.5) is REPORTED,
    not asserted: E also correlates with foreign presence through other channels."""
    seeds = range(9400, 9450)
    b1 = _fit(_run_table(dict(gamma=0.12), seeds), "cap")[0][0]
    b0 = _fit(_run_table(dict(gamma=0.12, eta=0.0), seeds), "cap")[0][0]
    assert abs(b0) < abs(b1), f"E->capture slope not smaller at eta=0: {b0:+.3f} vs {b1:+.3f}"
    ratio = abs(b0) / max(abs(b1), 1e-12)
    INFO["test_D3_eta0_ablation"] = (f"E->capture slope eta=0.15: {b1:+.3f}, eta=0: {b0:+.3f}, ratio {ratio:.2f} "
                                     + ("(collapsed)" if ratio < 0.5 else "(NOT collapsed: report as a finding)"))


# ---------------------------------------------------------------------------- D4
def test_D4_backlash_resets_discrete():
    """[D4] Backlash B resets to exactly 0 only on collapse ticks; otherwise it never falls faster than
    decay + 6 shock sd; B just before a collapse is far above its typical level; collapses are rare events."""
    pre, non, n_coll, n_cty_ticks = [], [], 0, 0
    for sd in (9500, 9501, 9502, 9503):
        p = P(gamma=0.16)
        w = M0World(p, sd)
        d = np.array([p.b_decay * (1.0 + c.institutional_resilience) for c in w.countries])
        prev = np.zeros(w.C)
        for _ in range(p.n_ticks):
            w.step()
            tk = w.tick
            col, Bp, Bq = tk["c_collapse"], tk["c_B_post"], tk["c_B_pre"]
            assert np.all(Bp[col] == 0.0), "backlash not reset to 0 on collapse"
            assert np.all(Bp[~col] >= (1.0 - d)[~col] * prev[~col] - 6 * p.shock_sd - 1e-12), \
                "non-collapse drop faster than decay"
            pre += list(Bq[col])
            non += list(Bp[~col])
            n_coll += int(col.sum())
            n_cty_ticks += w.C
            prev = Bp
    assert n_coll > 0, "no collapses: test not informative"
    assert np.mean(pre) > 1.5 * np.mean(non), f"pre-collapse B {np.mean(pre):.3f} not well above typical {np.mean(non):.3f}"
    assert n_coll / n_cty_ticks < 0.25, "collapses not rare/discrete"
    INFO["test_D4_backlash_resets_discrete"] = (f"{n_coll} collapses; B before collapse {np.mean(pre):.2f} vs typical "
                                                f"{np.mean(non):.2f}; rate {n_coll / n_cty_ticks:.3f}/country-tick")


def test_D4_marginal_extraction_declines():
    """[D4] One oligarch alone in one pool, tick-1 extraction vs position x: equals G(1-exp(-q x)) exactly,
    positive, strictly concave (marginal extraction falls with total position)."""
    p = P()
    w0 = M0World(p, 9510)
    c, i = np.unravel_index(int(np.argmax(w0.endow)), w0.endow.shape)
    xs, ext = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]), []
    for x in xs:
        w = M0World(p, 9510)
        for o in w.oligarchs:
            o.positions[:] = 0.0
            o.capture[:] = 0.0
        w.oligarchs[0].positions[c, i] = x
        w.step()
        ext.append(w.tick["extraction"][0])
    ext = np.array(ext)
    G = w0.endow[c, i] * p.yield_scale * p.price_bar[i]
    assert np.allclose(ext, G * (1.0 - np.exp(-p.q * xs)), rtol=1e-9, atol=1e-12), "extraction != G(1-exp(-qX))"
    mr = np.diff(ext) / np.diff(xs)
    assert (mr > 0).all() and (np.diff(mr) < 0).all(), f"marginal extraction not strictly decreasing: {np.round(mr, 4)}"
    INFO["test_D4_marginal_extraction_declines"] = "marginal extraction " + " > ".join(f"{m:.3f}" for m in mr)


def _home_collapse_losses(p, seeds):
    """Relative (seizure + resident liquid hit) / start-of-tick active capital for ACTIVE residents of a country
    on the tick it collapses, split by type."""
    rel = {True: [], False: []}
    for sd in seeds:
        w = M0World(p, sd)
        prev = np.array([o.initial_active for o in w.oligarchs])
        for _ in range(p.n_ticks):
            w.step()
            tk = w.tick
            for c in np.flatnonzero(tk["c_collapse"]):
                for o in np.flatnonzero((w.home == c) & tk["active"]):
                    if prev[o] > 1e-9:
                        rel[bool(w.ruling[o])].append((tk["loss_seized"][o] + tk["loss_home_hit"][o]) / prev[o])
            prev = tk["active_capital"].copy()
    return np.array(rel[True]), np.array(rel[False])


def test_D4_ruling_lose_more_at_home():
    """[D4] After a collapse, resident ruling oligarchs lose more (relative) than resident civil ones (ruling_home_mult
    on seizure). Type-blind gap reported for contrast."""
    seeds = range(9520, 9550)
    r, c = _home_collapse_losses(P(gamma=0.16), seeds)
    assert len(r) >= 100 and len(c) >= 100, f"too few events ({len(r)} ruling, {len(c)} civil)"
    assert r.mean() > c.mean(), f"ruling {r.mean():.3f} <= civil {c.mean():.3f}"
    rb, cb = _home_collapse_losses(P(gamma=0.16, ruling_home_mult=1.0), seeds)
    INFO["test_D4_ruling_lose_more_at_home"] = (f"rel. loss ruling {r.mean():.3f} (n={len(r)}) vs civil {c.mean():.3f} "
                                                f"(n={len(c)}); type-blind {rb.mean():.3f} vs {cb.mean():.3f}")


def test_D4_goodwill_nonzero_and_unsaturated():
    """[D4] Every tick: mean |G| and mean |S| nonzero, entries not pinned at +-1, G not degenerate."""
    lo = dict(G=9.0, S=9.0, sat=0.0)
    for sd in (9530, 9531, 9532):
        p = P(gamma=0.12)
        w = M0World(p, sd)
        off = ~np.eye(w.O, dtype=bool)
        for _ in range(p.n_ticks):
            w.step()
            g, S = w.relations.G[off], w.relations.S
            mg, ms = float(np.abs(g).mean()), float(np.abs(S).mean())
            sat = float((np.abs(g) >= 1 - 1e-12).mean())
            lo["G"], lo["S"], lo["sat"] = min(lo["G"], mg), min(lo["S"], ms), max(lo["sat"], sat)
            assert mg > 1e-3 and ms > 1e-3, f"goodwill collapsed to ~0 (|G|={mg:.4f}, |S|={ms:.4f})"
            assert sat < 0.5, f"G saturated ({sat:.2f} of pairs at +-1)"
            assert g.std() > 1e-3, "G degenerate (no spread)"
    INFO["test_D4_goodwill_nonzero_and_unsaturated"] = (f"min mean|G| {lo['G']:.3f}, min mean|S| {lo['S']:.3f}, "
                                                        f"max saturated share {lo['sat']:.3f}")


# ---------------------------------------------------------------------------- D5
def _run_checked(p, seed):
    w = M0World(p, seed)
    for _ in range(p.n_ticks):
        w.step()
        _check_state(w, p)
    return w


def _traj(p, seed):
    """Trajectory arrays (no RunLog) for exact-equality comparisons."""
    w = M0World(p, seed)
    out = {k: [] for k in ("wealth", "E_foreign", "extraction", "c_B_post")}
    for _ in range(p.n_ticks):
        w.step()
        for k in out:
            out[k].append(np.array(w.tick[k], dtype=float))
    return {k: np.array(v) for k, v in out.items()}


def test_D5_capture_rent_zero():
    p = P(capture_rent=0.0, n_ticks=40)
    for sd in (9600, 9601):
        w = M0World(p, sd)
        for _ in range(p.n_ticks):
            w.step()
            _check_state(w, p)
            assert np.all(w.tick["rent"] == 0.0), "rent nonzero with capture_rent = 0"


def test_D5_foreign_exposed_false():
    """[D5] foreign_exposed=False: seizure only ever hits the oligarch's OWN home-country collapse."""
    n_home, n_foreign_exposed = 0, 0
    for sd in (9610, 9611, 9612):
        for fe in (False, True):
            p = P(gamma=0.20, foreign_exposed=fe, n_ticks=60)
            w = M0World(p, sd)
            for _ in range(p.n_ticks):
                w.step()
                _check_state(w, p)
                tk = w.tick
                seized = tk["loss_seized"] > 0
                own = tk["c_collapse"][w.home]
                if not fe:
                    assert not np.any(seized & ~own), "foreign positions seized with foreign_exposed=False"
                    n_home += int((seized & own).sum())
                else:
                    n_foreign_exposed += int((seized & ~own).sum())
    assert n_foreign_exposed > 0, "foreign_exposed=True never seized foreign positions: test not informative"
    INFO["test_D5_foreign_exposed_false"] = f"home-only seizures {n_home}; foreign seizures when exposed {n_foreign_exposed}"


def test_D5_extreme_gamma():
    cc = {}
    for g in (0.0, 1.0, 5.0):
        p = P(gamma=g, n_ticks=60)
        w = _run_checked(p, 9620)
        cc[g] = sum(c.collapse_count for c in w.countries) / w.C
    assert cc[1.0] > cc[0.0], f"collapses not higher at gamma=1 ({cc})"
    INFO["test_D5_extreme_gamma"] = "collapses/country at gamma 0/1/5: " + " / ".join(f"{cc[g]:.2f}" for g in (0.0, 1.0, 5.0))


def test_D5_short_runs():
    for n in (1, 2, 5, 12):
        p = P(n_ticks=n)
        w = _run_checked(p, 9630)
        if n < p.exit_window:
            assert int(w.exited_array().sum()) == 0, f"exit before exit_window in a {n}-tick run"
        rl = M0World(p, 9630).run()
        assert rl.T == n, f"RunLog.T={rl.T} for n_ticks={n}"


def test_D5_all_civil_all_ruling():
    for frac, n in ((0.0, 0), (1.0, 25)):
        w = M0World(P(ruling_frac=frac), 9640)
        assert int(w.ruling.sum()) == n, f"ruling count {int(w.ruling.sum())} for ruling_frac={frac}"
        _run_checked(P(ruling_frac=frac, gamma=0.12, n_ticks=60), 9640)
    # all-civil: ruling_home_mult has nothing to act on => trajectories identical
    a, b = _traj(P(ruling_frac=0.0, ruling_home_mult=1.5, n_ticks=40), 9641), _traj(P(ruling_frac=0.0, ruling_home_mult=1.0, n_ticks=40), 9641)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"all-civil: {k} depends on ruling_home_mult"
    # all-ruling: civil_cost_mult has nothing to act on => identical
    a, b = _traj(P(ruling_frac=1.0, civil_cost_mult=1.0, n_ticks=40), 9642), _traj(P(ruling_frac=1.0, civil_cost_mult=1.3, n_ticks=40), 9642)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"all-ruling: {k} depends on civil_cost_mult"
    INFO["test_D5_all_civil_all_ruling"] = "ruling_home_mult inert when all civil; civil_cost_mult inert when all ruling"


# ---------------------------------------------------------------------------- flow-based static-price check (5.4)
def _instrument_deploy(w):
    """Wrap each oligarch's deploy() to record NEW capital placed (C,NI) this tick. Returns the dict the wrappers fill."""
    cur = {}
    for o in w.oligarchs:
        def wrap(net, p, o=o, orig=o.deploy):
            before = o.positions.copy()
            orig(net, p)
            cur[o.idx] = o.positions - before
        o.deploy = wrap
    return cur


def flow_price_check(gammas=(0.08, 0.12, 0.16), seeds=range(9700, 9708), verbose=True):
    """Where does NEW capital go? Post-burn-in (ticks >= 20%), both overlap signs. Metrics:
      top_change : share of oligarch-ticks whose single largest new-capital target (country, industry) differs
                   from that oligarch's previous-tick target (only ticks where it deployed on both)
      tv         : mean total-variation distance between consecutive ticks' aggregate new-capital allocation
      mr_cv      : median over pools of the time-CV of marginal extraction return q*G*exp(-qX)
    PROPOSED decision rule (fixed before running; confirm or change): FROZEN if top_change < 0.05 AND tv < 0.05;
    DYNAMIC if top_change >= 0.10; otherwise AMBIGUOUS. FROZEN => add a deterministic supply-responsive price rule
    (fixed, unswept elasticity); DYNAMIC => keep static prices.
    Usage: python -c "import test_batchD as t; t.flow_price_check()"      (~10-20 s)"""
    ch = pairs = 0
    tvs, cvs = [], []
    for g in gammas:
        for s in (-1.0, 1.0):
            for sd in seeds:
                p = P(gamma=g, overlap_sign=s)
                w = M0World(p, sd)
                cur = _instrument_deploy(w)
                t0, prev_f, prev_top, M = int(0.2 * p.n_ticks), None, None, []
                for t in range(p.n_ticks):
                    cur.clear()
                    w.step()
                    D = np.zeros((w.O, w.C, NI))
                    for k, v in cur.items():
                        D[k] = v
                    if t < t0:
                        continue
                    tot = float(D.sum())
                    f = D.sum(0).ravel() / tot if tot > 1e-12 else None
                    if f is not None and prev_f is not None:
                        tvs.append(0.5 * float(np.abs(f - prev_f).sum()))
                    prev_f = f
                    flat = D.reshape(w.O, -1)
                    top = np.where(flat.sum(1) > 1e-12, flat.argmax(1), -1)
                    if prev_top is not None:
                        both = (top >= 0) & (prev_top >= 0)
                        ch += int((top[both] != prev_top[both]).sum())
                        pairs += int(both.sum())
                    prev_top = top
                    X = w.stack_positions().sum(0)
                    H = np.stack([c.resource_health for c in w.countries])
                    Gp = w.endow * p.yield_scale * np.array(p.price_bar)[None, :] * H
                    M.append(p.q * Gp * np.exp(-p.q * X))
                M = np.array(M).reshape(len(M), -1)
                mu = M.mean(0)
                okp = mu > 1e-9
                if okp.any():
                    cvs.append(float(np.median(M[:, okp].std(0) / mu[okp])))
    res = dict(top_change=ch / max(pairs, 1), tv=float(np.mean(tvs)) if tvs else float("nan"),
               mr_cv=float(np.median(cvs)) if cvs else float("nan"), n_pairs=pairs)
    res["verdict"] = ("FROZEN" if res["top_change"] < 0.05 and res["tv"] < 0.05
                      else "DYNAMIC" if res["top_change"] >= 0.10 else "AMBIGUOUS")
    if verbose:
        print("flow price check:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    return res


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