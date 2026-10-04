"""
m0_model.py -- M0: minimal executable core of the transnational oligarchic
capture model. Built ON the agents defined in model_core.py:

    Country   (the state)    -- produces, accumulates backlash, collapses
    Oligarch                 -- earns, flees/divests, deploys, absorbs seizure
    Relations                -- symmetric goodwill: oligarch<->oligarch (G)
                                and state<->oligarch (S)

M0World(World) only ORCHESTRATES the flows between agents each tick. All
state lives on the agents; all agent behavior lives in model_core.py.

TICK ORDER
----------
 1 prices (AR(1) shocks) -> Country.produce
 2 extraction with congestion, Tullock split                    [Gordon54, Schaefer54, Clark90, Tullock80]
 3 capture contest (persistent stock) + capture rent            [Tullock80, HR89, HJK00]
 4 Oligarch.settle (income, costs, offshore return, insolvency)
 5 Country.update_backlash (capture load weighted by foreignness and state goodwill)
 6 Country.check_collapse -> Oligarch.suffer_collapse (seizure, goodwill-dampened)
 6b goodwill updates (G symmetric; one S per state-oligarch relationship)
 7 noisy perception -> Oligarch.flee_and_divest
 8 Oligarch.net_attractiveness -> Oligarch.deploy (one logit over ALL country-industry pairs)
 9 record (Oligarch.record_wealth drives the exit rule)

There is no strategy dial. Differences between oligarchs come from their
preferences, risk tolerance, type, home country and RELATIONSHIPS; outcomes
(who stays an oligarch) are emergent.

STOCHASTICITY: AR(1) prices, backlash shocks, noisy risk perception,
probabilistic collapse, goodwill shocks (symmetric for G). Fixed-shape draws
from separate streams => common random numbers [Law15].

SOURCES: see model_core.py header and m0_notes.md.

USAGE
-----
    python m0_model.py --baseline   # diagnostics + goodwill + survival profile
    python m0_model.py --demo       # baseline + gamma sweep + overlap-sign contrast
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict

import numpy as np

from model_core import (INDUSTRIES, NI, Country, Oligarch, OligarchType, Params,  # noqa: F401
                        Relations, World, gini)


class M0World(World):
    def __init__(self, p: Params, seed: int):
        super().__init__(p, seed)
        self.u = np.zeros(NI)                       # log price deviation (environment state)
        self.t = 0
        self.hist: Dict[str, list] = {k: [] for k in
            ["W", "B", "collapse", "load", "foreign_share", "offshore_share",
             "S_home", "S_foreign", "G_mean"]}
        # H-b outcome accumulators (per oligarch, cumulative over the run):
        self.cum_foreign_capture = np.zeros(self.O)   # "profit achieved abroad"
        self.cum_foreign_havoc = np.zeros(self.O)     # "backlash caused abroad"
        self.cum_domestic_capture = np.zeros(self.O)  # for capacity-normalized (share) comparisons
        self.cum_domestic_havoc = np.zeros(self.O)
        self.cum_foreign_position = np.zeros(self.O)  # TIME-INTEGRATED foreign capital (fixes rate-measure
                                                        # artifact: final-tick position is a biased denominator
                                                        # when positions shrink differentially by type/run)

    @property
    def W0(self) -> np.ndarray:
        return np.array([o.initial_wealth for o in self.oligarchs])

    # ------------------------------------------------------------------
    def _update_goodwill(self, A, cap, outshare, eps_oo, eps_os):
        """Own update equations (see m0_notes.md), trimmed to 5 shared parameters
        (gw_decay, lam_oo, lam_inv, lam_cap, sigma_gw). G stays symmetric, zero
        diagonal. halo (vouching) DROPPED -- not load-bearing for H-b/H-c."""
        p, R = self.p, self.relations
        total = A.sum((1, 2), keepdims=True)
        f = np.divide(A, total, out=np.zeros_like(A), where=total > 0)     # position shares
        Fc = f.sum(2)
        copres = Fc @ Fc.T
        flat = f.reshape(self.O, -1)
        ovl = flat @ flat.T                                                 # same (country, industry) overlap
        joint = copres - ovl                                                # same country, different industry
        dG = (p.gw_decay * (R.G0 - R.G) + p.lam_oo * joint
              + p.overlap_sign * p.lam_oo * ovl + p.sigma_gw * eps_oo)
        R.G = np.clip(R.G + dG, -1.0, 1.0)
        np.fill_diagonal(R.G, 0.0)

        Pn = np.tanh(A.sum(2) / p.a_ref)
        capshare = (cap * outshare[None]).sum(2)
        dS = (p.gw_decay * (R.S0 - R.S) + p.lam_inv * Pn - p.lam_cap * capshare
              + p.sigma_gw * eps_os)
        R.S = np.clip(R.S + dS, -1.0, 1.0)

    # ------------------------------------------------------------------
    def step(self):
        p, R, Os, Cs = self.p, self.relations, self.oligarchs, self.countries
        eps_price, eps_b, eps_perc, u_coll, eps_oo, eps_os = self.noise.draw()

        # 1. prices and output pools
        self.u = p.price_ar * self.u + p.price_sd * eps_price
        price = np.array(p.price_bar) * np.exp(self.u)
        G = np.stack([c.produce(price, p) for c in Cs])

        # 2. extraction: pool = G*(1-exp(-qX)), Tullock split
        A = self.stack_positions()
        Xtot = A.sum(0)
        pool = G * (1.0 - np.exp(-p.q * Xtot))
        w = A ** p.tullock_r
        share = w / np.maximum(w.sum(0), 1e-12)[None]
        extraction = (share * pool[None]).sum((1, 2))

        # 3. capture contest (persistent stock) and capture rent
        target = self.capture_terms(A)[3]
        cap = self.stack_capture()
        cap = cap + p.capture_adj * (target - cap)
        self.write_capture(cap)
        rent = p.capture_rent * (cap * G[None]).sum((1, 2))

        # 3b. FOREIGN vs DOMESTIC decomposition for H-b, read-only -- does not
        # feed back into any dynamics, purely diagnostic tracking. Domestic is
        # tracked alongside foreign so we can compare SHARES (capacity-adjusted),
        # not just absolute amounts, when checking for a type effect.
        foreign_mask = (~self.home_mask)[:, :, None]                        # (O,C,1)
        home_mask3 = self.home_mask[:, :, None]
        foreign_extraction = (share * pool[None] * foreign_mask).sum((1, 2))
        foreign_rent = p.capture_rent * (cap * G[None] * foreign_mask).sum((1, 2))
        foreign_capture_gained = foreign_extraction + foreign_rent           # "profit from abroad"
        domestic_extraction = (share * pool[None] * home_mask3).sum((1, 2))
        domestic_rent = p.capture_rent * (cap * G[None] * home_mask3).sum((1, 2))
        domestic_capture_gained = domestic_extraction + domestic_rent

        # 4. income, costs, insolvency (agent behavior)
        for o in Os:
            o.settle(extraction[o.idx], rent[o.idx], p)

        # 5. backlash: capture load weighted by foreignness, (1 - zeta_bl * state goodwill), and
        # ruling_home_mult AT HOME for ruling oligarchs (reused param -- direct state control is
        # more visible/attributable, so it generates disproportionate domestic backlash pressure).
        outshare = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        home_vuln = np.where(self.home_mask & self.ruling[:, None], p.ruling_home_mult, 1.0)
        wt = home_vuln * (1.0 + p.foreign_backlash_mult * (~self.home_mask)) * np.clip(1.0 - p.zeta_bl * R.S, 0.1, None)
        per_olig_load = (outshare[None] * cap * wt[:, :, None]).sum(2)       # (O,C): each oligarch's push on each country's load
        load = per_olig_load.sum(0)
        cap_by_industry = cap.sum(0)
        for c in Cs:
            c.update_backlash(load[c.idx], eps_b[c.idx], p)
            c.industry_capture = cap_by_industry[c.idx].copy()
            c.domestic_capture = float((outshare[c.idx] * c.industry_capture).sum())

        # 5b. "HAVOC ABROAD" for H-b: each oligarch's contribution to FOREIGN
        # countries' backlash load, i.e. destabilization caused specifically by
        # foreign activity. Read-only diagnostic; does not alter c.update_backlash.
        foreign_havoc = (per_olig_load * (~self.home_mask)).sum(1)          # (O,)
        domestic_havoc = (per_olig_load * self.home_mask).sum(1)

        # 6. collapse and seizure
        collapse = np.zeros(self.C, dtype=bool)
        for c in Cs:
            if c.check_collapse(u_coll[c.idx], p):
                collapse[c.idx] = True
                for o in Os:
                    o.suffer_collapse(c.idx, R.S[o.idx, c.idx], p)

        # 6b. goodwill updates
        self._update_goodwill(self.stack_positions(), self.stack_capture(), outshare, eps_oo, eps_os)

        # 7. noisy perception, flight, divestment
        Bp = np.array([c.backlash_pressure for c in Cs])[None, :] + p.perc_sd * eps_perc
        ratio = Bp / self.theta[None, :]
        hz = np.stack([Cs[c].hazard(Bp[:, c], p) for c in range(self.C)], axis=1)     # (O,C)
        for o in Os:
            o.flee_and_divest(ratio[o.idx], p)

        # 8. deployment: relationship-aware, risk-adjusted, logit over all (country, industry)
        A = self.stack_positions()
        Xtot = A.sum(0)
        mult, partner, _, _, Xeff = self.capture_terms(A)
        m_ext = p.q * G * np.exp(-p.q * Xtot)
        m_cap_shared = p.capture_rent * G * p.kmax * p.kappa * np.exp(-p.kappa * Xeff)
        for o in Os:
            net = o.net_attractiveness(m_ext, m_cap_shared, mult[o.idx], hz[o.idx], partner[o.idx], p)
            o.deploy(net, p)

        # 9. record
        self.t += 1
        Wl = np.array([o.record_wealth() for o in Os])
        pos = np.array([float(o.positions.sum()) for o in Os])
        for_pos = np.array([float(o.positions.sum() - o.positions[o.home_country].sum()) for o in Os])
        off = ~np.eye(self.O, dtype=bool)
        h = self.hist
        h["W"].append(Wl)
        h["B"].append(np.array([c.backlash_pressure for c in Cs]))
        h["collapse"].append(collapse)
        h["load"].append(load)
        h["foreign_share"].append(np.divide(for_pos, pos, out=np.zeros_like(pos), where=pos > 0))
        h["offshore_share"].append(np.array([o.offshore_capital for o in Os]) / np.maximum(Wl, 1e-9))
        h["S_home"].append(R.S[self.home_mask].mean())
        h["S_foreign"].append(R.S[~self.home_mask].mean())
        h["G_mean"].append(R.G[off].mean())
        self.cum_foreign_capture += foreign_capture_gained
        self.cum_foreign_havoc += foreign_havoc
        self.cum_domestic_capture += domestic_capture_gained
        self.cum_domestic_havoc += domestic_havoc
        self.cum_foreign_position += for_pos

    def run(self):
        for _ in range(self.p.n_ticks):
            self.step()
        return {k: np.array(v) for k, v in self.hist.items()}

    def exited_array(self) -> np.ndarray:
        return np.array([o.exited(self.p) for o in self.oligarchs])

    def profile(self) -> Dict[str, np.ndarray]:
        """End-of-run relationship/footprint features per oligarch (descriptive)."""
        R, O = self.relations, self.O
        A = self.stack_positions()
        pos = A.sum((1, 2))
        for_pos = (A * (~self.home_mask)[:, :, None]).sum((1, 2))
        foreign_share = np.divide(for_pos, pos, out=np.zeros(O), where=pos > 0)
        Wc = A.sum(2) * (~self.home_mask)
        S_for = np.divide((R.S * Wc).sum(1), Wc.sum(1), out=np.zeros(O), where=Wc.sum(1) > 0)
        total = pos[:, None, None]
        f = np.divide(A, total, out=np.zeros_like(A), where=total > 0).reshape(O, -1)
        ovl = f @ f.T
        gpos = np.clip(R.G, 0.0, None)
        partner_overlap = np.divide((gpos * ovl).sum(1), gpos.sum(1), out=np.zeros(O), where=gpos.sum(1) > 0)

        # FOREIGN-RESTRICTED network embeddedness (fixes the earlier diagnostic,
        # which averaged over partners in unrelated countries): only count a
        # partner's goodwill where that partner is actually PRESENT in the same
        # foreign country as this oligarch.
        presence = np.tanh(A.sum(2) / self.p.a_ref)                          # (O,C)
        foreign_presence = presence * (~self.home_mask)
        shared_foreign_presence = foreign_presence @ foreign_presence.T      # (O,O): co-presence weight
        np.fill_diagonal(shared_foreign_presence, 0.0)
        w = shared_foreign_presence
        foreign_network_embeddedness = np.divide((R.G * w).sum(1), w.sum(1),
                                                   out=np.zeros(O), where=w.sum(1) > 1e-9)

        # CAPACITY-NORMALIZED versions (share of total gains/havoc that is foreign),
        # controlling for the wealth-spillover effect the verification check found:
        # ruling oligarchs have more capital overall (from ruling_home_mult's DOMESTIC
        # boost), so their absolute foreign numbers are larger even if the underlying
        # domestic-vs-foreign ALLOCATION logic is identical across types.
        tot_cap = self.cum_foreign_capture + self.cum_domestic_capture
        tot_hav = self.cum_foreign_havoc + self.cum_domestic_havoc
        foreign_capture_share = np.divide(self.cum_foreign_capture, tot_cap,
                                           out=np.zeros(O), where=tot_cap > 1e-9)
        foreign_havoc_share = np.divide(self.cum_foreign_havoc, tot_hav,
                                         out=np.zeros(O), where=tot_hav > 1e-9)
        # FOREIGN-ONLY rate (excludes domestic from the denominator entirely, unlike
        # the share measures above, which stay contaminated by ruling_home_mult's
        # DOMESTIC inflation diluting/distorting the ratio). This is the cleanest
        # type-neutral check: per unit of foreign capital deployed, do civil and
        # ruling oligarchs achieve the same foreign capture/havoc? Uses the TIME-
        # INTEGRATED AVERAGE foreign position (cum_foreign_position / ticks elapsed),
        # not the final-tick snapshot -- the final-tick version is a biased
        # denominator when positions shrink differentially across the run (e.g. from
        # civil_cost_mult-driven liquidation), which artificially inflates the
        # weaker type's apparent "rate." See m0_notes.md for the diagnosis.
        avg_foreign_capital = self.cum_foreign_position / max(self.t, 1)
        foreign_capture_rate = np.divide(self.cum_foreign_capture, avg_foreign_capital,
                                          out=np.zeros(O), where=avg_foreign_capital > 1e-9)
        foreign_havoc_rate = np.divide(self.cum_foreign_havoc, avg_foreign_capital,
                                        out=np.zeros(O), where=avg_foreign_capital > 1e-9)

        return dict(foreign_share=foreign_share, S_foreign=S_for,
                    S_home=R.S[np.arange(O), self.home],
                    mean_partner_gw=R.G.sum(1) / (O - 1),
                    partner_industry_overlap=partner_overlap,
                    foreign_network_embeddedness=foreign_network_embeddedness,
                    foreign_capture_gained=self.cum_foreign_capture.copy(),   # H-b outcome 1 (absolute): "profit abroad"
                    foreign_havoc=self.cum_foreign_havoc.copy(),              # H-b outcome 2 (absolute): "destabilization abroad"
                    foreign_capture_share=foreign_capture_share,              # H-b outcome 1 (share of total)
                    foreign_havoc_share=foreign_havoc_share,                  # H-b outcome 2 (share of total)
                    foreign_capture_rate=foreign_capture_rate,                # H-b outcome 1 (foreign-only rate; cleanest)
                    foreign_havoc_rate=foreign_havoc_rate,                    # H-b outcome 2 (foreign-only rate; cleanest)
                    oligarch_type=np.array([o.oligarch_type for o in self.oligarchs]))


def type_verification_check(p: Params, seeds=range(60)):
    """VERIFICATION, not a hypothesis test: civil and ruling oligarchs should NOT
    differ meaningfully in foreign capture/havoc or exit rate, since
    ruling_home_mult only advantages ruling types in DOMESTIC capture (see
    model_core.py's RESEARCH FRAMING note). A large, significant gap here would
    indicate a bug or an unintended channel, not a finding about H-b/H-c."""
    from scipy import stats
    rows = {"foreign_capture_gained": [], "foreign_havoc": [],
            "foreign_capture_share": [], "foreign_havoc_share": [],
            "foreign_capture_rate": [], "foreign_havoc_rate": [], "exited": []}
    types = []
    for s in seeds:
        w = M0World(p, s)
        h = w.run()
        prof = w.profile()
        for k in rows:
            rows[k].append(prof[k] if k != "exited" else w.exited_array().astype(float))
        types.append(prof["oligarch_type"])
    types = np.concatenate(types)
    civil = types == OligarchType.CIVIL
    ruling = types == OligarchType.RULING
    print("civil-vs-ruling VERIFICATION (expect NO significant difference; see note above)")
    for k, lst in rows.items():
        v = np.concatenate(lst)
        t, pval = stats.ttest_ind(v[civil], v[ruling], equal_var=False)
        print(f"  {k:>22}: civil={v[civil].mean():8.3f}  ruling={v[ruling].mean():8.3f}"
              f"   Welch t={t:+.2f}  p={pval:.3f}")


# --------------------------------------------------------------------------
# Diagnostics / experiments (no strategy dial: outcomes are emergent)
# --------------------------------------------------------------------------
def baseline_diagnostics(p: Params, seeds=range(30)):
    rows, prof_all, ex_all = [], {}, []
    for s in seeds:
        w = M0World(p, s)
        h = w.run()
        W = h["W"]
        ex = w.exited_array()
        ex_all.append(ex)
        for k, v in w.profile().items():
            prof_all.setdefault(k, []).append(v)
        rows.append(dict(
            growth=np.mean(W[-1] / w.W0), gini=gini(W[-1]),
            collapses_per_country=h["collapse"].sum() / p.n_countries,
            exit_rate=ex.mean(),
            offshore=float(h["offshore_share"][-1].mean()),
            foreign=float(h["foreign_share"][-1].mean()),
            S_home_end=float(h["S_home"][-1]), S_foreign_end=float(h["S_foreign"][-1]),
            G_end=float(h["G_mean"][-1]), maxB=float(h["B"].max())))
    print("baseline (mean over seeds):")
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"  {k:>22}: {v.mean():8.3f}   (sd {v.std():.3f})")
    ex = np.concatenate(ex_all)
    print(f"\nsurvival profile (descriptive; features are endogenous). exited={ex.sum()} survivors={(~ex).sum()}")
    print(f"  {'feature':>26}  {'survivors':>10}  {'exited':>10}")
    for k, lst in prof_all.items():
        if k == "oligarch_type":   # non-numeric; see type_verification_check for the type comparison
            continue
        v = np.concatenate(lst)
        a = v[~ex].mean() if (~ex).any() else float('nan')
        b = v[ex].mean() if ex.any() else float('nan')
        print(f"  {k:>26}  {a:10.3f}  {b:10.3f}")
    return rows


def gamma_sweep(p: Params, gammas=(0.04, 0.06, 0.08, 0.10, 0.12, 0.16), seeds=range(30)):
    """PRIMARY SWEEP: backlash sensitivity vs collapse frequency, exit rate,
    wealth concentration and average state goodwill."""
    print("gamma sweep")
    print("  gamma  collapses/country  exit_rate  gini   S_foreign  S_home")
    for g in gammas:
        pg = replace(p, gamma=g)
        r = []
        for s in seeds:
            w = M0World(pg, s)
            h = w.run()
            r.append([h["collapse"].sum() / p.n_countries, w.exited_array().mean(), gini(h["W"][-1]),
                      h["S_foreign"][-1], h["S_home"][-1]])
        m = np.mean(r, 0)
        print(f"  {g:5.2f}  {m[0]:16.2f}  {m[1]:9.3f}  {m[2]:.3f}  {m[3]:9.3f}  {m[4]:6.3f}")


def overlap_sign_contrast(p: Params, seeds=range(30)):
    """Open question: does same-industry overlap breed rivalry or affinity?"""
    print("overlap_sign contrast (-1 rivalry vs +1 affinity)")
    for sgn in (-1.0, 1.0):
        ps = replace(p, overlap_sign=sgn)
        ex, po, W = [], [], []
        for s in seeds:
            w = M0World(ps, s)
            h = w.run()
            ex.append(w.exited_array().mean())
            po.append(w.profile()["partner_industry_overlap"].mean())
            W.append(gini(h["W"][-1]))
        print(f"  sign={sgn:+.0f}: exit_rate={np.mean(ex):.3f}  partner_industry_overlap={np.mean(po):.4f}  gini={np.mean(W):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    P = Params()
    if args.baseline or args.demo:
        baseline_diagnostics(P)
    if args.demo:
        print()
        gamma_sweep(P)
        print()
        overlap_sign_contrast(P)
        print()
        type_verification_check(P)