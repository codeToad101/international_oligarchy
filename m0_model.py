"""
m0_model.py -- M0: minimal executable core of the transnational oligarchic
capture model. Built ON the agents defined in model_core.py:

    Country   (the state)    -- produces, depletes, accumulates backlash, collapses
    Oligarch                 -- earns, flees/divests, deploys, absorbs seizure, may exit
    Relations                -- symmetric goodwill: oligarch<->oligarch (G)
                                and state<->oligarch (S)

M0World(World) only ORCHESTRATES the flows between agents each tick. All
state lives on the agents; all agent behavior lives in model_core.py.
Per-tick logging lives in runlog.py; calibration/diagnostics in pilot.py;
the experimental design runner in run_design.py. The spec (slides = ground
truth) is project_overview.txt.

TICK ORDER  (slide 4 -> code mapping is in project_overview.txt 4.6)
----------
 1 prices (STATIC unless p.price_dynamics) -> Country.produce
 2 extraction with congestion, Tullock split              [Gordon54, Schaefer54, Clark90, Tullock80]
 2b resource depletion / abuse, then scarcity; pool havoc value
 3 capture contest (persistent stock) + capture rent      [Tullock80, HR89, HJK00]
 4 Oligarch.settle (income, costs, offshore return, insolvency)   [ACTIVE agents only]
 5 Country.update_backlash (capture load weighted by foreignness and state
   goodwill, PLUS scarcity)
 5b havoc = Tullock effort share x pool abuse damage (abuse x prior health x V); foreign/domestic
 6 Country.check_collapse -> Oligarch.suffer_collapse (seizure, goodwill-dampened)   [ACTIVE only]
 6b goodwill updates (G symmetric; one S per state-oligarch relationship)
 7 noisy perception -> Oligarch.flee_and_divest                    [ACTIVE only]
 8 Oligarch.net_attractiveness -> Oligarch.deploy (one logit over ALL country-industry pairs) [ACTIVE only]
 9 record (wealth, embeddedness, active capital, S, country/world fields -> self.tick),
   THEN exit check (floor-and-window on ACTIVE capital; exited agents leave the stage)

There is no strategy dial. Differences between oligarchs come from their
preferences, divestment propensity, type, home country and RELATIONSHIPS; outcomes
(who stays an oligarch) are emergent.

STOCHASTICITY: AR(1) prices (only if price_dynamics), backlash shocks, noisy
risk perception, probabilistic collapse, goodwill shocks (symmetric for G).
Fixed-shape draws from separate streams => common random numbers [Law15].
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from model_core import NI, Params, World
from runlog import RunLog


class M0World(World):
    def __init__(self, p: Params, seed: int):
        super().__init__(p, seed)
        self.u = np.zeros(NI)                       # log price deviation (stays 0 while prices are static)
        self.t = 0
        self.tick: Dict[str, np.ndarray] = {}       # last tick's arrays (consumed by RunLog.record)

    # ------------------------------------------------------------------
    def _update_goodwill(self, A, cap, outshare, eps_oo, eps_os):
        """Own update equations, 5 shared parameters (gw_decay, lam_oo, lam_inv,
        lam_cap, sigma_gw). G stays symmetric, zero diagonal."""
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
        for o in Os:                                                         # zero per-tick loss trackers
            o.begin_tick()
        active0 = np.array([o.active for o in Os])                           # on the stage at tick start

        # 1. prices and output pools: STATIC unless price_dynamics (stream is drawn either way)
        if p.price_dynamics:
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

        # 2b. resource depletion / abuse, then scarcity. Uses THIS tick's extraction fraction; the
        # updated health affects NEXT tick's output pool. Scarcity is set here so step 5 sees it.
        h_ext = 1.0 - np.exp(-p.q * Xtot)                                   # (C,NI) extraction fraction per pool
        for c in Cs:
            c.update_resource(h_ext[c.idx], price, p)
        # value of stock destroyed by ABUSE in each pool this tick: abuse x prior health x V
        abuse = np.stack([c.abuse_rate for c in Cs])                        # (C,NI)
        h_prev = np.stack([c.health_prev for c in Cs])                      # (C,NI)
        V_full = np.stack([c.full_value(price, p) for c in Cs])             # (C,NI)
        havoc_pool = abuse * h_prev * V_full

        # 3. capture contest (persistent stock) and capture rent
        target = self.capture_terms(A)[3]
        cap = self.stack_capture()
        cap = cap + p.capture_adj * (target - cap)
        self.write_capture(cap)
        rent = p.capture_rent * (cap * G[None]).sum((1, 2))

        # 3b. FOREIGN vs DOMESTIC capture gained this tick (extraction + rent). Exited agents hold
        # no positions/capture, so their values are 0 (mask with tick["active"]).
        foreign_mask = (~self.home_mask)[:, :, None]                        # (O,C,1)
        home_mask3 = self.home_mask[:, :, None]
        foreign_extraction = (share * pool[None] * foreign_mask).sum((1, 2))
        foreign_rent = p.capture_rent * (cap * G[None] * foreign_mask).sum((1, 2))
        foreign_capture_gained = foreign_extraction + foreign_rent
        domestic_extraction = (share * pool[None] * home_mask3).sum((1, 2))
        domestic_rent = p.capture_rent * (cap * G[None] * home_mask3).sum((1, 2))
        domestic_capture_gained = domestic_extraction + domestic_rent

        # 4. income, costs, insolvency (agent behavior) -- ACTIVE agents only
        for o in Os:
            if o.active:
                o.settle(extraction[o.idx], rent[o.idx], p)

        # 5. backlash: capture load weighted by foreignness, (1 - zeta_bl * state goodwill), and
        # ruling_home_mult AT HOME for ruling oligarchs. Country.update_backlash ALSO adds
        # w_scar * scarcity (set in 2b); gamma scales both.
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
        B_pre = np.array([c.backlash_pressure for c in Cs])

        # 5b. HAVOC (DV for P): each oligarch's Tullock effort share of the abuse damage in each pool,
        # summed over FOREIGN pools (and, as a counterpart, over HOME pools).
        foreign_havoc = (share * havoc_pool[None] * foreign_mask).sum((1, 2))
        domestic_havoc = (share * havoc_pool[None] * home_mask3).sum((1, 2))
        # diagnostic only (the OLD "havoc"): contribution to countries' backlash LOAD
        foreign_load = (per_olig_load * (~self.home_mask)).sum(1)
        domestic_load = (per_olig_load * self.home_mask).sum(1)

        # 6. collapse and seizure (ACTIVE agents only)
        collapse = np.zeros(self.C, dtype=bool)
        for c in Cs:
            if c.check_collapse(u_coll[c.idx], p):
                collapse[c.idx] = True
                for o in Os:
                    if o.active:
                        o.suffer_collapse(c.idx, R.S[o.idx, c.idx], p)

        # 6b. goodwill updates
        self._update_goodwill(self.stack_positions(), self.stack_capture(), outshare, eps_oo, eps_os)

        # 7. noisy perception, flight, divestment (ACTIVE only)
        Bp = np.array([c.backlash_pressure for c in Cs])[None, :] + p.perc_sd * eps_perc
        ratio = Bp / self.theta[None, :]
        hz = np.stack([Cs[c].hazard(Bp[:, c], p) for c in range(self.C)], axis=1)     # (O,C)
        for o in Os:
            if o.active:
                o.flee_and_divest(ratio[o.idx], p)

        # 8. deployment: relationship-aware, risk-adjusted, logit over all (country, industry) (ACTIVE only)
        A = self.stack_positions()
        Xtot = A.sum(0)
        mult, partner, _, _, Xeff = self.capture_terms(A)
        m_ext = p.q * G * np.exp(-p.q * Xtot)
        m_cap_shared = p.capture_rent * G * p.kmax * p.kappa * np.exp(-p.kappa * Xeff)
        for o in Os:
            if o.active:
                net = o.net_attractiveness(m_ext, m_cap_shared, mult[o.idx], hz[o.idx], partner[o.idx], p)
                o.deploy(net, p)

        # 9. record, THEN exits
        self.t += 1
        A_end = self.stack_positions()
        emb = self.embeddedness(A_end)                                       # from end-of-tick positions
        capital = np.array([o.capital for o in Os])
        offshore = np.array([o.offshore_capital for o in Os])
        pos = A_end.sum((1, 2))
        for_pos = (A_end * foreign_mask).sum((1, 2))
        Wl = capital + offshore + pos
        act_cap = capital + pos                                              # value tested by the exit rule
        # state goodwill summaries
        S = R.S
        fm = ~self.home_mask
        S_home = S[self.home_mask]                                           # (O,) one home country each, row-major
        S_foreign_mean = (S * fm).sum(1) / np.maximum(fm.sum(1), 1)
        Pm = np.tanh(A_end.sum(2) / p.a_ref) * fm
        den = Pm.sum(1)
        S_foreign_pw = np.divide((Pm * S).sum(1), den, out=np.full(self.O, np.nan), where=den > 1e-9)
        offd = ~np.eye(self.O, dtype=bool)
        h_mean = np.array([c.resource_health.mean() for c in Cs])
        abuse_frac = np.array([(c.abuse_rate > 0).mean() for c in Cs])

        # exit check AFTER the records above, so the exit tick's row holds real (pre-exit) values
        exited_now = np.zeros(self.O, dtype=bool)
        for o in Os:
            if o.check_exit(self.t, p):
                exited_now[o.idx] = True

        self.tick = dict(
            t=self.t, active=active0, exited_now=exited_now,
            wealth=Wl, capital=capital, offshore=offshore, pos_home=pos - for_pos, pos_foreign=for_pos,
            active_capital=act_cap, extraction=extraction, rent=rent,
            foreign_capture=foreign_capture_gained, domestic_capture=domestic_capture_gained,
            foreign_havoc=foreign_havoc, domestic_havoc=domestic_havoc,
            foreign_load=foreign_load, domestic_load=domestic_load,
            loss_seized=np.array([o.t_seized for o in Os]),
            loss_home_hit=np.array([o.t_home_hit for o in Os]),
            loss_forced_liq=np.array([o.t_forced_liq for o in Os]),
            loss_divest=np.array([o.t_divest for o in Os]),
            flight=np.array([o.t_flight for o in Os]),
            exit_liq_loss=np.array([o.exit_liq_loss if exited_now[o.idx] else 0.0 for o in Os]),
            E_foreign=emb["E_foreign"], E_all=emb["E_all"], E_pos=emb["E_pos"], E_neg=emb["E_neg"],
            S_home=S_home, S_foreign_mean=S_foreign_mean, S_foreign_pw=S_foreign_pw,
            c_B_pre=B_pre, c_B_post=np.array([c.backlash_pressure for c in Cs]), c_scarcity=np.array([c.scarcity_pressure for c in Cs]),
            c_collapse=collapse, c_load=load, c_health_mean=h_mean, c_abuse_frac=abuse_frac,
            c_recovery=np.array([c.in_recovery for c in Cs]),
            w_G_mean=float(R.G[offd].mean()), w_S_home_mean=float(S_home.mean()),
            w_S_foreign_mean=float(S[fm].mean()), w_n_active=float(sum(o.active for o in Os)),
        )

    def run(self, log: bool = True) -> Optional[RunLog]:
        rl = RunLog(self) if log else None
        for _ in range(self.p.n_ticks):
            self.step()
            if rl is not None:
                rl.record(self)
        if rl is not None:
            rl.finalize(self)
        return rl

    def exited_array(self) -> np.ndarray:
        return np.array([o.exited() for o in self.oligarchs])

    def exit_tick_array(self) -> np.ndarray:
        """exit tick per oligarch (nan if never exited)."""
        return np.array([np.nan if o.exit_tick is None else o.exit_tick for o in self.oligarchs], dtype=float)