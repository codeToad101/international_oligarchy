"""
model_core.py -- agent definitions and shared structure for the transnational
oligarchic capture model.

TWO CORE AGENTS (every model version, M0 and beyond, uses these classes):

  Country   (the STATE)     owns: endowment, institutions (resilience,
                            collapse threshold), backlash pressure, recovery
                            status, aggregate capture.
                            does: produces output, accumulates backlash,
                            collapses (probabilistically).

  Oligarch                  owns: type, home country, industry preferences,
                            risk tolerance, liquid capital, offshore capital,
                            positions and capture stock in every country.
                            does: earns, settles insolvency, flees/divests
                            under perceived risk, chooses deployment, absorbs
                            seizure when a country collapses.

RELATIONSHIPS (goodwill on EVERY relationship; symmetric for now):
  oligarch <-> oligarch   Relations.G[a,b]  (O x O, G[a,b] == G[b,a])
  state    <-> oligarch   Relations.S[o,c]  (O x C, incl. home country)
Relations is a shared object that both agents reference, so the two ends of
a relationship always see the same value (that is what "symmetric" means
here). Agents read their own relationships through goodwill_with_* methods.
Directed goodwill (A's view of B != B's view of A) is a parked extension.

M0 SIMPLIFICATIONS (documented, revisitable):
  * industry_investments and illiquid_assets are merged into ONE sunk,
    seizable stock, Oligarch.positions[c, i].
  * Capture is a dimensionless FRACTION of output (Oligarch.capture[c, i]).
  * Country output is a per-tick pool from endowment x price (no depletion).
  * DEFERRED fields kept on Country for continuity (unused in M0):
    unclaimed_assets, scarcity_pressure, wealth_returned_to_population.

RESEARCH FRAMING (as of the current project scope; see m0_notes.md and
project_overview.txt for the full discussion):
  The domestic capture contest (kmax, kappa, capture_adj, tullock_r) is
  SCAFFOLDING, not a finding: its only job is to produce a plausible
  baseline level of domestic capture so there is something for foreign
  behavior to depart from. It is held constant, never swept.

  UPDATE (civil/ruling asymmetry, own design, not from any paper): the type
  distinction is now deliberately asymmetric, built entirely by REUSING
  ruling_home_mult for three home-entanglement effects plus one new
  parameter (civil_cost_mult) for civil's offsetting cost:
    RULING, at home only: cheaper/more effective capture (existing),
      MORE backlash contribution (visible, attributable to direct state
      control), and MORE seizure exposure if their OWN home country
      collapses (entangled with the fallen regime).
    CIVIL, everywhere: pays civil_cost_mult x higher running costs on all
      captured positions (legal defense, lobbying, intermediaries), but
      gets relative protection at home on collapse BY COMPARISON to
      ruling (civil_cost_mult does not appear in the collapse-exposure
      term at all -- civil's home protection is implicit, not a separate
      dial), reflecting that popular anger deflects to intermediaries and
      the regime itself rather than to civil oligarchs directly.
  This makes civil vs. ruling a real, asymmetric, two-dimensional tradeoff
  (cost/vulnerability vs. effectiveness/exposure) rather than a null
  result. A civil/ruling comparison is still run as a VERIFICATION check,
  but the expectation changes accordingly (see type_verification_check in
  m0_model.py): foreign-only rates (capture/havoc per unit of foreign
  capital, which this asymmetry does not touch) should still show no
  meaningful difference, since all of it is wired through HOME-country
  effects; exit rate, domestic capture, and domestic backlash contribution
  ARE now expected to differ by type, and should be interpreted as the
  deliberate tradeoff above, not a bug.

SOURCES (details/verification caveats in m0_notes.md):
  Gordon (1954); Schaefer (1954); Clark (1990)  -- extraction
  Tullock (1980); Hillman & Riley (1989)        -- contest share
  Hellman, Jones & Kaufmann (2000)              -- bounded capture (motivation)
  Turchin (2003); Turchin & Nefedov (2009)      -- stress-indicator structure
  McFadden (1974); McKelvey & Palfrey (1995)    -- logit choice
  Salton & McGill (1983)                        -- cosine similarity
  Axelrod (1984)                                -- goodwill via joint activity
                                                   (motivation only)
  Law (2015)                                    -- common random numbers
  OWN ASSUMPTIONS (not from any paper): collapse hazard, capture persistence,
  capture-rent stream, backlash accumulation, flight/divest rules, seizure
  fractions, AR(1) prices, the exit rule, ALL goodwill formulas and effect
  sizes, and all numeric parameter values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

INDUSTRIES = ["agriculture", "energy", "mining", "manufacturing", "finance", "technology"]
NI = len(INDUSTRIES)


class OligarchType:
    CIVIL = "civil"      # wealth-based influence, no direct control of the state
    RULING = "ruling"    # holds/controls state office; home-capture advantage


# ---------------------------------------------------------------------------
# Parameters (all PLACEHOLDERS unless a source is noted)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # --- structure ---
    n_countries: int = 5
    oligs_per_country: int = 2
    n_ticks: int = 100
    n_specialty: int = 2
    specialty_range: Tuple[float, float] = (0.6, 1.0)
    nonspecialty_range: Tuple[float, float] = (0.0, 0.4)
    ruling_prob: float = 0.3
    capital_scale: float = 10.0
    init_home_frac: float = 0.5         # share of initial capital placed at home (=> initial domestic influence)
    a_ref: float = 2.0                  # position size at which "presence" ~ 0.76 (tanh scale)

    # --- output pools / extraction [Gordon54, Schaefer54, Clark90] ---
    price_bar: Tuple[float, ...] = (1.0, 1.6, 2.0, 1.4, 2.2, 2.5)
    yield_scale: float = 0.7
    q: float = 0.3
    op_cost: float = 0.08
    tullock_r: float = 1.0              # [Tullock80, HR89]

    # --- capture [Tullock80, HR89, HJK00] ---
    kmax: float = 0.85
    kappa: float = 0.5
    capture_adj: float = 0.3
    capture_rent: float = 0.5
    ruling_home_mult: float = 1.5       # REUSED for 3 ruling home-entanglement effects: capture
                                         # effectiveness (existing), backlash vulnerability, and
                                         # collapse exposure at home (both new; see m0_notes.md)
    civil_cost_mult: float = 1.3        # ONE new param: civil oligarchs pay this multiple on running
                                         # costs (legal defense/lobbying/intermediaries), domestic+foreign

    # --- backlash [structure only: Turchin03] ---
    gamma: float = 0.08                 # PRIMARY SWEEP PARAMETER
    b_decay: float = 0.10
    foreign_backlash_mult: float = 1.0
    shock_sd: float = 0.03

    # --- collapse / seizure (own) ---
    h_max: float = 0.35
    h_width: float = 0.08
    seize_frac: float = 0.8
    foreign_exposed: bool = True
    resident_liquid_hit: float = 0.3
    recovery_ticks: int = 5
    recovery_output: float = 0.5

    # --- prices ---
    price_ar: float = 0.8
    price_sd: float = 0.10

    # --- goodwill: initialization (own) ---
    home_bonus: float = 0.6
    s_scale: float = 2.0
    foreign_stigma: float = 0.1
    init_overlap_coef: float = 3.0
    same_country_bonus: float = 0.3
    overlap_sign: float = -1.0          # -1: same-industry overlap breeds RIVALRY; +1: AFFINITY

    # --- goodwill: dynamics (own; trimmed from 9 params to 5 -- see m0_notes.md) ---
    gw_decay: float = 0.07              # shared reversion rate, G_oo and S both revert toward G0/S0 at this rate
    lam_oo: float = 0.05                # shared build rate: drives BOTH the complementary-co-presence term
                                         # (+lam_oo*joint) and the same-pool overlap term (overlap_sign*lam_oo*ovl)
    lam_inv: float = 0.04                # S rises with visible investment
    lam_cap: float = 0.15                # S falls with captured share of output
    sigma_gw: float = 0.02               # shared shock sd for G_oo (symmetric) and S
    # DROPPED: halo (vouching by liked locals). Not load-bearing for H-b/H-c; listed in
    # m0_notes.md as a future extension (alongside propaganda) rather than defended here.

    # --- goodwill: effects (own) ---
    zeta_cap: float = 0.5
    zeta_bl: float = 0.5
    zeta_seize: float = 0.5
    eta: float = 0.15
    goodwill_pull: float = 0.05

    # --- oligarch behavior ---
    deploy_frac: float = 0.5
    logit_lambda: float = 5.0           # [McFadden74, MP95]
    perc_sd: float = 0.10
    pref_alpha: float = 2.0
    flight_gain: float = 0.5
    flight_max: float = 0.5
    divest_gain: float = 0.5
    divest_max: float = 0.3
    fire_sale_loss: float = 0.3
    offshore_return: float = 0.02
    exit_drawdown: float = 0.5          # exit if wealth falls >= this fraction below own recorded peak


# ---------------------------------------------------------------------------
# Random streams: fixed-shape draws each tick (common random numbers [Law15])
# ---------------------------------------------------------------------------
class Noise:
    def __init__(self, seed: int, p: Params, n_olig: int):
        s = np.random.SeedSequence(seed).spawn(7)
        self.init = np.random.default_rng(s[0])
        self._price, self._back, self._perc, self._coll, self._goo, self._gos = (
            np.random.default_rng(x) for x in s[1:])
        self.p, self.n = p, n_olig

    def draw(self):
        p, O = self.p, self.n
        e_oo = np.triu(self._goo.standard_normal((O, O)), 1)
        e_oo = e_oo + e_oo.T                                   # symmetric pair shocks
        return (
            self._price.standard_normal(NI),
            self._back.standard_normal(p.n_countries),
            self._perc.standard_normal((O, p.n_countries)),
            self._coll.random(p.n_countries),
            e_oo,
            self._gos.standard_normal((O, p.n_countries)),
        )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a and rows of b [Salton83]."""
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return an @ bn.T


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------------------------------------------------------------------
# Relationships: goodwill on every relationship (symmetric)
# ---------------------------------------------------------------------------
class Relations:
    """Shared goodwill store referenced by both agent types.

    G[a,b]  oligarch<->oligarch, symmetric, zero diagonal, in [-1,1]
    S[o,c]  state<->oligarch (one value per relationship), in [-1,1]
    G0, S0  initial attitudes; goodwill reverts toward these.
    """

    def __init__(self, G0: np.ndarray, S0: np.ndarray):
        self.G0, self.S0 = G0, S0
        self.G, self.S = G0.copy(), S0.copy()

    @staticmethod
    def initialize(p: Params, pref_share: np.ndarray, payoff_profile: np.ndarray,
                   home: np.ndarray, home_mask: np.ndarray) -> "Relations":
        """Initial attitudes from industrial interests and potential payoffs (own formulas)."""
        O = pref_share.shape[0]
        align = cosine_similarity(pref_share, payoff_profile)              # (O,C)
        S0 = np.clip(p.home_bonus * home_mask + p.s_scale * (align - align.mean())
                     - p.foreign_stigma * (~home_mask), -1.0, 1.0)
        ov = cosine_similarity(pref_share, pref_share)
        off = ~np.eye(O, dtype=bool)
        same = (home[:, None] == home[None, :]).astype(float)
        G0 = p.overlap_sign * p.init_overlap_coef * (ov - ov[off].mean()) + p.same_country_bonus * same
        G0 = np.clip((G0 + G0.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(G0, 0.0)
        return Relations(G0, S0)


# ---------------------------------------------------------------------------
# AGENT 1: Country (the state)
# ---------------------------------------------------------------------------
@dataclass
class Country:
    """The state as an agent.

    Fixed traits
      idx, name
      baseline_wealth          scales resident oligarchs' starting capital
      institutional_resilience in [0.2,1]: makes capture harder and lets the
                               state absorb pressure (faster backlash decay)
      collapse_threshold       backlash level at which collapse hazard is 50% of h_max
      industry_endowment (NI,) specialist/generalist endowments
      payoff_profile (NI,)     endowment x price: the "potential payoffs" that
                               initialize goodwill
    Stocks / state
      backlash_pressure        accumulated popular pressure from capture
      recovery_left            ticks of reduced output after a collapse
      collapse_count
      industry_output (NI,)    this tick's output pool
      industry_capture (NI,)   total captured fraction of each industry (all oligarchs)
      domestic_capture         output-weighted captured fraction of the economy
    DEFERRED (M1+, unused in M0): unclaimed_assets, scarcity_pressure,
      wealth_returned_to_population
    Relationships: goodwill_with_oligarchs() -> S[:, idx] (shared, symmetric).
    """
    idx: int
    name: str
    baseline_wealth: float
    institutional_resilience: float
    collapse_threshold: float
    industry_endowment: np.ndarray
    payoff_profile: np.ndarray
    backlash_pressure: float = 0.0
    recovery_left: int = 0
    collapse_count: int = 0
    industry_output: np.ndarray = field(default_factory=lambda: np.zeros(NI))
    industry_capture: np.ndarray = field(default_factory=lambda: np.zeros(NI))
    domestic_capture: float = 0.0
    unclaimed_assets: np.ndarray = field(default_factory=lambda: np.zeros(NI))      # DEFERRED
    scarcity_pressure: float = 0.0                                                   # DEFERRED
    wealth_returned_to_population: float = 0.0                                       # DEFERRED
    resident_oligarch_ids: List[str] = field(default_factory=list)
    relations: Optional[Relations] = field(default=None, repr=False)

    # ---- properties ----
    @property
    def in_recovery(self) -> bool:
        return self.recovery_left > 0

    def goodwill_with_oligarchs(self) -> np.ndarray:
        return self.relations.S[:, self.idx]

    # ---- behavior ----
    def produce(self, price: np.ndarray, p: Params) -> np.ndarray:
        """Per-tick output pool: endowment x yield x price, reduced during recovery."""
        mult = p.recovery_output if self.in_recovery else 1.0
        self.industry_output = self.industry_endowment * p.yield_scale * price * mult
        return self.industry_output

    def hazard(self, backlash, p: Params):
        """Probabilistic collapse hazard (own logistic form; width->0 is a hard threshold)."""
        return p.h_max * sigmoid((backlash - self.collapse_threshold) / p.h_width)

    def update_backlash(self, load: float, shock_eps: float, p: Params) -> None:
        """Backlash stock: decays (faster with resilience), fed by capture load + shock [Turchin03 structure]."""
        decay = p.b_decay * (1.0 + self.institutional_resilience)
        self.backlash_pressure = max(0.0, (1 - decay) * self.backlash_pressure
                                     + p.gamma * load + p.shock_sd * shock_eps)

    def check_collapse(self, u: float, p: Params) -> bool:
        """Collapse if the uniform draw falls under the hazard. Resets pressure, starts recovery."""
        if u < self.hazard(self.backlash_pressure, p):
            self.backlash_pressure = 0.0
            self.recovery_left = p.recovery_ticks
            self.collapse_count += 1
            return True
        self.recovery_left = max(self.recovery_left - 1, 0)
        return False


# ---------------------------------------------------------------------------
# AGENT 2: Oligarch
# ---------------------------------------------------------------------------
@dataclass
class Oligarch:
    """The oligarch as an agent.

    Fixed traits
      oligarch_id, idx, home_country (country idx)
      oligarch_type            CIVIL or RULING (ruling => home capture multiplier)
      risk_tolerance in [0,1]  maps to flight/divest threshold on B/theta: 0.4 + 0.8*tol
      industry_preferences (NI,) shares summing to 1; weight allocation choices
    Stocks / state
      capital                  liquid, deployable
      offshore_capital         safe from seizure, earns offshore_return, one-way in M0
      positions (C,NI)         sunk, seizable stock per (country, industry). Stands in for
                               both industry_investments and illiquid_assets in M0.
      capture (C,NI)           captured fraction of each (country, industry) output
      seized_total             cumulative position value lost to seizure
      initial_wealth, peak_wealth, max_drawdown   (exit bookkeeping)
    Relationships: goodwill_with_state() -> S[idx,:]; goodwill_with_peers() -> G[idx,:]
    """
    idx: int
    oligarch_id: str
    home_country: int
    oligarch_type: str
    risk_tolerance: float
    industry_preferences: np.ndarray
    capital: float
    n_countries: int
    offshore_capital: float = 0.0
    positions: Optional[np.ndarray] = None
    capture: Optional[np.ndarray] = None
    seized_total: float = 0.0
    initial_wealth: float = 0.0
    peak_wealth: float = 0.0
    max_drawdown: float = 0.0
    relations: Optional[Relations] = field(default=None, repr=False)

    def __post_init__(self):
        if self.positions is None:
            self.positions = np.zeros((self.n_countries, NI))
        if self.capture is None:
            self.capture = np.zeros((self.n_countries, NI))

    # ---- properties ----
    @property
    def is_ruling(self) -> bool:
        return self.oligarch_type == OligarchType.RULING

    @property
    def flight_threshold(self) -> float:
        return 0.4 + 0.8 * self.risk_tolerance

    @property
    def pref_log(self) -> np.ndarray:
        return np.log(np.maximum(self.industry_preferences * NI, 1e-6))

    @property
    def wealth(self) -> float:
        return self.capital + self.offshore_capital + float(self.positions.sum())

    def goodwill_with_state(self) -> np.ndarray:
        return self.relations.S[self.idx]

    def goodwill_with_peers(self) -> np.ndarray:
        return self.relations.G[self.idx]

    # ---- behavior ----
    def settle(self, extraction: float, rent: float, p: Params) -> None:
        """Book income, pay running costs, earn offshore return; liquidate positions if insolvent.
        Civil oligarchs pay civil_cost_mult x op_cost on ALL positions (domestic+foreign) --
        the transaction-cost cost of relying on legal defense/lobbying/intermediaries
        rather than direct state control."""
        cost_mult = p.civil_cost_mult if not self.is_ruling else 1.0
        self.capital += extraction + rent - cost_mult * p.op_cost * float(self.positions.sum())
        self.offshore_capital *= 1.0 + p.offshore_return
        if self.capital < 0:
            tot = float(self.positions.sum())
            frac = min(1.0, -self.capital / max(tot * (1 - p.fire_sale_loss), 1e-9))
            self.capital += frac * tot * (1 - p.fire_sale_loss)
            self.positions *= (1.0 - frac)
            self.capital = max(self.capital, 0.0)

    def suffer_collapse(self, c: int, state_goodwill: float, p: Params) -> None:
        """Country c collapsed: seize positions there (foreign holdings too if foreign_exposed),
        reduced by the polity's goodwill toward this oligarch; residents also lose liquid capital.
        Ruling oligarchs face ruling_home_mult EXTRA exposure specifically when their OWN home
        country collapses (direct entanglement with the fallen regime) -- civil oligarchs get
        relative protection here by comparison (they deflect blame to intermediaries/the regime
        itself rather than being seen as the regime). Combined fraction is clipped to 1."""
        exposed = 1.0 if p.foreign_exposed else float(self.home_country == c)
        home_vuln = p.ruling_home_mult if (self.home_country == c and self.is_ruling) else 1.0
        protection = 1.0 - p.zeta_seize * float(np.clip(state_goodwill, 0.0, 1.0))
        frac = np.clip(p.seize_frac * exposed * home_vuln * protection, 0.0, 1.0)
        loss = self.positions[c] * frac
        self.seized_total += float(loss.sum())
        self.positions[c] -= loss
        if self.home_country == c:
            self.capital *= (1 - p.resident_liquid_hit)
        self.capture[c] = 0.0

    def flee_and_divest(self, ratio: np.ndarray, p: Params) -> None:
        """Perceived backlash ratio B/theta per country -> capital flight (liquid -> offshore)
        and divestment of positions (at a fire-sale loss) when above the risk threshold."""
        Wc = self.positions.sum(1)
        risk = ((Wc * ratio).sum() + self.capital * ratio[self.home_country]) / max(Wc.sum() + self.capital, 1e-9)
        flight = float(np.clip(p.flight_gain * (risk - self.flight_threshold), 0.0, p.flight_max))
        moved = flight * self.capital
        self.capital -= moved
        self.offshore_capital += moved
        dv = np.clip(p.divest_gain * (ratio - self.flight_threshold), 0.0, p.divest_max)
        amt = self.positions * dv[:, None]
        self.positions -= amt
        self.capital += (1 - p.fire_sale_loss) * float(amt.sum())

    def net_attractiveness(self, m_ext: np.ndarray, m_cap_shared: np.ndarray, mult_row: np.ndarray,
                           hazard_row: np.ndarray, partner_row: np.ndarray, p: Params) -> np.ndarray:
        """Myopic risk-adjusted marginal return per (country, industry):
        marginal extraction + marginal capture rent - running cost - expected seizure loss
        + pull from relationships (state goodwill and partners' presence)."""
        S_row = self.goodwill_with_state()
        base = np.ones(self.n_countries) if p.foreign_exposed else (np.arange(self.n_countries) == self.home_country).astype(float)
        home_vuln = np.where((np.arange(self.n_countries) == self.home_country) & self.is_ruling,
                              p.ruling_home_mult, 1.0)                       # ruling: more exposed AT HOME (reused param)
        exposure = base * home_vuln * (1.0 - p.zeta_seize * np.clip(S_row, 0.0, 1.0))
        cost_mult = p.civil_cost_mult if not self.is_ruling else 1.0
        pull = p.goodwill_pull * (S_row + np.tanh(partner_row))
        return (m_ext + m_cap_shared * mult_row[:, None] - cost_mult * p.op_cost
                - (hazard_row * exposure * p.seize_frac)[:, None] + pull[:, None])

    def deploy(self, net: np.ndarray, p: Params) -> None:
        """Logit allocation of deploy_frac*capital over ALL (country, industry) pairs with net>0
        [McFadden74, MP95]; industry preferences enter as log weights."""
        D = p.deploy_frac * self.capital
        logits = np.where(net > 0, self.pref_log[None, :] + p.logit_lambda * net, -np.inf)
        mx = logits.max()
        if not np.isfinite(mx):
            return
        ex = np.exp(logits - mx)
        frac = ex / ex.sum()
        dep = D * frac
        self.positions += dep
        self.capital -= float(dep.sum())

    def record_wealth(self) -> float:
        w = self.wealth
        self.peak_wealth = max(self.peak_wealth, w)
        self.max_drawdown = max(self.max_drawdown, 1.0 - w / max(self.peak_wealth, 1e-12))
        return w

    def exited(self, p: Params) -> bool:
        """Exits if wealth ever fell >= exit_drawdown below its own recorded peak (own definition)."""
        return self.max_drawdown >= p.exit_drawdown


# ---------------------------------------------------------------------------
# World: builds the agents and relations; shared structural computations.
# Model versions (M0, M1, ...) subclass World and implement step().
# ---------------------------------------------------------------------------
class World:
    def __init__(self, p: Params, seed: int):
        self.p = p
        C = p.n_countries
        O = C * p.oligs_per_country
        self.C, self.O = C, O
        self.noise = Noise(seed, p, O)
        r = self.noise.init

        # ---- draws (order fixed for reproducibility) ----
        baseline = r.uniform(0.2, 1.0, C)
        rho = r.uniform(0.2, 1.0, C)
        theta = r.uniform(0.5, 1.0, C)
        endow = np.zeros((C, NI))
        for c in range(C):
            spec = r.choice(NI, size=p.n_specialty, replace=False)
            e = r.uniform(*p.nonspecialty_range, NI)
            e[spec] = r.uniform(*p.specialty_range, p.n_specialty)
            endow[c] = e
        payoff = endow * np.array(p.price_bar)[None, :]
        home = np.repeat(np.arange(C), p.oligs_per_country)
        ruling = r.random(O) < p.ruling_prob
        capital = baseline[home] * r.uniform(0.5, 1.5, O) * p.capital_scale
        tol = r.uniform(0.0, 1.0, O)
        pref = r.dirichlet(np.full(NI, p.pref_alpha), O)

        home_mask = np.zeros((O, C), dtype=bool)
        home_mask[np.arange(O), home] = True
        self.home, self.home_mask = home, home_mask
        self.relations = Relations.initialize(p, pref, payoff, home, home_mask)

        # ---- agents ----
        self.countries: List[Country] = [
            Country(idx=c, name=f"country_{c}", baseline_wealth=float(baseline[c]),
                    institutional_resilience=float(rho[c]), collapse_threshold=float(theta[c]),
                    industry_endowment=endow[c].copy(), payoff_profile=payoff[c].copy(),
                    relations=self.relations)
            for c in range(C)]
        self.oligarchs: List[Oligarch] = []
        for o in range(O):
            ol = Oligarch(idx=o, oligarch_id=f"oligarch_{o}", home_country=int(home[o]),
                          oligarch_type=OligarchType.RULING if ruling[o] else OligarchType.CIVIL,
                          risk_tolerance=float(tol[o]), industry_preferences=pref[o].copy(),
                          capital=float(capital[o]), n_countries=C, relations=self.relations)
            self.oligarchs.append(ol)
            self.countries[home[o]].resident_oligarch_ids.append(ol.oligarch_id)

        # cached read-only views of fixed traits (for vectorized cross-agent computation)
        self.rho = rho
        self.theta = theta
        self.endow = endow
        self.ruling = ruling

        # ---- initial domestic influence: home positions, capture at contest-implied level ----
        w0 = pref * payoff[home]
        w0 = w0 / np.maximum(w0.sum(1, keepdims=True), 1e-12)
        for o, ol in enumerate(self.oligarchs):
            put = p.init_home_frac * ol.capital
            ol.positions[ol.home_country, :] = put * w0[o]
            ol.capital -= put
            ol.initial_wealth = ol.wealth
        self.write_capture(self.capture_terms(self.stack_positions())[3])

    # ---- stacked views (agents remain the source of truth) ----
    def stack_positions(self) -> np.ndarray:
        return np.stack([o.positions for o in self.oligarchs])

    def stack_capture(self) -> np.ndarray:
        return np.stack([o.capture for o in self.oligarchs])

    def write_capture(self, cap: np.ndarray) -> None:
        for o, ol in enumerate(self.oligarchs):
            ol.capture = cap[o].copy()

    # ---- capture contest [Tullock80, HR89, HJK00 motivation] ----
    def capture_multiplier(self, A: np.ndarray):
        """Effectiveness of each oligarch's effort in each country: institutions, ruling
        status at home, state goodwill, and partners' signed presence (own formulas)."""
        p, R = self.p, self.relations
        P = np.tanh(A.sum(2) / p.a_ref)                          # presence (O,C)
        partner = R.G @ P                                        # sum_b G_ab * presence_bc
        base = np.where(self.home_mask & self.ruling[:, None], p.ruling_home_mult, 1.0)
        mult = (base * (1.0 - self.rho)[None, :]
                * np.clip(1.0 + p.zeta_cap * R.S, 0.2, None)
                * np.clip(1.0 + p.eta * partner, 0.3, 2.0))
        return mult, partner, P

    def capture_terms(self, A: np.ndarray):
        p = self.p
        mult, partner, P = self.capture_multiplier(A)
        e = mult[:, :, None] * A
        ew = e ** p.tullock_r
        Xeff = e.sum(0)
        K = p.kmax * (1.0 - np.exp(-p.kappa * Xeff))
        cshare = ew / np.maximum(ew.sum(0), 1e-12)[None]
        return mult, partner, P, K[None] * cshare, Xeff

    def step(self):  # implemented by model versions (see m0_model.py)
        raise NotImplementedError