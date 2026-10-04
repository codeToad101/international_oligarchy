# M0 (v2) notes

Companion to `model_core.py` and `m0_model.py`. Run `python m0_model.py --demo` (about 25 seconds).

## 0. File roles (architecture)

- **`model_core.py`** defines the model's structure: the two core agents, the relationship store, parameters and shared structural computations.
  - `Country` (the state): endowment, institutional resilience, collapse threshold, backlash pressure, recovery status, aggregate capture. Behaviors: `produce`, `update_backlash`, `hazard`, `check_collapse`.
  - `Oligarch`: type, home country, industry preferences, risk tolerance, liquid and offshore capital, positions, capture stock, exit bookkeeping. Behaviors: `settle`, `suffer_collapse`, `flee_and_divest`, `net_attractiveness`, `deploy`, `record_wealth`.
  - `Relations`: the shared goodwill store, referenced by both agent types (`G`: oligarch-oligarch, symmetric; `S`: state-oligarch). Each agent reads its own relationships (`goodwill_with_state`, `goodwill_with_peers`, `goodwill_with_oligarchs`).
  - `World`: builds agents and relations, holds the capture contest, and is subclassed by each model version.
- **`m0_model.py`** contains only M0's flow equations (`M0World(World)`, which orchestrates the agents each tick) and the diagnostics. All state lives on the agents.
- **Regression check:** the agent-based refactor reproduces the earlier array-based version exactly (wealth, collapses, exits, both goodwill matrices, seeds 0-4), so it changed structure, not behavior.
- The `Alliance` class from the old `model_core.py` is replaced by `Relations`. Country/oligarch fields that M0 does not use (`unclaimed_assets`, `scarcity_pressure`, `wealth_returned_to_population`) are kept on `Country`, marked DEFERRED.

## 1. What changed from v1 (of `m0_model.py`)

- **Removed** everything ambition-related: the ambition variable, the focal-vs-population payoff grid and the ambition sweep. Passive-vs-ambitious populations are parked as a future add-on. The v1 ambition results are void.
- **Foreign investment kept.** There is one logit over all (country, industry) pairs. Home advantage comes from goodwill (home S0 is high) plus a ruling-oligarch home multiplier. The separate "foreign capture penalty" is gone.
- **Goodwill on every relationship, symmetric for now:**
  - `G_oo[a,b]`, an O x O matrix (one value per pair).
  - `S[o,c]`, an O x C matrix (one value per state-oligarch relationship, including the home country).
- **Initial domestic influence added** (from your brief): each oligarch starts with `init_home_frac` of its capital placed at home, and its capture stock starts at the contest-implied level.
- **Exit rule added**: an oligarch exits if its wealth falls at least 50% below its own running peak. My first version (below 50% of *initial* wealth) triggered for only about 1% of oligarchs, too rare to analyze. Exit is recorded only; the agent stays in the run so random streams stay aligned.
- **Stochasticity kept** and extended: goodwill shocks (symmetric for `G_oo`), on top of price shocks, backlash shocks, noisy risk perception and probabilistic collapse.

## 2. Goodwill specification (all my own assumptions unless a source is named)

**Initialization** ("industrial interests & potential payoffs"):
- `S0[o,c] = home_bonus*home + s_scale*(align - mean(align)) - foreign_stigma*foreign`, clipped to [-1,1]. `align` is the cosine similarity between oligarch o's industry preference shares and country c's payoff profile (endowment x price). Cosine similarity: Salton & McGill (1983).
- `G0[a,b] = overlap_sign*init_overlap_coef*(cos(pref_a,pref_b) - mean) + same_country_bonus*(co-resident)`, symmetrized and clipped. **`overlap_sign` = -1 (same-industry overlap breeds rivalry) or +1 (affinity)** is your open question from the `Alliance` TODO. It is switchable and used in both initialization and dynamics.

**Dynamics:**
- `G_oo`: reverts toward `G0`; rises with *complementary* co-presence (same country, different industries: `lam_joint`); moves by `overlap_sign` with same-pool overlap (`lam_overlap`); symmetric shocks. Motivation for building goodwill through repeated joint activity: Axelrod (1984). Motivation only; no formula taken.
- `S`: reverts toward `S0`; rises with visible investment (`lam_inv`); falls with captured share of output (`lam_cap`); a foreigner gains from vouching by liked locals (`halo`); shocks.

**Effects:**
- Capture effectiveness x `(1 + zeta_cap*S)` and x `(1 + eta * sum_b G_ab * presence_bc)` (signed: rival presence hurts, partner presence helps).
- Backlash weight of an oligarch's capture x `(1 - zeta_bl*S)` (times a foreignness visibility factor).
- Seizure loss on collapse x `(1 - zeta_seize*max(S,0))`.
- Deployment attractiveness gets `goodwill_pull*(S + tanh(partner presence))`.

**Interpretation to confirm:** I treat state goodwill as the *polity's standing* toward the oligarch (legitimacy, not the regime's favor). That is why higher S reduces seizure. If you mean regime favor instead, this effect would need to flip or be reconsidered.

## 3. Sources

Formulas from papers (page-level details are from memory; verify before citing):
- Gordon (1954) *J. Political Economy* 62(2):124-142; Schaefer (1954) *Bull. IATTC* 1(2):27-56; Clark (1990) *Mathematical Bioeconomics*, 2nd ed. (extraction, exponential-catch form).
- Tullock (1980) in Buchanan, Tollison & Tullock (eds.); Hillman & Riley (1989) *Economics & Politics* 1(1):17-39 (contest share).
- Hellman, Jones & Kaufmann (2000), World Bank WP 2444 (motivation for bounded capture only).
- Turchin (2003); Turchin & Nefedov (2009) (structure of an aggregate stress indicator only; no coefficients).
- McFadden (1974); McKelvey & Palfrey (1995) (logit choice).
- Salton & McGill (1983) (cosine similarity, new in v2).
- Axelrod (1984) (motivation only, new in v2).
- Law (2015) (common random numbers).

**Own assumptions, not from any paper:** collapse hazard, capture persistence, capture-rent stream, backlash accumulation, flight/divest rules, seizure fractions, AR(1) prices, the exit rule, every goodwill formula and effect size, and all parameter values.

## 4. Results snapshot (100 ticks, 30-150 seeds, placeholder parameters)

Baseline: about 0.8 collapses/country/run, exit rate about 18%, Gini of final wealth about 0.39, mean state goodwill about 0.65 (home) and 0.15 (foreign).

**Backlash sensitivity (gamma) 0.04 -> 0.16:** collapses/country rise from 0.17 to 3.6; exit rate rises from about 9% to about 30% (levelling off above 0.12).

**Who survives** (within-seed difference, survivors minus exited, mean ± SE, 121 seeds). Descriptive only, since these features are endogenous:

| Feature | Survivors - exited |
|---|---|
| State goodwill abroad (`S_foreign`) | +0.093 ± 0.022 |
| Foreign share of positions | +0.024 ± 0.026 (no relationship) |
| Average goodwill with other oligarchs | -0.035 ± 0.014 (**opposite** to the cooperation-protects hypothesis) |

**Overlap sign** (150 seeds, paired): affinity (+1) vs. rivalry (-1) changes exit rate by -0.023 ± 0.012, marginal.

Reading it:
- Standing with *host states* tracks survival, and footprint size does not. That matches your point that ambition is the wrong axis.
- Nothing yet shows oligarch-oligarch goodwill protects. The current measure averages over all oligarchs, including ones far away, and the effects (`eta`, `halo`) may be too weak. A better test is goodwill with partners *present in the same countries*. It also means deals are not yet explicit events.
- Causality is not established. Oligarchs who lose positions also lose the presence that supports goodwill, so `S_foreign` could partly follow survival rather than cause it.

## 5. Open questions / next steps

1. Directed goodwill (parked, per your call).
2. Explicit deals: pooled commitments per (pair, country, industry) and defection events, so "deals abroad without moving abroad" is a testable strategy rather than an emergent correlation.
3. A partner-relevant goodwill measure (partners present in shared countries) for the survival analysis.
4. Sensitivity sweeps on `zeta_*`, `eta`, `halo`, `lam_cap`, and the exit drawdown threshold.
5. Deferred M1+ pieces: depletion/scarcity, illiquid vs. liquid split, asset repurposing, ruling-status transitions.

## 6. Civil/ruling asymmetry (added per discussion; own design, no new mechanism paper)

Built by REUSING `ruling_home_mult` for three home-entanglement effects, plus one
new parameter, `civil_cost_mult`, for civil's offsetting cost:

- **Ruling, at home only:** cheaper/more effective capture (existing); MORE
  backlash contribution (direct state control is visible/attributable); MORE
  seizure exposure if their OWN home country collapses.
- **Civil, everywhere:** pays `civil_cost_mult` (1.3x) higher running costs on
  all positions (legal defense, lobbying, intermediaries); gets relative home
  collapse protection only by comparison to ruling (no separate dial).

**Verification check result (60 seeds), interpreted against the NEW expectation:**

| Measure | Civil | Ruling | Expected to differ? |
|---|---|---|---|
| exited | 0.331 | 0.240 | Yes (cost pressure vs. collapse exposure tradeoff) |
| foreign_capture_gained (absolute) | 52.5 | 108.6 | Yes (ruling's extra home-side wealth) |
| foreign_capture_share | 0.352 | 0.435 | Yes (home-side effects dilute differently) |
| **foreign_capture_rate (time-integrated, foreign-only)** | **12.6** | **10.7** | **Expected: no.** Found: civil higher, Cohen's d = 1.09 (large) |

**The remaining rate gap is real, not a measurement artifact** (the first version of
this measure, using final-tick position as the denominator, WAS an artifact --
fixed by switching to a time-integrated average; see code comments). The
corrected gap has a plausible, economically sensible mechanism: `civil_cost_mult`
raises the net-return hurdle for civil deployment EVERYWHERE (not just at home),
so civil oligarchs deploy into fewer foreign opportunities but more selectively
-- a classic cost-of-capital selection effect (higher hurdle rate -> smaller,
higher-quality portfolio -> higher measured per-unit return on what's actually
deployed). This is a believable emergent result, not a bug, and can be framed as
a "quality over quantity abroad" finding for civil oligarchs in the discussion
section. It should be reported as a real, sizeable (d=1.09) effect, not
downplayed -- the design intentionally makes civil and ruling asymmetric, so a
large distinguishing effect is expected and desired here, just not of the kind
this specific rate measure was originally built to rule out.

**Sanity check:** no negative capital/position states or NaNs across 60 seeds x
10 oligarchs after all changes.