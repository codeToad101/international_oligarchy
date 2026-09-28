"""
model_core.py

Baseline class and variable scaffolding for the transnational oligarchic
capture model. Countries and Oligarchs are agents (ABM); the internal
attributes on each are the SD-style stocks discussed in the design
sessions (see project_overview.txt for the full stock-and-flow diagram
this maps onto).

This file defines STRUCTURE ONLY -- no update/step equations yet.
World.step() is a stub. Use this as the scaffold to build the actual
flow equations onto once the open questions in project_overview.txt
(Section 5) are resolved.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Fixed industry set. Kept small and deliberately generic for the baseline
# model rather than trying to be exhaustive -- extend later if the sweep
# results suggest granularity matters. Every Country gets an
# industry_endowment and industry_resource_stock entry for each of these.
INDUSTRIES: List[str] = [
    "agriculture",
    "energy",
    "mining",
    "manufacturing",
    "finance",
    "technology",
]

# How many industries a country is a genuine "specialist" in (high
# endowment) vs merely competent/weak at. Placeholder -- see
# _sample_industry_profile docstring below for the reasoning and the
# open question this raises.
N_SPECIALTY_INDUSTRIES = 2

# Bounds used when sampling endowment for specialty vs non-specialty
# industries -- this now governs RESOURCE ABUNDANCE only (how big a
# country's stock/capacity is in an industry), not extraction efficiency.
# See CATCHABILITY_COEFFICIENT below for why efficiency was pulled out
# into its own, non-country-specific constant.
SPECIALTY_ENDOWMENT_RANGE = (0.6, 1.0)
NONSPECIALTY_ENDOWMENT_RANGE = (0.0, 0.4)
RESOURCE_STOCK_SCALE = 100.0

# --- Extraction: Gordon-Schaefer bioeconomic harvest model -----------------
# (Schaefer 1954 on the biological stock dynamics; Gordon 1954 on the
# economics.) Harvest is a function of the HARVESTER's own effort and the
# CURRENT stock, not a country-specific efficiency dial:
#     harvest(oligarch, c, i) = CATCHABILITY_COEFFICIENT * investment * stock
# and profit = INDUSTRY_PRICE[i] * harvest - investment (cost of effort).
# CATCHABILITY_COEFFICIENT is a single constant shared by everyone -- no
# country gets to be inherently "better" at converting money into
# extracted value; the only thing that varies by country is how much
# stock is actually there (industry_resource_stock/capacity, sampled via
# industry_endowment above). This deliberately replaces an earlier design
# where a country-level "endowment" multiplier played double duty as both
# stock size AND extraction efficiency AND (via a third use) synthetic
# GDP value -- collapsing three distinct things into one number. Not yet
# wired into World.step().
CATCHABILITY_COEFFICIENT = 0.02

# Per-industry price used to (a) convert harvested quantity into oligarch
# profit/capital, and (b) size industry_value (see Country.industry_value)
# as capacity * price rather than from endowment. Deliberately varied
# across industries so that a small-but-precious resource (e.g. a
# diamonds-style industry) and a large-but-cheap one (e.g. a wheat-style
# industry) can be told apart even at similar physical abundance --
# PLACEHOLDER relative values, not calibrated to anything real.
INDUSTRY_PRICE: Dict[str, float] = {
    "agriculture": 1.0,
    "energy": 3.0,
    "mining": 4.0,
    "manufacturing": 2.0,
    "finance": 5.0,
    "technology": 6.0,
}

# --- Capture mechanics (see Country.industry_value / industry_max_capture
# / industry_capture docstrings) -----------------------------------------
# Tullock (1980) contest-intensity parameter for splitting a captured
# industry's value among competing oligarchs: share_i = x_i^r / sum(x_j^r).
# r=1 is the proportional "lottery" contest; r>1 skews toward winner-take-
# most; r<1 flattens toward an even split regardless of investment gap.
# Starting at r=1 (the field's standard baseline -- direct empirical
# measurement of real-world capture/rent-seeking contests is notoriously
# hard to get, so most applied work defaults here for tractability rather
# than from a settled estimate). Candidate future refinement: tie r to
# Country.institutional_resilience instead of a global constant -- some
# of the contest-success-function literature treats "discriminatory
# power" as characterizing the institutional environment (how decisively
# effort/money buys outcome) rather than a fixed universal number, which
# would mean weak-institution countries get r > 1 (money buys decisive
# dominance) and strong-institution countries stay near r = 1 or lower.
# Not implemented -- flagged for later, per project_overview.txt Section 5.
CAPTURE_CONTEST_INTENSITY = 1.0

# Ceiling on how much of ANY industry's synthetic value can ever be
# oligarchically captured, however much is invested -- represents an
# irreducible informal/independent sector that capture cannot fully
# absorb (empirically, even heavily "captured" economies keep functioning
# rather than freezing entirely). Placeholder value, deliberately left
# as-is for now -- revisit later.
MAX_CAPTURABLE_FRACTION = 0.85

# --- Resource stock regeneration / abuse (see
# Country.industry_resource_capacity docstring) ---------------------------
# Fraction of the gap between current stock and full capacity that
# regenerates each tick when extraction stays below the abuse threshold.
# Placeholder.
RESOURCE_REGEN_RATE = 0.05

# If a single tick's extraction from an industry's resource stock exceeds
# this fraction of the REMAINING stock, treat it as abuse: rather than
# just drawing the stock down (which would regenerate back), permanently
# shrink industry_resource_capacity itself (the pool is damaged, not just
# temporarily drawn on). Placeholder.
RESOURCE_ABUSE_THRESHOLD_FRACTION = 0.5

# --- Unclaimed-asset repurposing (post-collapse) --------------------------
# Deliberately reuses the Gordon-Schaefer harvest SHAPE rather than
# inventing a third mechanism: claiming freed illiquid assets is treated
# as another harvest, against a different pool.
#     claim = ASSET_CLAIM_RATE * investment * Country.unclaimed_assets[i]
# This gets "nothing forces claiming to happen" for free -- if nobody
# invests in that (country, industry), claim = 0 and the pool just sits
# there indefinitely, no separate decay/expiry logic needed. It also
# means multiple oligarchs investing in the same freshly-collapsed
# industry compete for the same shrinking pool automatically (a
# common-pool-resource / commons-scramble dynamic -- see Ostrom, and
# module docstring further down), without a bespoke contest function.
# Placeholder value, deliberately close to CATCHABILITY_COEFFICIENT in
# order of magnitude since it's the same kind of "effort meets an
# available pool" process -- not derived from anything.
ASSET_CLAIM_RATE = 0.02

# Once claimed, value can either be ABSORBED (added to the claimant's own
# illiquid_assets -- they commit to running it, exposed to being seized
# again in a future collapse) or STRIPPED (converted straight to liquid
# capital -- asset-strip and move on, safer but no long-run upside).
# Modeled as a linear function of the claimant's risk_tolerance rather
# than a separate trait: risk_tolerance=1 -> fully absorbed;
# risk_tolerance=0 -> fully stripped to liquid capital. Reuses an
# existing Oligarch trait instead of adding new agent psychology, per
# the instruction to aim for realistic bottom-up behavior without
# building out a full state-internal ABM. Not yet wired into
# World.step().


class OligarchType:
    """Distinguishes civil vs. ruling oligarchs (Winters' typology).

    CIVIL  -- exercises power through legal/economic institutions rather
              than direct rule (e.g. Rothschild-style dynasties,
              company-states like United Fruit/Chiquita).
    RULING -- holds direct political power in a state (the more familiar
              "technical" sense of oligarch in a captured state).

    A civil oligarch can transition to ruling during the simulation via
    the seizure flow (see Oligarch.ruling_status below).
    """
    CIVIL = "civil"
    RULING = "ruling"


@dataclass
class Country:
    """An agent representing a single state.

    Stocks (change over the simulation)
    ------------------------------------
    domestic_capture : float
        Aggregate policy/regulatory control held by oligarchs operating
        in this country. Fed by capture investment, drained by decay and
        by collapse events.
    backlash_pressure : float
        Aggregate instability pressure. Two feeder channels are intended:
        immiseration (extraction from the general population) and
        counter-elite exclusion (rival elites shut out of captured
        channels). The functional FORM is loosely inspired by the
        structure of Turchin's Political Stress Indicator -- explicitly
        NOT his calibrated coefficients, which are fitted to a different
        historical dataset and shouldn't be imported as-is.
    industry_resource_stock : Dict[str, float]
        CURRENT extractable quantity of each industry's resource, keyed
        by industry name. Unlike a one-way depleting stock, this now
        REGENERATES each tick toward industry_resource_capacity at
        RESOURCE_REGEN_RATE, as long as extraction in that tick stayed
        below RESOURCE_ABUSE_THRESHOLD_FRACTION of the remaining stock --
        i.e. ordinary use is sustainable and the pool recovers. Only
        extraction ABOVE that threshold in a given tick ("abuse")
        permanently damages industry_resource_capacity itself (see
        below), which is what should eventually manifest as
        Country.scarcity_pressure. Deliberately kept simple (a single
        threshold, not a full stock-flow-with-lookback model) per the
        instruction not to go too deep into scarcity mechanics.
    industry_resource_capacity : Dict[str, float]
        The CEILING industry_resource_stock regenerates toward -- think
        of it as the industry's sustainable carrying capacity rather
        than a one-time finite reserve. Starts equal to the initial
        resource stock; only shrinks when abuse (see above) occurs.
        This is what makes scarcity a real, if rare, one-way ratchet:
        ordinary extraction is fully recoverable, but abusive extraction
        erodes the ceiling itself and that erosion is not undone by the
        regeneration flow.
    industry_value : Dict[str, float]
        Synthetic per-industry economic value (a GDP-style proxy),
        keyed by industry name. Used only to size industry_max_capture
        below -- NOT itself a capture target. Computed at init as
        industry_resource_capacity[i] * INDUSTRY_PRICE[i] -- i.e. from
        physical abundance and price, NOT from industry_endowment
        directly. This was changed from an earlier design where value
        was computed straight from endowment, which made one number
        (endowment) simultaneously set stock size, extraction
        efficiency, AND value -- collapsing three axes that should be
        able to vary independently (e.g. a scarce-but-precious resource
        vs. an abundant-but-cheap one) into one. Endowment now only
        determines abundance (stock/capacity); price is what lets two
        similarly-abundant industries differ in value.
    industry_max_capture : Dict[str, float]
        Ceiling on how much of industry_value can ever be under
        oligarchic control in that industry, = industry_value[i] *
        MAX_CAPTURABLE_FRACTION. Leaves a deliberate, permanent
        irreducible independent/informal-sector floor so that "100%
        oligarchic capture" of a country never literally means zero
        other economic activity.
    industry_capture : Dict[str, float]
        CURRENT total captured value in each industry -- a stock, not
        yet updated by any flow (World.step() is still a stub). Intended
        update rule once implemented: total captured value saturates
        toward industry_max_capture as aggregate oligarch investment in
        that industry rises (a diminishing-returns curve, not a hard
        cliff), and is then SPLIT among the individual oligarchs
        investing there via a Tullock-style contest share (see
        CAPTURE_CONTEST_INTENSITY): each oligarch's realized captured
        amount is proportional to their investment raised to that
        power, divided by the sum across all oligarchs investing in
        that (country, industry) pair. Two consequences worth flagging:
        (1) domestic_capture (below) can then simply be DEFINED as
        sum(industry_capture.values()) rather than tracked as an
        independent stock, resolving the open question about how
        industry-level investment relates to the aggregate; (2) when a
        new entrant invests in an already-captured industry, the
        incumbent's realized share drops even though the incumbent
        changed nothing -- that drop is the natural trigger for the
        "existing oligarch responds unfavorably" mechanic, feeding
        Alliance.goodwill (see Alliance docstring) or an equivalent
        rivalry stock for un-allied competitors. NOTE this is a
        distinct mechanism from resource EXTRACTION (see
        Oligarch.industry_investments / CATCHABILITY_COEFFICIENT):
        industry_capture is about regulatory/policy control, extraction
        is about physical resource and profit. OPEN QUESTION, not yet
        decided: should an oligarch's realized industry_capture share
        also boost their effective extraction efficiency (i.e. political
        capture buys preferential resource access, a reinforcing loop
        directly relevant to the paper's thesis), or should the two
        stay fully independent for v1?
    unclaimed_assets : Dict[str, float]
        Illiquid physical capital (factories, offices, equipment) freed
        by a collapse event (see is_collapsed below), keyed by industry.
        Populated when Country.is_collapsed triggers: every resident
        oligarch's Oligarch.illiquid_assets in this country get zeroed
        out and their value moves here. Deliberately NOT automatically
        reclaimed by anyone -- claiming is modeled as another Gordon-
        Schaefer-style harvest (see ASSET_CLAIM_RATE in the module
        docstring): claim = ASSET_CLAIM_RATE * investment * this pool,
        drawn down only by oligarchs who choose to invest here. If
        nobody does, it sits idle indefinitely -- no forced flow, no
        decay/expiry built in (deliberately, to avoid over-modeling a
        state-internal process that isn't this paper's focus). Any
        oligarch can claim here, including foreign ones already
        investing in this country's industries -- a domestic collapse
        becoming a cross-border capture opportunity falls directly out
        of this without any extra machinery, which is thesis-relevant.
        Not yet implemented in World.step().
    domestic_capture : float
        Aggregate policy/regulatory control held by oligarchs operating
        in this country. Under the mechanism above this is intended to
        equal sum(industry_capture.values()) once industry_capture is
        actually computed by World.step() -- kept as its own field for
        now since that flow isn't implemented yet.
    backlash_pressure : float
        Aggregate instability pressure, fed by THREE channels (not two):
        immiseration, exclusion, and scarcity. Immiseration is now
        explicitly a COMPOSITE of (a) the shortfall in
        wealth_returned_to_population relative to extraction volume and
        (b) the resource-consumption rate itself -- i.e. "does the
        country still get the product of its own labor" AND "how much
        is being taken out of the ground regardless." Exact combination
        (sum vs. one scaling the other) is not yet decided -- see
        project_overview.txt Section 5. Scarcity is kept as its own,
        separate channel (Country.scarcity_pressure) rather than folded
        into immiseration, since it should be able to trigger revolt
        from pure resource exhaustion even if extraction was otherwise
        being "fairly" shared -- functional FORM is loosely inspired by
        the structure of Turchin's Political Stress Indicator --
        explicitly NOT his calibrated coefficients, which are fitted to
        a different historical dataset and shouldn't be imported as-is.
    is_collapsed : bool
        Whether a collapse event has been triggered. On collapse:
        domestic_capture drains, the triggering (and other resident)
        oligarchs' liquid capital takes a hit, and -- new -- every
        resident oligarch's Oligarch.illiquid_assets in this country
        are seized/abandoned and their value moves to
        Country.unclaimed_assets (see below), where they sit available
        for repurposing but are not automatically claimed by anyone.
        Exact drain magnitudes and the claiming mechanism are not yet
        implemented.
    wealth_returned_to_population : float
        Cumulative stock tracking how much of the wealth generated by
        capture/extraction in this country has flowed back to the
        general population (public goods, wages, reinvestment) versus
        been extracted as oligarch capital/offshore capital. Feeds the
        immiseration channel above.
    scarcity_pressure : float
        Backlash-pressure contribution specifically from abusive
        resource depletion (industry_resource_capacity being damaged --
        see industry_resource_stock/capacity above), kept per-industry
        in principle but currently exposed as a single aggregate float
        per country. NOT YET wired into backlash_pressure -- flow
        equation not implemented.

    Parameters (heterogeneous, sampled once at init -- NOT stocks)
    ----------------------------------------------------------------
    baseline_wealth : float
        Starting macroeconomic wealth level. Determines the starting
        capital of oligarchs originating here.
    institutional_resilience : float
        Country-specific resistance to capture and to backlash
        escalating into collapse. Higher = harder to capture, harder to
        destabilize.
    collapse_threshold : float
        Backlash pressure level at which a collapse event triggers.
        Currently assumed to be a hard threshold -- whether it should
        instead be a probability that rises with pressure is an open
        question (see project_overview.txt Section 5).
    industry_endowment : Dict[str, float]
        Country's comparative advantage per industry, 0-1, keyed by
        industry name (see module-level INDUSTRIES). Sampled ONCE at
        init so countries are heterogeneous by design -- every country
        gets 1-2 "specialty" industries with high endowment and the
        rest low/mediocre (see World._sample_industry_profile). Governs
        RESOURCE ABUNDANCE ONLY (how big industry_resource_stock/
        capacity start out) -- it is deliberately NOT an extraction
        efficiency multiplier: extraction efficiency is now a single
        global CATCHABILITY_COEFFICIENT shared by every oligarch in
        every country (see module docstring), so that "better resources"
        means "more of it," never "objectively better at extracting it."
    """

    name: str
    baseline_wealth: float
    institutional_resilience: float
    collapse_threshold: float

    domestic_capture: float = 0.0
    backlash_pressure: float = 0.0
    is_collapsed: bool = False

    industry_endowment: Dict[str, float] = field(default_factory=dict)
    industry_resource_stock: Dict[str, float] = field(default_factory=dict)
    industry_resource_capacity: Dict[str, float] = field(default_factory=dict)
    industry_value: Dict[str, float] = field(default_factory=dict)
    industry_max_capture: Dict[str, float] = field(default_factory=dict)
    industry_capture: Dict[str, float] = field(default_factory=dict)
    wealth_returned_to_population: float = 0.0
    scarcity_pressure: float = 0.0
    unclaimed_assets: Dict[str, float] = field(default_factory=dict)

    # populated by World once oligarchs are assigned to their home country
    resident_oligarch_ids: List[str] = field(default_factory=list)


@dataclass
class Oligarch:
    """An agent representing a single oligarch (civil or ruling).

    Stocks
    ------
    capital : float
        LIQUID wealth held onshore -- cash/fluid capital an oligarch can
        move, invest, or flee with. Previously documented as "liquid +
        illiquid combined"; split out because collapse needs to treat
        the two differently (see illiquid_assets below and
        Country.unclaimed_assets).
    illiquid_assets : Dict[str, Dict[str, float]]
        Physical/fixed capital (factories, offices, equipment) this
        oligarch holds in each (country, industry): {country_name:
        {industry_name: value}}. Distinct from capital (liquid) and from
        industry_investments (ongoing effort/spend, see below) -- this
        represents capital that has hardened into sunk, illiquid plant.
        Exact accumulation rule not yet decided (e.g. some fraction of
        cumulative industry_investments converts to illiquid_assets over
        time) -- flagged as an open question. What IS decided: only
        illiquid_assets get seized/frozen on a Country collapse event
        (moved to Country.unclaimed_assets) -- liquid capital and
        offshore_capital are a separate concern the oligarch may already
        have moved out of reach.
    offshore_capital : float
        Capital moved out of reach of the home country via capital
        flight. One-way in this baseline (nothing flows back in) --
        flagged as a simplification to revisit.
    foreign_capture : Dict[str, float]
        Degree of captured influence held over each foreign country,
        keyed by country name.
    industry_investments : Dict[str, Dict[str, float]]
        Amount invested by this oligarch into each industry, broken
        out per country: {country_name: {industry_name: amount}}. This
        includes the oligarch's home country (domestic investment) as
        well as any foreign countries they invest in. This single stock
        now has to feed TWO separate mechanisms, which is worth being
        explicit about:
        (1) EXTRACTION (Gordon-Schaefer bioeconomic harvest model --
        Schaefer 1954 / Gordon 1954): harvest = CATCHABILITY_COEFFICIENT
        * investment * Country.industry_resource_stock[i]; profit =
        INDUSTRY_PRICE[i] * harvest - investment. Effort (this oligarch's
        own investment) and current stock are the only two inputs --
        there is no country-specific "efficiency" dial (see module
        docstring for why that was deliberately removed). This is what
        feeds Oligarch.capital and Country.wealth_returned_to_population.
        (2) CAPTURE (Tullock-style contest, see Country.industry_capture
        and CAPTURE_CONTEST_INTENSITY): the same investment figure is
        also the effort input to a separate contest determining this
        oligarch's share of regulatory/policy capture in that industry,
        which is what domestic_capture/foreign_capture are defined from.
        OPEN QUESTION, not yet decided: does one investment figure
        really drive both mechanisms identically, or should extraction
        effort and capture effort be split into two separate stocks
        (an oligarch might spend to extract resources without bothering
        to buy political influence, or vice versa)? Kept as ONE stock
        for now since that's the simpler v1 default, not because the
        question is resolved.
    ruling_status : float
        0.0 = purely civil, rises toward 1.0 via the seizure flow, fed
        by domestic_capture and suppressed by the home country's
        backlash_pressure (higher perceived risk = harder to seize
        power outright). Not yet clear what changes once ruling_status
        is high -- open question.

    Parameters (heterogeneous, sampled once at init)
    ------------------------------------------------
    oligarch_type : str
        One of OligarchType.CIVIL / OligarchType.RULING at init; can
        change via ruling_status crossing a threshold during the run.
    home_country : str
        Name of the originating Country -- sets starting capital scale.
    risk_tolerance : float
        Shapes willingness to pursue capital flight / power seizure
        under a given backlash_pressure level. Also now governs the
        absorb-vs-strip split when claiming unclaimed_assets (see
        ASSET_CLAIM_RATE in the module docstring): higher risk_tolerance
        -> more of what's claimed is absorbed into illiquid_assets
        (commit to running it); lower -> more is stripped straight to
        liquid capital (safer, no long-run upside). Reused deliberately
        rather than adding a separate trait for this.
    """

    oligarch_id: str
    home_country: str
    oligarch_type: str
    capital: float
    risk_tolerance: float

    offshore_capital: float = 0.0
    ruling_status: float = 0.0
    industry_investments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    illiquid_assets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    foreign_capture: Dict[str, float] = field(default_factory=dict)


@dataclass
class Alliance: #NEED TO ADD COMPARISON of industry interests, alliances go better if invested
                #in different industries, or if they are in the same industry?  Need to think about this more
                #want to avoid competition, goal is monopoly... but the ends could alter the means
    """A relationship between two oligarchs pooling capital for joint
    foreign capture -- the "feudal lord" mutual-consolidation mechanic.

    Stocks
    ------
    goodwill : float
        Trust built up through joint deals. Intended to feed back into
        the effectiveness of future joint foreign-capture investment,
        and to decay without continued activity. Whether goodwill can
        also be actively spent/broken by a discrete defection event
        (vs. only passive decay) is an open question -- see
        project_overview.txt Section 5.

        NEW candidate source of a NEGATIVE goodwill shock (grievance):
        under the capture-contest mechanism (see Country.industry_capture),
        when one oligarch's investment in a (country, industry) another
        oligarch already operates in causes that incumbent's realized
        captured share to drop, that drop is a natural, already-computed
        trigger for "the existing oligarch responds unfavorably." NOT
        YET wired into any flow. Open question this raises: does this
        only apply to already-ALLIED oligarch pairs (goodwill going
        negative within an existing Alliance), or does displacement
        between two oligarchs who were never allied need its own
        lightweight rivalry/grievance relation instead of overloading
        Alliance (which was designed for deliberately formed
        partnerships, not adversarial pairs)?
    """

    oligarch_a: str
    oligarch_b: str
    goodwill: float = 0.0

    @property
    def key(self) -> Tuple[str, str]:
        return tuple(sorted((self.oligarch_a, self.oligarch_b)))


class World:
    """Container for the full model: countries, oligarchs, and alliances.

    Baseline scope: 5 countries, 0-2 oligarchs per country, sampled at
    initialization. No step/update logic yet -- see World.step().
    """

    def __init__(self, n_countries: int = 5, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.countries: Dict[str, Country] = {}
        self.oligarchs: Dict[str, Oligarch] = {}
        self.alliances: Dict[Tuple[str, str], Alliance] = {}
        self._init_countries(n_countries)
        self._init_oligarchs()

    def _init_countries(self, n_countries: int) -> None:
        for i in range(n_countries):
            name = f"country_{i}"
            country = Country(
                name=name,
                baseline_wealth=self.rng.uniform(0.2, 1.0),
                institutional_resilience=self.rng.uniform(0.2, 1.0),
                collapse_threshold=self.rng.uniform(0.5, 1.0),
            )
            self._sample_industry_profile(country)
            self.countries[name] = country

    def _sample_industry_profile(self, country: Country) -> None:
        """Sample industry_endowment, industry_resource_stock/capacity,
        and industry_value/industry_max_capture for one country so it's
        heterogeneous by construction: good at a couple of things,
        mediocre/weak at the rest -- never uniformly good (or uniformly
        bad) across the board.

        Mechanism (PLACEHOLDER, deliberately simple for v1): pick
        N_SPECIALTY_INDUSTRIES industries at random to be this country's
        "specialties" and sample their endowment from
        SPECIALTY_ENDOWMENT_RANGE; sample every other industry's
        endowment from NONSPECIALTY_ENDOWMENT_RANGE. This is a discrete
        specialist/generalist split rather than a smooth distribution
        (e.g. a Dirichlet over industries) -- flagging that choice
        explicitly since it affects how "comparative advantage" reads
        in the sweep (hard specialization vs. graded advantage).

        industry_resource_stock/capacity is set proportional to
        endowment (specialists start with deeper reserves in their
        specialty, consistent with "better resources" meaning more of
        it -- NOT better at extracting it; see CATCHABILITY_COEFFICIENT
        in the module docstring for why extraction efficiency is no
        longer a country-specific number).

        industry_value (the synthetic-GDP proxy used only to size the
        capture ceiling) is derived from industry_resource_capacity *
        INDUSTRY_PRICE[i], NOT from endowment directly -- this is what
        lets a scarce-but-valuable industry and an abundant-but-cheap
        one be told apart, rather than endowment simultaneously setting
        stock size, extraction efficiency, and value. industry_capture
        and unclaimed_assets both start at 0.0 since no capture or
        collapse has occurred yet.
        """
        specialties = self.rng.sample(INDUSTRIES, k=min(N_SPECIALTY_INDUSTRIES, len(INDUSTRIES)))
        for industry in INDUSTRIES:
            if industry in specialties:
                endowment = self.rng.uniform(*SPECIALTY_ENDOWMENT_RANGE)
            else:
                endowment = self.rng.uniform(*NONSPECIALTY_ENDOWMENT_RANGE)
            country.industry_endowment[industry] = endowment

            resource_stock = endowment * RESOURCE_STOCK_SCALE
            country.industry_resource_stock[industry] = resource_stock
            country.industry_resource_capacity[industry] = resource_stock

            industry_value = resource_stock * INDUSTRY_PRICE[industry]
            country.industry_value[industry] = industry_value
            country.industry_max_capture[industry] = industry_value * MAX_CAPTURABLE_FRACTION
            country.industry_capture[industry] = 0.0
            country.unclaimed_assets[industry] = 0.0

    def _init_oligarchs(self) -> None:
        """Sample 0-2 oligarchs per country. Starting capital and type
        distributions are PLACEHOLDERS -- both should ultimately be a
        deliberate function of baseline_wealth per the "originating
        country changes both kind and amount" design decision, not the
        arbitrary weights used here."""
        oligarch_counter = 0
        for country in self.countries.values():
            n_oligarchs = self.rng.choice([0, 1, 1, 2])  # placeholder weighting
            for _ in range(n_oligarchs):
                oligarch_id = f"oligarch_{oligarch_counter}"
                oligarch_counter += 1
                oligarch_type = (
                    OligarchType.RULING
                    if self.rng.random() < 0.3  # placeholder split
                    else OligarchType.CIVIL
                )
                capital = country.baseline_wealth * self.rng.uniform(0.5, 1.5)
                self.oligarchs[oligarch_id] = Oligarch(
                    oligarch_id=oligarch_id,
                    home_country=country.name,
                    oligarch_type=oligarch_type,
                    capital=capital,
                    risk_tolerance=self.rng.uniform(0.0, 1.0),
                )
                country.resident_oligarch_ids.append(oligarch_id)

    def step(self) -> None:
        """One simulation tick. NOT YET IMPLEMENTED.

        Needs to encode, per oligarch/country, the flows from the
        stock-and-flow design:
          1. return_on_capital        -> Oligarch.capital
          2. capital_flight           -> Oligarch.offshore_capital
          3. capture_contest          -> Oligarch.industry_investments[c][i]
             (input/effort) determines each oligarch's realized share of
             Country.industry_capture[i], via a two-stage rule:
               (a) aggregate captured value in (c,i) saturates toward
                   Country.industry_max_capture[i] as total investment
                   from all oligarchs there rises (diminishing returns,
                   never reaches the ceiling);
               (b) that captured value is split among the investing
                   oligarchs via a Tullock-style contest share:
                   x_o^r / sum_j(x_j^r), r = CAPTURE_CONTEST_INTENSITY.
             Country.domestic_capture and Oligarch.foreign_capture[c]
             are then DEFINED as sums of industry_capture / realized
             shares across industries -- not independently updated.
          4. capture_decay            -> drains industry_capture (and
             therefore domestic_capture/foreign_capture) absent
             continued investment
          5. joint_capture_investment -> pooled capital -> foreign_capture
             (boosted by goodwill: R2 loop)
          6. joint_deals              -> foreign_capture activity -> goodwill
          7. goodwill_decay           -> drains Alliance.goodwill
          8. displacement_grievance   -> when flow #3 causes an
             incumbent's realized industry_capture share to drop because
             a rival increased their own investment, push
             Alliance.goodwill negative (or an equivalent rivalry stock
             for non-allied pairs -- open question, see Alliance
             docstring) rather than only ever decaying passively.
          9. resource_extraction      -> Gordon-Schaefer harvest per
             oligarch per (country, industry): harvest =
             CATCHABILITY_COEFFICIENT * industry_investments[c][i] *
             Country.industry_resource_stock[i]; profit =
             INDUSTRY_PRICE[i] * harvest - industry_investments[c][i].
             Draws down industry_resource_stock[i] by the total harvest
             across all oligarchs there; splits profit between
             Oligarch.capital and Country.wealth_returned_to_population
             (split ratio is the "does the country still enjoy the
             product of its own labor" lever -- not yet defined).
          10. resource_regeneration   -> Country.industry_resource_stock
              recovers toward industry_resource_capacity at
              RESOURCE_REGEN_RATE each tick, UNLESS a tick's extraction
              exceeded RESOURCE_ABUSE_THRESHOLD_FRACTION of the
              remaining stock, in which case industry_resource_capacity
              itself is permanently reduced instead (abuse damages the
              pool; ordinary use does not).
          11. immiseration channel    -> COMPOSITE of (a) the shortfall
              in wealth_returned_to_population relative to extraction
              volume and (b) the resource-consumption rate itself ->
              Country.backlash_pressure. Exact combination (summed vs.
              one scaling the other) not yet decided -- see
              project_overview.txt Section 5.
          12. exclusion channel       -> foreign_capture -> backlash_pressure
              (rival elites shut out of captured channels)
          13. scarcity_channel        -> industry_resource_capacity being
              damaged by abuse (flow #10) -> Country.scarcity_pressure ->
              backlash_pressure. Kept separate from immiseration: a
              country can revolt from pure resource exhaustion even if
              extraction was otherwise being "fairly" shared.
          14. seizure                 -> Oligarch.ruling_status
              (fed by domestic_capture, suppressed by backlash_pressure)
          15. collapse_event          -> backlash_pressure crosses
              collapse_threshold; drains domestic_capture + resident
              oligarchs' liquid capital; sets Country.is_collapsed; every
              resident oligarch's illiquid_assets[this country] are
              zeroed out and their value moves to Country.unclaimed_assets
              per industry.
          16. asset_repurposing       -> OPTIONAL, not guaranteed: any
              oligarch (new entrant, survivor, or foreign investor
              already active in this country) MAY claim unclaimed_assets
              via the SAME Gordon-Schaefer harvest shape used for
              resource extraction, against a different pool: claim =
              ASSET_CLAIM_RATE * industry_investments[c][i] *
              Country.unclaimed_assets[i]. If nobody invests there,
              claim = 0 and the pool sits idle indefinitely -- no forced
              flow, no decay/expiry. Multiple oligarchs investing in the
              same freshly-collapsed industry draw down the same
              shrinking pool automatically (a commons-scramble dynamic,
              no separate contest function needed). Claimed value then
              splits between Oligarch.illiquid_assets (absorbed -- commit
              to running it) and Oligarch.capital (stripped -- convert
              to liquid and move on), linearly by risk_tolerance (see
              Oligarch docstring): risk_tolerance=1 fully absorbs,
              risk_tolerance=0 fully strips.
        """
        raise NotImplementedError


if __name__ == "__main__":
    world = World(n_countries=5, seed=0)
    print(f"{len(world.countries)} countries, {len(world.oligarchs)} oligarchs")
    for cname, country in world.countries.items():
        endowment_str = ", ".join(
            f"{k}={v:.2f}" for k, v in country.industry_endowment.items()
        )
        value_str = ", ".join(
            f"{k}={v:.1f}" for k, v in country.industry_value.items()
        )
        ceiling_str = ", ".join(
            f"{k}={v:.1f}" for k, v in country.industry_max_capture.items()
        )
        print(f"  {cname}: wealth={country.baseline_wealth:.2f}")
        print(f"      endowment:   {endowment_str}")
        print(f"      value:       {value_str}")
        print(f"      max_capture: {ceiling_str}")
    for oid, olig in world.oligarchs.items():
        print(f"  {oid}: type={olig.oligarch_type}, home={olig.home_country}, "
              f"capital={olig.capital:.2f}")