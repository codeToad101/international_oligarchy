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
    is_collapsed : bool
        Whether a collapse event has been triggered. On collapse,
        domestic_capture (and the triggering oligarch's capital) should
        drain, and the freed assets become available to "new entrant"
        oligarchs -- mechanism not yet implemented.

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
    """

    name: str
    baseline_wealth: float
    institutional_resilience: float
    collapse_threshold: float

    domestic_capture: float = 0.0
    backlash_pressure: float = 0.0
    is_collapsed: bool = False

    # populated by World once oligarchs are assigned to their home country
    resident_oligarch_ids: List[str] = field(default_factory=list)


@dataclass
class Oligarch:
    """An agent representing a single oligarch (civil or ruling).

    Stocks
    ------
    capital : float
        Liquid + illiquid wealth held onshore.
    offshore_capital : float
        Capital moved out of reach of the home country via capital
        flight. One-way in this baseline (nothing flows back in) --
        flagged as a simplification to revisit.
    foreign_capture : Dict[str, float]
        Degree of captured influence held over each foreign country,
        keyed by country name.
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
        under a given backlash_pressure level.
    """

    oligarch_id: str
    home_country: str
    oligarch_type: str
    capital: float
    risk_tolerance: float

    offshore_capital: float = 0.0
    ruling_status: float = 0.0
    foreign_capture: Dict[str, float] = field(default_factory=dict)


@dataclass
class Alliance:
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
            self.countries[name] = Country(
                name=name,
                baseline_wealth=self.rng.uniform(0.2, 1.0),
                institutional_resilience=self.rng.uniform(0.2, 1.0),
                collapse_threshold=self.rng.uniform(0.5, 1.0),
            )

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
          3. capture_investment       -> Country.domestic_capture
          4. capture_decay            -> (drains domestic_capture)
          5. joint_capture_investment -> Oligarch.foreign_capture[target]
          6. joint_deals              -> Alliance.goodwill
          7. goodwill_decay           -> (drains Alliance.goodwill)
          8. immiseration channel     -> Country.backlash_pressure
          9. exclusion channel        -> Country.backlash_pressure
          10. seizure                 -> Oligarch.ruling_status
              (fed by domestic_capture, suppressed by backlash_pressure)
          11. collapse_event          -> drains domestic_capture + capital,
              sets Country.is_collapsed, frees assets for new entrants
        """
        raise NotImplementedError


if __name__ == "__main__":
    world = World(n_countries=5, seed=0)
    print(f"{len(world.countries)} countries, {len(world.oligarchs)} oligarchs")
    for oid, olig in world.oligarchs.items():
        print(f"  {oid}: type={olig.oligarch_type}, home={olig.home_country}, "
              f"capital={olig.capital:.2f}")