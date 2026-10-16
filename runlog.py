"""
runlog.py -- [C1] per-tick logging for the transnational oligarchic capture model.

One RunLog per run. Arrays are (T, O) for oligarch fields, (T, C) for country fields and (T,) for
world fields, plus static per-oligarch / per-country tables. A run is saved as ONE compressed .npz
(dependency-free); to_frames() rebuilds long-format pandas DataFrames for analysis.

CONVENTIONS
  * t index 0..T-1 corresponds to tick 1..T.
  * Oligarch fields are END-OF-TICK values taken BEFORE the exit check that tick, so an exiting
    oligarch's exit-tick row still holds real (pre-exit) values; from the next tick it is inactive.
  * `active` = on the stage at the START of the tick. For inactive oligarchs the flow fields are 0
    (and E = 0): MASK WITH `active` IN ANALYSIS. Level fields (wealth, offshore) stay meaningful.
  * Loss fields are value DESTROYED to the oligarch: (1) loss_seized, (2) loss_home_hit,
    (3) loss_forced_liq, (4) loss_divest, plus exit_liq_loss at the exit tick. `flight` is a transfer.
    S headline covariate = (1)+(2); (4) is excluded as circular (see project_overview.txt 6.3).
  * E_foreign (PRIMARY), E_all (robustness), E_pos/E_neg: see World.embeddedness.
  * S_home = S[o, home]; S_foreign_mean = simple mean of S over foreign countries;
    S_foreign_pw = presence-weighted mean over foreign countries (NaN if no foreign presence).
  * Country fields: B_pre (backlash after this tick's update, BEFORE the collapse check), B_post
    (after; 0 on collapse ticks), scarcity, collapse flag, capture load, mean resource health,
    abuse_frac (fraction of the 6 industries with abuse > 0), recovery flag.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np

OLIG_BOOL = ("active", "exited_now")
OLIG_FLOAT = ("wealth", "capital", "offshore", "pos_home", "pos_foreign", "active_capital",
              "extraction", "rent",
              "foreign_capture", "domestic_capture", "foreign_havoc", "domestic_havoc",
              "foreign_load", "domestic_load",
              "loss_seized", "loss_home_hit", "loss_forced_liq", "loss_divest", "flight", "exit_liq_loss",
              "E_foreign", "E_all", "E_pos", "E_neg",
              "S_home", "S_foreign_mean", "S_foreign_pw")
COUNTRY_BOOL = ("collapse", "recovery")
COUNTRY_FLOAT = ("B_pre", "B_post", "scarcity", "load", "health_mean", "abuse_frac")
WORLD_FLOAT = ("G_mean", "S_home_mean", "S_foreign_mean", "n_active")


class RunLog:
    def __init__(self, world=None, T: Optional[int] = None, O: Optional[int] = None, C: Optional[int] = None):
        if world is not None:
            T, O, C = world.p.n_ticks, world.O, world.C
        self.T, self.O, self.C = T, O, C
        self.olig: Dict[str, np.ndarray] = {}
        for k in OLIG_BOOL:
            self.olig[k] = np.zeros((T, O), dtype=bool)
        for k in OLIG_FLOAT:
            self.olig[k] = np.full((T, O), np.nan, dtype=np.float32)
        self.country: Dict[str, np.ndarray] = {}
        for k in COUNTRY_BOOL:
            self.country[k] = np.zeros((T, C), dtype=bool)
        for k in COUNTRY_FLOAT:
            self.country[k] = np.full((T, C), np.nan, dtype=np.float32)
        self.world: Dict[str, np.ndarray] = {k: np.full(T, np.nan, dtype=np.float32) for k in WORLD_FLOAT}
        self.static: Dict[str, np.ndarray] = {}
        self.country_static: Dict[str, np.ndarray] = {}
        self.meta: Dict = {}
        self.n_recorded = 0

    # ------------------------------------------------------------------
    def record(self, world) -> None:
        tk, i = world.tick, world.t - 1
        for k in OLIG_BOOL + OLIG_FLOAT:
            self.olig[k][i] = tk[k]
        for k in COUNTRY_BOOL + COUNTRY_FLOAT:
            self.country[k][i] = tk["c_" + k]
        for k in WORLD_FLOAT:
            self.world[k][i] = tk["w_" + k]
        self.n_recorded = world.t

    def finalize(self, world) -> None:
        Os, Cs = world.oligarchs, world.countries
        self.static = dict(
            ruling=np.array([o.is_ruling for o in Os]),
            home=np.array([o.home_country for o in Os]),
            divestment_propensity=np.array([o.divestment_propensity for o in Os], dtype=np.float32),
            initial_wealth=np.array([o.initial_wealth for o in Os], dtype=np.float32),
            initial_active=np.array([o.initial_active for o in Os], dtype=np.float32),
            exit_tick=np.array([np.nan if o.exit_tick is None else o.exit_tick for o in Os], dtype=np.float32),
            exit_transfer=np.array([o.exit_transfer for o in Os], dtype=np.float32),
            cum_seized=np.array([o.seized_total for o in Os], dtype=np.float32),
            cum_home_hit=np.array([o.cum_home_hit for o in Os], dtype=np.float32),
            cum_forced_liq=np.array([o.cum_forced_liq for o in Os], dtype=np.float32),
            cum_divest=np.array([o.cum_divest for o in Os], dtype=np.float32),
            cum_flight=np.array([o.cum_flight for o in Os], dtype=np.float32),
        )
        self.country_static = dict(
            resilience=np.array([c.institutional_resilience for c in Cs], dtype=np.float32),
            threshold=np.array([c.collapse_threshold for c in Cs], dtype=np.float32),
            baseline_wealth=np.array([c.baseline_wealth for c in Cs], dtype=np.float32),
            collapse_count=np.array([c.collapse_count for c in Cs]),
        )

    # ------------------------------------------------------------------
    def save(self, path: str, meta: Optional[Dict] = None) -> None:
        meta = dict(self.meta if meta is None else meta)
        meta.update(T=self.T, O=self.O, C=self.C)
        arrs = {}
        for k, v in self.olig.items():
            arrs["olig__" + k] = v
        for k, v in self.country.items():
            arrs["country__" + k] = v
        for k, v in self.world.items():
            arrs["world__" + k] = v
        for k, v in self.static.items():
            arrs["static__" + k] = v
        for k, v in self.country_static.items():
            arrs["cstatic__" + k] = v
        arrs["meta"] = np.array(json.dumps(meta, default=str))
        np.savez_compressed(path, **arrs)

    @staticmethod
    def load(path: str) -> "RunLog":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        rl = RunLog(T=meta["T"], O=meta["O"], C=meta["C"])
        rl.meta = meta
        for name in z.files:
            if name == "meta":
                continue
            grp, key = name.split("__", 1)
            target = dict(olig=rl.olig, country=rl.country, world=rl.world,
                          static=rl.static, cstatic=rl.country_static)[grp]
            target[key] = z[name]
        rl.n_recorded = meta["T"]
        return rl

    # ------------------------------------------------------------------
    def to_frames(self):
        """Long-format pandas DataFrames: (olig, country, world). Oligarch frame carries the static
        columns (ruling, home, initial_*, exit_tick) on every row."""
        import pandas as pd
        T, O, C = self.T, self.O, self.C
        t = np.repeat(np.arange(1, T + 1), O)
        oid = np.tile(np.arange(O), T)
        d = {"run": self.meta.get("run_id", ""), "t": t, "oligarch": oid}
        for k, v in self.olig.items():
            d[k] = v.reshape(-1)
        olig = pd.DataFrame(d)
        for k, v in self.static.items():
            olig[k] = np.tile(v, T)
        tc = np.repeat(np.arange(1, T + 1), C)
        cd = {"run": self.meta.get("run_id", ""), "t": tc, "country": np.tile(np.arange(C), T)}
        for k, v in self.country.items():
            cd[k] = v.reshape(-1)
        country = pd.DataFrame(cd)
        for k, v in self.country_static.items():
            country[k] = np.tile(v, T)
        world = pd.DataFrame({"run": self.meta.get("run_id", ""), "t": np.arange(1, T + 1),
                              **{k: v for k, v in self.world.items()}})
        return olig, country, world