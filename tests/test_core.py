"""Core checks for the simulator, the demand generator and the daily MILP baseline."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.demand import DemandConfig, DemandGenerator
from src.env import CTLEnv, EnvConfig
from src.milp import solve_day_milp
from src.utils import Catalog, load_patterns_csv

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
SCENARIOS = [(5, "catalog_5", "patterns_5"), (20, "catalog_20", "patterns_20")]


def load(n, cat_key, pat_key):
    catalog = Catalog.from_json(str(ROOT / CFG["paths"][cat_key]), expect_n=n)
    patterns = load_patterns_csv(str(ROOT / CFG["paths"][pat_key]))
    return catalog, patterns


def demand(catalog, seed=4242):
    d = CFG["demand"]
    cfg = DemandConfig(model=d["model"], group_probs=d["group_probs"], mean_units_by_group=d["mean_units_by_group"],
                       nb_dispersion_k=d["nb_dispersion_k"], noise_sigma_frac=d["noise_sigma_frac"],
                       cap_fraction_of_capacity_mm=d["cap_fraction_of_capacity_mm"],
                       target_utilization=d["target_utilization"], tolerance=d["tolerance"])
    e = CFG["env"]
    return DemandGenerator(cfg, seed_groups=777, seed_demands=seed, n_skus=len(catalog.sku_ids),
                           widths_mm=catalog.widths_mm, capacity_sheets_per_day=e["capacity_sheets_per_day"],
                           coil_width_mm=e["coil_width_mm"])


def env(catalog, patterns, horizon=5):
    e = CFG["env"]
    cfg = EnvConfig(horizon_days=horizon, coil_width_mm=e["coil_width_mm"],
                    capacity_sheets_per_day=e["capacity_sheets_per_day"], max_blocks_per_day=e["max_blocks_per_day"],
                    action_qty_max=e["action_qty_max"], kerf_mm=e["kerf_mm"], use_all_patterns=e["use_all_patterns"],
                    costs=CFG["costs"], reward_scale=e["reward_scale"],
                    inventory_cap_units_per_sku=e["inventory_cap_units_per_sku"],
                    inventory_penalty_per_mm=e["inventory_penalty_per_mm"], auto_scrap_excess=e["auto_scrap_excess"])
    return CTLEnv(cfg, demand(catalog), catalog, patterns, rng_seed=0)


@pytest.mark.parametrize("n,cat_key,pat_key", SCENARIOS)
def test_every_pattern_fits_in_the_coil(n, cat_key, pat_key):
    catalog, patterns = load(n, cat_key, pat_key)
    used = patterns @ catalog.widths_mm
    assert patterns.shape[1] == n
    assert (used <= CFG["env"]["coil_width_mm"]).all()


def test_demand_is_reproducible_and_non_negative():
    catalog, _ = load(*SCENARIOS[1])
    a = demand(catalog, seed=1).sample_day_units()
    b = demand(catalog, seed=1).sample_day_units()
    assert np.array_equal(a, b)
    assert (a >= 0).all() and a.dtype.kind == "i"


@pytest.mark.parametrize("n,cat_key,pat_key", SCENARIOS)
def test_env_conserves_units_and_costs_are_non_negative(n, cat_key, pat_key):
    catalog, patterns = load(n, cat_key, pat_key)
    e = env(catalog, patterns)
    obs, _ = e.reset(seed=0)
    assert obs.shape == e.observation_space.shape
    days = 0
    while days < 3:
        obs, reward, terminated, _, info = e.step(e.action_space.sample())
        assert np.isfinite(obs).all()
        if info:
            days += 1
            assert np.array_equal(info["sales_units"] + info["lost_units"], info["demand_units"])
            assert info["cost_day"] >= 0 and reward <= 0
            assert info["sheets_used"] <= CFG["env"]["capacity_sheets_per_day"]
        if terminated:
            break


def test_milp_meets_demand_when_capacity_allows():
    catalog, patterns = load(*SCENARIOS[0])
    demand_units = np.ones(len(catalog.sku_ids), dtype=np.int32)
    sol = solve_day_milp(stock_units=np.zeros_like(demand_units), demand_units=demand_units, patterns=patterns,
                         widths_mm=catalog.widths_mm, coil_width_mm=CFG["env"]["coil_width_mm"],
                         capacity_sheets=CFG["env"]["capacity_sheets_per_day"],
                         max_blocks=CFG["env"]["max_blocks_per_day"], costs=CFG["costs"],
                         solver_options={"timeLimit": 30, "msg": False})
    assert sol["status"] == "Optimal"
    assert sol["lost_mm"] == 0
    assert np.array_equal(sol["sales_units"], demand_units)
