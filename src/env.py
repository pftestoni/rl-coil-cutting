import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .utils import Catalog, load_patterns_csv, ensure_shapes
from .demand import DemandGenerator, DemandConfig

@dataclass
class EnvConfig:
    horizon_days: int
    coil_width_mm: int
    capacity_sheets_per_day: int
    max_blocks_per_day: int
    action_qty_max: int
    kerf_mm: int
    use_all_patterns: bool
    costs: Dict[str, float]
    reward_scale: float
    inventory_cap_units_per_sku: int | None = None
    inventory_penalty_per_mm: float = 0.0
    auto_scrap_excess: bool = False

class CTLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_cfg: EnvConfig, demand_gen: DemandGenerator, catalog: Catalog,
                 patterns: np.ndarray, rng_seed: int = 0):
        super().__init__()
        self.cfg = env_cfg
        self.demand_gen = demand_gen
        self.catalog = catalog
        self.patterns = ensure_shapes(patterns.astype(np.int32), catalog)
        self.rng = np.random.default_rng(rng_seed)
        self.n = self.catalog.widths_mm.shape[0]
        self.trim_per_pattern = (self.cfg.coil_width_mm - self.patterns @ self.catalog.widths_mm).astype(np.int32)
        keep = self.trim_per_pattern >= 0     # no upper bound on trim; accept all feasible combos
        self.patterns = self.patterns[keep]
        self.trim_per_pattern = self.trim_per_pattern[keep]
        self.num_patterns = self.patterns.shape[0]
        if self.num_patterns == 0:
            raise ValueError("Nenhum padrão viável (scrap >= 0)")

        # Action: [pattern_index, quantity]
        self.action_space = spaces.MultiDiscrete([self.num_patterns, self.cfg.action_qty_max + 1])
        # Observation: [stock(N), demand(N), sheets_left, blocks_left, day_norm]
        obs_dim = 2 * self.n + 3
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.day = 0
        self.stock_units = np.zeros(self.n, dtype=np.int32)
        self.demand_units = np.zeros(self.n, dtype=np.int32)
        self.blocks_used = 0
        self.sheets_left = self.cfg.capacity_sheets_per_day
        self.prod_units = np.zeros(self.n, dtype=np.int32)
        self.scrap_mm_today = 0
        self.pattern_counts_today = np.zeros(self.num_patterns, dtype=np.int32)

    def _obs(self) -> np.ndarray:
        day_norm = self.day / max(1, (self.cfg.horizon_days - 1))
        blocks_left = self.cfg.max_blocks_per_day - self.blocks_used
        v = np.concatenate([
            self.stock_units.astype(np.float32),
            self.demand_units.astype(np.float32),
            np.array([self.sheets_left, blocks_left, day_norm], dtype=np.float32),
        ])
        return v

    def _start_new_day(self):
        self.blocks_used = 0
        self.sheets_left = self.cfg.capacity_sheets_per_day
        self.prod_units[:] = 0
        self.scrap_mm_today = 0
        self.pattern_counts_today[:] = 0
        self.demand_units = self.demand_gen.sample_day_units()

    def _end_day_flow_and_cost(self) -> Dict[str, Any]:
        stock_units_start = self.stock_units.copy()
        available_units = stock_units_start + self.prod_units
        sales_units = np.minimum(available_units, self.demand_units)
        lost_units = self.demand_units - sales_units
        stock_out_units = available_units - sales_units
        served_from_inv_units = np.minimum(stock_units_start, sales_units)
        sheets_used_today = int(self.cfg.capacity_sheets_per_day - self.sheets_left)

        lost_mm = int(np.dot(lost_units, self.catalog.widths_mm))
        scrap_mm = int(self.scrap_mm_today)
        scrap_from_inventory_mm = 0
        excess_units_total = 0
        excess_penalty_cost = 0.0
        if self.cfg.inventory_cap_units_per_sku is not None:
            cap = int(self.cfg.inventory_cap_units_per_sku)
            if cap >= 0:
                excess_units = np.maximum(stock_out_units - cap, 0)
                if np.any(excess_units):
                    excess_units_total = int(np.sum(excess_units))
                    excess_mm = int(np.dot(excess_units, self.catalog.widths_mm))
                    if self.cfg.auto_scrap_excess:
                        stock_out_units = stock_out_units - excess_units
                        scrap_from_inventory_mm = excess_mm
                        scrap_mm += scrap_from_inventory_mm
                    if self.cfg.inventory_penalty_per_mm > 0:
                        excess_penalty_cost = float(self.cfg.inventory_penalty_per_mm) * excess_mm
        served_from_inv_mm = int(np.dot(served_from_inv_units, self.catalog.widths_mm))
        served_mm = int(np.dot(sales_units, self.catalog.widths_mm))
        inv_mm = int(np.dot(stock_out_units, self.catalog.widths_mm))

        c_inv = float(self.cfg.costs["inventory_cost_per_mm"]) * inv_mm
        c_scr = float(self.cfg.costs["scrap_cost_per_mm"]) * scrap_mm
        c_los = float(self.cfg.costs["lost_sale_cost_per_mm"]) * lost_mm
        cost_day = c_inv + c_scr + c_los + excess_penalty_cost

        self.stock_units = stock_out_units
        metrics = {
            "inv_mm": inv_mm,
            "scrap_mm": scrap_mm,
            "lost_mm": lost_mm,
            "cost_day": cost_day,
            "served_mm": served_mm,
            "served_from_inv_mm": served_from_inv_mm,
            "sheets_used": sheets_used_today,
            "sales_units": sales_units.copy(),
            "lost_units": lost_units.copy(),
            "prod_units": self.prod_units.copy(),
            "demand_units": self.demand_units.copy(),
            "stock_units_start": stock_units_start,
            "stock_units_end": stock_out_units.copy(),
            "served_from_inv_units": served_from_inv_units.copy(),
            "pattern_counts": self.pattern_counts_today.copy(),
            "scrap_from_inventory_mm": scrap_from_inventory_mm,
            "inventory_excess_units": excess_units_total,
            "inventory_excess_penalty": excess_penalty_cost,
        }
        self.pattern_counts_today[:] = 0
        return metrics

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.day = 0
        self.stock_units[:] = 0
        reset_fn = getattr(self.demand_gen, "reset", None)
        if callable(reset_fn):
            reset_fn()
        self._start_new_day()
        return self._obs(), {}

    def step(self, action):
        pat_idx, qty = int(action[0]), int(action[1])
        if pat_idx < 0 or pat_idx >= self.num_patterns:
            pat_idx = int(self.rng.integers(0, self.num_patterns))
        qty = max(0, min(qty, self.cfg.action_qty_max, self.sheets_left))

        if qty > 0:
            self.prod_units += self.patterns[pat_idx] * qty
            self.scrap_mm_today += int(self.trim_per_pattern[pat_idx]) * qty
            self.sheets_left -= qty
            self.pattern_counts_today[pat_idx] += qty
        self.blocks_used += 1  # first block doesn't consume setup; modeled via max 6 blocks/day

        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        if self.blocks_used >= self.cfg.max_blocks_per_day or self.sheets_left <= 0:
            metrics = self._end_day_flow_and_cost()
            scale = self.cfg.reward_scale if self.cfg.reward_scale > 0 else 1.0
            reward = -float(metrics["cost_day"]) / scale
            info.update(metrics)
            self.day += 1
            if self.day >= self.cfg.horizon_days:
                terminated = True
            else:
                self._start_new_day()
        return self._obs(), reward, terminated, truncated, info
