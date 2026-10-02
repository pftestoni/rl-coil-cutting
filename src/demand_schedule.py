"""Utilities for deterministic demand schedules shared between PPO and MILP."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .demand import DemandConfig, DemandGenerator
from .utils import Catalog


@dataclass
class ScheduleResult:
    dataframe: pd.DataFrame
    counts: np.ndarray
    metadata: dict


class FixedScheduleDemand:
    """Deterministic generator that replays a precomputed schedule."""

    def __init__(self, counts: np.ndarray):
        arr = np.asarray(counts, dtype=np.int32)
        if arr.ndim != 2:
            raise ValueError("Counts array must be 2D [days, skus]")
        self.counts = arr
        self._ptr = 0

    def reset(self) -> None:
        self._ptr = 0

    def sample_day_units(self) -> np.ndarray:
        if self._ptr >= self.counts.shape[0]:
            raise IndexError("Schedule exhausted before horizon completed")
        result = self.counts[self._ptr].copy()
        self._ptr += 1
        return result


def build_schedule(
    catalog: Catalog,
    demand_cfg: DemandConfig,
    capacity_sheets_per_day: int,
    coil_width_mm: int,
    horizon_days: int,
    seed_groups: int,
    seed_demands: int,
) -> ScheduleResult:
    """Generate a deterministic demand schedule enforcing utilisation tolerance."""

    generator = DemandGenerator(
        demand_cfg,
        seed_groups=seed_groups,
        seed_demands=seed_demands,
        n_skus=catalog.widths_mm.shape[0],
        widths_mm=catalog.widths_mm,
        capacity_sheets_per_day=int(capacity_sheets_per_day),
        coil_width_mm=int(coil_width_mm),
    )

    rows = []
    counts: list[np.ndarray] = []

    for day in range(1, horizon_days + 1):
        units = generator.sample_day_units()
        counts.append(units.copy())

        demand_total_units = int(units.sum())
        demand_total_mm = int(np.dot(units, catalog.widths_mm))
        appeared = int(np.count_nonzero(units))

        row = {
            "Dia": day,
            "DemandTotal_units": demand_total_units,
            "DemandTotal_mm": demand_total_mm,
            "DemandAppeared_SKUs": appeared,
        }
        for width, qty in zip(catalog.widths_mm, units):
            row[f"Demanda_{int(width)}mm"] = int(qty)
        rows.append(row)

    schedule_df = pd.DataFrame(rows)
    counts_array = np.asarray(counts, dtype=np.int32)

    target_mm = generator.daily_target_mm
    tolerance = float(demand_cfg.tolerance)
    lower_bound = int(round(target_mm * (1.0 - tolerance)))
    upper_bound = int(round(target_mm * (1.0 + tolerance)))

    totals_mm = schedule_df["DemandTotal_mm"].to_numpy()
    if ((totals_mm < lower_bound) | (totals_mm > upper_bound)).any():
        raise ValueError(
            "Generated schedule violates utilisation tolerance: "
            f"expected [{lower_bound}, {upper_bound}] mm per day, got {totals_mm.min()}-{totals_mm.max()}"
        )

    max_expected_units = math.ceil(upper_bound / int(catalog.widths_mm.min()))
    if int(counts_array.max(initial=0)) > max_expected_units * 2:
        raise ValueError(
            "Generated schedule has implausibly high coil count for at least one SKU. "
            "Check demand configuration."
        )

    metadata = {
        "target_utilization": float(demand_cfg.target_utilization),
        "tolerance": tolerance,
        "daily_target_mm": int(target_mm),
        "lower_bound_mm": int(lower_bound),
        "upper_bound_mm": int(upper_bound),
        "seed_groups": int(seed_groups),
        "seed_demands": int(seed_demands),
        "horizon_days": int(horizon_days),
        "max_units_per_sku": int(counts_array.max(initial=0)),
    }

    return ScheduleResult(schedule_df, counts_array, metadata)


def save_schedule_artifacts(run_dir: str, schedule: ScheduleResult, meta_extra: Optional[dict] = None) -> None:
    """Persist schedule CSV and accompanying metadata JSON."""

    meta = dict(schedule.metadata)
    if meta_extra:
        meta.update(meta_extra)

    schedule_path = f"{run_dir}/demand.csv"
    meta_path = f"{run_dir}/demand_meta.json"

    schedule.dataframe.to_csv(schedule_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_schedule_from_csv(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    demand_cols = [col for col in df.columns if col.startswith("Demanda_")]
    counts = df[demand_cols].to_numpy(dtype=np.int32)
    return df, counts