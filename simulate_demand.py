import argparse
import os
from typing import Optional

import pandas as pd
import yaml

from src.demand import DemandConfig
from src.demand_schedule import build_schedule
from src.utils import Catalog


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _select_catalog(cfg: dict, env_name: str) -> Catalog:
    paths = cfg["paths"]
    if env_name == "pequeno":
        return Catalog.from_json(paths["catalog_5"], expect_n=5)
    if env_name == "medio":
        return Catalog.from_json(paths["catalog_20"], expect_n=20)
    raise ValueError("--env must be 'pequeno' or 'medio'")


def _build_demand_config(
    cfg: dict,
    target_utilization: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> DemandConfig:
    demand_cfg = cfg["demand"]
    target = target_utilization if target_utilization is not None else float(demand_cfg.get("target_utilization", 0.8))
    tol = tolerance if tolerance is not None else float(demand_cfg.get("tolerance", 0.05))
    return DemandConfig(
        model=str(demand_cfg["model"]),
        group_probs=list(demand_cfg["group_probs"]),
        mean_units_by_group=list(demand_cfg["mean_units_by_group"]),
        nb_dispersion_k=float(demand_cfg["nb_dispersion_k"]),
        noise_sigma_frac=float(demand_cfg["noise_sigma_frac"]),
        cap_fraction_of_capacity_mm=float(demand_cfg.get("cap_fraction_of_capacity_mm", 0.0)),
        target_utilization=float(target),
        tolerance=float(tol),
    )


def _resolve_seed(value: Optional[int], default: int, offset: int) -> int:
    base = default if value is None else int(value)
    return base + offset


def simulate_demand(
    cfg: dict,
    env_name: str,
    days: int,
    seed_groups: Optional[int],
    seed_demands: Optional[int],
    seed_offset: int,
    target_utilization: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> pd.DataFrame:
    catalog = _select_catalog(cfg, env_name)
    env_cfg = cfg["env"]
    dem_cfg = _build_demand_config(cfg, target_utilization=target_utilization, tolerance=tolerance)

    rand_cfg = cfg.get("random", {})
    sg = _resolve_seed(seed_groups, int(rand_cfg.get("seed_groups", 0)), seed_offset)
    sd = _resolve_seed(seed_demands, int(rand_cfg.get("seed_demands", 0)), seed_offset)
    schedule = build_schedule(
        catalog=catalog,
        demand_cfg=dem_cfg,
        capacity_sheets_per_day=int(env_cfg["capacity_sheets_per_day"]),
        coil_width_mm=int(env_cfg["coil_width_mm"]),
        horizon_days=int(days),
        seed_groups=sg,
        seed_demands=sd,
    )

    return schedule.dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate daily demand using the legacy generator")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--env", choices=["pequeno", "medio"], default="medio", help="Environment catalog to use")
    parser.add_argument("--days", type=int, default=7, help="Number of days to simulate")
    parser.add_argument("--seed-groups", type=int, default=None, help="Override base seed for group assignment")
    parser.add_argument("--seed-demands", type=int, default=None, help="Override base seed for demand sampling")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset applied to both seeds")
    parser.add_argument("--target-utilization", type=float, default=None, help="Override target utilisation (default from config)")
    parser.add_argument("--tolerance", type=float, default=None, help="Override utilisation tolerance (default from config)")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path to persist the simulated demand")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    df = simulate_demand(
        cfg,
        env_name=args.env,
        days=int(args.days),
        seed_groups=args.seed_groups,
        seed_demands=args.seed_demands,
        seed_offset=int(args.seed_offset),
        target_utilization=args.target_utilization,
        tolerance=args.tolerance,
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved demand simulation to {args.output}")


if __name__ == "__main__":
    main()
