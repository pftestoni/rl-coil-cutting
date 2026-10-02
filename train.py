import os
import re
import argparse
import time
import shutil
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn

from src.utils import Catalog, load_patterns_csv
from src.env import CTLEnv, EnvConfig
from src.demand import DemandGenerator, DemandConfig
from src.milp import solve_day_milp
from src.io_reports import write_csv, ensure_dir

RUN_NAME_PATTERN = re.compile(r"ppo(\d+)$")
SEED_DIR_PATTERN = re.compile(r"seed-(\d+)$")


def _run_name_to_index(run_name: str) -> int:
    match = RUN_NAME_PATTERN.fullmatch(run_name)
    if not match:
        raise ValueError(f"Run name '{run_name}' is not in the expected format 'ppo<N>'")
    return int(match.group(1))


def _seed_dir_name(run_seed: int) -> str:
    return f"seed-{run_seed}"


def _learn_with_progress(model: PPO, total_timesteps: int, **kwargs):
    try:
        return model.learn(total_timesteps=total_timesteps, progress_bar=True, **kwargs)
    except (TypeError, ImportError):
        print("[warn] progress bar unavailable (missing support); continuing without live bar")
        return model.learn(total_timesteps=total_timesteps, **kwargs)


def _runs_root(env_name: str) -> str:
    return os.path.join("reports", "runs", env_name)


def _seed_runs_dir(env_name: str, run_seed: int) -> str:
    return os.path.join(_runs_root(env_name), _seed_dir_name(run_seed))


def _run_dir_path(env_name: str, run_seed: int, run_name: str) -> str:
    return os.path.join(_seed_runs_dir(env_name, run_seed), run_name)


def _model_path(env_name: str, run_seed: int, run_name: str) -> str:
    return os.path.join("reports", "models", env_name, _seed_dir_name(run_seed), f"{run_name}.zip")


def _vecnorm_path(env_name: str, run_seed: int, run_name: str) -> str:
    return os.path.join(
        "reports",
        "models",
        env_name,
        _seed_dir_name(run_seed),
        f"vecnorm_{run_name}.pkl",
    )


def _run_artifacts_exist(env_name: str, run_seed: int, run_name: str) -> bool:
    run_dir = _run_dir_path(env_name, run_seed, run_name)
    rl_csv = os.path.join(run_dir, "episode_rl.csv")
    model_path = _model_path(env_name, run_seed, run_name)
    vec_path = _vecnorm_path(env_name, run_seed, run_name)
    return os.path.isdir(run_dir) and os.path.isfile(rl_csv) and os.path.isfile(model_path) and os.path.isfile(vec_path)


def _find_incomplete_run(env_name: str, run_seed: int) -> tuple[str | None, int | None]:
    indices = _list_run_indices(env_name, run_seed)
    for idx in indices:
        run_name = f"ppo{idx}"
        if not _run_artifacts_exist(env_name, run_seed, run_name):
            return run_name, idx
    return None, None


def _cleanup_run_artifacts(run_dir: str, model_path: str, vec_path: str) -> None:
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir, ignore_errors=True)
    for art in (model_path, vec_path):
        if os.path.isfile(art):
            try:
                os.remove(art)
            except OSError:
                pass


def _list_run_indices(env_name: str, run_seed: int | None) -> list[int]:
    root = _runs_root(env_name)
    if run_seed is None:
        indices: set[int] = set()
        if not os.path.isdir(root):
            return []
        for entry in os.listdir(root):
            full = os.path.join(root, entry)
            if not os.path.isdir(full):
                continue
            seed_match = SEED_DIR_PATTERN.fullmatch(entry)
            if seed_match:
                indices.update(_list_run_indices(env_name, int(seed_match.group(1))))
            elif RUN_NAME_PATTERN.fullmatch(entry):
                indices.add(_run_name_to_index(entry))
        return sorted(indices)

    seed_dir = _seed_runs_dir(env_name, run_seed)
    if not os.path.isdir(seed_dir):
        return []
    indices: list[int] = []
    for entry in os.listdir(seed_dir):
        if RUN_NAME_PATTERN.fullmatch(entry):
            full = os.path.join(seed_dir, entry)
            if os.path.isdir(full):
                indices.append(_run_name_to_index(entry))
    return sorted(indices)


def _next_run_name(env_name: str, run_seed: int) -> tuple[str, int]:
    indices = _list_run_indices(env_name, run_seed)
    next_idx = indices[-1] + 1 if indices else 1
    run_name = f"ppo{next_idx}"
    return run_name, next_idx


def _latest_run_name(env_name: str, run_seed: int) -> tuple[str | None, int | None]:
    indices = _list_run_indices(env_name, run_seed)
    if not indices:
        return None, None
    last_idx = indices[-1]
    run_name = f"ppo{last_idx}"
    return run_name, last_idx


def _build_demand_columns(widths_mm: np.ndarray) -> list[str]:
    return [f"Demanda_{int(w)}mm" for w in widths_mm]


def _build_stock_columns(widths_mm: np.ndarray) -> list[str]:
    return [f"Estoque_{int(w)}mm" for w in widths_mm]


def _build_rl_fieldnames(widths_mm: np.ndarray) -> list[str]:
    demand_cols = _build_demand_columns(widths_mm)
    stock_cols = _build_stock_columns(widths_mm)
    return [
        "RunName",
        "RunSeed",
        "Dia",
        *demand_cols,
        "DemandTotal_units",
        "DemandTotal_mm",
        "DemandAppeared_SKUs",
        "RL_Patterns",
        *stock_cols,
        "RL_DailyCost",
        "RL_CumCost",
        "RL_DemandMet",
        "SheetsUsed",
        "SetupsUsed",
        "Inv_mm",
        "Scrap_mm",
        "ScrapCost",
        "InvCost",
        "LostSale_mm",
        "LostSaleCost",
        "ServedFromInv",
        "Served_mm",
        "ScrapFromInventory_mm",
        "InventoryExcessUnits",
        "InventoryExcessPenalty",
    ]


def _build_milp_fieldnames(widths_mm: np.ndarray) -> list[str]:
    demand_cols = _build_demand_columns(widths_mm)
    stock_cols = _build_stock_columns(widths_mm)
    return [
        "RunName",
        "RunSeed",
        "Dia",
        *demand_cols,
        "DemandTotal_units",
        "DemandTotal_mm",
        "DemandAppeared_SKUs",
        "MILP_Patterns",
        *stock_cols,
        "MILP_DailyCost",
        "MILP_CumCost",
        "MILP_DemandMet",
        "MILP_SheetsUsed",
        "MILP_SetupsUsed",
        "MILP_Inv_mm",
        "MILP_Scrap_mm",
        "MILP_ScrapCost",
        "MILP_InvCost",
        "MILP_LostSale_mm",
        "MILP_LostSaleCost",
        "MILP_ServedFromInv",
        "MILP_Served_mm",
    ]


def _format_pattern_usage(pattern_counts: dict[int, int]) -> str:
    items = [(idx, qty) for idx, qty in pattern_counts.items() if qty > 0]
    if not items:
        return "N/A"
    items.sort(key=lambda x: x[0])
    return " | ".join(f"{idx}x{qty}" for idx, qty in items)


def _update_total_cost_summary(
    env_name: str,
    run_seed: int,
    run_name: str,
    rl_total: float,
    rl_met_rate: float,
    milp_total: float | None,
    milp_met_rate: float | None,
):
    summary_dir = os.path.join("reports", "runs", env_name, "summary")
    ensure_dir(summary_dir)
    summary_csv = os.path.join(summary_dir, "total_costs.csv")

    milp_total_val = float(milp_total) if milp_total is not None else np.nan
    milp_met_val = float(milp_met_rate) if milp_met_rate is not None else np.nan
    record = {
        "RunName": run_name,
        "RunSeed": int(run_seed),
        "RL_TotalCost": float(rl_total),
        "MILP_TotalCost": milp_total_val,
        "Cost_Delta": float(rl_total) - milp_total_val if not np.isnan(milp_total_val) else np.nan,
        "RL_DemandMet_Rate": float(rl_met_rate),
        "MILP_DemandMet_Rate": milp_met_val,
    }

    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)
        if "RunName" not in summary_df.columns:
            summary_df = summary_df.reset_index(drop=True)
            legacy_names = [f"ppo{i+1}" for i in range(len(summary_df))]
            summary_df.insert(0, "RunName", legacy_names)
        if "RunSeed" not in summary_df.columns:
            summary_df["RunSeed"] = np.nan
        mask_dupe = (summary_df["RunName"] == run_name) & (summary_df["RunSeed"] == run_seed)
        summary_df = summary_df[~mask_dupe]
        summary_df = pd.concat([summary_df, pd.DataFrame([record])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([record])

    summary_df["RunSeed"] = pd.to_numeric(summary_df["RunSeed"], errors="coerce")
    summary_df["RunIndex"] = summary_df["RunName"].apply(_run_name_to_index)
    summary_df.sort_values(["RunSeed", "RunIndex"], inplace=True, na_position="last")
    summary_df["RunSeed"] = summary_df["RunSeed"].astype("Int64")
    summary_df.drop(columns="RunIndex", inplace=True)
    summary_df.to_csv(summary_csv, index=False)

    _plot_env_total_costs(env_name, summary_df, summary_dir)
    seed_summary_dir = os.path.join("reports", "runs", env_name, _seed_dir_name(run_seed), "summary")
    ensure_dir(seed_summary_dir)
    _plot_seed_total_costs(env_name, run_seed, summary_df, seed_summary_dir)


def _plot_env_total_costs(env_name: str, summary_df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["RL_TotalCost"], width, label="RL (PPO)")

    milp_mask = summary_df["MILP_TotalCost"].notna()
    if milp_mask.any():
        ax.bar(x[milp_mask] + width / 2, summary_df.loc[milp_mask, "MILP_TotalCost"], width, label="MILP")

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["RunName"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("Run Name")
    ax.set_ylabel("Total Cumulative Cost")
    ax.set_title(f"Total Cumulative Cost per Run – {env_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "total_costs.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_seed_total_costs(
    env_name: str,
    run_seed: int,
    summary_df: pd.DataFrame,
    out_dir: str,
) -> None:
    seed_df = summary_df[summary_df["RunSeed"] == run_seed]
    if seed_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seed_df))
    width = 0.35
    ax.bar(x - width / 2, seed_df["RL_TotalCost"], width, label="RL (PPO)")
    milp_mask = seed_df["MILP_TotalCost"].notna()
    if milp_mask.any():
        ax.bar(x[milp_mask] + width / 2, seed_df.loc[milp_mask, "MILP_TotalCost"], width, label="MILP")

    ax.set_xticks(x)
    ax.set_xticklabels(seed_df["RunName"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("Run Name")
    ax.set_ylabel("Total Cumulative Cost")
    ax.set_title(f"Seed {run_seed} Total Costs – {env_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "seed_total_costs.png"), bbox_inches="tight")
    plt.close(fig)


def make_env(env_name: str, cfg: dict, seed: int, run_seed_offset: int = 0):
    if env_name not in ("pequeno", "medio"):
        raise ValueError("--env must be 'pequeno' (5 SKUs) or 'medio' (20 SKUs)")

    paths = cfg["paths"]
    if env_name == "pequeno":
        catalog_path = paths["catalog_5"]
        patterns_path = paths["patterns_5"]
        expect_n = 5
    else:
        catalog_path = paths["catalog_20"]
        patterns_path = paths["patterns_20"]
        expect_n = 20

    catalog = Catalog.from_json(catalog_path, expect_n=expect_n)
    patterns = load_patterns_csv(patterns_path)

    env_cfg = EnvConfig(
        horizon_days=int(cfg["env"]["horizon_days"]),
        coil_width_mm=int(cfg["env"]["coil_width_mm"]),
        capacity_sheets_per_day=int(cfg["env"]["capacity_sheets_per_day"]),
        max_blocks_per_day=int(cfg["env"]["max_blocks_per_day"]),
        action_qty_max=int(cfg["env"]["action_qty_max"]),
        kerf_mm=int(cfg["env"]["kerf_mm"]),
        use_all_patterns=bool(cfg["env"]["use_all_patterns"]),
        costs=cfg["costs"],
        reward_scale=float(cfg["env"].get("reward_scale", 10000.0)),
        inventory_cap_units_per_sku=(
            None
            if cfg["env"].get("inventory_cap_units_per_sku") in (None, "null")
            else int(cfg["env"].get("inventory_cap_units_per_sku"))
        ),
        inventory_penalty_per_mm=float(cfg["env"].get("inventory_penalty_per_mm", 0.0)),
        auto_scrap_excess=bool(cfg["env"].get("auto_scrap_excess", False)),
    )

    demand_resource: dict[str, object] = {"mode": "legacy"}

    base_seed_groups = int(cfg["random"]["seed_groups"]) + run_seed_offset
    base_seed_demands = int(cfg["random"]["seed_demands"]) + run_seed_offset

    dem_cfg = DemandConfig(
        model=str(cfg["demand"]["model"]),
        group_probs=list(cfg["demand"]["group_probs"]),
        mean_units_by_group=list(cfg["demand"]["mean_units_by_group"]),
        nb_dispersion_k=float(cfg["demand"]["nb_dispersion_k"]),
        noise_sigma_frac=float(cfg["demand"]["noise_sigma_frac"]),
        cap_fraction_of_capacity_mm=float(cfg["demand"].get("cap_fraction_of_capacity_mm", 0.0)),
        target_utilization=float(cfg["demand"].get("target_utilization", 0.8)),
        tolerance=float(cfg["demand"].get("tolerance", 0.05)),
    )
    demand_resource.update(
        {
            "config": dem_cfg,
            "base_seed_groups": base_seed_groups,
            "base_seed_demands": base_seed_demands,
        }
    )

    def _factory_builder(seed_offset: int = 0):
        def _factory():
            gen = DemandGenerator(
                dem_cfg,
                seed_groups=base_seed_groups + seed_offset,
                seed_demands=base_seed_demands + seed_offset,
                n_skus=catalog.widths_mm.shape[0],
                widths_mm=catalog.widths_mm,
                capacity_sheets_per_day=env_cfg.capacity_sheets_per_day,
                coil_width_mm=env_cfg.coil_width_mm,
            )
            e = CTLEnv(env_cfg, gen, catalog, patterns, rng_seed=seed + seed_offset)
            return Monitor(e)

        return _factory

    return _factory_builder, catalog, patterns, env_cfg, demand_resource


def rollout_rl_episode(
    model: PPO,
    vecnorm: VecNormalize,
    env_cfg: EnvConfig,
    catalog: Catalog,
    costs_cfg: dict,
    run_seed: int,
    run_name: str,
):
    widths = catalog.widths_mm.astype(int)
    demand_cols = _build_demand_columns(widths)
    stock_cols = _build_stock_columns(widths)

    reset_out = vecnorm.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    day = 0
    rows = []
    cum_cost = 0.0
    while not done and day < env_cfg.horizon_days:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action)
            if action_arr.ndim == 1:
                act = action_arr
            elif action_arr.ndim >= 2:
                act = action_arr[0]
            else:
                act = action_arr.reshape(-1)
            pat_idx = int(act[0])
            qty = int(act[1]) if act.shape[0] > 1 else 0

            step_out = vecnorm.step(action)
            if len(step_out) == 5:
                obs, rew, term, trunc, info = step_out
                done_flags = np.logical_or(term, trunc)
            else:
                obs, rew, done_flags, info = step_out
                term = done_flags
                trunc = np.zeros_like(done_flags, dtype=bool)

            i0 = info[0]
            if "cost_day" in i0:
                demand_units = i0["demand_units"].astype(int)
                stock_units_end = i0["stock_units_end"].astype(int)
                demand_mm = (demand_units * widths).astype(int)
                demand_total_units = int(demand_units.sum())
                demand_total_mm = int(demand_mm.sum())
                demand_appeared = int(np.count_nonzero(demand_units))

                inv_mm = int(i0["inv_mm"])
                scrap_mm = int(i0["scrap_mm"])
                lost_mm = int(i0["lost_mm"])
                served_from_inv_mm = int(i0.get("served_from_inv_mm", 0))
                served_mm = int(i0.get("served_mm", demand_total_mm - lost_mm))
                sheets_used = int(i0.get("sheets_used", 0))
                pattern_counts = i0.get("pattern_counts")
                if pattern_counts is not None:
                    pattern_counts = np.asarray(pattern_counts, dtype=int)
                    pattern_usage_dict = {idx: int(cnt) for idx, cnt in enumerate(pattern_counts) if cnt > 0}
                    setups_used_today = int(np.count_nonzero(pattern_counts))
                else:
                    pattern_usage_dict = {}
                    setups_used_today = 0

                cost_day = float(i0["cost_day"])
                cum_cost += cost_day

                scrap_cost = float(costs_cfg["scrap_cost_per_mm"]) * scrap_mm
                inv_cost = float(costs_cfg["inventory_cost_per_mm"]) * inv_mm
                lost_cost = float(costs_cfg["lost_sale_cost_per_mm"]) * lost_mm

                row = {
                    "RunName": run_name,
                    "RunSeed": run_seed,
                    "Dia": day + 1,
                    "DemandTotal_units": demand_total_units,
                    "DemandTotal_mm": demand_total_mm,
                    "DemandAppeared_SKUs": demand_appeared,
                    "RL_Patterns": _format_pattern_usage(pattern_usage_dict),
                    "RL_DailyCost": cost_day,
                    "RL_CumCost": cum_cost,
                    "RL_DemandMet": "Y" if lost_mm == 0 else "N",
                    "SheetsUsed": sheets_used,
                    "SetupsUsed": setups_used_today,
                    "Inv_mm": inv_mm,
                    "Scrap_mm": scrap_mm,
                    "ScrapCost": scrap_cost,
                    "InvCost": inv_cost,
                    "LostSale_mm": lost_mm,
                    "LostSaleCost": lost_cost,
                    "ServedFromInv": served_from_inv_mm,
                    "Served_mm": served_mm,
                    "ScrapFromInventory_mm": int(i0.get("scrap_from_inventory_mm", 0)),
                    "InventoryExcessUnits": int(i0.get("inventory_excess_units", 0)),
                    "InventoryExcessPenalty": float(i0.get("inventory_excess_penalty", 0.0)),
                }

                for col_name, value in zip(demand_cols, demand_units):
                    row[col_name] = int(value)
                for col_name, value in zip(stock_cols, stock_units_end):
                    row[col_name] = int(value)

                rows.append(row)
                day += 1
                break

            if bool(done_flags[0]):
                break
        done = bool(done_flags[0])

    # Ensure columns for CSV have consistent ordering
    fieldnames = _build_rl_fieldnames(widths)
    ordered_rows = []
    for r in rows:
        ordered = {key: r.get(key) for key in fieldnames}
        ordered_rows.append(ordered)
    return ordered_rows


def rebuild_demands_and_run_milp(
    env_name: str,
    cfg: dict,
    env_cfg: EnvConfig,
    catalog: Catalog,
    patterns: np.ndarray,
    seed: int,
    run_name: str,
    demand_resource: dict,
    run_offset: int,
):
    widths = catalog.widths_mm.astype(int)
    demand_cols = _build_demand_columns(widths)
    stock_cols = _build_stock_columns(widths)

    demand_sequence: list[np.ndarray] = []

    dem_cfg = demand_resource.get("config")  # type: ignore[arg-type]
    if dem_cfg is None:
        dem_cfg = DemandConfig(
            model=str(cfg["demand"]["model"]),
            group_probs=list(cfg["demand"]["group_probs"]),
            mean_units_by_group=list(cfg["demand"]["mean_units_by_group"]),
            nb_dispersion_k=float(cfg["demand"]["nb_dispersion_k"]),
            noise_sigma_frac=float(cfg["demand"]["noise_sigma_frac"]),
            cap_fraction_of_capacity_mm=float(cfg["demand"].get("cap_fraction_of_capacity_mm", 0.0)),
            target_utilization=float(cfg["demand"].get("target_utilization", 0.8)),
            tolerance=float(cfg["demand"].get("tolerance", 0.05)),
        )
    base_seed_groups = int(cfg["random"]["seed_groups"]) + run_offset
    base_seed_demands = int(cfg["random"]["seed_demands"]) + run_offset
    gen = DemandGenerator(
        dem_cfg,
        seed_groups=base_seed_groups,
        seed_demands=base_seed_demands,
        n_skus=catalog.widths_mm.shape[0],
        widths_mm=catalog.widths_mm,
        capacity_sheets_per_day=env_cfg.capacity_sheets_per_day,
        coil_width_mm=env_cfg.coil_width_mm,
    )
    for _ in range(env_cfg.horizon_days):
        demand_sequence.append(gen.sample_day_units().astype(np.int32))

    stock_units = np.zeros_like(widths, dtype=np.int32)
    rows = []
    cum_cost = 0.0

    milp_cfg = cfg.get("milp", {})
    solver_options: dict[str, object] = {}
    time_limit = milp_cfg.get("cbc_time_limit_sec")
    if time_limit is not None:
        solver_options["timeLimit"] = float(time_limit)
    threads = milp_cfg.get("cbc_threads")
    if threads is not None:
        solver_options["threads"] = int(threads)
    solver_options["msg"] = bool(milp_cfg.get("cbc_msg", False))

    for day_idx, day_dem in enumerate(demand_sequence):
        start_solve = time.perf_counter()
        day_units = day_dem.astype(np.int32)
        print(
            f"[milp] Day {day_idx + 1}/{env_cfg.horizon_days} | status=solving...",
            flush=True,
        )
        sol = solve_day_milp(
            stock_units,
            day_units,
            patterns,
            catalog.widths_mm,
            env_cfg.coil_width_mm,
            env_cfg.capacity_sheets_per_day,
            env_cfg.max_blocks_per_day,
            env_cfg.costs,
            solver_options=solver_options,
            inventory_cap_units=env_cfg.inventory_cap_units_per_sku,
            inventory_penalty_per_mm=env_cfg.inventory_penalty_per_mm,
            auto_scrap_excess=env_cfg.auto_scrap_excess,
        )
        solve_time = time.perf_counter() - start_solve
        inv_mm = int(sol["inv_mm"])
        scrap_mm = int(sol["scrap_mm"])
        lost_mm = int(sol["lost_mm"])
        cost_day = float(sol["cost_day"])
        status = str(sol.get("status", "unknown"))
        print(
            f"[milp] Day {day_idx + 1}/{env_cfg.horizon_days} | status={status} | cost={cost_day:,.2f} | "
            f"inv={inv_mm}mm | scrap={scrap_mm}mm | lost={lost_mm}mm | time={solve_time:.2f}s",
            flush=True,
        )

        sheets_used = int(sol["x"].sum())
        setups_used = int(np.count_nonzero(sol["x"]))
        prod_units = sol["prod_units"].astype(int)

        sales_units = sol["sales_units"].astype(int)
        lost_units = sol["lost_units"].astype(int)
        stock_units_end = sol["stock_units_end"].astype(int)
        stock_units_start = stock_units.copy()
        served_from_inv_units = np.minimum(stock_units_start, sales_units)

        demand_mm = (day_units * widths).astype(int)
        demand_total_units = int(day_units.sum())
        demand_total_mm = int(demand_mm.sum())
        demand_appeared = int(np.count_nonzero(day_units))

        inv_mm = int(sol["inv_mm"])
        scrap_mm = int(sol["scrap_mm"])
        lost_mm = int(sol["lost_mm"])
        scrap_from_inventory_mm = int(sol.get("scrap_from_inventory_mm", 0))
        inventory_excess_units = int(sol.get("inventory_excess_units", 0))
        inventory_excess_penalty = float(sol.get("inventory_excess_penalty", 0.0))
        served_mm = int(np.dot(sales_units, widths))
        served_from_inv_mm = int(np.dot(served_from_inv_units, widths))

        cost_day = float(sol["cost_day"])
        cum_cost += cost_day

        scrap_cost = float(env_cfg.costs["scrap_cost_per_mm"]) * scrap_mm
        inv_cost = float(env_cfg.costs["inventory_cost_per_mm"]) * inv_mm
        lost_cost = float(env_cfg.costs["lost_sale_cost_per_mm"]) * lost_mm

        pattern_counts = {idx: int(qty) for idx, qty in enumerate(sol["x"]) if int(qty) > 0}

        row = {
            "RunName": run_name,
            "RunSeed": seed,
            "Dia": day_idx + 1,
            "DemandTotal_units": demand_total_units,
            "DemandTotal_mm": demand_total_mm,
            "DemandAppeared_SKUs": demand_appeared,
            "MILP_Patterns": _format_pattern_usage(pattern_counts),
            "MILP_DailyCost": cost_day,
            "MILP_CumCost": cum_cost,
            "MILP_DemandMet": "Y" if lost_mm == 0 else "N",
            "MILP_SheetsUsed": sheets_used,
            "MILP_SetupsUsed": setups_used,
            "MILP_Inv_mm": inv_mm,
            "MILP_Scrap_mm": scrap_mm,
            "MILP_ScrapCost": scrap_cost,
            "MILP_InvCost": inv_cost,
            "MILP_LostSale_mm": lost_mm,
            "MILP_LostSaleCost": lost_cost,
            "MILP_ServedFromInv": served_from_inv_mm,
            "MILP_Served_mm": served_mm,
            "MILP_ScrapFromInventory_mm": scrap_from_inventory_mm,
            "MILP_InventoryExcessUnits": inventory_excess_units,
            "MILP_InventoryExcessPenalty": inventory_excess_penalty,
        }

        for col_name, value in zip(demand_cols, day_units):
            row[col_name] = int(value)
        for col_name, value in zip(stock_cols, stock_units_end):
            row[col_name] = int(value)

        rows.append(row)
        stock_units = stock_units_end

    fieldnames = _build_milp_fieldnames(widths)
    ordered_rows = []
    for r in rows:
        ordered = {key: r.get(key) for key in fieldnames}
        ordered_rows.append(ordered)
    return ordered_rows


def make_plots(out_dir: str, compare_csv_path: str, costs_cfg: dict):
    ensure_dir(out_dir)
    comp_df = pd.read_csv(compare_csv_path)
    day_col = "day" if "day" in comp_df.columns else "Dia"
    days = comp_df[day_col]

    inv_rate = 1.0
    scr_rate = 35.0
    los_rate = 700.0

    rl_inv_costs = comp_df["inv_mm_rl"] * inv_rate
    rl_scr_costs = comp_df["scrap_mm_rl"] * scr_rate
    rl_lost_costs = comp_df["lost_mm_rl"] * los_rate

    ml_inv_costs = comp_df["inv_mm_milp"] * inv_rate
    ml_scr_costs = comp_df["scrap_mm_milp"] * scr_rate
    ml_lost_costs = comp_df["lost_mm_milp"] * los_rate

    # cum_cost.png
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, comp_df["cum_cost_rl"], label="RL (PPO)")
    ax.plot(days, comp_df["cum_cost_milp"], label="MILP")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Cost")
    ax.set_title("Cumulative Cost – RL vs MILP")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "cum_cost.png"), bbox_inches="tight")
    plt.close(fig)

    # avg_cost_components.png
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.35
    rl_values = [rl_inv_costs.mean(), rl_scr_costs.mean(), rl_lost_costs.mean()]
    ml_values = [ml_inv_costs.mean(), ml_scr_costs.mean(), ml_lost_costs.mean()]
    ax.bar(x - width / 2, rl_values, width, label="RL (PPO)")
    ax.bar(x + width / 2, ml_values, width, label="MILP")
    ax.set_xticks(x)
    ax.set_xticklabels(["Inventory", "Scrap", "Lost"])
    ax.set_ylabel("Average Daily Cost")
    ax.set_title("Average Daily Cost Components")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "avg_cost_components.png"), bbox_inches="tight")
    plt.close(fig)

    # daily_costs.png
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, comp_df["cost_day_rl"], label="RL (PPO)")
    ax.plot(days, comp_df["cost_day_milp"], label="MILP")
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Cost")
    ax.set_title("Daily Costs – RL vs MILP")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "daily_costs.png"), bbox_inches="tight")
    plt.close(fig)

    # cum_cost_diff.png
    cum_diff = comp_df["cum_cost_rl"] - comp_df["cum_cost_milp"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, cum_diff, label="RL - MILP")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Cost Difference")
    ax.set_title("Cumulative Cost Difference (RL - MILP)")
    fig.savefig(os.path.join(out_dir, "cum_cost_diff.png"), bbox_inches="tight")
    plt.close(fig)

    # rolling_avg_cost_30d.png
    rl_roll = comp_df["cost_day_rl"].rolling(window=30, min_periods=1).mean()
    ml_roll = comp_df["cost_day_milp"].rolling(window=30, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, rl_roll, label="RL (30d avg)")
    ax.plot(days, ml_roll, label="MILP (30d avg)")
    ax.set_xlabel("Day")
    ax.set_ylabel("30-Day Rolling Average Cost")
    ax.set_title("30-Day Rolling Average Daily Cost")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "rolling_avg_cost_30d.png"), bbox_inches="tight")
    plt.close(fig)

    # daily_components_rl.png
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(days, rl_inv_costs, label="Inventory")
    ax.bar(days, rl_scr_costs, bottom=rl_inv_costs, label="Scrap")
    ax.bar(days, rl_lost_costs, bottom=rl_inv_costs + rl_scr_costs, label="Lost Sales")
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Cost")
    ax.set_title("RL Daily Cost Components")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "daily_components_rl.png"), bbox_inches="tight")
    plt.close(fig)

    # daily_components_milp.png
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(days, ml_inv_costs, label="Inventory")
    ax.bar(days, ml_scr_costs, bottom=ml_inv_costs, label="Scrap")
    ax.bar(days, ml_lost_costs, bottom=ml_inv_costs + ml_scr_costs, label="Lost Sales")
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Cost")
    ax.set_title("MILP Daily Cost Components")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "daily_components_milp.png"), bbox_inches="tight")
    plt.close(fig)

    # cumulative_components_rl.png
    fig, ax = plt.subplots(figsize=(10, 6))
    rl_totals = [rl_inv_costs.sum(), rl_scr_costs.sum(), rl_lost_costs.sum()]
    ax.bar(["Inventory", "Scrap", "Lost"], rl_totals, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Total Cost")
    ax.set_title("RL Cumulative Cost Components")
    fig.savefig(os.path.join(out_dir, "cumulative_components_rl.png"), bbox_inches="tight")
    plt.close(fig)

    # cumulative_components_milp.png
    fig, ax = plt.subplots(figsize=(10, 6))
    ml_totals = [ml_inv_costs.sum(), ml_scr_costs.sum(), ml_lost_costs.sum()]
    ax.bar(["Inventory", "Scrap", "Lost"], ml_totals, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Total Cost")
    ax.set_title("MILP Cumulative Cost Components")
    fig.savefig(os.path.join(out_dir, "cumulative_components_milp.png"), bbox_inches="tight")
    plt.close(fig)

    # lost_vs_inventory_scatter.png
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(comp_df["inv_mm_rl"], comp_df["lost_mm_rl"], alpha=0.6, label="RL (PPO)")
    ax.scatter(comp_df["inv_mm_milp"], comp_df["lost_mm_milp"], alpha=0.6, label="MILP")
    ax.set_xlabel("Inventory (mm)")
    ax.set_ylabel("Lost Sales (mm)")
    ax.set_title("Inventory vs Lost Sales")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "lost_vs_inventory_scatter.png"), bbox_inches="tight")
    plt.close(fig)

    # inventory_vs_scrap_tradeoff.png
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(comp_df["inv_mm_rl"], comp_df["scrap_mm_rl"], alpha=0.6, label="RL (PPO)")
    ax.scatter(comp_df["inv_mm_milp"], comp_df["scrap_mm_milp"], alpha=0.6, label="MILP")
    ax.set_xlabel("Inventory (mm)")
    ax.set_ylabel("Scrap (mm)")
    ax.set_title("Inventory vs Scrap Trade-off")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "inventory_vs_scrap_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def train_and_eval(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Global seed
    seed_global = int(cfg["random"]["seed_global"])
    run_seed = int(args.seed) if args.seed is not None else seed_global

    runs_root = _runs_root(args.env)
    ensure_dir(runs_root)
    models_root = os.path.join("reports", "models", args.env)
    ensure_dir(models_root)
    runtime_cfg = cfg.get("runtime", {})

    if args.mode == "train":
        if args.run_name is not None:
            raise ValueError("--run-name should not be provided in train mode; runs are auto-numbered.")
        incompleted_run_name, incompleted_idx = _find_incomplete_run(args.env, run_seed)
        if incompleted_run_name is not None and incompleted_idx is not None:
            run_name, run_index = incompleted_run_name, incompleted_idx
            run_offset = run_index - 1
            resume_incomplete = True
            print(f"[resume] Detected incomplete run {run_name}; restarting this run")
        else:
            run_name, run_index = _next_run_name(args.env, run_seed)
            run_offset = run_index - 1
            resume_incomplete = False
    else:
        if args.seed is None:
            raise ValueError("Evaluation mode requires --seed to select the run directory.")
        if args.run_name is not None:
            run_name = args.run_name.strip()
            run_index = _run_name_to_index(run_name)
        else:
            run_name, run_index = _latest_run_name(args.env, run_seed)
            if run_name is None:
                raise FileNotFoundError(
                    f"No PPO runs found for environment '{args.env}' with seed {run_seed}."
                )
        resume_incomplete = False
        run_offset = run_index - 1

    np.random.seed(run_seed)

    seed_dir_name = _seed_dir_name(run_seed)
    seed_dir = os.path.join(runs_root, seed_dir_name)
    ensure_dir(seed_dir)
    runs_dir = os.path.join(seed_dir, run_name)
    if args.mode == "train":
        if resume_incomplete:
            if os.path.isdir(runs_dir):
                shutil.rmtree(runs_dir)
        elif os.path.exists(runs_dir):
            raise FileExistsError(
                f"Run directory '{runs_dir}' already exists. Delete it or run evaluation mode."
            )
    else:
        if not os.path.exists(runs_dir):
            raise FileNotFoundError(
                f"Run directory '{runs_dir}' not found. Provide the correct --seed/--run-name pair."
            )
    ensure_dir(runs_dir)

    print(
        f"[run] env={args.env} | mode={args.mode} | run={run_name} | seed={run_seed} | output={runs_dir}"
    )

    plot_every = max(0, int(args.plot_milp_every or 0))
    freq_hit = plot_every > 0 and args.mode == "train" and (run_index % plot_every == 0)
    do_eval_milp = bool(args.eval_milp or freq_hit)
    do_make_plots = bool(args.make_plots or freq_hit)

    milp_disabled = bool(runtime_cfg.get("disable_milp", False))
    if milp_disabled:
        if do_eval_milp or do_make_plots:
            print(
                f"[milp-disabled] Runtime configuration disabled MILP for {run_name}; skipping baseline/plots"
            )
        freq_hit = False
        do_eval_milp = False
        do_make_plots = False
    elif freq_hit:
        print(
            f"[milp-frequency] Run {run_name} meets {plot_every}-run interval; enabling MILP and plots"
        )

    models_seed_dir = os.path.join(models_root, seed_dir_name)
    ensure_dir(models_seed_dir)

    cleanup_on_interrupt = bool(runtime_cfg.get("cleanup_on_interrupt", False))

    if args.mode == "train" and resume_incomplete:
        for path in (_model_path(args.env, run_seed, run_name), _vecnorm_path(args.env, run_seed, run_name)):
            if os.path.exists(path):
                os.remove(path)
    current_model_path = os.path.join(models_seed_dir, f"{run_name}.zip")
    current_vecnorm_path = os.path.join(models_seed_dir, f"vecnorm_{run_name}.pkl")
    latest_model_alias_seed = os.path.join(models_seed_dir, "latest.zip")
    latest_vecnorm_alias_seed = os.path.join(models_seed_dir, "vecnorm_latest.pkl")
    latest_model_alias_env = os.path.join("reports", "models", f"ppo_{args.env}_seed-{run_seed}.zip")
    latest_vecnorm_alias_env = os.path.join(
        "reports", "models", f"vecnorm_{args.env}_seed-{run_seed}.pkl"
    )
    latest_model_alias_global = os.path.join("reports", "models", f"ppo_{args.env}.zip")
    latest_vecnorm_alias_global = os.path.join("reports", "models", f"vecnorm_{args.env}.pkl")

    error_cleanup_enabled = args.mode == "train" and cleanup_on_interrupt
    cleanup_targets = (runs_dir, current_model_path, current_vecnorm_path)

    try:
        prev_model_path = None
        prev_vecnorm_path = None
        if args.mode == "train":
            prev_idx = run_index - 1
            while prev_idx >= 1:
                candidate_name = f"ppo{prev_idx}"
                candidate_model = os.path.join(models_seed_dir, f"{candidate_name}.zip")
                candidate_vec = os.path.join(models_seed_dir, f"vecnorm_{candidate_name}.pkl")
                if os.path.exists(candidate_model) and os.path.exists(candidate_vec):
                    prev_model_path = candidate_model
                    prev_vecnorm_path = candidate_vec
                    break
                prev_idx -= 1

        if args.mode == "train" and args.fresh_model and prev_model_path is not None:
            print("[train] Fresh model requested; ignoring previous checkpoints for this run")
            prev_model_path = None
            prev_vecnorm_path = None

        env_builder, catalog, patterns, env_cfg, demand_resource = make_env(args.env, cfg, run_seed, run_offset)

        ppo_cfg = cfg.get("ppo", {}) if isinstance(cfg.get("ppo"), dict) else {}

        n_envs_overrides = runtime_cfg.get("n_envs_by_env", {}) if isinstance(runtime_cfg.get("n_envs_by_env"), dict) else {}
        default_n_envs_cfg = runtime_cfg.get("default_n_envs")

        if args.mode == "train":
            cpu_total = os.cpu_count() or 1
            if args.n_envs is not None:
                n_envs = max(1, int(args.n_envs))
            else:
                cfg_override = n_envs_overrides.get(args.env)
                if cfg_override is not None:
                    n_envs = max(1, int(cfg_override))
                elif default_n_envs_cfg is not None:
                    n_envs = max(1, int(default_n_envs_cfg))
                else:
                    n_envs = max(1, cpu_total - 1)
            print(f"[train] n_envs={n_envs} | timesteps={int(args.timesteps)} | progress_bar=on")
            env_fns = [env_builder(i) for i in range(n_envs)]
            if n_envs > 1:
                venv = SubprocVecEnv(env_fns)
            else:
                venv = DummyVecEnv(env_fns)
            if prev_vecnorm_path is not None:
                vecnorm_train = VecNormalize.load(prev_vecnorm_path, venv)
                vecnorm_train.training = True
                vecnorm_train.norm_reward = True
                vecnorm_train.clip_reward = 10.0
            else:
                vecnorm_train = VecNormalize(
                    venv,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0,
                )

            try:
                if prev_model_path is not None:
                    model_train = PPO.load(prev_model_path, env=vecnorm_train, device="auto")
                    new_lr = float(ppo_cfg.get("learning_rate", model_train.learning_rate))
                    model_train.learning_rate = new_lr
                    model_train.lr_schedule = get_schedule_fn(new_lr)
                    if hasattr(model_train, "policy") and hasattr(model_train.policy, "optimizer"):
                        for group in model_train.policy.optimizer.param_groups:
                            group["lr"] = new_lr

                    new_clip = float(ppo_cfg.get("clip_range", 0.2))
                    model_train.clip_range = get_schedule_fn(new_clip)
                    if hasattr(model_train, "clip_range_vf") and model_train.clip_range_vf is not None:
                        model_train.clip_range_vf = get_schedule_fn(float(ppo_cfg.get("clip_range_vf", new_clip)))

                    model_train.gamma = float(ppo_cfg.get("gamma", model_train.gamma))
                    model_train.ent_coef = float(ppo_cfg.get("ent_coef", model_train.ent_coef))
                    model_train.n_steps = int(ppo_cfg.get("n_steps", model_train.n_steps))
                    model_train.batch_size = int(ppo_cfg.get("batch_size", model_train.batch_size))
                    model_train.n_epochs = int(ppo_cfg.get("n_epochs", model_train.n_epochs))
                    _learn_with_progress(
                        model_train,
                        int(args.timesteps),
                        reset_num_timesteps=False,
                    )
                else:
                    model_train = PPO(
                        "MlpPolicy",
                        vecnorm_train,
                        verbose=1,
                        gamma=float(ppo_cfg.get("gamma", 0.995)),
                        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
                        n_steps=int(ppo_cfg.get("n_steps", 4096)),
                        batch_size=int(ppo_cfg.get("batch_size", 4096)),
                        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
                        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
                        ent_coef=float(ppo_cfg.get("ent_coef", 0.005)),
                        seed=run_seed,
                    )
                    _learn_with_progress(model_train, int(args.timesteps))

                ensure_dir("reports/models")
                model_train.save(current_model_path)
                vecnorm_train.save(current_vecnorm_path)
                model_train.save(latest_model_alias_seed)
                vecnorm_train.save(latest_vecnorm_alias_seed)
                model_train.save(latest_model_alias_env)
                vecnorm_train.save(latest_vecnorm_alias_env)
                model_train.save(latest_model_alias_global)
                vecnorm_train.save(latest_vecnorm_alias_global)
            finally:
                venv.close()

            eval_env = DummyVecEnv([env_builder(0)])
            eval_vecnorm = VecNormalize.load(current_vecnorm_path, eval_env)
            eval_vecnorm.training = False
            eval_vecnorm.norm_reward = False
            eval_model = PPO.load(current_model_path)
            eval_model.policy.set_training_mode(False)
        else:
            if not os.path.exists(current_vecnorm_path):
                raise FileNotFoundError(f"VecNormalize stats not found for run '{run_name}'.")
            if not os.path.exists(current_model_path):
                raise FileNotFoundError(f"Model checkpoint not found for run '{run_name}'.")
            print(f"[eval] Loading existing policy for run={run_name} (seed={run_seed})")
            eval_env = DummyVecEnv([env_builder(0)])
            eval_vecnorm = VecNormalize.load(current_vecnorm_path, eval_env)
            eval_vecnorm.training = False
            eval_vecnorm.norm_reward = False
            eval_model = PPO.load(current_model_path)
            eval_model.policy.set_training_mode(False)

        print("[eval] Rolling out RL policy for reporting")
        rl_rows = rollout_rl_episode(eval_model, eval_vecnorm, env_cfg, catalog, cfg["costs"], run_seed, run_name)
        rl_fieldnames = _build_rl_fieldnames(catalog.widths_mm)
        write_csv(
            os.path.join(runs_dir, "episode_rl.csv"),
            rl_rows,
            rl_fieldnames,
        )

        rl_df = pd.DataFrame(rl_rows)
        rl_total = float(rl_df["RL_CumCost"].iloc[-1])
        rl_met_series = rl_df["RL_DemandMet"]
        rl_met_mask = rl_met_series.isin(["Y", "y", True])
        rl_met_rate = float(rl_met_mask.mean())

        milp_rows = None
        milp_df = None
        milp_total = None
        milp_met_rate = None

        if do_eval_milp:
            print("[milp] Solving MILP baseline against the same demands")
            milp_rows = rebuild_demands_and_run_milp(
                args.env,
                cfg,
                env_cfg,
                catalog,
                patterns,
                run_seed,
                run_name,
                demand_resource,
                run_offset,
            )
            milp_fieldnames = _build_milp_fieldnames(catalog.widths_mm)
            write_csv(
                os.path.join(runs_dir, "episode_milp.csv"),
                milp_rows,
                milp_fieldnames,
            )

            milp_df = pd.DataFrame(milp_rows)
            milp_total = float(milp_df["MILP_CumCost"].iloc[-1])
            milp_met_series = milp_df["MILP_DemandMet"]
            milp_met_mask = milp_met_series.isin(["Y", "y", True])
            milp_met_rate = float(milp_met_mask.mean())

            inv_rate = float(cfg["costs"]["inventory_cost_per_mm"])
            scr_rate = float(cfg["costs"]["scrap_cost_per_mm"])
            los_rate = float(cfg["costs"]["lost_sale_cost_per_mm"])

            compare_df = pd.DataFrame(
                {
                    "RunName": rl_df["RunName"],
                    "RunSeed": rl_df["RunSeed"],
                    "day": rl_df["Dia"],
                    "DemandTotal_mm": rl_df["DemandTotal_mm"],
                    "inv_mm_rl": rl_df["Inv_mm"],
                    "scrap_mm_rl": rl_df["Scrap_mm"],
                    "lost_mm_rl": rl_df["LostSale_mm"],
                    "cost_day_rl": rl_df["RL_DailyCost"],
                    "cum_cost_rl": rl_df["RL_CumCost"],
                    "inv_cost_rl": rl_df["Inv_mm"] * inv_rate,
                    "scrap_cost_rl": rl_df["Scrap_mm"] * scr_rate,
                    "lost_cost_rl": rl_df["LostSale_mm"] * los_rate,
                    "inv_mm_milp": milp_df["MILP_Inv_mm"],
                    "scrap_mm_milp": milp_df["MILP_Scrap_mm"],
                    "lost_mm_milp": milp_df["MILP_LostSale_mm"],
                    "cost_day_milp": milp_df["MILP_DailyCost"],
                    "cum_cost_milp": milp_df["MILP_CumCost"],
                    "inv_cost_milp": milp_df["MILP_Inv_mm"] * inv_rate,
                    "scrap_cost_milp": milp_df["MILP_Scrap_mm"] * scr_rate,
                    "lost_cost_milp": milp_df["MILP_LostSale_mm"] * los_rate,
                    "RL_DemandMet": rl_df["RL_DemandMet"],
                    "MILP_DemandMet": milp_df["MILP_DemandMet"],
                }
            )
            compare_path = os.path.join(runs_dir, "compare_day.csv")
            compare_df.to_csv(compare_path, index=False)

            if do_make_plots:
                make_plots(runs_dir, compare_path, cfg["costs"])
        else:
            if plot_every > 0 and args.mode == "train":
                print(
                    f"[skip-milp] Run {run_name} not on {plot_every}-run interval; MILP baseline skipped"
                )
            else:
                print("[skip-milp] Skipping MILP baseline for faster PPO turnaround")

        _update_total_cost_summary(
            args.env,
            run_seed,
            run_name,
            rl_total,
            rl_met_rate,
            milp_total,
            milp_met_rate,
        )

        eval_env.close()

        if milp_df is not None:
            print("[done] Run {run} (seed {seed}) complete -> {dir}".format(run=run_name, seed=run_seed, dir=runs_dir))
        else:
            print(
                "[done] Run {run} (seed {seed}) complete without MILP -> {dir}".format(
                    run=run_name, seed=run_seed, dir=runs_dir
                )
            )
    except KeyboardInterrupt:
        if error_cleanup_enabled:
            print(f"[cleanup] Keyboard interrupt caught; removing incomplete artifacts for {run_name}")
            _cleanup_run_artifacts(*cleanup_targets)
        raise
    except Exception:
        if error_cleanup_enabled:
            print(f"[cleanup] Exception raised during run {run_name}; removing incomplete artifacts")
            _cleanup_run_artifacts(*cleanup_targets)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["pequeno", "medio"], required=True)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--make-plots", action="store_true", help="Save comparison plots")
    p.add_argument("--run-name", type=str, default=None, help="Specify an existing run (e.g. ppo3) for evaluation")
    p.add_argument("--n-envs", type=int, default=None, help="Number of vectorized environments to use during training")
    p.add_argument(
        "--fresh-model",
        action="store_true",
        help="Start training with a new PPO model even if previous checkpoints exist",
    )
    p.add_argument("--eval-milp", action="store_true", help="Rebuild demands and solve MILP baseline after training")
    p.add_argument(
        "--plot-milp-every",
        type=int,
        default=0,
        help="When >0, every Nth training run triggers MILP evaluation and plot generation",
    )
    args = p.parse_args()
    train_and_eval(args)


if __name__ == "__main__":
    main()
