import math


def compute_daily_capacity_mm(cfg: dict) -> int:
    """Derive the daily rolling capacity in millimetres.

    Prefers an explicit value when provided, otherwise falls back to
    line speed multiplied by available minutes.
    """

    demand_cfg = cfg.get("demand", {})
    explicit = demand_cfg.get("daily_capacity_mm") or cfg.get("daily_capacity_mm")
    if explicit is not None:
        return int(explicit)

    env_cfg = cfg.get("env", {})
    sheets_per_day = env_cfg.get("capacity_sheets_per_day")
    coil_width_mm = env_cfg.get("coil_width_mm")
    if sheets_per_day is not None and coil_width_mm is not None:
        return int(sheets_per_day) * int(coil_width_mm)

    speed_mpm = float(cfg.get("speed_m_per_min", 20.0))
    weekly_minutes = float(cfg.get("weekly_minutes", 4200.0))
    days_per_week = int(demand_cfg.get("days_per_week", 7)) or 1
    daily_minutes = weekly_minutes / days_per_week
    capacity_mm = speed_mpm * daily_minutes * 1000.0
    return int(math.floor(capacity_mm + 0.5))
