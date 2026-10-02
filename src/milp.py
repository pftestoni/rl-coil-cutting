import numpy as np
import pulp as pl
from typing import Dict, Optional, Mapping, Any

def solve_day_milp(
    stock_units: np.ndarray,
    demand_units: np.ndarray,
    patterns: np.ndarray,
    widths_mm: np.ndarray,
    coil_width_mm: int,
    capacity_sheets: int,
    max_blocks: int,
    costs: Dict[str, float],
    solver_options: Optional[Mapping[str, Any]] = None,
    inventory_cap_units: Optional[int] = None,
    inventory_penalty_per_mm: float = 0.0,
    auto_scrap_excess: bool = False,
) -> Dict[str, np.ndarray | float | int | str]:
    n = widths_mm.shape[0]
    P = patterns.shape[0]
    trim = coil_width_mm - patterns @ widths_mm
    trim = np.maximum(trim, 0)

    m = pl.LpProblem("CTL_Daily", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", range(P), lowBound=0, cat=pl.LpInteger)
    y = pl.LpVariable.dicts("y", range(P), lowBound=0, upBound=1, cat=pl.LpBinary)
    sales = pl.LpVariable.dicts("sales", range(n), lowBound=0)
    s_out = pl.LpVariable.dicts("s_out", range(n), lowBound=0)
    lost = pl.LpVariable.dicts("lost", range(n), lowBound=0)

    cap_val = int(inventory_cap_units) if inventory_cap_units is not None else None
    scrap_from_inv: Optional[dict[int, pl.LpVariable]] = None
    excess_units_vars: Optional[dict[int, pl.LpVariable]] = None
    if cap_val is not None and auto_scrap_excess:
        scrap_from_inv = pl.LpVariable.dicts("scrap_from_inv", range(n), lowBound=0)
    elif cap_val is not None:
        excess_units_vars = pl.LpVariable.dicts("excess_units", range(n), lowBound=0)

    prod_i = [pl.lpSum(patterns[p, i] * x[p] for p in range(P)) for i in range(n)]

    for i in range(n):
        if cap_val is not None and auto_scrap_excess:
            m += stock_units[i] + prod_i[i] == sales[i] + s_out[i] + scrap_from_inv[i]  # type: ignore[index]
            if cap_val >= 0:
                m += s_out[i] <= cap_val
        else:
            m += stock_units[i] + prod_i[i] == sales[i] + s_out[i]
            if cap_val is not None and cap_val >= 0 and excess_units_vars is not None:
                m += excess_units_vars[i] >= s_out[i] - cap_val
        m += demand_units[i] == sales[i] + lost[i]

    m += pl.lpSum(x[p] for p in range(P)) <= capacity_sheets

    M = capacity_sheets
    for p in range(P):
        m += x[p] <= M * y[p]
    m += pl.lpSum(y[p] for p in range(P)) <= max_blocks

    inv_mm = pl.lpSum(s_out[i] * widths_mm[i] for i in range(n))
    lost_mm = pl.lpSum(lost[i] * widths_mm[i] for i in range(n))
    scrap_pattern_mm = pl.lpSum(trim[p] * x[p] for p in range(P))
    scrap_inv_mm = (
        pl.lpSum(scrap_from_inv[i] * widths_mm[i] for i in range(n))  # type: ignore[index]
        if scrap_from_inv is not None
        else 0
    )
    scrap_mm = scrap_pattern_mm + scrap_inv_mm
    excess_mm_expr = (
        scrap_inv_mm
        if scrap_from_inv is not None
        else pl.lpSum(excess_units_vars[i] * widths_mm[i] for i in range(n))  # type: ignore[index]
        if excess_units_vars is not None
        else 0
    )

    c_inv = float(costs["inventory_cost_per_mm"]) * inv_mm
    c_scr = float(costs["scrap_cost_per_mm"]) * scrap_mm
    c_los = float(costs["lost_sale_cost_per_mm"]) * lost_mm
    c_exc = float(inventory_penalty_per_mm) * excess_mm_expr

    m += c_inv + c_scr + c_los + c_exc

    solver_kwargs: dict[str, Any] = {}
    if solver_options:
        solver_kwargs.update({k: v for k, v in solver_options.items() if v is not None})
    msg_flag = bool(solver_kwargs.get("msg", False))
    solver_kwargs.setdefault("msg", msg_flag)
    m.solve(pl.PULP_CBC_CMD(**solver_kwargs))
    status = pl.LpStatus.get(m.status, str(m.status))

    x_sol = np.array([x[p].value() or 0 for p in range(P)], dtype=np.int32)
    prod_units = np.zeros(n, dtype=np.int32)
    for i in range(n):
        prod_units[i] = int(sum(patterns[p, i] * x_sol[p] for p in range(P)))
    sales_units = np.array([int(round(sales[i].value() or 0)) for i in range(n)], dtype=np.int32)
    stock_units_end = np.array([int(round(s_out[i].value() or 0)) for i in range(n)], dtype=np.int32)
    lost_units = np.array([int(round(lost[i].value() or 0)) for i in range(n)], dtype=np.int32)

    inv_mm_val = int(sum(stock_units_end[i] * widths_mm[i] for i in range(n)))
    lost_mm_val = int(sum(lost_units[i] * widths_mm[i] for i in range(n)))
    scrap_pattern_mm_val = int(sum(trim[p] * x_sol[p] for p in range(P)))
    scrap_inv_mm_val = int(
        sum((scrap_from_inv[i].value() or 0) * widths_mm[i] for i in range(n))  # type: ignore[index]
    ) if scrap_from_inv is not None else 0
    scrap_mm_val = scrap_pattern_mm_val + scrap_inv_mm_val

    excess_units_total = int(
        sum((scrap_from_inv[i].value() or 0) for i in range(n))  # type: ignore[index]
    ) if scrap_from_inv is not None else int(
        sum((excess_units_vars[i].value() or 0) for i in range(n))  # type: ignore[index]
    ) if excess_units_vars is not None else 0
    excess_mm_val = scrap_inv_mm_val if scrap_from_inv is not None else int(
        sum((excess_units_vars[i].value() or 0) * widths_mm[i] for i in range(n))  # type: ignore[index]
    ) if excess_units_vars is not None else 0
    excess_penalty_val = float(inventory_penalty_per_mm) * excess_mm_val

    total_cost = (
        float(costs["inventory_cost_per_mm"]) * inv_mm_val
        + float(costs["scrap_cost_per_mm"]) * scrap_mm_val
        + float(costs["lost_sale_cost_per_mm"]) * lost_mm_val
        + excess_penalty_val
    )

    return {
        "x": x_sol,
        "prod_units": prod_units,
        "sales_units": sales_units,
        "lost_units": lost_units,
        "stock_units_end": stock_units_end,
        "inv_mm": inv_mm_val,
        "lost_mm": lost_mm_val,
        "scrap_mm": scrap_mm_val,
        "scrap_from_inventory_mm": scrap_inv_mm_val,
        "inventory_excess_units": excess_units_total,
        "inventory_excess_penalty": excess_penalty_val,
        "cost_day": total_cost,
        "status": status,
    }
