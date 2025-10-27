from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def _load_run_pair(run_dir: Path) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    rl = _safe_read_csv(run_dir / "episode_rl_details.csv")
    milp = _safe_read_csv(run_dir / "episode_milp_details.csv")
    if rl is None or milp is None:
        return None
    # Normalize column names
    if "Dia" not in rl.columns:
        return None
    if "Dia" not in milp.columns:
        return None
    return rl, milp


def _ensure_figs_dir(reports_root: Path, figs_root: Path | None = None) -> Path:
    if figs_root is not None:
        figs = Path(figs_root)
    else:
        figs = Path(reports_root) / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return figs


def _join_daywise(rl: pd.DataFrame, milp: pd.DataFrame) -> pd.DataFrame:
    keep_rl = [c for c in rl.columns if c in (
        "Dia",
        "RL_DailyCost", "RL_CumCost",
        "ScrapCost", "InvCost",
        "LostSaleCost"
    )]
    keep_m = [c for c in milp.columns if c in ("Dia", "MILP_DailyCost", "MILP_CumCost")]
    r = rl[keep_rl].copy()
    m = milp[keep_m].copy()
    # Coerce numeric
    for df in (r, m):
        for col in df.columns:
            if col != "Dia":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    r["Dia"] = pd.to_numeric(r["Dia"], errors="coerce").astype("Int64")
    m["Dia"] = pd.to_numeric(m["Dia"], errors="coerce").astype("Int64")
    merged = pd.merge(r, m, on="Dia", how="inner").sort_values("Dia")
    merged = merged.reset_index(drop=True)
    return merged


def _plot_latest_run(latest_run: Path, figs_dir: Path) -> None:
    pair = _load_run_pair(latest_run)
    if pair is None:
        return
    rl, milp = pair
    df = _join_daywise(rl, milp)
    if df.empty:
        return

    days = df["Dia"].astype(int).to_numpy()
    # Daily comparison: RL vs MILP
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, df["RL_DailyCost"], label="RL - Custo Diário", lw=2)
    plt.plot(days, df["MILP_DailyCost"], label="MILP - Custo Diário", lw=2)
    plt.xlabel("Dia"); plt.ylabel("Custo")
    plt.title("Comparação diária: RL vs MILP")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "comparacao_diaria_rl_milp.png", dpi=150)
    plt.close()

    # Cumulative comparison
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, df["RL_CumCost"], label="RL - Cumulativo", lw=2)
    plt.plot(days, df["MILP_CumCost"], label="MILP - Cumulativo", lw=2)
    plt.xlabel("Dia"); plt.ylabel("Custo Acumulado")
    plt.title("Custo acumulado: RL vs MILP")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "comparacao_rl_milp_acumulado.png", dpi=150)
    plt.close()

    # Percent difference per day (relative to MILP)
    milp_daily = df["MILP_DailyCost"].replace(0, np.nan)
    pct = (df["RL_DailyCost"] - df["MILP_DailyCost"]) / milp_daily * 100.0
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, pct, label="Diferença % (RL vs MILP)", lw=2)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Dia"); plt.ylabel("%")
    plt.title("Diferença percentual diária (RL - MILP)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "diferenca_percentual_diaria.png", dpi=150)
    plt.close()

    # Final total comparison for latest run
    rl_final = float(pd.to_numeric(df["RL_CumCost"], errors="coerce").dropna().iloc[-1])
    milp_final = float(pd.to_numeric(df["MILP_CumCost"], errors="coerce").dropna().iloc[-1])
    plt.figure(figsize=(6, 4.5))
    plt.bar(["RL", "MILP"], [rl_final, milp_final], color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Custo Final Acumulado"); plt.title("Comparação Final - Última Execução")
    plt.tight_layout()
    plt.savefig(figs_dir / "comparacao_rl_milp.png", dpi=150)
    plt.close()

    # Cost breakdown by type (RL) summed over days if available
    components = [
        ("Scrap", "ScrapCost"),
        ("Estoque", "InvCost"),
        ("PerdaVenda", "LostSaleCost"),
    ]
    totals = []
    labels = []
    for lbl, col in components:
        if col in df.columns:
            val = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
            if val != 0:
                labels.append(lbl)
                totals.append(val)
    if totals:
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, totals, color="#1f77b4")
        plt.ylabel("Custo acumulado no episódio")
        plt.title("Custo acumulado por tipo (RL) - Última Execução")
        plt.tight_layout()
        plt.savefig(figs_dir / "custo_acumulado_por_tipo.png", dpi=150)
        plt.close()

    # Demand service (service rate per day)
    try:
        # Identify demand columns and widths
        dem_cols = [c for c in rl.columns if c.startswith("Demanda_") and c.endswith("mm")]
        widths = []
        for c in dem_cols:
            try:
                w = int(c.split("_")[1].replace("mm", ""))
                widths.append(w)
            except Exception:
                widths.append(0)
        R = rl.copy()
        R[dem_cols] = R[dem_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        demand_mm = (R[dem_cols] * np.array(widths)).sum(axis=1)
        served_mm = pd.to_numeric(R.get("Served_mm", pd.Series([np.nan]*len(R))), errors="coerce").fillna(0)
        service_rate = (served_mm / demand_mm.replace(0, np.nan)) * 100.0
        plt.figure(figsize=(9, 4.5))
        plt.plot(days, service_rate, lw=2)
        plt.xlabel("Dia"); plt.ylabel("Atendimento da Demanda (%)")
        plt.title("Taxa de atendimento da demanda (RL)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(figs_dir / "taxa_atendimento_demanda.png", dpi=150); plt.close()
    except Exception:
        pass

    # End-of-day inventory (mm)
    try:
        stock_cols = [c for c in rl.columns if c.startswith("Estoque_") and c.endswith("mm")]
        widths_s = []
        for c in stock_cols:
            try:
                w = int(c.split("_")[1].replace("mm", ""))
                widths_s.append(w)
            except Exception:
                widths_s.append(0)
        S = rl.copy()
        S[stock_cols] = S[stock_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        inv_mm = (S[stock_cols] * np.array(widths_s)).sum(axis=1)
        plt.figure(figsize=(9, 4.5))
        plt.plot(days, inv_mm, lw=2)
        plt.xlabel("Dia"); plt.ylabel("Estoque (mm)")
        plt.title("Estoque ao final do dia (RL)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(figs_dir / "estoque_final_dia.png", dpi=150); plt.close()
    except Exception:
        pass

    # Sheets and setups usage (RL vs MILP)
    try:
        rl_sheets = pd.to_numeric(rl.get("SheetsUsed", pd.Series([np.nan]*len(rl))), errors="coerce").fillna(0)
        rl_setups = pd.to_numeric(rl.get("SetupsUsed", pd.Series([np.nan]*len(rl))), errors="coerce").fillna(0)
        milp_sheets = pd.to_numeric(milp.get("SheetsUsed", pd.Series([np.nan]*len(milp))), errors="coerce").fillna(0)
        # Infer MILP setups from MILP_Patterns string if available
        milp_pat_col = None
        for cand in ("MILP_Patterns", "Patterns"):
            if cand in milp.columns:
                milp_pat_col = cand
                break
        if milp_pat_col is not None:
            def _count_patterns(s: str) -> int:
                if not isinstance(s, str) or not s:
                    return 0
                parts = [p for p in s.split(";") if ":" in p]
                return len(parts)
            milp_setups = milp[milp_pat_col].fillna("").apply(_count_patterns).astype(int)
        else:
            milp_setups = pd.Series([np.nan]*len(milp))
        plt.figure(figsize=(10, 5))
        plt.subplot(2,1,1)
        plt.plot(days, rl_sheets, label="RL", lw=2)
        plt.plot(days, milp_sheets, label="MILP", lw=2)
        plt.ylabel("Chapas usadas"); plt.title("Uso de chapas por dia"); plt.grid(True, alpha=0.3); plt.legend()
        plt.subplot(2,1,2)
        plt.plot(days, rl_setups, label="RL", lw=2)
        try:
            plt.plot(days, milp_setups.iloc[:len(days)], label="MILP", lw=2)
        except Exception:
            pass
        plt.ylabel("Setups usados"); plt.xlabel("Dia"); plt.title("Uso de setups por dia"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(figs_dir / "uso_chapas_setups.png", dpi=150); plt.close()
    except Exception:
        pass


def _plot_across_runs(run_dirs: List[Path], figs_dir: Path) -> None:
    rows = []
    for rd in run_dirs:
        pair = _load_run_pair(rd)
        if pair is None:
            continue
        rl, milp = pair
        # final cumulative values
        rl_final = None
        milp_final = None
        if "RL_CumCost" in rl.columns and not rl.empty:
            rl_final = pd.to_numeric(rl["RL_CumCost"], errors="coerce").dropna()
            rl_final = float(rl_final.iloc[-1]) if not rl_final.empty else None
        if "MILP_CumCost" in milp.columns and not milp.empty:
            milp_final = pd.to_numeric(milp["MILP_CumCost"], errors="coerce").dropna()
            milp_final = float(milp_final.iloc[-1]) if not milp_final.empty else None
        if rl_final is None and milp_final is None:
            continue
        rows.append({
            "run": rd.name,
            "RL_Final": rl_final if rl_final is not None else np.nan,
            "MILP_Final": milp_final if milp_final is not None else np.nan,
        })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("run")
    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(max(8, len(df)*0.7), 5))
    plt.bar(x - width/2, df["RL_Final"], width, label="RL")
    plt.bar(x + width/2, df["MILP_Final"], width, label="MILP")
    plt.xticks(x, df["run"], rotation=45, ha="right")
    plt.ylabel("Custo final acumulado")
    plt.title("Custo acumulado por run (RL vs MILP)")
    plt.legend(); plt.tight_layout()
    plt.savefig(figs_dir / "custo_acumulado_por_run.png", dpi=150)
    plt.close()

    # Alias view focusing on final totals per run (same data, different name)
    plt.figure(figsize=(max(8, len(df)*0.7), 5))
    plt.bar(x - width/2, df["RL_Final"], width, label="RL")
    plt.bar(x + width/2, df["MILP_Final"], width, label="MILP")
    plt.xticks(x, df["run"], rotation=45, ha="right")
    plt.ylabel("Custo final acumulado")
    plt.title("Total final por run (RL vs MILP)")
    plt.legend(); plt.tight_layout()
    plt.savefig(figs_dir / "final_total_per_run.png", dpi=150)
    plt.close()


def _has_required_csvs(d: Path) -> bool:
    return (d / "episode_rl_details.csv").exists() and (d / "episode_milp_details.csv").exists()


def _discover_run_dirs(reports_root: Path, mode: Optional[str]) -> List[Path]:
    """
    Find per-run snapshot directories. We look in both:
      - reports/runs/<mode>/ppo_*
      - runs/<mode>/ppo_*
    to support both legacy and current snapshot locations.
    """
    candidates_roots: List[Path] = []
    # reports/runs/<mode>
    rr = reports_root / "runs"
    if mode in ("train", "deploy"):
        candidates_roots.append(rr / mode)
    else:
        candidates_roots.extend([rr / "train", rr / "deploy"]) 
    # top-level runs/<mode>
    tl = Path("runs")
    if mode in ("train", "deploy"):
        candidates_roots.append(tl / mode)
    else:
        candidates_roots.extend([tl / "train", tl / "deploy"]) 

    found: List[Path] = []
    for root in candidates_roots:
        if not root.exists():
            continue
        for p in root.glob("ppo_*"):
            if p.is_dir() and _has_required_csvs(p):
                found.append(p)

    # De-duplicate and sort
    uniq = {p.resolve(): p for p in found if p.name.startswith("ppo_")}
    return sorted(uniq.values(), key=lambda p: p.name)


def refresh_all_figs(
    reports_root: str | Path = "reports",
    prefer_run: Optional[Path] = None,
    figs_root: Optional[Path] = None,
    mode: Optional[str] = None,
) -> None:
    """
    Regenerate figures under reports/figs using data in reports/runs.
    If prefer_run is provided, ensure the "latest" plots use that run.
    If mode is provided ('train' or 'deploy'), across-run comparisons are filtered to that mode.
    """
    reports_root = Path(reports_root)
    figs_dir = _ensure_figs_dir(reports_root, figs_root)

    # Discover candidate run directories (including archived) filtered by mode
    run_dirs = _discover_run_dirs(reports_root, mode)
    # Also consider the per-mode latest directory as a source for latest plots only (not counted as a run)
    latest = None
    if prefer_run and prefer_run.exists():
        latest = prefer_run

    if not run_dirs and latest is None:
        return

    # Pick a latest run if not explicitly provided
    if latest is None:
        run_dirs_sorted = sorted(run_dirs, key=lambda p: p.name, reverse=True)
        latest = run_dirs_sorted[0]

    # Generate per-run (latest) plots
    try:
        _plot_latest_run(latest, figs_dir)
    except Exception:
        pass

    # Generate across-runs plots (only if we have >1 run or want a summary)
    try:
        _plot_across_runs(run_dirs, figs_dir)
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh figures under reports/figs")
    parser.add_argument("--reports-root", type=str, default="reports")
    parser.add_argument("--mode", choices=["train", "deploy", "all"], default=None, help="Generate figures for selected mode")
    parser.add_argument("--figs-root", type=str, default=None, help="Optional explicit figs directory; defaults to reports/figs/<mode> when mode is set")
    args = parser.parse_args()

    reports_root = Path(args.reports_root)
    if args.mode in ("train", "deploy"):
        figs_root = Path(args.figs_root) if args.figs_root else (reports_root / "figs" / args.mode)
        prefer = reports_root / args.mode
        refresh_all_figs(reports_root=reports_root, prefer_run=prefer, figs_root=figs_root, mode=args.mode)
    else:
        # Run for both modes
        for m in ("train", "deploy"):
            figs_root = Path(args.figs_root) if args.figs_root else (reports_root / "figs" / m)
            prefer = reports_root / m
            refresh_all_figs(reports_root=reports_root, prefer_run=prefer, figs_root=figs_root, mode=m)
