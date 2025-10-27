import sys
import csv
from pathlib import Path
from statistics import mean, pstdev


def read_last_rl_cum_cost(csv_path: Path):
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header:
                return None
            try:
                j = header.index("RL_CumCost")
            except ValueError:
                j = None
            last = None
            for row in r:
                if not row:
                    continue
                if j is not None:
                    try:
                        last = float(row[j])
                    except Exception:
                        continue
                else:
                    # fallback: last numeric
                    for cell in reversed(row):
                        try:
                            last = float(cell)
                            break
                        except Exception:
                            continue
            return last
    except Exception:
        return None


def main(argv):
    if len(argv) < 2:
        print("Usage: python scripts/analyze_tune.py <session_name> [--root <repo_root>]")
        print("  Example: python scripts/analyze_tune.py 20251010_134103")
        sys.exit(2)
    session = argv[1]
    root = Path(".")
    if "--root" in argv:
        i = argv.index("--root")
        if i + 1 < len(argv):
            root = Path(argv[i + 1])
    tune_runs = root / "reports" / "runs" / "tune" / session
    tune_sess = root / "reports" / "tune-sessions" / session
    if not tune_runs.exists() and not tune_sess.exists():
        print(f"Session not found under either: {tune_runs} or {tune_sess}")
        sys.exit(1)

    results = []
    for cand_dir in sorted(tune_runs.glob("candidate-*/rung-*/")):
        cand_name = cand_dir.parts[-2]  # candidate-XX
        # Prefer last rung if multiple
        # We'll collect only one entry per candidate: the highest rung number
    
    # Build candidate sources: prefer rung copies, fallback to tune-sessions per-candidate CSVs
    cand_sources = {}
    if tune_runs.exists():
        for rung_csv in tune_runs.glob("candidate-*/rung-*/episode_rl_details.csv"):
            parts = rung_csv.parts
            cand = parts[-3]  # candidate-XX
            rung = parts[-2]  # rung-YY
            try:
                rung_num = int(rung.split("-")[-1])
            except Exception:
                rung_num = 0
            prev = cand_sources.get(cand)
            if prev is None or rung_num > prev[0]:
                cand_sources[cand] = (rung_num, rung_csv)
    if tune_sess.exists():
        # Fallback: per-candidate RL CSV directly under tune-sessions/<session>/candidate-XX/<mode>/episode_rl_details.csv
        for cand_dir in sorted(tune_sess.glob("candidate-*/")):
            cand = cand_dir.name
            csv_path = cand_dir / "train" / "episode_rl_details.csv"
            if not csv_path.exists():
                csv_path = cand_dir / "deploy" / "episode_rl_details.csv"
            if csv_path.exists():
                # Only set if not already present from rung copies
                cand_sources.setdefault(cand, (0, csv_path))

    for cand, (_rn, csv_path) in sorted(cand_sources.items()):
        metric = read_last_rl_cum_cost(csv_path)
        if metric is not None:
            try:
                idx = int(cand.split("-")[-1])
            except Exception:
                idx = cand
            results.append((idx, metric, str(csv_path)))

    if not results:
        print("No candidate metrics found.")
        sys.exit(1)

    results.sort(key=lambda x: x[1])  # lower cost is better
    metrics = [m for _i, m, _p in results]
    best = results[0]
    worst = results[-1]
    avg = mean(metrics)
    sd = pstdev(metrics) if len(metrics) > 1 else 0.0
    spread = worst[1] - best[1]
    pct = (spread / best[1] * 100.0) if best[1] != 0 else float('inf')

    # Write summary CSV
    out_csv = tune_runs / "summary_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Candidate", "RL_CumCost", "SourceCSV"])
        for i, m, p in results:
            w.writerow([i, f"{m:.2f}", p])
        w.writerow([])
        w.writerow(["COUNT", len(metrics)])
        w.writerow(["BEST", f"{best[1]:.2f}", best[0]])
        w.writerow(["WORST", f"{worst[1]:.2f}", worst[0]])
        w.writerow(["MEAN", f"{avg:.2f}"])
        w.writerow(["STD", f"{sd:.2f}"])
        w.writerow(["SPREAD", f"{spread:.2f}"])
        w.writerow(["PCT_SPREAD", f"{pct:.2f}%"]) 

    # Print concise report
    print(f"Session: {session}")
    print(f"Candidates: {len(metrics)}")
    print(f"Best:   cand {best[0]} RL_CumCost={best[1]:.2f}")
    print(f"Worst:  cand {worst[0]} RL_CumCost={worst[1]:.2f}")
    print(f"Mean:   {avg:.2f}  Std: {sd:.2f}")
    print(f"Spread: {spread:.2f}  ({pct:.2f}%)")
    print("Top 5:")
    for i, (ci, cm, _p) in enumerate(results[:5], start=1):
        print(f"  {i:>2}. cand {ci}  {cm:.2f}")
    print("Bottom 5:")
    for i, (ci, cm, _p) in enumerate(reversed(results[-5:]), start=1):
        print(f"  {i:>2}. cand {ci}  {cm:.2f}")
    print(f"Summary CSV: {out_csv}")


if __name__ == "__main__":
    main(sys.argv)
