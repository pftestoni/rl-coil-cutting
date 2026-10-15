"""Builds the English summary chart (docs/figures/summary_en.png) from results/summary_*.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"Custo_medio_diario_mm": "Average daily cost", "Perda_de_vendas_media_diaria_mm": "Average daily lost sales (mm)"}


def main():
    milp = pd.read_csv(ROOT / "results" / "summary_milp.csv").set_index("Metric")
    rl = pd.read_csv(ROOT / "results" / "summary_rl.csv").set_index("Metric")
    keys = [k for k in LABELS if k in milp.index]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, k in zip(axes if len(keys) > 1 else [axes], keys):
        data = pd.DataFrame({"MILP (day by day)": milp.loc[k], "PPO agent": rl.loc[k]})
        data.index = ["5 SKUs", "20 SKUs"]
        data.plot.bar(ax=ax, rot=0, color=["#8c8c8c", "#2f6db5"])
        ax.set_title(LABELS[k])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "summary_en.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
