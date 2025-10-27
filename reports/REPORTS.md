# Reports folder layout

Goal: keep it clear and simple. Only two per-mode data folders and two per-mode figure folders.

- train/ — CSVs for the latest training cycle
  - episode_rl_details.csv
  - episode_milp_details.csv
  - comparison_rl_milp_per_day_detailed.csv
- deploy/ — CSVs for the latest deploy cycle
  - episode_rl_details.csv
  - episode_milp_details.csv
  - comparison_rl_milp_per_day_detailed.csv
- figs/
  - train/ — images directly inside (no subfolders)
  - deploy/ — images directly inside (no subfolders)
- runs/
  - archive/ — all older per-run folders (ppo_*) stored here; nothing else under runs/
- models/ — saved PPO model and VecNormalize

Notes
- Figures are regenerated each cycle; any stray subfolders are flattened by `scripts/housekeep_figs.py`.
- All historical per-run artifacts are preserved in `runs/archive/` to avoid clutter.
- `train.py` always writes the latest CSVs into `reports/train` or `reports/deploy`.
