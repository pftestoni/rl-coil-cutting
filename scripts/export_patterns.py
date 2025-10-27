import os, sys, csv, yaml
sys.path.insert(0, os.getcwd())

from src.csp_rl.envs.ctl_env import CtlEnv

cfg = yaml.safe_load(open(os.path.join("configs", "default.yaml"), encoding="utf-8"))
env_cfg = cfg.get("environment", cfg)

env = CtlEnv(env_cfg)

out_dir = os.path.join("reports")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "patterns.csv")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    header = ["pattern_idx"]
    for i, L in enumerate(env.strip_lengths, start=1):
        header.append(f"count_type{i}({int(L)}mm)")
    header += ["used_length_mm", "scrap_mm", "scrap_cost", "pattern_str"]
    w.writerow(header)

    for idx, pat in enumerate(env.patterns):
        counts = [int(c) for c in pat]
        used_length = int(sum(c * int(L) for c, L in zip(counts, env.strip_lengths)))
        scrap_mm = int(env.coil_length - used_length)
        scrap_cost = float(scrap_mm) * float(env.scrap_cost_per_mm)
        pat_str = "; ".join(f"t{i+1}:{q}" for i, q in enumerate(counts) if q > 0)
        row = [idx] + counts + [used_length, scrap_mm, round(scrap_cost, 3), pat_str]
        w.writerow(row)

print("patterns exported to", out_path)