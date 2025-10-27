import os, sys, csv, yaml
sys.path.insert(0, os.getcwd())

from src.csp_rl.envs.ctl_env import CtlEnv

cfg = yaml.safe_load(open(os.path.join("configs", "default.yaml"), encoding="utf-8"))
env_cfg = cfg.get("environment", cfg)

env = CtlEnv(env_cfg)

out_dir = os.path.join("reports")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "patterns_pipe.csv")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="|")
    header = ["pattern_idx"] + [f"count_type{i+1}" for i,_ in enumerate(env.strip_lengths)]
    w.writerow(header)
    for idx, pat in enumerate(env.patterns):
        counts = [int(c) for c in pat]
        row = [idx] + counts
        w.writerow(row)

print("patterns (pipe-separated) exported to", out_path)