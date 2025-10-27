import argparse
import subprocess
import sys
from pathlib import Path


"""
Deprecated standalone tuner. This wrapper now delegates to train.py --tune so there's a single source of truth.
CLI remains mostly compatible. Notable mappings:
  --model-root -> --tune-model-root
  --out-root   -> --tune-out-root
Other args are passed through unchanged when possible.
"""


def main():
    parser = argparse.ArgumentParser(description="Wrapper: delegates tuning to train.py --tune", allow_abbrev=False)
    # Common flags preserved for backward compatibility
    parser.add_argument("--mode", choices=["train", "deploy"], default="train")
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--rungs", type=str, default=None)
    parser.add_argument("--keep-fraction", type=float, default=None)
    parser.add_argument("--session-name", type=str, default=None)
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--lrs", type=str, default=None)
    parser.add_argument("--ent-coefs", type=str, default=None)
    parser.add_argument("--finalize-best", action="store_true")
    parser.add_argument("--n-workers", type=int, default=None)
    # Generic train.py args pass-through (subset)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--drop-day-index", action="store_true")
    parser.add_argument("--no-norm-reward", action="store_true")
    parser.add_argument("--clip-reward", type=float, default=None)
    parser.add_argument("--demand-frac", type=float, default=None)
    parser.add_argument("--train-shaping", action="store_true")
    parser.add_argument("--env-seed", type=int, default=None)
    args, unknown = parser.parse_known_args()

    cmd = [
        sys.executable,
        str(Path("train.py").name),
        "--tune",
        "--mode",
        str(args.mode),
    ]

    def add_flag(name, val):
        if val is None:
            return
        if isinstance(val, bool):
            if val:
                cmd.append(name)
        else:
            cmd.extend([name, str(val)])

    add_flag("--num-candidates", args.num_candidates)
    add_flag("--rungs", args.rungs)
    add_flag("--keep-fraction", args.keep_fraction)
    add_flag("--session-name", args.session_name)
    # Map legacy roots to new flags
    if args.model_root:
        cmd.extend(["--tune-model-root", args.model_root])
    if args.out_root:
        cmd.extend(["--tune-out-root", args.out_root])
    add_flag("--seeds", args.seeds)
    add_flag("--lrs", args.lrs)
    add_flag("--ent-coefs", args.ent_coefs)
    add_flag("--n-workers", args.n_workers)
    add_flag("--finalize-best", args.finalize_best)
    # Pass-through trainer-related flags
    add_flag("--gamma", args.gamma)
    add_flag("--n-envs", args.n_envs)
    add_flag("--subproc", args.subproc)
    add_flag("--config", args.config)
    add_flag("--drop-day-index", args.drop_day_index)
    add_flag("--no-norm-reward", args.no_norm_reward)
    add_flag("--clip-reward", args.clip_reward)
    add_flag("--demand-frac", args.demand_frac)
    add_flag("--train-shaping", args.train_shaping)
    add_flag("--env-seed", args.env_seed)

    # Preserve any extra args unknown to this wrapper
    cmd.extend(unknown)

    print("[TUNE] Delegating to:", " ".join(cmd))
    rc = subprocess.call(cmd)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
