"""Authoritative research-world command line interface."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def output_directory(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise ValueError(f"output already exists: {path}; use --overwrite explicitly")
        if path.is_symlink() or path.resolve() in (Path.cwd().resolve(), Path.home().resolve(), Path('/')):
            raise ValueError(f"unsafe overwrite destination: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="write a labeled offline synthetic scenario")
    synthetic.add_argument("--out", type=Path, required=True)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--overwrite", action="store_true")
    build = commands.add_parser("build", help="import explicit provenance-tagged environmental sources")
    build.add_argument("--inputs", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    run = commands.add_parser("run", help="run an authoritative fire/operations episode")
    run.add_argument("--scenario", type=Path, required=True)
    run.add_argument("--longitude", type=float)
    run.add_argument("--latitude", type=float)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--policy", choices=("none", "heuristic", "checkpoint"), default="none")
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--overwrite", action="store_true")
    train = commands.add_parser("train", help="train an experimental masked PPO policy")
    train.add_argument("--scenario", type=Path, required=True)
    train.add_argument("--steps", type=int, default=512)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--overwrite", action="store_true")
    evaluate = commands.add_parser("evaluate", help="paired held-out policy evaluation")
    evaluate.add_argument("--scenario", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--seeds", default=",".join(str(seed) for seed in range(10000, 10032)))
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            from .scenario import save_scenario, load_scenario
            from .synthetic import make_synthetic
            scenario = make_synthetic(seed=args.seed, duration_s=21600, size=96)
            scenario.validate()
            output_directory(args.out, args.overwrite)
            save_scenario(scenario, args.out)
            loaded = load_scenario(args.out)
            print(f"Saved validated synthetic scenario {loaded.name}: {args.out}")
        elif args.command == "build":
            from .prepare import build
            output_directory(args.out, args.overwrite)
            scenario = build(args.inputs, args.out)
            print(f"Saved imported scenario {scenario.name}: {args.out}")
        elif args.command == "run":
            from .scenario import load_scenario
            from .rollout import run_episode
            scenario = load_scenario(args.scenario)
            output_directory(args.out, args.overwrite)
            metrics = run_episode(scenario, args.seed, args.policy, checkpoint=args.checkpoint,
                                  longitude=args.longitude, latitude=args.latitude, out=args.out)
            print(f"Completed {metrics['time_s']} s; burned {metrics['burned_area_m2']:.1f} m²; "
                  f"dispatches {metrics['dispatch_count']}; censored={metrics['area_censored']}")
        elif args.command == "train":
            from .scenario import load_scenario
            from .learning import train
            scenario = load_scenario(args.scenario)
            output_directory(args.out, args.overwrite)
            result = train(scenario, args.steps, args.seed, args.out)
            print(f"Saved experimental PPO checkpoint: {args.out}")
        elif args.command == "evaluate":
            from .scenario import load_scenario
            from .evaluation import evaluate
            scenario = load_scenario(args.scenario)
            seeds = [int(value) for value in args.seeds.split(",")]
            output_directory(args.out, args.overwrite)
            evaluate(scenario, args.checkpoint, seeds, args.out)
            print(f"Saved paired evaluation: {args.out}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
