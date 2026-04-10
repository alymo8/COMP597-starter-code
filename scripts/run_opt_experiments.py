#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

from log_utils import Logger, LogLevel


def nearest_pow2_floor(x: int) -> int:
    if x < 1:
        return 1
    return 1 << (x.bit_length() - 1)


def parse_launcher(launcher: str) -> List[str]:
    return shlex.split(launcher)


def detect_oom(text: str) -> bool:
    low = text.lower()
    return "out of memory" in low or "cuda error: out of memory" in low or "cuda out of memory" in low


def run_command(cmd: List[str], cwd: str, log_dir: str, timeout_sec: float = 0) -> Dict:
    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_sec if timeout_sec > 0 else None,
    )
    elapsed = time.perf_counter() - start
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(proc.stderr or "")
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "oom_detected": detect_oom((proc.stdout or "") + "\n" + (proc.stderr or "")),
    }


def get_git_commit(cwd: str) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        return out
    except Exception:
        return "unknown"


def get_gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def check_duration_validity(elapsed_sec: float, target_sec: float, tolerance_sec: float) -> bool:
    if target_sec <= 0:
        return True
    return abs(elapsed_sec - target_sec) <= tolerance_sec


def build_matrix_status(
    runs: List[Dict],
    batch_sweep: List[int],
    profiles: List[str],
    repeats: int,
    require_valid_duration: bool,
) -> Dict:
    cells: Dict[str, Dict] = {}
    matrix_complete = True
    for batch_size in batch_sweep:
        for profile in profiles:
            matching = [r for r in runs if r["batch_size"] == batch_size and r["profile"] == profile]
            run_count = len(matching)
            success_count = sum(1 for r in matching if int(r.get("returncode", 1)) == 0)
            valid_duration_count = sum(1 for r in matching if bool(r.get("valid_duration", False)))
            good_count = sum(
                1
                for r in matching
                if int(r.get("returncode", 1)) == 0
                and (bool(r.get("valid_duration", False)) or not require_valid_duration)
            )
            cell_complete = (
                run_count == repeats
                and success_count == repeats
                and (valid_duration_count == repeats if require_valid_duration else True)
            )
            matrix_complete = matrix_complete and cell_complete
            key = f"batch_{batch_size}_profile_{profile}"
            cells[key] = {
                "batch_size": batch_size,
                "profile": profile,
                "expected_repeats": repeats,
                "run_count": run_count,
                "success_count": success_count,
                "valid_duration_count": valid_duration_count,
                "good_count": good_count,
                "missing_runs": max(0, repeats - run_count),
                "complete": cell_complete,
            }
    return {"matrix_complete": matrix_complete, "cells": cells}


def build_common_args(args: argparse.Namespace, batch_size: int, repeat_id: int) -> List[str]:
    return [
        "--logging.level",
        args.logging_level,
        "--model",
        "opt",
        "--data",
        "opt",
        "--trainer",
        "simple",
        "--learning_rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--repeat_id",
        str(repeat_id),
        "--model_configs.opt.hf_name",
        args.hf_name,
        "--model_configs.opt.dtype",
        args.dtype,
        "--data_configs.opt.batch_size",
        str(batch_size),
        "--data_configs.opt.seq_len",
        str(args.seq_len),
        "--data_configs.opt.num_workers",
        str(args.num_workers),
        "--data_configs.opt.max_samples",
        str(args.max_samples),
        "--trainer_configs.simple.max_duration_sec",
        str(args.duration_sec),
        "--trainer_configs.simple.warmup_steps",
        str(args.warmup_steps),
        "--trainer_configs.simple.warmup_sec",
        str(args.warmup_sec),
    ]


def profile_args(profile: str, run_dir: str) -> List[str]:
    if profile == "A":
        return ["--trainer_stats", "noop"]
    if profile == "B":
        # CodeCarbon end-to-end only: disable step/substep task tracking.
        return [
            "--trainer_stats", "codecarbon",
            "--trainer_stats_configs.codecarbon.output_dir",
            run_dir,
            "--trainer_stats_configs.codecarbon.project_name",
            "opt_e2e_energy",
            "--trainer_stats_configs.codecarbon.track_steps",
            "0",
            "--trainer_stats_configs.codecarbon.track_substeps",
            "0",
        ]
    if profile == "C":
        return [
            "--trainer_stats",
            "composite",
            "--trainer_stats_configs.composite.components",
            "simple,resource",
            "--trainer_stats_configs.simple.output_dir",
            run_dir,
            "--trainer_stats_configs.simple.output_file_prefix",
            "simple",
            "--trainer_stats_configs.simple.plot_metrics",
            "0",
            "--trainer_stats_configs.simple.flush_every_n",
            "50",
            "--trainer_stats_configs.resource.output_dir",
            run_dir,
            "--trainer_stats_configs.resource.output_file_prefix",
            "resource",
            "--trainer_stats_configs.resource.plot_metrics",
            "0",
            "--trainer_stats_configs.resource.include_gpu",
            "1",
            "--trainer_stats_configs.resource.include_energy",
            "1",
            "--trainer_stats_configs.resource.include_system",
            "1",
            "--trainer_stats_configs.resource.include_process",
            "1",
            "--trainer_stats_configs.resource.include_cpu",
            "1",
            "--trainer_stats_configs.resource.include_io",
            "0",
            "--trainer_stats_configs.resource.include_torch_cuda_memory",
            "1",
            "--trainer_stats_configs.resource.sample_interval_ms",
            "500",
            "--trainer_stats_configs.resource.flush_every_n",
            "50",
        ]
    raise ValueError(f"Unknown profile {profile}")


def probe_max_batch(
    args: argparse.Namespace, cwd: str, launcher: List[str], out_dir: str
) -> Tuple[int, List[Dict]]:
    logger = Logger("PROBE", LogLevel.INFO)
    records: List[Dict] = []
    low_success = 0
    high_fail = None
    batch = max(1, args.probe_start_batch)

    logger.info(f"Starting batch size probe from {args.probe_start_batch}...")

    while True:
        run_dir = os.path.join(out_dir, "probe", f"batch_{batch}")
        logger.info(f"Testing batch size {batch}...")
        cmd = launcher + [
            "--logging.level",
            args.logging_level,
            "--model",
            "opt",
            "--data",
            "opt",
            "--trainer",
            "simple",
            "--trainer_stats",
            "noop",
            "--seed",
            str(args.seed),
            "--repeat_id",
            "-1",
            "--learning_rate",
            str(args.learning_rate),
            "--model_configs.opt.hf_name",
            args.hf_name,
            "--model_configs.opt.dtype",
            args.dtype,
            "--data_configs.opt.batch_size",
            str(batch),
            "--data_configs.opt.seq_len",
            str(args.seq_len),
            "--data_configs.opt.num_workers",
            str(args.num_workers),
            "--data_configs.opt.max_samples",
            str(args.max_samples),
            "--trainer_configs.simple.max_duration_sec",
            str(args.probe_duration_sec),
            "--trainer_configs.simple.warmup_steps",
            "0",
            "--trainer_configs.simple.warmup_sec",
            "0",
        ]
        result = run_command(cmd, cwd=cwd, log_dir=run_dir)
        result.update({"batch_size": batch, "command": cmd})
        records.append(result)

        if result["returncode"] == 0:
            logger.info(f"Batch {batch} succeeded ({result['elapsed_sec']:.1f}s)")
            low_success = batch
            if batch >= args.probe_max_batch:
                logger.info(f"Reached max probe batch {args.probe_max_batch}")
                break
            next_batch = batch * 2
            if next_batch > args.probe_max_batch:
                next_batch = args.probe_max_batch
            if next_batch == batch:
                logger.info(f"Max batch is {batch}")
                break
            batch = next_batch
            continue

        if not result["oom_detected"]:
            raise RuntimeError(
                f"Probe failed with non-OOM error at batch {batch}. See logs: {run_dir}"
            )
        logger.info(f"Batch {batch} failed with OOM")
        high_fail = batch
        break

    if high_fail is None:
        return max(1, low_success), records

    lo = low_success + 1
    hi = high_fail - 1
    logger.info(f"Binary search between {lo} and {hi}")
    best = low_success
    while lo <= hi:
        mid = (lo + hi) // 2
        run_dir = os.path.join(out_dir, "probe", f"batch_{mid}")
        logger.info(f"Binary search: testing batch {mid} ({lo}-{hi})")
        cmd = launcher + [
            "--logging.level",
            args.logging_level,
            "--model",
            "opt",
            "--data",
            "opt",
            "--trainer",
            "simple",
            "--trainer_stats",
            "noop",
            "--seed",
            str(args.seed),
            "--repeat_id",
            "-1",
            "--learning_rate",
            str(args.learning_rate),
            "--model_configs.opt.hf_name",
            args.hf_name,
            "--model_configs.opt.dtype",
            args.dtype,
            "--data_configs.opt.batch_size",
            str(mid),
            "--data_configs.opt.seq_len",
            str(args.seq_len),
            "--data_configs.opt.num_workers",
            str(args.num_workers),
            "--data_configs.opt.max_samples",
            str(args.max_samples),
            "--trainer_configs.simple.max_duration_sec",
            str(args.probe_duration_sec),
            "--trainer_configs.simple.warmup_steps",
            "0",
            "--trainer_configs.simple.warmup_sec",
            "0",
        ]
        result = run_command(cmd, cwd=cwd, log_dir=run_dir)
        result.update({"batch_size": mid, "command": cmd})
        records.append(result)
        if result["returncode"] == 0:
            logger.info(f"Batch {mid} succeeded")
            best = mid
            lo = mid + 1
        elif result["oom_detected"]:
            logger.info(f"Batch {mid} failed with OOM")
            hi = mid - 1
        else:
            raise RuntimeError(
                f"Probe binary search failed with non-OOM error at batch {mid}. See logs: {run_dir}"
            )

    logger.info(f"Probe complete. Max batch size: {best}")
    return max(1, best), records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OPT experiment matrix with batch-size sweep and repeats.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, default="./results/opt_experiments")
    parser.add_argument("--launcher", type=str, default="python3 launch.py")
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-sec", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--logging-level", type=str, default="INFO")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--hf-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=100000000)
    parser.add_argument("--probe-start-batch", type=int, default=16)
    parser.add_argument("--probe-max-batch", type=int, default=1024)
    parser.add_argument("--probe-duration-sec", type=float, default=20.0)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--duration-tolerance-sec", type=float, default=5.0)
    parser.add_argument(
        "--require-valid-duration",
        type=int,
        default=1,
        help="Require all runs to satisfy duration tolerance for matrix completeness (1=yes, 0=no).",
    )
    parser.add_argument(
        "--strict-matrix",
        type=int,
        default=1,
        help="Exit non-zero when matrix is incomplete (1=yes, 0=no).",
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=0,
        help="If >0, skip probing and use this value as the max batch size.",
    )
    parser.add_argument("--profiles", type=str, default="A,B,C")
    args = parser.parse_args()

    cwd = os.path.abspath(args.repo_root)
    launcher = parse_launcher(args.launcher)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.abspath(os.path.join(args.output_root, f"opt-run-{ts}"))
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "created_at": ts,
        "repo_root": cwd,
        "launcher": launcher,
        "git_commit": get_git_commit(cwd),
        "host": socket.gethostname(),
        "gpu_info": get_gpu_info(),
        "params": vars(args),
        "hyperparameters": {
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "hf_name": args.hf_name,
            "duration_sec": args.duration_sec,
            "duration_tolerance_sec": args.duration_tolerance_sec,
            "launcher": " ".join(launcher),
        },
        "probe_records": [],
        "batch_sweep": [],
        "runs": [],
        "matrix_status": {},
    }

    if args.fixed_batch_size > 0:
        max_batch = int(args.fixed_batch_size)
    elif args.skip_probe:
        if args.max_batch_size <= 0:
            raise ValueError("--max-batch-size must be >0 when --skip-probe is used")
        max_batch = args.max_batch_size
    else:
        max_batch, probe_records = probe_max_batch(args=args, cwd=cwd, launcher=launcher, out_dir=out_dir)
        manifest["probe_records"] = probe_records

    max_pow2 = nearest_pow2_floor(max_batch)
    batch_sweep = [max_pow2, max_pow2 // 2, max_pow2 // 4]
    batch_sweep = [b for b in batch_sweep if b >= 1]
    batch_sweep = sorted(set(batch_sweep), reverse=True)
    manifest["batch_sweep"] = batch_sweep

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    total_runs = len(batch_sweep) * len(profiles) * args.repeats
    current_run = 0
    start_time = time.time()

    logger = Logger("MAIN", LogLevel.INFO)
    logger.info(f"Starting experiment matrix with {len(batch_sweep)} batch sizes, {len(profiles)} profiles, {args.repeats} repeats = {total_runs} total runs")

    for batch_size in batch_sweep:
        for profile in profiles:
            for repeat in range(1, args.repeats + 1):
                current_run += 1
                elapsed_total = time.time() - start_time
                logger.info(f"Run {current_run}/{total_runs} (batch={batch_size}, profile={profile}, repeat={repeat}) - {elapsed_total:.1f}s elapsed")
                run_dir = os.path.join(
                    out_dir,
                    f"batch_{batch_size}",
                    f"profile_{profile}",
                    f"repeat_{repeat}",
                )
                os.makedirs(run_dir, exist_ok=True)
                cmd = launcher + build_common_args(args, batch_size=batch_size, repeat_id=repeat)
                cmd += profile_args(profile, run_dir=run_dir)
                if profile == "B":
                    # Deterministic file names for profile-B CodeCarbon outputs.
                    codecarbon_run_num = batch_size * 1000 + repeat
                    cmd += ["--trainer_stats_configs.codecarbon.run_num", str(codecarbon_run_num)]

                result = run_command(cmd=cmd, cwd=cwd, log_dir=run_dir)
                valid_duration = check_duration_validity(
                    elapsed_sec=result["elapsed_sec"],
                    target_sec=args.duration_sec,
                    tolerance_sec=args.duration_tolerance_sec,
                )
                record = {
                    "batch_size": batch_size,
                    "profile": profile,
                    "repeat": repeat,
                    "run_dir": run_dir,
                    "command": cmd,
                    "returncode": result["returncode"],
                    "elapsed_sec": result["elapsed_sec"],
                    "oom_detected": result["oom_detected"],
                    "stdout_path": result["stdout_path"],
                    "stderr_path": result["stderr_path"],
                    "valid_duration": valid_duration,
                    "duration_target_sec": args.duration_sec,
                    "duration_tolerance_sec": args.duration_tolerance_sec,
                    "hyperparameters": {
                        "seq_len": args.seq_len,
                        "dtype": args.dtype,
                        "learning_rate": args.learning_rate,
                        "num_workers": args.num_workers,
                        "max_samples": args.max_samples,
                        "seed": args.seed,
                        "hf_name": args.hf_name,
                    },
                }
                with open(os.path.join(run_dir, "run_timing.json"), "w", encoding="utf-8") as fp:
                    json.dump(record, fp, indent=2)
                manifest["runs"].append(record)

                status_str = "SUCCESS" if result["returncode"] == 0 else "FAILED"
                logger.info(
                    f"Run {current_run}/{total_runs} {status_str} "
                    f"({result['elapsed_sec']:.1f}s, valid_duration={valid_duration})"
                )

                if result["returncode"] != 0:
                    manifest["matrix_status"] = build_matrix_status(
                        runs=manifest["runs"],
                        batch_sweep=batch_sweep,
                        profiles=profiles,
                        repeats=args.repeats,
                        require_valid_duration=bool(args.require_valid_duration),
                    )
                    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fp:
                        json.dump(manifest, fp, indent=2)
                    logger.error(f"Run failed: batch={batch_size}, profile={profile}, repeat={repeat}")
                    logger.error(f"See logs under: {run_dir}")
                    return 1

    manifest["matrix_status"] = build_matrix_status(
        runs=manifest["runs"],
        batch_sweep=batch_sweep,
        profiles=profiles,
        repeats=args.repeats,
        require_valid_duration=bool(args.require_valid_duration),
    )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    total_elapsed = time.time() - start_time
    logger.info(f"Experiment matrix complete!")
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    logger.info(f"Output root: {out_dir}")
    logger.info(f"Max batch discovered: {max_batch}, using sweep: {batch_sweep}")
    logger.info(f"Matrix complete: {manifest['matrix_status'].get('matrix_complete', False)}")
    if bool(args.strict_matrix) and not bool(manifest["matrix_status"].get("matrix_complete", False)):
        logger.error("Matrix is incomplete under current validation settings.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
