#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd: str, env=None, tee=None):
    print(f"\n$ {cmd}\n", flush=True)
    p = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env or os.environ.copy(),
    )
    for line in p.stdout:
        sys.stdout.write(line)
        if tee:
            tee.write(line)
            tee.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="M-sweep for IVF+Adapter vs Dual+Adapter with m1==m3 and m2 sweep."
    )

    # dataset & paths
    ap.add_argument("--dataset", default="beir_nq")
    ap.add_argument("--work_dir", default="/mnt/work")
    ap.add_argument("--data_root", default="/mnt/work/datasets")
    ap.add_argument("--run_root", default="/mnt/work/runs/m_sweep")
    ap.add_argument("--intermediate_root", default=None)

    # retrieval knobs
    ap.add_argument("--bits_sq", type=int, default=4)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--select_nprobe", type=int, default=64)
    ap.add_argument("--k2_fixed", type=int, default=1000)
    ap.add_argument("--alphas", default="2,2,2")

    # adapter hyperparams (shared adapter across all configs in this job)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--cands", type=int, default=2048)
    ap.add_argument("--teacher", default="cos")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", type=int, default=50000)
    ap.add_argument("--q_batch", type=int, default=64)
    ap.add_argument("--slug", default="auto")

    # M sweep: m1 == m3 fixed per job, sweep m2 list
    ap.add_argument(
        "--m13",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        required=True,
        help="Shared m for stages 1 and 3 (s1 = s3 = m13).",
    )
    ap.add_argument(
        "--s2_values",
        default="1,2,4,8",
        help="Comma-separated list of m2 values to sweep (subset of 1,2,4,8).",
    )

    # group: 4 configs per job
    ap.add_argument(
        "--group",
        type=int,
        required=True,
        help="0-based group index; each group runs up to 4 (m1,m2,m3) configs.",
    )

    # optional tag
    ap.add_argument(
        "--results_tag",
        default="",
        help="Optional custom tag for this job's results directory.",
    )

    return ap.parse_args()


def auto_slug(a):
    if a.slug != "auto":
        return a.slug
    s = (
        f"tau{a.tau}_b{a.beta}_c{a.cands}_{a.teacher}_"
        f"lr{a.lr}_e{a.epochs}_s{a.subset}"
    )
    return s.replace(".", "")


def main():
    a = parse_args()
    a.slug = auto_slug(a)

    # ----- Build (m1,m2,m3) grid: m1=m3=m13, m2 from s2_values -----
    s2_raw = [x.strip() for x in a.s2_values.split(",") if x.strip()]
    allowed = {"1", "2", "4", "8", "16", "32"}
    s2_list = [int(x) for x in s2_raw if x in allowed]
    if not s2_list:
        raise ValueError(
            "No valid m2 values from --s2_values; must be subset of 1,2,4,8."
        )

    full_grid = [(a.m13, m2, a.m13) for m2 in s2_list]

    start = a.group * 4
    end = min(start + 4, len(full_grid))
    ms_list = full_grid[start:end]
    if not ms_list:
        raise ValueError(
            f"Group {a.group} has no configs to run (grid size={len(full_grid)})."
        )

    # ----- Directories -----
    run_root = Path(a.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = a.results_tag.strip() or f"m13{a.m13}_g{a.group}_{ts}"
    results_dir = run_root / "results" / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"m_sweep_{tag}.log"

    # ----- Environment -----
    env = os.environ.copy()
    env.setdefault("WORK_DIR", a.work_dir)
    env.setdefault("DATA_ROOT", a.data_root)
    env.setdefault("RUN_ROOT", str(run_root))
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")

    with open(log_path, "w") as tee:
        print(f"Logging to {log_path}", flush=True)
        tee.write(f"# M-sweep {tag}\n")
        tee.flush()

        for (m1, m2, m3) in ms_list:
            ms_str = f"{m1},{m2},{m3}"
            ms_tag = f"m{m1}-{m2}-{m3}"

            tee.write(f"\n## Config {ms_tag}\n")
            tee.flush()

            # ---------------- IVF + Adapter ----------------
            ivf_csv = results_dir / f"ivf_adapter_{ms_tag}.csv"
            if not ivf_csv.exists():
                cmd = (
                    "python -u nq_cli.py "
                    f"--dataset {a.dataset} "
                    f"--work_dir {a.work_dir} "
                    f"--data_root {a.data_root} "
                    f"--run_root {run_root} "
                    f"--bits_sq {a.bits_sq} "
                    f"--nlist {a.nlist} "
                    f"--select_nprobe {a.select_nprobe} "
                    f"--k2_fixed {a.k2_fixed} "
                    f"--kfinal 10 25 50 100 "
                    f"--ms_infer {ms_str} "
                    f"--alphas {a.alphas} "
                    f"--tau {a.tau} "
                    f"--beta {a.beta} "
                    f"--cands {a.cands} "
                    f"--teacher {a.teacher} "
                    f"--lr {a.lr} "
                    f"--epochs {a.epochs} "
                    f"--subset {a.subset} "
                    f"--q_batch {a.q_batch} "
                    f"--slug {a.slug} "
                    f"--mode ivf_fp32_adapter "
                    f"--out_csv {ivf_csv}"
                )
                run_cmd(cmd, env=env, tee=tee)
            else:
                msg = f"[skip] {ivf_csv} exists"
                print(msg, flush=True)
                tee.write(msg + "\n")

            # ---------------- Dual (D2-BAM) + Adapter ----------------
        #     dual_csv = results_dir / f"dual_adapter_{ms_tag}.csv"
        #     if not dual_csv.exists():
        #         cmd = (
        #             "python -u nq_cli.py "
        #             f"--dataset {a.dataset} "
        #             f"--work_dir {a.work_dir} "
        #             f"--data_root {a.data_root} "
        #             f"--run_root {run_root} "
        #             f"--bits_sq {a.bits_sq} "
        #             f"--nlist {a.nlist} "
        #             f"--select_nprobe {a.select_nprobe} "
        #             f"--k2_fixed {a.k2_fixed} "
        #             f"--kfinal 10 25 50 100 "
        #             f"--ms_infer {ms_str} "
        #             f"--alphas {a.alphas} "
        #             f"--tau {a.tau} "
        #             f"--beta {a.beta} "
        #             f"--cands {a.cands} "
        #             f"--teacher {a.teacher} "
        #             f"--lr {a.lr} "
        #             f"--epochs {a.epochs} "
        #             f"--subset {a.subset} "
        #             f"--q_batch {a.q_batch} "
        #             f"--slug {a.slug} "
        #             f"--mode dbam_dual_adapter "
        #             f"--out_csv {dual_csv}"
        #         )
        #         run_cmd(cmd, env=env, tee=tee)
        #     else:
        #         msg = f"[skip] {dual_csv} exists"
        #         print(msg, flush=True)
        #         tee.write(msg + "\n")

        # print("\n✅ m-sweep finished.", flush=True)
        # tee.write("\n✅ m-sweep finished.\n")


if __name__ == "__main__":
    main()
