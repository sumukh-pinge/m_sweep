#!/usr/bin/env python3
import argparse, os, shlex, subprocess, sys
from pathlib import Path
from datetime import datetime

GRID_MS = [
    (1,1,1), (1,2,1), (1,4,1), (1,8,1),
    (2,1,2), (2,2,2), (2,4,2), (2,8,2),
    (4,1,4), (4,2,4), (4,4,4), (4,8,4),
    (8,1,8), (8,2,8), (8,4,8), (8,8,8),
]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir_nq")
    ap.add_argument("--work_dir", default="/mnt/work")
    ap.add_argument("--data_root", default="/mnt/work/datasets")
    ap.add_argument("--run_root", default="/mnt/work/runs/nq_m_sweep")
    ap.add_argument("--bits_sq", type=int, default=4)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--select_nprobe", type=int, default=64)
    ap.add_argument("--k2_fixed", type=int, default=1000)
    ap.add_argument("--alphas", default="2,2,2")

    # adapter hyperparams (shared; same adapter used across all M configs)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--cands", type=int, default=2048)
    ap.add_argument("--teacher", default="cos")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", type=int, default=50000)
    ap.add_argument("--q_batch", type=int, default=64)
    ap.add_argument("--slug", default="auto")

    # which 4 configs this job should run: 0,1,2,3
    ap.add_argument("--group", type=int, required=True,
                    help="0..3; selects which 4 (ms1,ms2,ms3) configs to run")

    return ap.parse_args()

def main():
    a = parse_args()

    # pick our 4 configs
    start = a.group * 4
    end = min(start + 4, len(GRID_MS))
    ms_list = GRID_MS[start:end]
    if not ms_list:
        raise ValueError(f"Group {a.group} has no configs")

    # base dirs
    run_root = Path(a.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"g{a.group}_{ts}"
    results_dir = run_root / "results" / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"m_sweep_g{a.group}_{ts}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")

    # common adapter suffix: SAME adapter across all M configs
    adapter_suffix = (
        f" --slug {a.slug} --tau {a.tau} --beta {a.beta} --cands {a.cands} "
        f"--teacher {a.teacher} --lr {a.lr} --epochs {a.epochs} "
        f"--subset {a.subset} --q_batch {a.q_batch}"
    )

    with open(log_path, "w") as tee:
        print(f"Logging to {log_path}")

        for (m1, m2, m3) in ms_list:
            ms_str = f"{m1},{m2},{m3}"
            ms_tag = f"m{m1}-{m2}-{m3}"
            print(f"\n==== MS = {ms_str} ({ms_tag}) ====\n", flush=True)
            tee.write(f"\n==== MS = {ms_str} ({ms_tag}) ====\n")

            # 1) IVF + Adapter (for reference; IVF ignores ms, but we save per config if you like)
            ivf_csv = results_dir / f"ivf_with_adapter_{ms_tag}.csv"
            if not ivf_csv.exists():
                cmd_ivf = (
                    "python -u nq_cli.py "
                    f"--dataset {a.dataset} "
                    f"--work_dir {a.work_dir} "
                    f"--data_root {a.data_root} "
                    f"--run_root {run_root} "
                    f"--bits_sq {a.bits_sq} "
                    f"--nlist {a.nlist} "
                    f"--select_nprobe {a.select_nprobe} "
                    f"--k2_fixed {a.k2_fixed} "
                    f"--ms_infer {ms_str} "
                    f"--alphas {a.alphas} "
                    f"--mode ivf_fp32_adapter "
                    f"--out_csv {ivf_csv}"
                    f"{adapter_suffix}"
                )
                run_cmd(cmd_ivf, env=env, tee=tee)

            # 2) Dual + Adapter (D2-BAM)
            dual_csv = results_dir / f"dbam_dual_with_adapter_{ms_tag}.csv"
            if not dual_csv.exists():
                cmd_dual = (
                    "python -u nq_cli.py "
                    f"--dataset {a.dataset} "
                    f"--work_dir {a.work_dir} "
                    f"--data_root {a.data_root} "
                    f"--run_root {run_root} "
                    f"--bits_sq {a.bits_sq} "
                    f"--nlist {a.nlist} "
                    f"--select_nprobe {a.select_nprobe} "
                    f"--k2_fixed {a.k2_fixed} "
                    f"--ms_infer {ms_str} "
                    f"--alphas {a.alphas} "
                    f"--mode dbam_dual_adapter "
                    f"--out_csv {dual_csv}"
                    f"{adapter_suffix}"
                )
                run_cmd(cmd_dual, env=env, tee=tee)

        print("\n✅ M-sweep group complete.")
        print(f"📂 Results in: {results_dir}")

if __name__ == "__main__":
    main()
