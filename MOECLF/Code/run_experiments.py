"""MOECLF 三组对照实验 —— 并行 + 断点续跑版。

用法:
  python run_experiments.py            # 跑全部 3 组（默认 Np=200/Ngen=1000/5 种子/6 实例）
  python run_experiments.py 1 2 3      # 只跑指定实验
  python run_experiments.py --workers 14 --population 200 --generations 1000

每个 (算法, 实例, 种子) 的结果存到 results/_cache/<sig>/<inst>/<seed>.npy，
中断后重跑会自动跳过已完成的；每个实验跑完立即写 results/<label>.md。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

# 单线程 numpy/torch，避免多进程下 OpenMP 抢核（必须在 import numpy 之前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from agents import PPOAgent, QLearningAgent
from baselines import NSGAII, MODJA
from ehfsp_hmc import fast_non_dominated_sort, load_instances
from metrics import gd, hv, igd, normalize_objs, set_coverage, wilcoxon_ranksum
from moeclf import MOECLF

DEVICE = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(1)

Np, Ngen, FSM = 200, 1000, 46
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = ["J15-S2-1", "J15-S4-1", "J30-S2-1", "J30-S4-1", "J45-S4-1", "J45-S6-1"]
CACHE_DIR = os.path.join("results", "_cache")
JSON = "EHFSP-HMC.json"

LABELS = {"1": "实验1 消融", "2": "实验2 控制器对比", "3": "实验3 算法对比"}

# 输出文件名带上规模，避免不同规模的结果互相覆盖，也便于直接辨认
OUT_NAMES = {"1": "实验1_消融", "2": "实验2_控制器对比", "3": "实验3_算法对比"}


def out_path_for(label: str) -> str:
    """按当前 Np/Ngen 生成结果文件名，例如 实验1_消融_Np200_Ngen1000.md"""
    stem = OUT_NAMES.get(label, LABELS.get(label, label))
    return os.path.join("results", f"{stem}_Np{Np}_Ngen{Ngen}.md")


# --------------------------------------------------------------------------
# 算法工厂：spec 字典 -> runner(inst, seed) -> (n,2) front
# --------------------------------------------------------------------------


def _moeclf(spec):
    controller = spec["controller"]
    use_msi = spec.get("use_msi", True)
    use_vns = spec.get("use_vns", True)
    use_ep = spec.get("use_ep", True)
    np_, ngen, fsm, dev = spec["Np"], spec["Ngen"], spec["FSM"], spec["device"]

    def runner(inst, seed):
        n_actions = 10 if use_vns else 3
        ppo = [PPOAgent(fsm, fsm, device=dev, n_actions=n_actions) for _ in range(2)]
        q = QLearningAgent(n_actions=n_actions)
        s = MOECLF(inst, Np=np_, Ngen=ngen, fsm_rows=fsm, fsm_cols=fsm, ppo=ppo, q_agent=q,
                   seed=seed, batch_size=128, update_epochs=10,
                   use_msi=use_msi, use_vns=use_vns, use_ep=use_ep, controller=controller)
        return np.array([x.obj for x in s.run(train=True)])

    return runner


def _nsga2(spec):
    np_, ngen = spec["Np"], spec["Ngen"]

    def runner(inst, seed):
        return np.array([s.obj for s in NSGAII(inst, Np=np_, Ngen=ngen, seed=seed).run()])

    return runner


def _modja(spec):
    np_, ngen = spec["Np"], spec["Ngen"]

    def runner(inst, seed):
        return np.array([s.obj for s in MODJA(inst, Np=np_, Ngen=ngen, seed=seed).run()])

    return runner


_RUNNER_BUILDERS = {"moeclf": _moeclf, "nsga2": _nsga2, "modja": _modja}


# --------------------------------------------------------------------------
# 三组实验定义（label -> {算法名: spec}）
# --------------------------------------------------------------------------

EXPERIMENTS = {
    "1": {
        "v1(无MSI/VNS/EP/PPO)": dict(type="moeclf", controller="q", use_msi=False, use_vns=False, use_ep=False),
        "v2(+MSI)": dict(type="moeclf", controller="q", use_msi=True, use_vns=False, use_ep=False),
        "v3(+VNS)": dict(type="moeclf", controller="q", use_msi=True, use_vns=True, use_ep=False),
        "v4(+EP=MOEQLF)": dict(type="moeclf", controller="q", use_msi=True, use_vns=True, use_ep=True),
        "MOECLF(+PPO)": dict(type="moeclf", controller="full", use_msi=True, use_vns=True, use_ep=True),
    },
    "2": {
        "Random": dict(type="moeclf", controller="random", use_msi=True, use_vns=True, use_ep=True),
        "Q-only": dict(type="moeclf", controller="q", use_msi=True, use_vns=True, use_ep=True),
        "PPO+Q": dict(type="moeclf", controller="full", use_msi=True, use_vns=True, use_ep=True),
    },
    "3": {
        "MODJA": dict(type="modja"),
        "NSGA-II": dict(type="nsga2"),
        "MOEQLF": dict(type="moeclf", controller="q", use_msi=True, use_vns=True, use_ep=True),
        "MOECLF": dict(type="moeclf", controller="full", use_msi=True, use_vns=True, use_ep=True),
    },
}


# --------------------------------------------------------------------------
# 缓存：结果按 (Np, Ngen, spec) 稳定签名去重，跨实验复用同一 (spec, inst, seed)
# --------------------------------------------------------------------------


def spec_sig(spec: dict) -> str:
    canon = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.md5(canon.encode()).hexdigest()[:16]


def cache_path(spec: dict, inst: str, seed: int) -> str:
    return os.path.join(CACHE_DIR, spec_sig(spec), inst, f"{seed}.npy")


def _finalize(spec: dict) -> dict:
    """把运行配置显式打包进 spec，随 job 传给子进程（spawn 会重新导入模块，
    模块级全局 Np/Ngen 在子进程里是默认值，不能依赖）。"""
    s = dict(spec)
    s["Np"] = Np
    s["Ngen"] = Ngen
    s["FSM"] = FSM
    s["device"] = DEVICE
    return s


# --------------------------------------------------------------------------
# worker（多进程，Windows spawn：模块级函数）
# --------------------------------------------------------------------------

_G = {}


def _init_worker():
    torch.set_num_threads(1)
    _G["by_name"] = dict(load_instances(JSON))


def _run_job(job):
    spec, inst_name, seed = job
    try:
        inst = _G["by_name"][inst_name]
        runner = _RUNNER_BUILDERS[spec["type"]](spec)
        front = runner(inst, seed)
        return spec, inst_name, seed, front, None
    except Exception:
        return spec, inst_name, seed, None, traceback.format_exc()


# --------------------------------------------------------------------------
# 主流程：逐实验运行（并行），每个实验跑完立即写 md
# --------------------------------------------------------------------------


def run_experiment(label: str, algos: dict, workers: int):
    fin = {name: _finalize(spec) for name, spec in algos.items()}

    jobs = []
    for algo_name, spec in fin.items():
        for inst in INSTANCES:
            for seed in SEEDS:
                if not os.path.exists(cache_path(spec, inst, seed)):
                    jobs.append((spec, inst, seed))

    total = len(fin) * len(INSTANCES) * len(SEEDS)
    print(f"\n[{LABELS[label]}] 需运行 {len(jobs)}/{total} 个任务（其余已缓存）", flush=True)

    if jobs:
        failures = 0
        done = 0
        with Pool(workers, initializer=_init_worker) as pool:
            for spec, inst_name, seed, front, err in pool.imap_unordered(_run_job, jobs):
                if err is not None:
                    failures += 1
                    print(f"ERROR {err}", flush=True)
                    continue
                cp = cache_path(spec, inst_name, seed)
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.save(cp, front)
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"  [{LABELS[label]}] {done}/{len(jobs)} 完成", flush=True)
        if failures:
            print(f"  [{LABELS[label]}] {failures} 个任务失败", flush=True)

    fronts = load_fronts(fin)
    text = render_experiment(label, fin, fronts)
    out_path = out_path_for(label)
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out_path}", flush=True)


def load_fronts(algos: dict):
    fronts = {}
    for algo_name, spec in algos.items():
        fronts[algo_name] = {}
        for inst in INSTANCES:
            fronts[algo_name][inst] = [
                np.load(cache_path(spec, inst, seed)) for seed in SEEDS
            ]
    return fronts


def nondominated(objs):
    _, fronts = fast_non_dominated_sort(objs)
    return objs[fronts[0]] if fronts else objs


def render_experiment(label: str, algos: dict, fronts: dict) -> str:
    names = list(algos)
    lines = [f"\n## {LABELS[label]}\n", "| 实例 | " + " | ".join(names) + " |"]
    lines.append("|" + "---|" * (len(names) + 1))
    hv_table = {n: [] for n in names}
    for i in INSTANCES:
        allpts = np.vstack([f for n in names for f in fronts[n][i]])
        P = nondominated(allpts)
        fmin, fmax = allpts.min(0), allpts.max(0)
        Pn = normalize_objs(P, fmin, fmax)

        row = [f"**{i}**"]
        for n in names:
            hs = []
            for f in fronts[n][i]:
                fn = normalize_objs(f, fmin, fmax)
                hs.append(hv(fn))
            hv_table[n].extend(hs)
            mean_hv = float(np.mean(hs))
            gs = [gd(normalize_objs(f, fmin, fmax), Pn) for f in fronts[n][i]]
            igs = [igd(Pn, normalize_objs(f, fmin, fmax)) for f in fronts[n][i]]
            row.append(
                f"HV {mean_hv:.3f}<br>GD {np.mean(gs):.3f}<br>IGD {np.mean(igs):.3f}"
            )
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n### 集合覆盖率 C（行算法支配列算法的比例）\n")
    lines.append("| C(U,V) | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for n1 in names:
        row = [f"**{n1}**"]
        for n2 in names:
            if n1 == n2:
                row.append("—")
            else:
                cs = [set_coverage(fronts[n1][i][s], fronts[n2][i][s])
                      for i in INSTANCES for s in range(len(SEEDS))]
                row.append(f"{np.mean(cs):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n### Wilcoxon rank-sum p 值（行 vs 列，基于 HV，5 种子×6 实例）\n")
    lines.append("| p | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for n1 in names:
        row = [f"**{n1}**"]
        for n2 in names:
            if n1 == n2:
                row.append("—")
            else:
                p = wilcoxon_ranksum(hv_table[n1], hv_table[n2])
                row.append(f"{p:.2e}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    global Np, Ngen, SEEDS, INSTANCES, DEVICE
    ap = argparse.ArgumentParser(description="MOECLF 三组对照实验（并行+断点续跑）")
    ap.add_argument("which", nargs="*", default=["1", "2", "3"])
    ap.add_argument("--population", type=int, default=Np)
    ap.add_argument("--generations", type=int, default=Ngen)
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    Np, Ngen = args.population, args.generations
    SEEDS = list(range(args.seeds))
    INSTANCES = args.instances
    DEVICE = args.device
    workers = args.workers

    labels = args.which if args.which else ["1", "2", "3"]
    print(f"DEVICE={DEVICE}, Np={Np}, Ngen={Ngen}, seeds={SEEDS}, "
          f"instances={INSTANCES}, workers={workers}", flush=True)

    for label in labels:
        if label not in EXPERIMENTS:
            print(f"未知实验: {label}", flush=True)
            continue
        t0 = time.time()
        run_experiment(label, EXPERIMENTS[label], workers)
        print(f"[{LABELS[label]}] 耗时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
