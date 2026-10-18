"""MOECLF 对照实验：训练阶段 + 测试阶段（对应论文 Table 5 / Table 6 协议）。

按论文的两阶段协议组织，不做"在测试实例上边运行边学习"的处理：

  训练阶段（training stage）
      在 10 个 45×6 训练实例上依次运行（每个实例 1 个 episode）。
      两个控制器（Q-learning 与 PPO）的 agent 跨 episode 持续累积经验、不重置。
      训练结束后保存学到的策略（Q 表 / PPO 权重）。

  测试阶段（testing stage）
      载入训练好的策略并冻结（train=False，不再更新参数），
      在 6 个代表实例上独立运行。非学习算法（Random / MODJA / NSGA-II）
      无需训练，直接在测试实例上运行。

  汇总阶段
      计算 HV / GD / IGD / 集合覆盖率 C / Wilcoxon 秩和检验，写出结果表。

用法:
  python run_experiments.py                  # 训练 + 测试 + 汇总（论文规模）
  python run_experiments.py 2                # 只汇总实验2
  python run_experiments.py --workers 9
  python run_experiments.py --population 30 --generations 40 --seeds 1 \\
      --instances J15-S2-1 --train-instances J45-S6-1   # 快速冒烟
"""
from __future__ import annotations

import argparse
import os
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
from baselines import MODJA, NSGAII
from ehfsp_hmc import fast_non_dominated_sort, load_instances
from metrics import gd, hv, igd, normalize_objs, set_coverage, wilcoxon_ranksum
from moeclf import MOECLF

DEVICE = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(1)

Np, Ngen, FSM = 200, 1000, 46
SEEDS = [0, 1, 2, 3, 4]
JSON = "EHFSP-HMC.json"

# 训练实例：论文取 10 个 45 工件 / 6 阶段的实例
TRAIN_INSTANCES = [f"J45-S6-{i}" for i in range(1, 11)]
# 测试实例：6 个代表实例（覆盖 J15/J30/J45 与 2/4/6 阶段）
TEST_INSTANCES = ["J15-S2-1", "J15-S4-1", "J30-S2-1", "J30-S4-1", "J45-S4-1", "J45-S6-1"]

MODEL_DIR = os.path.join("results", "_model")   # 训练好的策略
CACHE_DIR = os.path.join("results", "_cache")   # 测试阶段的每个 (算法, 实例, 种子) 前沿

# 需要训练的控制器：'q' = Q-learning；'full' = PPO 管边界个体 + Q 管其余
LEARNED = ["q", "full"]
# 无需训练的算法：'random' = 随机选算子；'modja' / 'nsga2' = 对比算法
NON_LEARNED = ["random", "modja", "nsga2"]

# 两个实验的算法组成与显示名
EXPERIMENTS = {
    "2": {"random": "Random", "q": "Q-only", "full": "PPO+Q"},
    "3": {"modja": "MODJA", "nsga2": "NSGA-II", "q": "MOEQLF", "full": "MOECLF"},
}

OUT_NAMES = {"2": "实验2_控制器对比", "3": "实验3_算法对比"}


def out_path_for(label: str) -> str:
    return os.path.join("results", f"{OUT_NAMES[label]}_Np{Np}_Ngen{Ngen}.md")


# --------------------------------------------------------------------------
# 训练阶段
# --------------------------------------------------------------------------


def make_agents(n_actions: int = 10):
    ppo = [PPOAgent(FSM, FSM, device=DEVICE, n_actions=n_actions) for _ in range(2)]
    q = QLearningAgent(n_actions=n_actions)
    return ppo, q


def model_dir_for(controller: str) -> str:
    return os.path.join(MODEL_DIR, f"Np{Np}_Ngen{Ngen}", controller)


def save_agents(controller: str, ppo, q):
    out = model_dir_for(controller)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "Q.npy"), q.Q)
    if controller == "full":
        for i, a in enumerate(ppo):
            torch.save(a.policy.state_dict(), os.path.join(out, f"ppo{i}.pt"))


def load_agents(controller: str):
    ppo, q = make_agents()
    d = model_dir_for(controller)
    q.Q = np.load(os.path.join(d, "Q.npy"))
    if controller == "full":
        for i, a in enumerate(ppo):
            a.policy.load_state_dict(
                torch.load(os.path.join(d, f"ppo{i}.pt"), map_location=DEVICE)
            )
    return ppo, q


def models_ready() -> bool:
    return all(os.path.exists(os.path.join(model_dir_for(c), "Q.npy")) for c in LEARNED)


def train_controller(controller: str):
    """训练阶段：多个 episode 共享同一组 agent，经验跨实例累积。"""
    ppo, q = make_agents()
    by_name = dict(load_instances(JSON))
    t0 = time.time()
    for ep, name in enumerate(TRAIN_INSTANCES):
        inst = by_name[name]
        solver = MOECLF(inst, Np=Np, Ngen=Ngen, fsm_rows=FSM, fsm_cols=FSM,
                        ppo=ppo, q_agent=q, seed=ep, batch_size=128, update_epochs=10,
                        controller=controller)
        solver.run(train=True)
        print(f"  [train {controller}] ep {ep + 1}/{len(TRAIN_INSTANCES)} ({name}) "
              f"done, {time.time() - t0:.0f}s", flush=True)
    save_agents(controller, ppo, q)
    print(f"[train {controller}] saved, total {time.time() - t0:.0f}s", flush=True)
    return controller


# --------------------------------------------------------------------------
# 测试阶段
# --------------------------------------------------------------------------

_G = {}


def _init_worker():
    torch.set_num_threads(1)
    _G["by_name"] = dict(load_instances(JSON))
    _G["agents"] = {c: load_agents(c) for c in LEARNED if models_ready()}


def _test_job(job):
    algo, inst_name, seed = job
    try:
        inst = _G["by_name"][inst_name]
        if algo == "modja":
            front = np.array([s.obj for s in MODJA(inst, Np=Np, Ngen=Ngen, seed=seed).run()])
        elif algo == "nsga2":
            front = np.array([s.obj for s in NSGAII(inst, Np=Np, Ngen=Ngen, seed=seed).run()])
        else:
            ppo, q = _G["agents"][algo] if algo in LEARNED else (None, None)
            solver = MOECLF(inst, Np=Np, Ngen=Ngen, fsm_rows=FSM, fsm_cols=FSM,
                            ppo=ppo, q_agent=q, seed=seed, batch_size=128,
                            update_epochs=10, controller=algo)
            # 测试阶段：冻结策略，不再更新参数
            front = np.array([x.obj for x in solver.run(train=False)])
        return algo, inst_name, seed, front, None
    except Exception:
        return algo, inst_name, seed, None, traceback.format_exc()


def cache_path(algo: str, inst: str, seed: int) -> str:
    return os.path.join(CACHE_DIR, f"Np{Np}_Ngen{Ngen}", algo, inst, f"{seed}.npy")


def load_fronts(names):
    fronts = {n: {} for n in names}
    for n in names:
        for inst in TEST_INSTANCES:
            fronts[n][inst] = [np.load(cache_path(n, inst, s)) for s in SEEDS]
    return fronts


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def nondominated(objs):
    _, fronts = fast_non_dominated_sort(objs)
    return objs[fronts[0]] if fronts else objs


def render(title: str, labels: dict, fronts: dict) -> str:
    names = list(labels)
    L = [f"\n## {title}\n"]
    L.append("| 实例 | " + " | ".join(labels[n] for n in names) + " |")
    L.append("|" + "---|" * (len(names) + 1))

    hv_table = {n: [] for n in names}
    for inst in TEST_INSTANCES:
        allpts = np.vstack([f for n in names for f in fronts[n][inst]])
        P = nondominated(allpts)
        fmin, fmax = allpts.min(0), allpts.max(0)
        Pn = normalize_objs(P, fmin, fmax)

        row = [f"**{inst}**"]
        for n in names:
            hs = [hv(normalize_objs(f, fmin, fmax)) for f in fronts[n][inst]]
            hv_table[n].extend(hs)
            gs = [gd(normalize_objs(f, fmin, fmax), Pn) for f in fronts[n][inst]]
            igs = [igd(Pn, normalize_objs(f, fmin, fmax)) for f in fronts[n][inst]]
            row.append(f"HV {np.mean(hs):.3f}<br>GD {np.mean(gs):.3f}<br>"
                       f"IGD {np.mean(igs):.3f}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n### 集合覆盖率 C（行算法支配列算法的比例）\n")
    L.append("| C(U,V) | " + " | ".join(labels[n] for n in names) + " |")
    L.append("|" + "---|" * (len(names) + 1))
    for n1 in names:
        row = [f"**{labels[n1]}**"]
        for n2 in names:
            if n1 == n2:
                row.append("—")
            else:
                cs = [set_coverage(fronts[n1][i][s], fronts[n2][i][s])
                      for i in TEST_INSTANCES for s in range(len(SEEDS))]
                row.append(f"{np.mean(cs):.3f}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n### Wilcoxon 秩和检验 p 值（行 vs 列，基于 HV，各种子×实例）\n")
    L.append("| p | " + " | ".join(labels[n] for n in names) + " |")
    L.append("|" + "---|" * (len(names) + 1))
    for n1 in names:
        row = [f"**{labels[n1]}**"]
        for n2 in names:
            row.append("—" if n1 == n2 else f"{wilcoxon_ranksum(hv_table[n1], hv_table[n2]):.2e}")
        L.append("| " + " | ".join(row) + " |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    global Np, Ngen, SEEDS, TEST_INSTANCES, TRAIN_INSTANCES
    ap = argparse.ArgumentParser(description="MOECLF 对照实验（训练阶段 + 测试阶段）")
    ap.add_argument("which", nargs="*", default=["2", "3"])
    ap.add_argument("--population", type=int, default=Np)
    ap.add_argument("--generations", type=int, default=Ngen)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--instances", nargs="+", default=TEST_INSTANCES)
    ap.add_argument("--train-instances", nargs="+", default=TRAIN_INSTANCES)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    Np, Ngen = args.population, args.generations
    SEEDS = list(range(args.seeds))
    TEST_INSTANCES = args.instances
    TRAIN_INSTANCES = args.train_instances
    DEVICE = args.device

    labels = args.which or ["2", "3"]
    print(f"DEVICE={DEVICE} Np={Np} Ngen={Ngen} seeds={SEEDS}\n"
          f"train={TRAIN_INSTANCES}\ntest={TEST_INSTANCES}", flush=True)

    overlap = [i for i in TEST_INSTANCES if i in TRAIN_INSTANCES]
    if overlap:
        print(f"警告：测试集含训练实例 {overlap}，严格意义上不属于 unseen 实例。", flush=True)

    # ---- 训练阶段 ----
    if not models_ready():
        print("\n=== 训练阶段：10 个训练实例 × 1 episode，agent 跨实例累积经验 ===", flush=True)
        t0 = time.time()
        with Pool(min(2, len(LEARNED))) as pool:
            pool.map(train_controller, LEARNED)
        print(f"训练阶段完成，耗时 {time.time() - t0:.0f}s", flush=True)
    else:
        print("\n训练阶段已完成，直接载入策略", flush=True)

    # ---- 测试阶段 ----
    needed = sorted({a for k in labels if k in EXPERIMENTS for a in EXPERIMENTS[k]})
    jobs = [(a, i, s) for a in needed for i in TEST_INSTANCES for s in SEEDS
            if not os.path.exists(cache_path(a, i, s))]
    total = len(needed) * len(TEST_INSTANCES) * len(SEEDS)
    print(f"\n=== 测试阶段：需运行 {len(jobs)}/{total} 个任务（策略冻结）===", flush=True)

    if jobs:
        t0 = time.time()
        done = 0
        with Pool(args.workers, initializer=_init_worker) as pool:
            for algo, inst, seed, front, err in pool.imap_unordered(_test_job, jobs):
                if err is not None:
                    print(f"ERROR {algo}/{inst}/{seed}\n{err}", flush=True)
                    continue
                p = cache_path(algo, inst, seed)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, front)
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  {done}/{len(jobs)}  已用 {el:.0f}s  "
                          f"ETA {el / done * (len(jobs) - done):.0f}s", flush=True)

    # ---- 汇总阶段 ----
    print("\n=== 汇总阶段 ===", flush=True)
    os.makedirs("results", exist_ok=True)
    for label in labels:
        if label not in EXPERIMENTS:
            print(f"未知实验: {label}", flush=True)
            continue
        labels_map = EXPERIMENTS[label]
        fronts = load_fronts(list(labels_map))
        text = render(f"实验{label} " + ("控制器对比" if label == "2" else "算法对比"),
                      labels_map, fronts)
        out = out_path_for(label)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
