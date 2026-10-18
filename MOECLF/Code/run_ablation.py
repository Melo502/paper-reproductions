"""消融实验（两阶段协议）：v1 → v2(+MSI) → v3(+VNS) → v4(+EP) → MOECLF(+PPO)。

与主实验相同，采用论文的两阶段协议：
  训练阶段：在 J45-S6-1..10 上依次训练，agent 跨实例累积经验（不重置）
  测试阶段：载入训练所得策略，在 6 个测试实例上冻结运行（train=False）

v4（= MOEQLF，全部模块开启、控制器为 Q）与 MOECLF（控制器为 full）
的两阶段结果与主实验共用同一份模型与前沿缓存，直接复用，不重复计算。

用法:
  python run_ablation.py                 # 训练 + 测试 + 汇总
  python run_ablation.py --workers 9
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
from ehfsp_hmc import load_instances
from moeclf import MOECLF
from report import render

DEVICE = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(1)

Np, Ngen, FSM = 200, 1000, 46
SEEDS = [0, 1, 2, 3, 4]
JSON = "EHFSP-HMC.json"

TRAIN_INSTANCES = [f"J45-S6-{i}" for i in range(1, 11)]
TEST_INSTANCES = ["J15-S2-1", "J15-S4-1", "J30-S2-1", "J30-S4-1", "J45-S4-1", "J45-S6-1"]

MODEL_ROOT = os.path.join("results", "_model", f"Np{Np}_Ngen{Ngen}")
CACHE_ROOT = os.path.join("results", "_cache", f"Np{Np}_Ngen{Ngen}")

# 消融配置：逐级开启 MSI / VNS / EP。控制器固定为 Q，
# 使差异只来自进化模块本身（论文 Table 6 的做法）。
CONFIGS = {
    "v1": {"label": "v1 基线", "use_msi": False, "use_vns": False, "use_ep": False},
    "v2": {"label": "v2 +MSI", "use_msi": True, "use_vns": False, "use_ep": False},
    "v3": {"label": "v3 +VNS", "use_msi": True, "use_vns": True, "use_ep": False},
    "v4": {"label": "v4 +EP (=MOEQLF)", "use_msi": True, "use_vns": True, "use_ep": True},
}

# v4 与 MOECLF 复用主实验的两阶段结果（同一协议、同一模型）
REUSE = {"v4": "q", "moeclf": "full"}
LABELS = {**{k: c["label"] for k, c in CONFIGS.items()}, "moeclf": "MOECLF +PPO"}


def n_actions_for(cfg: dict) -> int:
    return 10 if cfg["use_vns"] else 3


def model_dir(key: str) -> str:
    return os.path.join(MODEL_ROOT, f"abl_{key}")


def cache_path(key: str, inst: str, seed: int) -> str:
    return os.path.join(CACHE_ROOT, f"abl_{key}", inst, f"{seed}.npy")


def make_agents(n_actions: int):
    ppo = [PPOAgent(FSM, FSM, device=DEVICE, n_actions=n_actions) for _ in range(2)]
    return ppo, QLearningAgent(n_actions=n_actions)


def save_agents(key: str, q):
    out = model_dir(key)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "Q.npy"), q.Q)


def load_agents(key: str):
    n_act = n_actions_for(CONFIGS[key])
    ppo, q = make_agents(n_act)
    q.Q = np.load(os.path.join(model_dir(key), "Q.npy"))
    return ppo, q


# --------------------------------------------------------------------------
# 训练阶段
# --------------------------------------------------------------------------


def train_one(key: str):
    cfg = CONFIGS[key]
    n_act = n_actions_for(cfg)
    ppo, q = make_agents(n_act)
    by_name = dict(load_instances(JSON))
    t0 = time.time()
    for ep, name in enumerate(TRAIN_INSTANCES):
        solver = MOECLF(by_name[name], Np=Np, Ngen=Ngen, fsm_rows=FSM, fsm_cols=FSM,
                        ppo=ppo, q_agent=q, seed=ep, batch_size=128, update_epochs=10,
                        controller="q", use_msi=cfg["use_msi"], use_vns=cfg["use_vns"],
                        use_ep=cfg["use_ep"])
        solver.run(train=True)
        print(f"  [train {key}] ep {ep + 1}/{len(TRAIN_INSTANCES)} ({name}) "
              f"{time.time() - t0:.0f}s", flush=True)
    save_agents(key, q)
    print(f"[train {key}] saved, {time.time() - t0:.0f}s", flush=True)
    return key


# --------------------------------------------------------------------------
# 测试阶段
# --------------------------------------------------------------------------

_G = {}


def _init_worker(keys):
    torch.set_num_threads(1)
    _G["by_name"] = dict(load_instances(JSON))
    _G["agents"] = {k: load_agents(k) for k in keys}


def _test_job(job):
    key, inst_name, seed = job
    try:
        cfg = CONFIGS[key]
        ppo, q = _G["agents"][key]
        solver = MOECLF(_G["by_name"][inst_name], Np=Np, Ngen=Ngen, fsm_rows=FSM,
                        fsm_cols=FSM, ppo=ppo, q_agent=q, seed=seed, batch_size=128,
                        update_epochs=10, controller="q", use_msi=cfg["use_msi"],
                        use_vns=cfg["use_vns"], use_ep=cfg["use_ep"])
        front = np.array([x.obj for x in solver.run(train=False)])  # 策略冻结
        return key, inst_name, seed, front, None
    except Exception:
        return key, inst_name, seed, None, traceback.format_exc()


def load_fronts(keys):
    fronts = {}
    for k in keys:
        fronts[k] = {}
        for inst in TEST_INSTANCES:
            if k in REUSE:
                p = os.path.join(CACHE_ROOT, REUSE[k], inst)
            else:
                p = os.path.join(CACHE_ROOT, f"abl_{k}", inst)
            fronts[k][inst] = [np.load(os.path.join(p, f"{s}.npy")) for s in SEEDS]
    return fronts


def main():
    global Np, Ngen, SEEDS, TEST_INSTANCES, TRAIN_INSTANCES, DEVICE
    ap = argparse.ArgumentParser(description="MOECLF 消融实验（两阶段协议）")
    ap.add_argument("--population", type=int, default=Np)
    ap.add_argument("--generations", type=int, default=Ngen)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    Np, Ngen = args.population, args.generations
    SEEDS = list(range(args.seeds))
    DEVICE = args.device

    # v4 与 MOECLF 复用主实验模型，只需训练 v1/v2/v3
    to_train = [k for k in ("v1", "v2", "v3")
                if not os.path.exists(os.path.join(model_dir(k), "Q.npy"))]

    print(f"DEVICE={DEVICE} Np={Np} Ngen={Ngen} seeds={SEEDS}", flush=True)
    print(f"train={TRAIN_INSTANCES}\ntest={TEST_INSTANCES}", flush=True)

    overlap = [i for i in TEST_INSTANCES if i in TRAIN_INSTANCES]
    if overlap:
        print(f"警告：测试集含训练实例 {overlap}", flush=True)

    if to_train:
        print(f"\n=== 训练阶段：{to_train} ===", flush=True)
        t0 = time.time()
        with Pool(len(to_train)) as pool:
            pool.map(train_one, to_train)
        print(f"训练阶段完成，耗时 {time.time() - t0:.0f}s", flush=True)
    else:
        print("\n训练阶段已完成，直接载入策略", flush=True)

    keys = ["v1", "v2", "v3"]
    jobs = [(k, i, s) for k in keys for i in TEST_INSTANCES for s in SEEDS
            if not os.path.exists(cache_path(k, i, s))]
    total = len(keys) * len(TEST_INSTANCES) * len(SEEDS)
    print(f"\n=== 测试阶段：需运行 {len(jobs)}/{total} 个任务（策略冻结）===", flush=True)

    if jobs:
        t0 = time.time()
        done = 0
        with Pool(args.workers, initializer=_init_worker, initargs=(keys,)) as pool:
            for k, inst, seed, front, err in pool.imap_unordered(_test_job, jobs):
                if err is not None:
                    print(f"ERROR {k}/{inst}/{seed}\n{err}", flush=True)
                    continue
                p = cache_path(k, inst, seed)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, front)
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  {done}/{len(jobs)}  已用 {el:.0f}s  "
                          f"ETA {el / done * (len(jobs) - done):.0f}s", flush=True)

    # ---- 汇总 ----
    print("\n=== 汇总阶段 ===", flush=True)
    os.makedirs("results", exist_ok=True)
    order = ["v1", "v2", "v3", "v4", "moeclf"]
    fronts = load_fronts(order)
    labels = {k: LABELS[k] for k in order}
    body = render("实验1 消融", labels, fronts, TEST_INSTANCES, SEEDS)
    header = (
        "\n## 实验1 消融\n\n"
        f"> 规模：Np={Np} / Ngen={Ngen}，{len(SEEDS)} 种子 × {len(TEST_INSTANCES)} 实例"
        f"（{', '.join(TEST_INSTANCES)}）\n"
        "> 协议：训练阶段（J45-S6-1..10）→ 测试阶段（策略冻结）\n"
        "> 控制器固定为 Q-learning，逐级开启 MSI / VNS / EP；v4 即 MOEQLF。\n"
    )
    out = os.path.join("results", f"实验1_消融_Np{Np}_Ngen{Ngen}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + body)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
