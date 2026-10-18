"""MOECLF 跨 episode 预训练 + 测试（复现论文 Table 5/6 协议）。

Phase A: 在 J45-S6-1..10 上预训练两个控制器（各 10 episode，agent 跨 episode 共享）：
  - "q"    = Q-only（MOEQLF / Q-only）
  - "full" = PPO 管边界个体 + Q 管其余（MOECLF / PPO+Q）
Phase B: 用预训练好的（冻结）策略在 6 个代表实例上测试（train=False），5 种子。
Phase C: 与非学习方法（Random / MODJA / NSGA-II，复用 run_experiments 已缓存结果）
         一起算 HV/GD/IGD/C/Wilcoxon，写 results/{2,3}_pretrain.md。

用法: python run_pretrain.py [--workers 14]
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from agents import PPOAgent, QLearningAgent
from ehfsp_hmc import fast_non_dominated_sort, load_instances
from metrics import gd, hv, igd, normalize_objs, set_coverage, wilcoxon_ranksum
from moeclf import MOECLF

import run_experiments as rx  # 复用 spec_sig / cache_path / 非学习方法缓存读取

DEVICE = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(1)

# 与 run_experiments 完全一致的规模（复用其缓存必须参数一致）
Np, Ngen, FSM = 200, 1000, 46
SEEDS = [0, 1, 2, 3, 4]
TEST_INSTANCES = rx.INSTANCES  # 6 个代表实例
TRAIN_INSTANCES = [f"J45-S6-{i}" for i in range(1, 11)]
JSON = "EHFSP-HMC.json"
PRETRAIN_DIR = os.path.join("results", "_pretrain")
CACHE_PRETRAIN = os.path.join("results", "_cache_pretrain")

CONTROLLERS = ["q", "full"]


# --------------------------------------------------------------------------
# 预训练 (Phase A)
# --------------------------------------------------------------------------


def make_agents():
    ppo = [PPOAgent(FSM, FSM, device=DEVICE, n_actions=10) for _ in range(2)]
    q = QLearningAgent(n_actions=10)
    return ppo, q


def save_agents(controller, ppo, q):
    out = os.path.join(PRETRAIN_DIR, controller)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "Q.npy"), q.Q)
    if controller == "full":
        for i, a in enumerate(ppo):
            torch.save(a.policy.state_dict(), os.path.join(out, f"ppo{i}.pt"))


def load_agents(controller):
    ppo, q = make_agents()
    pdir = os.path.join(PRETRAIN_DIR, controller)
    q.Q = np.load(os.path.join(pdir, "Q.npy"))
    if controller == "full":
        for i, a in enumerate(ppo):
            a.policy.load_state_dict(
                torch.load(os.path.join(pdir, f"ppo{i}.pt"), map_location=DEVICE)
            )
    return ppo, q


def pretrain_controller(controller):
    """10 个 episode 共享 agent 训练，返回控制器名。"""
    ppo, q = make_agents()
    by_name = dict(load_instances(JSON))
    t0 = time.time()
    for ep, name in enumerate(TRAIN_INSTANCES):
        inst = by_name[name]
        s = MOECLF(inst, Np=Np, Ngen=Ngen, fsm_rows=FSM, fsm_cols=FSM,
                   ppo=ppo, q_agent=q, seed=ep, batch_size=128, update_epochs=10,
                   controller=controller)
        s.run(train=True)
        print(f"  [pretrain {controller}] ep {ep + 1}/{len(TRAIN_INSTANCES)} "
              f"({name}) done, {time.time() - t0:.0f}s", flush=True)
    save_agents(controller, ppo, q)
    print(f"[pretrain {controller}] saved, total {time.time() - t0:.0f}s", flush=True)
    return controller


# --------------------------------------------------------------------------
# 测试 (Phase B)
# --------------------------------------------------------------------------

_G = {}


def _init_worker():
    _G["by_name"] = dict(load_instances(JSON))
    _G["agents"] = {c: load_agents(c) for c in CONTROLLERS}


def _test_job(job):
    controller, inst_name, seed = job
    inst = _G["by_name"][inst_name]
    ppo, q = _G["agents"][controller]
    s = MOECLF(inst, Np=Np, Ngen=Ngen, fsm_rows=FSM, fsm_cols=FSM,
               ppo=ppo, q_agent=q, seed=seed, batch_size=128, update_epochs=10,
               controller=controller)
    front = np.array([x.obj for x in s.run(train=False)])
    return controller, inst_name, seed, front


# --------------------------------------------------------------------------
# 渲染 (Phase C)
# --------------------------------------------------------------------------


def nondominated(objs):
    _, fronts = fast_non_dominated_sort(objs)
    return objs[fronts[0]] if fronts else objs


def render(title: str, algos: dict, fronts: dict) -> str:
    names = list(algos)
    lines = [f"\n## {title}\n", "| 实例 | " + " | ".join(names) + " |"]
    lines.append("|" + "---|" * (len(names) + 1))
    hv_table = {n: [] for n in names}
    for i in TEST_INSTANCES:
        allpts = np.vstack([f for n in names for f in fronts[n][i]])
        P = nondominated(allpts)
        fmin, fmax = allpts.min(0), allpts.max(0)
        Pn = normalize_objs(P, fmin, fmax)
        row = [f"**{i}**"]
        for n in names:
            hs = []
            for f in fronts[n][i]:
                hs.append(hv(normalize_objs(f, fmin, fmax)))
            hv_table[n].extend(hs)
            gs = [gd(normalize_objs(f, fmin, fmax), Pn) for f in fronts[n][i]]
            igs = [igd(Pn, normalize_objs(f, fmin, fmax)) for f in fronts[n][i]]
            row.append(f"HV {float(np.mean(hs)):.3f}<br>GD {np.mean(gs):.3f}<br>IGD {np.mean(igs):.3f}")
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
                      for i in TEST_INSTANCES for s in range(len(SEEDS))]
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
                row.append(f"{wilcoxon_ranksum(hv_table[n1], hv_table[n2]):.2e}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def load_old_fronts(spec: dict):
    """从 run_experiments 的缓存读取非学习方法 front（参数一致）。"""
    spec = rx._finalize(spec)
    fronts = {}
    for inst in TEST_INSTANCES:
        fronts[inst] = [np.load(rx.cache_path(spec, inst, seed)) for seed in SEEDS]
    return fronts


def load_pretrain_fronts(controller: str):
    fronts = {}
    for inst in TEST_INSTANCES:
        p = os.path.join(CACHE_PRETRAIN, controller, inst)
        fronts[inst] = [np.load(os.path.join(p, f"{seed}.npy")) for seed in SEEDS]
    return fronts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    # Phase A: 预训练（2 控制器并行，各 10 episode 串行）
    if not all(os.path.exists(os.path.join(PRETRAIN_DIR, c, "Q.npy")) for c in CONTROLLERS):
        print("=== Phase A: 预训练（J45-S6-1..10 × 10 episode）===", flush=True)
        t0 = time.time()
        with Pool(2) as pool:
            pool.map(pretrain_controller, CONTROLLERS)
        print(f"预训练完成，耗时 {time.time() - t0:.0f}s", flush=True)
    else:
        print("预训练 agent 已存在，跳过 Phase A", flush=True)

    # Phase B: 测试（冻结策略）
    jobs = [(c, i, s) for c in CONTROLLERS for i in TEST_INSTANCES for s in SEEDS]
    print(f"=== Phase B: 测试（{len(jobs)} 个任务，冻结策略 train=False）===", flush=True)
    todo = []
    for c, i, s in jobs:
        p = os.path.join(CACHE_PRETRAIN, c, i, f"{s}.npy")
        if not os.path.exists(p):
            todo.append((c, i, s))
    print(f"需运行 {len(todo)}/{len(jobs)}（其余已缓存）", flush=True)
    if todo:
        done = 0
        with Pool(args.workers, initializer=_init_worker) as pool:
            for controller, inst_name, seed, front in pool.imap_unordered(_test_job, todo):
                p = os.path.join(CACHE_PRETRAIN, controller, inst_name, f"{seed}.npy")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, front)
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"  测试 {done}/{len(todo)} 完成", flush=True)

    # Phase C: 渲染
    print("=== Phase C: 渲染结果 ===", flush=True)
    os.makedirs("results", exist_ok=True)

    exp2 = {
        "Random": load_old_fronts(dict(type="moeclf", controller="random", use_msi=True, use_vns=True, use_ep=True)),
        "Q-only(预训练)": load_pretrain_fronts("q"),
        "PPO+Q(预训练)": load_pretrain_fronts("full"),
    }
    out2 = os.path.join("results", f"实验2_控制器对比_预训练_Np{Np}_Ngen{Ngen}.md")
    with open(out2, "w", encoding="utf-8") as f:
        f.write(render("实验2 控制器对比（跨 episode 预训练，J45-S6-1..10）", exp2, exp2))

    exp3 = {
        "MODJA": load_old_fronts(dict(type="modja")),
        "NSGA-II": load_old_fronts(dict(type="nsga2")),
        "MOEQLF(预训练)": load_pretrain_fronts("q"),
        "MOECLF(预训练)": load_pretrain_fronts("full"),
    }
    out3 = os.path.join("results", f"实验3_算法对比_预训练_Np{Np}_Ngen{Ngen}.md")
    with open(out3, "w", encoding="utf-8") as f:
        f.write(render("实验3 算法对比（跨 episode 预训练，J45-S6-1..10）", exp3, exp3))

    print(f"wrote {out2} 和 {out3}", flush=True)


if __name__ == "__main__":
    main()
