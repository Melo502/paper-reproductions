"""MOECLF 入口：训练 + 测试 + 冒烟测试。

用法:
  python run.py smoke      # 小规模冒烟测试（秒级，验证全链路）
  python run.py train      # 训练 10 个 45-6 实例（论文 Table 5 参数，CPU 上较慢）
  python run.py test       # 用训练好的策略测试 unseen 实例
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from agents import PPOAgent, QLearningAgent
from ehfsp_hmc import generate_instance, load_instances
from moeclf import MOECLF
from metrics import hv, igd, normalize_objs, set_coverage

torch.set_num_threads(1)  # CPU 上单线程最快（实测多线程反而更慢）


def _default_device() -> str:
    """自动选择可用设备：CUDA > MPS > CPU；可用环境变量 DEVICE 覆盖。"""
    env = os.environ.get("DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _default_device()

# 训练参数（论文 Table 5）
PARAMS = dict(
    Np=200,
    Ngen=1000,
    train_episodes=10,
    ppo_lr=0.001,
    ppo_gamma=0.98,
    ppo_lam=0.98,
    ppo_clip=0.2,
    c_theta=2.0,
    c_phi=1.0,
    c_entropy=0.01,
    dropout=0.2,
    q_alpha=0.15,
    q_gamma=0.98,
    q_eps=0.4,
    update_epochs=10,
    batch_size=128,
)


def make_agents(fsm_rows: int, fsm_cols: int):
    ppo = [
        PPOAgent(
            fsm_rows, fsm_cols,
            alpha=PARAMS["ppo_lr"], gamma=PARAMS["ppo_gamma"], lam=PARAMS["ppo_lam"],
            clip=PARAMS["ppo_clip"], c_theta=PARAMS["c_theta"], c_phi=PARAMS["c_phi"],
            c_entropy=PARAMS["c_entropy"], dropout=PARAMS["dropout"], device=DEVICE,
        )
        for _ in range(2)
    ]
    q = QLearningAgent(alpha=PARAMS["q_alpha"], gamma=PARAMS["q_gamma"], eps=PARAMS["q_eps"])
    return ppo, q


def fsm_dims(insts, min_dim: int = 46):
    """FSM 固定尺寸：至少 min_dim(46，论文训练尺寸)，保证 MobileNetV2 不退化。"""
    rows = max(min_dim, max(i.n_machines() + i.n_workers() for i in insts))
    cols = max(min_dim, max(i.I + 1 for i in insts))
    return rows, cols


def train(train_insts):
    """在训练实例上跑多个 episode，返回训练好的智能体与 FSM 尺寸。"""
    fsm_rows, fsm_cols = fsm_dims(train_insts)
    ppo, q = make_agents(fsm_rows, fsm_cols)

    for ep, inst in enumerate(train_insts):
        solver = MOECLF(
            inst, Np=PARAMS["Np"], Ngen=PARAMS["Ngen"],
            fsm_rows=fsm_rows, fsm_cols=fsm_cols, ppo=ppo, q_agent=q, seed=ep,
            batch_size=PARAMS["batch_size"], update_epochs=PARAMS["update_epochs"],
        )
        front = solver.run(train=True)
        objs = np.array([s.obj for s in front])
        print(f"episode {ep + 1}/{len(train_insts)} done, front size={len(objs)}", flush=True)
    return ppo, q, fsm_rows, fsm_cols


def test(inst, ppo, q, fsm_rows, fsm_cols, seed=0):
    solver = MOECLF(
        inst, Np=PARAMS["Np"], Ngen=PARAMS["Ngen"],
        fsm_rows=fsm_rows, fsm_cols=fsm_cols, ppo=ppo, q_agent=q, seed=seed,
        batch_size=PARAMS["batch_size"], update_epochs=PARAMS["update_epochs"],
    )
    front = solver.run(train=False)
    return np.array([s.obj for s in front])


def smoke():
    """小规模冒烟测试：验证 MSI -> LLH -> Q-learning/PPO -> EP 全链路。"""
    print("=== smoke test ===")
    train_insts = [generate_instance(15, 2, seed=i) for i in range(2)]
    fsm_rows, fsm_cols = fsm_dims(train_insts)
    ppo, q = make_agents(fsm_rows, fsm_cols)
    print(f"FSM dims: {fsm_rows} x {fsm_cols}")

    solver = MOECLF(
        train_insts[0], Np=16, Ngen=20, fsm_rows=fsm_rows, fsm_cols=fsm_cols,
        ppo=ppo, q_agent=q, seed=0,
    )
    front = solver.run(train=True)
    for a in ppo:
        a.update(epochs=3, batch_size=8)
    objs = np.array([s.obj for s in front])
    print(f"train front size={len(objs)}")
    print("front objs:", objs[:3].tolist())

    # 测试 unseen 实例（小规模，直接用 MOECLF 而非 test() 的完整参数）
    test_inst = generate_instance(15, 2, seed=99)
    s2 = MOECLF(test_inst, Np=16, Ngen=20, fsm_rows=fsm_rows, fsm_cols=fsm_cols,
                ppo=ppo, q_agent=q, seed=0)
    test_front = np.array([s.obj for s in s2.run(train=False)])
    print(f"test front size={len(test_front)}")

    # 指标（自身对照）
    if len(objs) and len(test_front):
        fmin = np.minimum(objs.min(axis=0), test_front.min(axis=0))
        fmax = np.maximum(objs.max(axis=0), test_front.max(axis=0))
        n1 = normalize_objs(objs, fmin, fmax)
        n2 = normalize_objs(test_front, fmin, fmax)
        print(f"HV(train)={hv(n1):.4f}  HV(test)={hv(n2):.4f}")
        print(f"C(train,test)={set_coverage(n1, n2):.4f}  IGD(test,train)={igd(n1, n2):.4f}")
    print("=== smoke OK ===")


def load_official(json_path: str = "EHFSP-HMC.json") -> dict:
    """加载官方数据集，返回 {实例名: Instance}。"""
    return dict(load_instances(json_path))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke()
    elif mode == "train":
        by_name = load_official()
        train_insts = [by_name[f"J45-S6-{i}"] for i in range(1, 11)]
        ppo, q, rows, cols = train(train_insts)
        print(f"trained agents, FSM {rows}x{cols}")
    elif mode == "test":
        by_name = load_official()
        train_insts = [by_name[f"J45-S6-{i}"] for i in range(1, 11)]
        ppo, q, rows, cols = train(train_insts)
        inst = by_name["J30-S4-1"]
        front = test(inst, ppo, q, rows, cols)
        print("test front:", front[:5].tolist())
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
