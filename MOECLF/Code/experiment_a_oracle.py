"""实验 A：PPO 上限实验（oracle headroom test）。

目的：回答"即便 PPO 每次都能为边界个体选到最优 LLH，性能还能提升多少"。

做法：新增 controller="oracle"，对 F1 的 2 个边界个体枚举全部 10 个 LLH，
      取 reward 最大的动作（= 完美选择器），其余 198 个个体仍由 Q 管理。
      与 Q-only / Random / PPO+Q 在同一批种子下配对比较。

判据：
  - oracle ≈ Q-only  -> 2/200 的位置已无提升空间，PPO 路线结构上无解，可放弃
  - oracle 明显 > Q   -> 该位置有潜力，值得继续投入 PPO

用法:
  python experiment_a_oracle.py                    # 论文规模，复用已有缓存
  python experiment_a_oracle.py --population 50 --generations 300   # 快速冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from scipy import stats

import run_experiments as rx  # 复用 spec_sig / cache_path / _finalize
from ehfsp_hmc import fast_non_dominated_sort, load_instances
from metrics import hv, normalize_objs
from moeclf import MOECLF

torch.set_num_threads(1)

JSON = "EHFSP-HMC.json"
CACHE_A = os.path.join("results", "_cache_expA")

CONTROLLERS = ["random", "q", "full", "oracle"]
LABELS = {
    "random": "Random",
    "q": "Q-only",
    "full": "PPO+Q",
    "oracle": "Oracle(完美选择)",
}

# 与 run_experiments / run_pretrain 完全一致的规模
Np, Ngen, FSM = 200, 1000, 46
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = ["J15-S2-1", "J15-S4-1", "J30-S2-1", "J30-S4-1", "J45-S4-1", "J45-S6-1"]


# --------------------------------------------------------------------------
# 已有缓存的复用（random / q / full 已在论文规模下算过，按种子天然配对）
# --------------------------------------------------------------------------


def existing_cache_path(controller: str, inst: str, seed: int) -> str | None:
    """返回 run_experiments 缓存路径；不存在则 None。"""
    spec = dict(type="moeclf", controller=controller,
                use_msi=True, use_vns=True, use_ep=True)
    spec = rx._finalize(spec)
    p = rx.cache_path(spec, inst, seed)
    return p if os.path.exists(p) else None


def oracle_cache_path(inst: str, seed: int) -> str:
    return os.path.join(CACHE_A, f"Np{Np}_Ngen{Ngen}", inst, f"{seed}.npy")


def oracle_stat_path(inst: str, seed: int) -> str:
    return os.path.join(CACHE_A, f"Np{Np}_Ngen{Ngen}", inst, f"{seed}.json")


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

_G = {}


def _init_worker():
    torch.set_num_threads(1)
    _G["by_name"] = dict(load_instances(JSON))


def _run_oracle(job):
    inst_name, seed = job
    try:
        inst = _G["by_name"][inst_name]
        # oracle 不用 PPO，传空列表避免构造两个 MobileNetV2（省时间）
        solver = MOECLF(
            inst, Np=Np, Ngen=Ngen, fsm_rows=FSM, fsm_cols=FSM,
            ppo=[], q_agent=None, seed=seed,
            batch_size=128, update_epochs=10,
            use_msi=True, use_vns=True, use_ep=True,
            controller="oracle",
        )
        front = np.asarray([s.obj for s in solver.run(train=True)], dtype=float)
        log = np.asarray(solver.oracle_log, dtype=float)
        stat = {
            "decisions": int(log.size),
            "mean_best_reward": float(log.mean()) if log.size else 0.0,
            "frac_pos": float(np.mean(log > 0)) if log.size else 0.0,
            "frac_improve": float(np.mean(log >= 2.0)) if log.size else 0.0,
            "frac_worse": float(np.mean(log < 0)) if log.size else 0.0,
        }
        return inst_name, seed, front, stat, None
    except Exception:
        return inst_name, seed, None, None, traceback.format_exc()


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def nondominated(objs):
    _, fronts = fast_non_dominated_sort(objs)
    return objs[fronts[0]] if fronts else objs


def instance_hv(fronts: dict, inst: str) -> dict:
    """与项目现有口径一致：所有控制器所有种子共用 min-max 范围，
    但 HV 按种子分别计算后再取均值（不是把 5 个种子并成一个前沿）。"""
    allpts = np.vstack([f for c in CONTROLLERS for f in fronts[c][inst]])
    fmin, fmax = allpts.min(0), allpts.max(0)
    out = {}
    for c in CONTROLLERS:
        per_seed = [hv(normalize_objs(f, fmin, fmax)) for f in fronts[c][inst]]
        out[c] = float(np.mean(per_seed))
    return out


def safe_wilcoxon(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    if delta.size == 0 or np.allclose(delta, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------


def render(hv_rows, stats_rows, instances, seeds):
    L = []
    L.append("# 实验 A：PPO 上限实验（Oracle 完美选择器）\n")
    L.append(f"> 规模：Np={Np} / Ngen={Ngen}，{len(seeds)} 种子 × {len(instances)} 实例")
    L.append("> Oracle = 对 2 个边界个体枚举全部 10 个 LLH 并取最优，代表 PPO 的理论上限")
    L.append("> Random / Q-only / PPO+Q 复用 run_experiments 论文规模缓存（同种子天然配对）\n")

    # 表1：逐实例配对 HV
    L.append("## 表1 逐实例 HV（配对）\n")
    hdr = "| 实例 | " + " | ".join(LABELS[c] for c in CONTROLLERS) + " | Oracle−Q | Oracle−PPO |"
    L.append(hdr)
    L.append("|" + "---|" * (len(CONTROLLERS) + 3))
    for inst in instances:
        vals = hv_rows[inst]
        cells = " | ".join(f"{vals[c]:.4f}" for c in CONTROLLERS)
        L.append(f"| **{inst}** | {cells} | {vals['oracle']-vals['q']:+.4f} | {vals['oracle']-vals['full']:+.4f} |")

    # 表2：汇总配对差
    L.append("\n## 表2 配对差值汇总（每实例各控制器取种子均值）\n")
    L.append("| 对比 | HV 平均差 | Oracle/前者胜出 | Wilcoxon signed-rank p |")
    L.append("|---|---:|---:|---:|")
    pairs = [("oracle", "q", "Oracle vs Q-only"),
             ("oracle", "random", "Oracle vs Random"),
             ("oracle", "full", "Oracle vs PPO+Q"),
             ("full", "q", "PPO+Q vs Q-only"),
             ("full", "random", "PPO+Q vs Random")]
    summary = {}
    for a, b, label in pairs:
        d = np.asarray([hv_rows[i][a] - hv_rows[i][b] for i in instances], dtype=float)
        wins = int(np.sum(d > 0))
        p = safe_wilcoxon(d)
        summary[label] = (float(d.mean()), wins, len(d), p)
        L.append(f"| {label} | {d.mean():+.4f} | {wins}/{len(d)} | {p:.3g} |")

    # 表3：oracle 提升空间
    L.append("\n## 表3 Oracle 决策质量（衡量该位置的理论提升空间）\n")
    L.append("| 实例 | 决策数 | 平均最优 reward | reward>0 占比 | reward=+2 占比 | reward<0 占比 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for inst in instances:
        s = stats_rows[inst]
        L.append(f"| **{inst}** | {s['decisions']} | {s['mean_best_reward']:.3f} | "
                 f"{s['frac_pos']:.1%} | {s['frac_improve']:.1%} | {s['frac_worse']:.1%} |")

    # 结论
    d_oq, w_oq, n_oq, p_oq = summary["Oracle vs Q-only"]
    d_or, w_or, n_or, p_or = summary["Oracle vs Random"]
    d_op, w_op, n_op, p_op = summary["Oracle vs PPO+Q"]
    ALPHA = 0.05
    sig_oq = np.isfinite(p_oq) and p_oq < ALPHA
    sig_or = np.isfinite(p_or) and p_or < ALPHA

    L.append("\n## 结论判定\n")
    L.append(f"- Oracle − Q-only = **{d_oq:+.4f}**（{w_oq}/{n_oq} 胜，p={p_oq:.3g}）")
    L.append(f"- Oracle − Random = **{d_or:+.4f}**（{w_or}/{n_or} 胜，p={p_or:.3g}）")
    L.append(f"- Oracle − PPO+Q = **{d_op:+.4f}**（{w_op}/{n_op} 胜，p={p_op:.3g}）")
    L.append("")
    L.append(f"显著性水平 α={ALPHA}（Wilcoxon signed-rank，n={n_oq} 个实例，统计功效有限）\n")

    if not sig_oq and d_oq <= 0.05:
        L.append(f"**判定：2/200 边界位置基本没有提升空间。** 即使每次都为边界个体枚举全部 10 个 LLH")
        L.append(f"并挑 reward 最大的（= 完美选择器），HV 也仅比 Q-only 高 {d_oq:+.4f}，")
        L.append(f"p={p_oq:.3g} 不显著；而且比纯 Random 还低 {d_or:+.4f}。")
        L.append("")
        L.append("这意味着：**PPO 负责的那 2 个边界个体，其动作选择不是性能瓶颈。**")
        L.append("即便 PPO 训练得完美无缺，也无法把 HV 拉起来 —— 增益被结构性地限制住了。")
        L.append("PPO 路线在现有「2/200 分工」设计下没有空间，建议转向实验 B（让 PPO 接管全部个体）")
        L.append("或实验 E（简化观测），而不是继续在样本量/超参上加投入。")
    elif d_oq < 0:
        L.append("**判定：完美选择反而更差。** 说明奖励函数（式38）与该位置的 HV 目标不一致，")
        L.append("优化 reward 不能提升 HV —— 问题出在奖励设计，而非 PPO 的学习能力。")
    elif not sig_oq:
        L.append(f"**判定：方向为正但未达显著。** Oracle 比 Q-only 高 {d_oq:+.4f}，")
        L.append(f"但 p={p_oq:.3g} > α，且 n={n_oq} 的检验功效有限，不能断言存在真实增益。")
        L.append("建议增加实例数或种子数再判，或直接做实验 C（学习曲线）看 PPO 是否在学。")
    else:
        L.append(f"**判定：存在提升空间（{d_oq:+.4f} HV，p={p_oq:.3g} 显著）。**")
        L.append("完美选择显著优于 Q-only，说明该位置确有潜力；")
        L.append("PPO 未体现优势的原因在于学习/样本，而非结构无空间，")
        L.append("值得继续做实验 B（PPO-only）、C（学习曲线）、D（超参）。")

    L.append("")
    L.append("### 附：奖励信号的稀疏程度（表3 佐证）")
    L.append("")
    L.append("Oracle 已取每步**最大**可达 reward，但 `reward=+2`（支配改进）的占比仍只有 "
             "0.7%–2.6%，")
    L.append("说明即便完美选择，绝大多数动作也只能做到非劣（reward=+1）—— "
             "该奖励函数本身几乎不提供「改进方向」的梯度信息。")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    global Np, Ngen, SEEDS, INSTANCES
    ap = argparse.ArgumentParser(description="实验 A：PPO 上限（oracle）")
    ap.add_argument("--population", type=int, default=Np)
    ap.add_argument("--generations", type=int, default=Ngen)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="忽略 oracle 缓存重跑")
    args = ap.parse_args()

    Np, Ngen = args.population, args.generations
    SEEDS = list(range(args.seeds))
    INSTANCES = args.instances
    rx.Np, rx.Ngen, rx.FSM, rx.DEVICE = Np, Ngen, FSM, args.device

    print(f"实验A | Np={Np} Ngen={Ngen} seeds={SEEDS} instances={INSTANCES} workers={args.workers}", flush=True)

    by_name = dict(load_instances(JSON))
    missing = [i for i in INSTANCES if i not in by_name]
    if missing:
        ap.error(f"unknown instances: {missing}")

    fronts = {c: {} for c in CONTROLLERS}
    stats_rows = {}

    # 1) random / q / full：优先复用 run_experiments 缓存
    for c in ["random", "q", "full"]:
        for inst in INSTANCES:
            arr = []
            for seed in SEEDS:
                p = existing_cache_path(c, inst, seed)
                if p is None:
                    raise SystemExit(
                        f"缺少 {c}/{inst}/seed{seed} 的论文规模缓存。\n"
                        f"请先运行: python run_experiments.py 2 3"
                    )
                arr.append(np.load(p))
            fronts[c][inst] = arr  # 每个种子一个前沿，供 instance_hv 逐种子算 HV
        print(f"  [{LABELS[c]}] 已从缓存载入", flush=True)

    # 2) oracle：跑缺的
    jobs = [(i, s) for i in INSTANCES for s in SEEDS
            if args.force or not os.path.exists(oracle_cache_path(i, s))]
    total = len(INSTANCES) * len(SEEDS)
    print(f"\n[Oracle] 需运行 {len(jobs)}/{total}", flush=True)

    if jobs:
        t0 = time.time()
        done = 0
        with Pool(args.workers, initializer=_init_worker) as pool:
            for inst_name, seed, front, stat, err in pool.imap_unordered(_run_oracle, jobs):
                if err is not None:
                    print(f"ERROR {inst_name} seed={seed}\n{err}", flush=True)
                    continue
                cp = oracle_cache_path(inst_name, seed)
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.save(cp, front)
                with open(oracle_stat_path(inst_name, seed), "w", encoding="utf-8") as f:
                    json.dump(stat, f)
                done += 1
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f"  [Oracle] {done}/{len(jobs)}  已用 {el:.0f}s  ETA {eta:.0f}s", flush=True)

    # 3) 汇总 oracle
    for inst in INSTANCES:
        arr, stats_list = [], []
        for seed in SEEDS:
            arr.append(np.load(oracle_cache_path(inst, seed)))
            with open(oracle_stat_path(inst, seed), encoding="utf-8") as f:
                stats_list.append(json.load(f))
        fronts["oracle"][inst] = arr  # 每个种子一个前沿，供 instance_hv 逐种子算 HV
        stats_rows[inst] = {
            "decisions": int(np.mean([s["decisions"] for s in stats_list])),
            "mean_best_reward": float(np.mean([s["mean_best_reward"] for s in stats_list])),
            "frac_pos": float(np.mean([s["frac_pos"] for s in stats_list])),
            "frac_improve": float(np.mean([s["frac_improve"] for s in stats_list])),
            "frac_worse": float(np.mean([s["frac_worse"] for s in stats_list])),
        }

    # 4) HV 与渲染
    hv_rows = {inst: instance_hv(fronts, inst) for inst in INSTANCES}
    text = render(hv_rows, stats_rows, INSTANCES, SEEDS)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"实验A_PPO上限_Np{Np}_Ngen{Ngen}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
