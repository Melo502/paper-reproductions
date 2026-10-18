"""EHFSP-HMC 问题层：实例生成、三层编码、启发式解码(HD)、疲劳模型、能耗、多目标基础。

论文: Wu, Liu, Zhang (2025) Swarm and Evolutionary Computation 95, 101932.
目标: 最小化 makespan (Cmax) 与总能耗 (TEC = CEC + SEC + PEC + IEC)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LN2 = math.log(2.0)

# 速度 -> 能耗转换因子的对应 (表3)
_SPEED_TO_Q = {1.2: 1.5, 1.0: 1.0, 0.8: 0.6}
_SPEEDS = [1.2, 1.0, 0.8]


# --------------------------------------------------------------------------
# 实例
# --------------------------------------------------------------------------


@dataclass
class Instance:
    """EHFSP-HMC 实例，stage 全部用 0 起始下标。"""

    I: int                                # 作业数
    K: int                                # 阶段数
    M: list[int]                          # M[k] 阶段 k 的机器数
    semi_auto: list[bool]                 # 阶段 k 是否半自动（有工人）
    W: list[int]                          # 半自动阶段 k 的工人数（自动阶段记 0）
    v: list[list[float]]                  # v[k][m] 机器处理速度
    q: list[list[float]]                  # q[k][m] 速度对应能耗转换因子
    t: np.ndarray                         # t[i][k] 标准 setup 时间 (I x K)
    p: np.ndarray                         # p[i][k] 标准处理时间 (I x K)
    E_set: list[float]                    # 每阶段 setup 能耗率
    E_proc: list[float]                   # 每阶段标准 processing 能耗率
    E_idle: list[float]                   # 每阶段 idle 能耗率
    E_ext: float                          # 公共能耗率
    lam: list[float]                      # 疲劳累积率
    mu: list[float]                       # 疲劳恢复率
    delta: list[float]                    # 疲劳影响因子
    T: list[float]                        # 工作周期（分钟）

    def n_machines(self) -> int:
        return sum(self.M)

    def n_workers(self) -> int:
        return sum(self.W)

    def proc_power(self, k: int, m: int) -> float:
        """式(16): processing 能耗率 = q_mk * E_proc_k。"""
        return self.q[k][m] * self.E_proc[k]


def generate_instance(
    I: int,
    K: int,
    seed: int = 0,
    semi_auto_mode: str = "random",
    delta_scale: float = 0.3 / LN2,
) -> Instance:
    """按表3生成一个 EHFSP-HMC 实例。

    semi_auto_mode 决定半自动阶段（官方规则 = "random"）：
      "random" -- 半自动阶段数 randint(K//2, K-1)，随机选阶段（官方 EHFSP-HMC.json 的规则）；
      "odd"    -- 奇数阶段半自动；
      "all"    -- 全部阶段半自动；
      "none"   -- 全部自动。
    """
    rng = np.random.default_rng(seed)
    M = [int(rng.integers(4, 7)) for _ in range(K)]          # U[4,6] 整数

    if semi_auto_mode == "random":
        n_semi = int(rng.integers(K // 2, K))                # randint(K//2, K-1)
        semi_stages = set(rng.choice(K, size=n_semi, replace=False))
        semi_auto = [k in semi_stages for k in range(K)]
    elif semi_auto_mode == "odd":
        semi_auto = [(k % 2 == 0) for k in range(K)]
    elif semi_auto_mode == "all":
        semi_auto = [True] * K
    elif semi_auto_mode == "none":
        semi_auto = [False] * K
    else:
        raise ValueError(f"unknown semi_auto_mode: {semi_auto_mode}")

    W = [int(rng.integers(1, 4)) if semi_auto[k] else 0 for k in range(K)]  # U[1,3]

    v, q = [], []
    for k in range(K):
        vm = [float(rng.choice(_SPEEDS)) for _ in range(M[k])]
        v.append(vm)
        q.append([_SPEED_TO_Q[s] for s in vm])

    t = rng.uniform(25, 99, size=(I, K))
    p = rng.uniform(25, 99, size=(I, K))

    return Instance(
        I=I, K=K, M=M, semi_auto=semi_auto, W=W, v=v, q=q, t=t, p=p,
        E_set=[float(rng.uniform(10, 30)) for _ in range(K)],
        E_proc=[float(rng.uniform(15, 40)) for _ in range(K)],
        E_idle=[float(rng.uniform(5, 8)) for _ in range(K)],
        E_ext=3.0,
        lam=[0.02] * K,
        mu=[0.05] * K,
        delta=[delta_scale] * K,
        T=[480.0] * K,
    )


def load_instances(json_path: str, delta_idx: int = 0) -> list[tuple[str, Instance]]:
    """从官方 EHFSP-HMC.json 加载 90 个实例，返回 [(名称, Instance), ...]。

    delta_idx 选择疲劳影响因子：0=0.3/ln2(主实验)，1=0，2=0.7/ln2。
    working_period 单位是小时，这里转成分钟与 t/p 一致。
    """
    import json

    with open(json_path) as f:
        data = json.load(f)
    proc_speed = data["proc_speed"]
    conversion = data["conversion_rate"]
    lam = data["fatigue_accumulation_rate"]
    mu = data["fatigue_recovery_rate"]
    influence = data["influence_factor"]
    T = data["working_period"] * 60.0           # 小时 -> 分钟
    E_ext = data["pub_energy"]

    out = []
    for key in sorted(data):
        if not key.startswith("J"):
            continue
        d = data[key]
        K = len(d["machine_list"])
        M = d["machine_list"]
        W = d["worker_list"]
        semi_auto = [w > 0 for w in W]
        gear = d["machine_gear"]
        v = [[proc_speed[gear[k][m]] for m in range(M[k])] for k in range(K)]
        q = [[conversion[gear[k][m]] for m in range(M[k])] for k in range(K)]
        t = np.array(d["set_time_list"], dtype=float)
        p = np.array(d["proc_time_list"], dtype=float)
        out.append((
            key,
            Instance(
                I=len(p), K=K, M=M, semi_auto=semi_auto, W=W, v=v, q=q, t=t, p=p,
                E_set=d["set_energy_list"], E_proc=d["proc_energy_list"],
                E_idle=d["idle_energy_list"], E_ext=E_ext,
                lam=[lam] * K, mu=[mu] * K, delta=[influence[delta_idx]] * K, T=[T] * K,
            ),
        ))
    return out


# --------------------------------------------------------------------------
# 解编码（三层）
# --------------------------------------------------------------------------


@dataclass
class Solution:
    """三层编码解。

    job_seq : 阶段 1 的作业序列（0..I-1 的排列）
    mach    : mach[i][k] 作业 i 在阶段 k 的机器索引 (I x K)
    worker  : worker[i][k] 作业 i 在阶段 k 的工人索引 (I x K)，自动阶段 -1
    obj     : [Cmax, TEC]，惰性
    """

    job_seq: list[int]
    mach: np.ndarray
    worker: np.ndarray
    obj: np.ndarray | None = None

    def copy(self) -> "Solution":
        return Solution(
            list(self.job_seq),
            self.mach.copy(),
            self.worker.copy(),
            None if self.obj is None else self.obj.copy(),
        )


def random_solution(inst: Instance, rng: np.random.Generator) -> Solution:
    """随机解：job_seq 随机排列，机器/工人随机分配。"""
    job_seq = rng.permutation(inst.I).tolist()
    mach = np.empty((inst.I, inst.K), dtype=int)
    worker = np.full((inst.I, inst.K), -1, dtype=int)
    for k in range(inst.K):
        mach[:, k] = rng.integers(0, inst.M[k], size=inst.I)
        if inst.semi_auto[k]:
            worker[:, k] = rng.integers(0, inst.W[k], size=inst.I)
    return Solution(job_seq, mach, worker)


# --------------------------------------------------------------------------
# 启发式解码 (Algorithm 2-4) + 疲劳模型 (式22-26) + 能耗 (式17-21)
# --------------------------------------------------------------------------


@dataclass
class Schedule:
    """解码结果：目标值 + 供 LLH 与 FSM 使用的调度细节。"""

    obj: np.ndarray                          # [Cmax, TEC]
    C: np.ndarray                            # C[i][k] 完成时间 (I x K)
    mach_work: list[list[float]]             # [k][m] 机器累计工作量 (t_hat+p_hat)
    mach_avail: list[list[float]]            # [k][m] 机器最早可用时间 EAT
    mach_ops: list[list[tuple[int, float, float]]]   # 每机器(平铺)操作序列 (job, time, energy)
    worker_ops: list[list[tuple[int, float, float]]]  # 每工人(平铺) setup 序列 (job, time, energy)


def decode_schedule(inst: Instance, sol: Solution) -> Schedule:
    """解码三层编码解，返回完整调度。

    阶段 1 按 job_seq 顺序，后续阶段按上一阶段完成时间非降序；
    机器/工人取自 sol.mach / sol.worker；半自动阶段实际 setup 时间受疲劳影响。
    """
    I, K = inst.I, inst.K
    C = np.zeros((I, K))

    # 机器/工人的全局平铺偏移
    m_off, w_off = _offsets(inst)

    # 机器状态（平铺）
    AT_m = [0.0] * inst.n_machines()
    mach_start = [math.inf] * inst.n_machines()
    mach_end = [0.0] * inst.n_machines()
    mach_work = [0.0] * inst.n_machines()
    mach_ops = [[] for _ in range(inst.n_machines())]

    # 工人状态（平铺，仅半自动阶段使用）
    AT_w = [0.0] * inst.n_workers()
    shift_start = [0.0] * inst.n_workers()
    F_w = [0.0] * inst.n_workers()
    worker_ops = [[] for _ in range(inst.n_workers())]

    stage_start = [math.inf] * K
    stage_end = [0.0] * K

    SEC = PEC = 0.0

    for k in range(K):
        order = list(sol.job_seq) if k == 0 else sorted(range(I), key=lambda i: C[i, k - 1])

        for i in order:
            rt = C[i, k - 1] if k > 0 else 0.0
            m = int(sol.mach[i, k])
            gm = m_off[k] + m

            if inst.semi_auto[k]:
                w = int(sol.worker[i, k])
                gw = w_off[k] + w
                S, t_hat = _worker_setup(
                    inst, k, inst.t[i, k], AT_m[gm], AT_w[gw], shift_start[gw], F_w[gw], rt,
                )
                # 写回工人状态
                F_w[gw], shift_start[gw], AT_w[gw] = _finish_worker_setup(
                    inst, k, F_w[gw], shift_start[gw], S, t_hat
                )
            else:
                t_hat = inst.t[i, k]
                S = max(AT_m[gm], rt)

            p_hat = inst.p[i, k] / inst.v[k][m]               # 式(15)
            end = S + t_hat + p_hat
            C[i, k] = end

            AT_m[gm] = end
            mach_start[gm] = min(mach_start[gm], S)
            mach_end[gm] = end
            mach_work[gm] += t_hat + p_hat
            stage_start[k] = min(stage_start[k], S)
            stage_end[k] = max(stage_end[k], end)

            SEC += t_hat * inst.E_set[k]                     # 式(18)
            PEC += p_hat * inst.proc_power(k, m)             # 式(19)

            mach_ops[gm].append((i, t_hat + p_hat, t_hat * inst.E_set[k] + p_hat * inst.proc_power(k, m)))
            if inst.semi_auto[k]:
                worker_ops[gw].append((i, t_hat, t_hat * inst.E_set[k]))

    # 式(20): idle 能耗
    IEC = 0.0
    for k in range(K):
        for m in range(inst.M[k]):
            gm = m_off[k] + m
            if mach_end[gm] >= mach_start[gm]:
                IEC += (mach_end[gm] - mach_start[gm] - mach_work[gm]) * inst.E_idle[k]

    # 式(17): 公共能耗
    CEC = 0.0
    for k in range(K):
        if stage_end[k] >= stage_start[k]:
            CEC += (stage_end[k] - stage_start[k]) * inst.E_ext

    Cmax = float(np.max(C))
    TEC = CEC + SEC + PEC + IEC

    # 平铺机器状态重组为 [k][m]
    mach_work_2d = [[mach_work[m_off[k] + m] for m in range(inst.M[k])] for k in range(K)]
    mach_avail_2d = [[AT_m[m_off[k] + m] for m in range(inst.M[k])] for k in range(K)]

    return Schedule(
        obj=np.array([Cmax, TEC]), C=C, mach_work=mach_work_2d, mach_avail=mach_avail_2d,
        mach_ops=mach_ops, worker_ops=worker_ops,
    )


def decode(inst: Instance, sol: Solution) -> np.ndarray:
    """解码三层编码解，返回 [Cmax, TEC]（轻量封装）。"""
    return decode_schedule(inst, sol).obj


def _offsets(inst: Instance) -> tuple[list[int], list[int]]:
    """每阶段机器/工人在平铺数组中的起始偏移。"""
    m_off, w_off = [0] * inst.K, [0] * inst.K
    for k in range(1, inst.K):
        m_off[k] = m_off[k - 1] + inst.M[k - 1]
        w_off[k] = w_off[k - 1] + inst.W[k - 1]
    return m_off, w_off


def _worker_setup(
    inst: Instance,
    k: int,
    t_ik: float,
    AT_m: float,
    AT_w: float,
    shift_start: float,
    F: float,
    rt: float,
) -> tuple[float, float]:
    """半自动阶段 setup 的开始时间与实际时长（不改状态，只算）。

    1) 就绪 S = max(机器可用, 工人可用, 作业释放)；
    2) 空闲疲劳恢复 式(23)；
    3) 实际 setup 式(26)；
    4) 班次约束 式(24)：若 setup 会跨工作周期，右移到下一班次起点并完全恢复。
    """
    S = max(AT_m, AT_w, rt)
    if S > AT_w:
        F = F * math.exp(-inst.mu[k] * (S - AT_w))            # 式(23) 恢复

    # 式(26): t̂ = t(1 + δ·λ·ln(1+F))，半自动阶段（论文原文 δ_k·λ_k·ln(1+F_wk)）
    t_hat = t_ik * (1 + inst.delta[k] * inst.lam[k] * math.log1p(F))
    while S + t_hat > shift_start + inst.T[k]:                # 式(24) 班次约束
        shift_start = shift_start + inst.T[k]
        F = 0.0
        S = max(shift_start, AT_m, rt)
        t_hat = t_ik * (1 + inst.delta[k] * inst.lam[k] * math.log1p(F))  # F=0 → t_ik

    return S, t_hat


def _finish_worker_setup(
    inst: Instance, k: int, F: float, shift_start: float, S: float, t_hat: float,
) -> tuple[float, float, float]:
    """setup 结束后的疲劳累积 式(22)，返回 (新疲劳, 新班次起点, 工人可用时间)。

    用连续时间精确积分 F+(1-F)(1-e^{-λt})，而非欧拉离散 F+(1-F)λt（λt 大时后者会超 1）。
    """
    F = F + (1 - F) * (1 - math.exp(-inst.lam[k] * t_hat))
    return F, shift_start, S + t_hat


# --------------------------------------------------------------------------
# 多目标基础：支配、快速非支配排序、拥挤距离
# --------------------------------------------------------------------------


def dominates(a, b) -> bool:
    """a 支配 b（最小化）：a 全部 <= b 且至少一个严格 <。"""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def fast_non_dominated_sort(obj: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """返回 (rank, fronts)。rank[i] 个体 i 的非支配层级；fronts 按层分组。

    支配关系用 numpy 向量化计算，与原始 O(n^2) 双重循环逐位等价，仅加速。
    """
    obj = np.asarray(obj, dtype=np.float64)
    n = len(obj)
    if n == 0:
        return [], []
    # i 支配 j：obj[i] <= obj[j]（全部目标）且至少一个严格 <
    le = (obj[:, 0, None] <= obj[None, :, 0]) & (obj[:, 1, None] <= obj[None, :, 1])
    lt = (obj[:, 0, None] < obj[None, :, 0]) | (obj[:, 1, None] < obj[None, :, 1])
    dom = le & lt
    np.fill_diagonal(dom, False)

    # n_p[i] = 支配 i 的个体数（列和）
    n_p = dom.sum(axis=0).astype(int).tolist()
    # S[i] = 被 i 支配的个体索引（升序，等价原始循环）
    S = [list(np.nonzero(dom[i])[0]) for i in range(n)]

    rank = [0] * n
    fronts: list[list[int]] = []
    F1 = [i for i in range(n) if n_p[i] == 0]
    fronts.append(F1)

    i_front = 0
    while fronts[i_front]:
        Q = []
        for i in fronts[i_front]:
            for j in S[i]:
                n_p[j] -= 1
                if n_p[j] == 0:
                    rank[j] = i_front + 1
                    Q.append(j)
        i_front += 1
        fronts.append(Q)
    fronts.pop()
    return rank, fronts


def crowding_distance(obj_front: np.ndarray) -> list[float]:
    """一个前沿上的拥挤距离（两个目标）。"""
    m = len(obj_front)
    if m <= 2:
        return [math.inf] * m
    dist = [0.0] * m
    for d in range(2):
        idx = sorted(range(m), key=lambda i: obj_front[i][d])
        dist[idx[0]] = dist[idx[-1]] = math.inf
        fmin, fmax = obj_front[idx[0]][d], obj_front[idx[-1]][d]
        if fmax - fmin == 0:
            continue
        for r in range(1, m - 1):
            dist[idx[r]] += (obj_front[idx[r + 1]][d] - obj_front[idx[r - 1]][d]) / (fmax - fmin)
    return dist
