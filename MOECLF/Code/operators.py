"""十个低层启发式 LLH1-LLH10（问题域算子）。

遗传(LLH1-3)与 GVNS(LLH4-6) 操作 job_seq，LVNS(LLH7-10) 操作 mach 分配层。
执行后不解码（obj 置 None），由调用方解码。
"""
from __future__ import annotations

import numpy as np

from ehfsp_hmc import Instance, Schedule, Solution


def apply_llh(
    inst: Instance,
    sol: Solution,
    llh_id: int,
    rng: np.random.Generator,
    sch: Schedule | None = None,
    other: Solution | None = None,
) -> Solution | None:
    """执行第 llh_id(1..10) 个 LLH，返回新解；返回 None 表示无法执行。"""
    if llh_id in (1, 3) and other is None:
        raise ValueError(f"LLH{llh_id} 需要另一父代 other")
    if llh_id in (4, 5, 6, 7, 8, 10) and sch is None:
        raise ValueError(f"LLH{llh_id} 需要调度信息 sch")

    fn = _LLH[llh_id]
    new = fn(inst, sol, rng, sch, other)
    if new is None:
        return None
    return new


def _new_from(sol: Solution) -> Solution:
    return Solution(list(sol.job_seq), sol.mach.copy(), sol.worker.copy())


# --------------------------------------------------------------------------
# 遗传算子（操作 job_seq）
# --------------------------------------------------------------------------


def _llh1_2p_oxo(inst, sol, rng, sch, other):
    """两点顺序交叉：P1[.] 取段，其余按另一父代顺序填入。"""
    I = inst.I
    r1 = int(rng.integers(1, I - 1))
    r2 = int(rng.integers(r1 + 1, I))
    if rng.integers(0, 2) == 0:  # R1=1: 取 P1 段，填 P2；R1=2 反之
        seg = sol.job_seq[r1:r2]
        donor = other.job_seq
    else:
        seg = other.job_seq[r1:r2]
        donor = sol.job_seq
    rest = [g for g in donor if g not in seg]
    child = Solution(rest[:r1] + seg + rest[r1:], sol.mach.copy(), sol.worker.copy())
    return child


def _llh2_2p_esm(inst, sol, rng, sch, other):
    """两点分割变异：切三段按 R2 排列。"""
    I = inst.I
    r3 = int(rng.integers(1, I - 1))
    r4 = int(rng.integers(r3 + 1, I))
    a, b, c = sol.job_seq[:r3], sol.job_seq[r3:r4], sol.job_seq[r4:]
    r = int(rng.integers(1, 4))
    if r == 1:
        seq = b + a + c
    elif r == 2:
        seq = a + c + b
    else:
        seq = c + b + a
    return Solution(seq, sol.mach.copy(), sol.worker.copy())


def _llh3_oxo_esm(inst, sol, rng, sch, other):
    """LLH1 与 LLH2 串行。"""
    mid = _llh1_2p_oxo(inst, sol, rng, sch, other)
    return _llh2_2p_esm(inst, mid, rng, sch, other)


# --------------------------------------------------------------------------
# GVNS（LLH4-6，基于最后阶段首尾完成作业 J_f / J_l）
# --------------------------------------------------------------------------


def _fl_pair(inst, sol, sch):
    last = sch.C[:, inst.K - 1]
    jf = int(np.argmin(last))
    jl = int(np.argmax(last))
    return jf, jl


def _llh4_fl_swap(inst, sol, rng, sch, other):
    jf, jl = _fl_pair(inst, sol, sch)
    seq = list(sol.job_seq)
    pf, pl = seq.index(jf), seq.index(jl)
    seq[pf], seq[pl] = seq[pl], seq[pf]
    return Solution(seq, sol.mach.copy(), sol.worker.copy())


def _llh5_fl_insert(inst, sol, rng, sch, other):
    jf, jl = _fl_pair(inst, sol, sch)
    seq = list(sol.job_seq)
    seq.remove(jl)
    seq.insert(seq.index(jf) + 1, jl)
    return Solution(seq, sol.mach.copy(), sol.worker.copy())


def _llh6_lr_reverse(inst, sol, rng, sch, other):
    _, jl = _fl_pair(inst, sol, sch)
    seq = list(sol.job_seq)
    pl = seq.index(jl)
    pr = int(rng.integers(0, inst.I))
    lo, hi = min(pl, pr), max(pl, pr)
    seq[lo:hi + 1] = seq[lo:hi + 1][::-1]
    return Solution(seq, sol.mach.copy(), sol.worker.copy())


# --------------------------------------------------------------------------
# LVNS（LLH7-10，操作 mach 分配层）
# --------------------------------------------------------------------------


def _llh7_mw_move(inst, sol, rng, sch, other):
    """随机阶段 k，把最大工作量机器上的一个作业移到最小工作量机器。"""
    k = int(rng.integers(0, inst.K))
    work = np.array(sch.mach_work[k])
    m_max = int(np.argmax(work))
    m_min = int(np.argmin(work))
    jobs = [i for i in range(inst.I) if sol.mach[i, k] == m_max]
    if not jobs or m_max == m_min:
        return None
    new = _new_from(sol)
    new.mach[int(rng.choice(jobs)), k] = m_min
    return new


def _llh8_ra_swap(inst, sol, rng, sch, other):
    """随机阶段 k，交换释放时间最近的两个作业的机器分配。"""
    k = int(rng.integers(0, inst.K))
    rt = sch.C[:, k - 1] if k > 0 else np.zeros(inst.I)
    order = np.argsort(rt)
    pos = int(np.argmin(np.diff(rt[order])))
    i1, i2 = int(order[pos]), int(order[pos + 1])
    new = _new_from(sol)
    new.mach[i1, k], new.mach[i2, k] = new.mach[i2, k], new.mach[i1, k]
    return new


def _llh9_sl_move(inst, sol, rng, sch, other):
    """随机作业 J_r 移到 SPT 机器；若已在最短/常规速度机器则移到 LPT 机器。"""
    k = int(rng.integers(0, inst.K))
    jr = int(rng.integers(0, inst.I))
    vk = np.array(inst.v[k])
    m_spt = int(np.argmax(vk))
    m_lpt = int(np.argmin(vk))
    new = _new_from(sol)
    new.mach[jr, k] = m_lpt if sol.mach[jr, k] == m_spt else m_spt
    return new


def _llh10_l_move(inst, sol, rng, sch, other):
    """关键作业 OP_l 依次尝试移到 EAT / EAT+SPT / EAT+LPT 机器。"""
    k = inst.K - 1
    jl = int(np.argmax(sch.C[:, k]))
    avail = np.array(sch.mach_avail[k])
    vk = np.array(inst.v[k])
    cur = sol.mach[jl, k]

    m_eat = int(np.argmin(avail))
    eat_set = [m for m in range(inst.M[k]) if avail[m] == avail[m_eat]]
    cands = [m_eat, max(eat_set, key=lambda m: vk[m]), min(eat_set, key=lambda m: vk[m])]

    target = cur
    for c in cands:
        if c != cur:
            target = c
            break
    new = _new_from(sol)
    new.mach[jl, k] = target
    return new


_LLH = {
    1: _llh1_2p_oxo,
    2: _llh2_2p_esm,
    3: _llh3_oxo_esm,
    4: _llh4_fl_swap,
    5: _llh5_fl_insert,
    6: _llh6_lr_reverse,
    7: _llh7_mw_move,
    8: _llh8_ra_swap,
    9: _llh9_sl_move,
    10: _llh10_l_move,
}
