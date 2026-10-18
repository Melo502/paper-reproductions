"""MOECLF 主框架：多策略初始化(MSI) + 多智能体超启发式搜索 + 精英保留(EP)。

Fig 8 / Step1-7。两个 PPO 管理 F1 的边界个体(BI)，Q-learning 管理其余个体。
"""
from __future__ import annotations

import heapq

import numpy as np

from agents import PPOAgent, QLearningAgent, N_ACTIONS, build_fsm
from ehfsp_hmc import (
    Instance,
    Solution,
    decode,
    decode_schedule,
    dominates,
    fast_non_dominated_sort,
    crowding_distance,
    random_solution,
    _worker_setup,
    _finish_worker_setup,
    _offsets,
)
from metrics import normalize_objs
from operators import apply_llh

# 启发式规则编号
R_WB, R_EAT, R_RAND = 0, 1, 2


# --------------------------------------------------------------------------
# 多策略初始化 (Algorithm 1)
# --------------------------------------------------------------------------


def _select(n: int, rule: int, avail, workload, off: int, rng) -> int:
    """在 [off, off+n) 内按规则选机器/工人，返回阶段内索引。"""
    cands = list(range(off, off + n))
    if rule == R_WB:
        best = min(cands, key=lambda c: workload[c])
    elif rule == R_EAT:
        best = min(cands, key=lambda c: avail[c])
    else:
        best = int(rng.choice(cands))
    return best - off


def heuristic_decode(inst: Instance, job_seq, r_m: int, r_w: int, rng) -> Solution:
    """用规则 (r_m, r_w) 动态选机器/工人，构造三层解。"""
    I, K = inst.I, inst.K
    C = np.zeros((I, K))
    m_off, w_off = _offsets(inst)

    AT_m = [0.0] * inst.n_machines()
    Wm = [0.0] * inst.n_machines()
    AT_w = [0.0] * inst.n_workers()
    Ww = [0.0] * inst.n_workers()
    shift_start = [0.0] * inst.n_workers()
    F_w = [0.0] * inst.n_workers()

    mach = np.empty((I, K), dtype=int)
    worker = np.full((I, K), -1, dtype=int)

    for k in range(K):
        order = list(job_seq) if k == 0 else sorted(range(I), key=lambda i: C[i, k - 1])
        for i in order:
            rt = C[i, k - 1] if k > 0 else 0.0
            m = _select(inst.M[k], r_m, AT_m, Wm, m_off[k], rng)
            mach[i, k] = m
            gm = m_off[k] + m
            if inst.semi_auto[k]:
                w = _select(inst.W[k], r_w, AT_w, Ww, w_off[k], rng)
                worker[i, k] = w
                gw = w_off[k] + w
                S, t_hat = _worker_setup(inst, k, inst.t[i, k], AT_m[gm], AT_w[gw], shift_start[gw], F_w[gw], rt)
                F_w[gw], shift_start[gw], AT_w[gw] = _finish_worker_setup(inst, k, F_w[gw], shift_start[gw], S, t_hat)
                Ww[gw] += t_hat
            else:
                t_hat = inst.t[i, k]
                S = max(AT_m[gm], rt)
            p_hat = inst.p[i, k] / inst.v[k][m]
            end = S + t_hat + p_hat
            C[i, k] = end
            AT_m[gm] = end
            Wm[gm] += t_hat + p_hat

    return Solution(list(job_seq), mach, worker)


def ms_init(inst: Instance, Np: int, rng) -> list[Solution]:
    """9 条组合规则(3x3)各生成 NER 个个体，剩余用随机规则补齐。"""
    SCR = 9
    NER = Np // SCR
    pop: list[Solution] = []
    for r in range(SCR):
        r_m, r_w = r % 3, r // 3
        for _ in range(NER):
            job_seq = rng.permutation(inst.I).tolist()
            sol = heuristic_decode(inst, job_seq, r_m, r_w, rng)
            sol.obj = decode(inst, sol)
            pop.append(sol)
    while len(pop) < Np:
        job_seq = rng.permutation(inst.I).tolist()
        sol = heuristic_decode(inst, job_seq, R_RAND, R_RAND, rng)
        sol.obj = decode(inst, sol)
        pop.append(sol)
    return pop


# --------------------------------------------------------------------------
# 环境选择 (NSGA-II 精英保留)
# --------------------------------------------------------------------------


def environmental_selection(pop: list[Solution], offspring: list[Solution], Np: int) -> list[Solution]:
    """合并父代与子代，按 rank 升序 / 拥挤距离降序选 Np 个。"""
    merged = pop + offspring
    objs = np.array([s.obj for s in merged])
    ranks, fronts = fast_non_dominated_sort(objs)

    selected: list[Solution] = []
    for front in fronts:
        if len(selected) + len(front) <= Np:
            selected.extend(merged[i] for i in front)
        else:
            cd = crowding_distance(objs[front])
            order = sorted(front, key=lambda i: -cd[front.index(i)])
            for i in order:
                if len(selected) >= Np:
                    break
                selected.append(merged[i])
            break
    return selected


# --------------------------------------------------------------------------
# 奖励 (式38)
# --------------------------------------------------------------------------


def reward(pi_obj, pi2_obj) -> float:
    if dominates(pi2_obj, pi_obj):
        return 2.0
    if dominates(pi_obj, pi2_obj):
        return -1.0
    if np.allclose(pi2_obj, pi_obj):
        return 0.0
    return 1.0


# --------------------------------------------------------------------------
# MOECLF
# --------------------------------------------------------------------------


class MOECLF:
    def __init__(
        self,
        inst: Instance,
        Np: int = 100,
        Ngen: int = 1000,
        fsm_rows: int = 64,
        fsm_cols: int = 64,
        seed: int = 0,
        ppo: list[PPOAgent] | None = None,
        q_agent: QLearningAgent | None = None,
        batch_size: int = 128,
        update_epochs: int = 10,
        use_msi: bool = True,
        use_vns: bool = True,
        use_ep: bool = True,
        use_ppo: bool = True,
        controller: str | None = None,
    ):
        self.inst = inst
        self.Np = Np
        self.Ngen = Ngen
        self.fsm_rows = fsm_rows
        self.fsm_cols = fsm_cols
        self.rng = np.random.default_rng(seed)
        # 消融开关（论文 Table 6）
        self.use_msi = use_msi
        self.use_vns = use_vns
        self.use_ep = use_ep
        # 控制器：'random'(随机选 LLH) / 'q'(纯 Q-learning) / 'full'(PPO 管 BI + Q 管其余)
        if controller is None:
            controller = "full" if use_ppo else "q"
        self.controller = controller
        self.use_ppo = controller == "full"
        self.n_actions = 10 if use_vns else 3  # 无 VNS 时仅遗传算子 LLH1-3
        self.ppo = ppo if ppo is not None else [
            PPOAgent(fsm_rows, fsm_cols, n_actions=self.n_actions)
            for _ in range(2)
        ]
        self.q_agent = q_agent if q_agent is not None else QLearningAgent(n_actions=self.n_actions)
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.z_ = None  # 理想点（各目标历史最优）
        self.a_ = None  # 最坏点（各目标历史最差）

    def run(self, train: bool = True) -> list[Solution]:
        """跑一个 episode（Ngen 代），返回非支配解集。"""
        if self.use_msi:
            pop = ms_init(self.inst, self.Np, self.rng)
        else:
            pop = [random_solution(self.inst, self.rng) for _ in range(self.Np)]
            for s in pop:
                s.obj = decode(self.inst, s)
        self.q_agent.reset_state(self.rng)
        objs = np.array([s.obj for s in pop])
        self.z_ = objs.min(axis=0).copy()  # 理想点初始 = 初始种群 min
        self.a_ = objs.max(axis=0).copy()  # 最坏点初始 = 初始种群 max
        for gen in range(self.Ngen):
            pop = self._one_generation(pop, train, gen)
        objs = np.array([s.obj for s in pop])
        _, fronts = fast_non_dominated_sort(objs)
        return [pop[i] for i in fronts[0]]

    def _one_generation(self, pop: list[Solution], train: bool, gen: int) -> list[Solution]:
        objs = np.array([s.obj for s in pop])
        ranks, fronts = fast_non_dominated_sort(objs)
        # 官方：理想点 z_ / 最坏点 a_ 归一化（随进化累积更新）
        norm = normalize_objs(objs, self.z_, self.a_)

        F1 = fronts[0]
        f1_cd = crowding_distance(norm[F1])
        # 官方：取拥挤距离最大的 2 个边界个体
        top2 = heapq.nlargest(2, range(len(F1)), key=lambda i: f1_cd[i])
        BI = [F1[i] for i in top2]
        # agent1(ppo[0]) 管 Cmax 小的，agent2(ppo[1]) 管 Cmax 大的
        if len(BI) == 2 and objs[BI[0]][0] > objs[BI[1]][0]:
            BI[0], BI[1] = BI[1], BI[0]
        BI_set = set(BI)

        done = gen == self.Ngen - 1  # 最后一代表记 done

        offspring: list[Solution] = []
        for idx, sol in enumerate(pop):
            sch = decode_schedule(self.inst, sol)
            sol.obj = sch.obj

            if self.controller == "random":
                # Random 控制器：均匀随机选 LLH，无学习
                action = int(self.rng.integers(0, self.n_actions))
                other = self._select_other(pop, ranks, fronts, idx, F1, BI) if action in (0, 2) else None
                new_sol = apply_llh(self.inst, sol, action + 1, self.rng, sch, other)
                if new_sol is None:
                    new_sol = sol.copy()
                new_sol.obj = decode(self.inst, new_sol)
                self.z_ = np.minimum(self.z_, new_sol.obj)
                self.a_ = np.maximum(self.a_, new_sol.obj)
                offspring.append(new_sol)
            elif idx in BI_set and self.use_ppo:
                # 单边界个体时两个 agent 各执行一次（官方复制 FSM）
                agent_list = [0, 1] if len(BI) == 1 else ([0] if idx == BI[0] else [1])
                for ai in agent_list:
                    agent = self.ppo[ai]
                    fsm = build_fsm(self.inst, sch, sol.obj, self.fsm_rows, self.fsm_cols)
                    action, logprob, _ = agent.act(fsm, self.rng)  # 动作 0..9
                    other = self._select_other(pop, ranks, fronts, idx, F1, BI) if action in (0, 2) else None
                    new_sol = apply_llh(self.inst, sol, action + 1, self.rng, sch, other)
                    if new_sol is None:
                        new_sol = sol.copy()
                    new_sch = decode_schedule(self.inst, new_sol)
                    new_sol.obj = new_sch.obj
                    r = reward(sol.obj, new_sol.obj)
                    # 更新理想点/最坏点
                    self.z_ = np.minimum(self.z_, new_sol.obj)
                    self.a_ = np.maximum(self.a_, new_sol.obj)
                    if train:
                        fsm_new = build_fsm(self.inst, new_sch, new_sol.obj, self.fsm_rows, self.fsm_cols)
                        agent.store(fsm, action, r, fsm_new, logprob, done)
                        # 官方：每 batch_size 条或 episode 结束 update
                        if len(agent.buffer) >= self.batch_size or done:
                            agent.update(epochs=self.update_epochs, batch_size=self.batch_size)
                    offspring.append(new_sol)
            else:
                action = self.q_agent.act(self.rng)  # 动作 0..9
                other = self._select_other(pop, ranks, fronts, idx, F1, BI) if action in (0, 2) else None
                new_sol = apply_llh(self.inst, sol, action + 1, self.rng, sch, other)
                if new_sol is None:
                    new_sol = sol.copy()
                new_sol.obj = decode(self.inst, new_sol)
                r = reward(sol.obj, new_sol.obj)
                # 更新理想点/最坏点
                self.z_ = np.minimum(self.z_, new_sol.obj)
                self.a_ = np.maximum(self.a_, new_sol.obj)
                if train:
                    self.q_agent.update(action, r)
                offspring.append(new_sol)

        # 精英保留 EP：有则按 rank/CD 选 Np，无则直接用子代（论文 Table 6 的 EP 开关）
        if self.use_ep:
            return environmental_selection(pop, offspring, self.Np)
        return offspring[: self.Np]

    def _select_other(self, pop, ranks, fronts, p1_idx, F1, BI) -> Solution:
        """Algorithm 5: 选交叉的另一父代。"""
        if ranks[p1_idx] > 1:
            return pop[int(self.rng.choice(F1))]
        if len(BI) == 1:
            second = fronts[1] if len(fronts) > 1 else F1
            return pop[int(self.rng.choice(second))]
        others = [b for b in BI if b != p1_idx]
        return pop[int(self.rng.choice(others if others else BI))]
