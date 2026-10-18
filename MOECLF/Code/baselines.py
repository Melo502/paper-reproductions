"""对比算法基线：NSGA-II 与 MODJA（多目标离散 Jaya）。

复用 ehfsp_hmc 的编码/解码/排序/拥挤距离、moeclf 的 ms_init/environmental_selection、
operators 的 LLH1/LLH2。论文 Table 8 参数：NSGA-II Cr=0.8/Mr=0.2。
"""
from __future__ import annotations

import numpy as np

from ehfsp_hmc import (
    Instance,
    Solution,
    decode,
    random_solution,
    fast_non_dominated_sort,
    crowding_distance,
)
from moeclf import ms_init, environmental_selection
from operators import apply_llh


class NSGAII:
    """标准 NSGA-II：MSI 初始化 + binary tournament + LLH1(交叉)/LLH2(变异) + EP。"""

    def __init__(self, inst: Instance, Np: int = 50, Ngen: int = 100, cr: float = 0.8, mr: float = 0.2, seed: int = 0):
        self.inst = inst
        self.Np = Np
        self.Ngen = Ngen
        self.cr = cr
        self.mr = mr
        self.rng = np.random.default_rng(seed)

    def run(self) -> list[Solution]:
        pop = ms_init(self.inst, self.Np, self.rng)
        for _ in range(self.Ngen):
            pop = self._one_generation(pop)
        objs = np.array([s.obj for s in pop])
        _, fronts = fast_non_dominated_sort(objs)
        return [pop[i] for i in fronts[0]]

    def _one_generation(self, pop: list[Solution]) -> list[Solution]:
        objs = np.array([s.obj for s in pop])
        ranks, fronts = fast_non_dominated_sort(objs)
        cd = {i: d for f in fronts for i, d in zip(f, crowding_distance(objs[f]))}

        offspring: list[Solution] = []
        for _ in range(self.Np):
            p1 = self._tournament(pop, ranks, cd)
            p2 = self._tournament(pop, ranks, cd)
            child = p1.copy()
            if self.rng.random() < self.cr:
                c = apply_llh(self.inst, child, 1, self.rng, None, p2)  # LLH1 2p-OXO
                child = c if c is not None else child
            if self.rng.random() < self.mr:
                c = apply_llh(self.inst, child, 2, self.rng)             # LLH2 2p-ESM
                child = c if c is not None else child
            child.obj = decode(self.inst, child)
            offspring.append(child)
        return environmental_selection(pop, offspring, self.Np)

    def _tournament(self, pop, ranks, cd) -> Solution:
        i, j = self.rng.choice(len(pop), 2, replace=False)
        if ranks[i] < ranks[j] or (ranks[i] == ranks[j] and cd[i] > cd[j]):
            return pop[i]
        return pop[j]


class MODJA:
    """多目标离散 Jaya：随机初始化 + 离散 Jaya 更新（向 best 靠拢、远离 worst）+ EP。"""

    def __init__(self, inst: Instance, Np: int = 50, Ngen: int = 100, seed: int = 0):
        self.inst = inst
        self.Np = Np
        self.Ngen = Ngen
        self.rng = np.random.default_rng(seed)

    def run(self) -> list[Solution]:
        pop = [random_solution(self.inst, self.rng) for _ in range(self.Np)]
        for s in pop:
            s.obj = decode(self.inst, s)
        for _ in range(self.Ngen):
            pop = self._one_generation(pop)
        objs = np.array([s.obj for s in pop])
        _, fronts = fast_non_dominated_sort(objs)
        return [pop[i] for i in fronts[0]]

    def _one_generation(self, pop: list[Solution]) -> list[Solution]:
        objs = np.array([s.obj for s in pop])
        ranks, fronts = fast_non_dominated_sort(objs)
        cd = {i: d for f in fronts for i, d in zip(f, crowding_distance(objs[f]))}

        best_idx = min(range(len(pop)), key=lambda i: (ranks[i], -cd[i]))
        worst_idx = max(range(len(pop)), key=lambda i: (ranks[i], cd[i]))
        best = pop[best_idx]
        worst = pop[worst_idx]

        offspring: list[Solution] = []
        for sol in pop:
            new_seq = self._jaya(sol.job_seq, best.job_seq, worst.job_seq)
            child = Solution(new_seq, sol.mach.copy(), sol.worker.copy())
            child.obj = decode(self.inst, child)
            offspring.append(child)
        return environmental_selection(pop, offspring, self.Np)

    def _jaya(self, x, x_best, x_worst) -> list[int]:
        """离散 Jaya：每个基因以概率向 best 靠拢（远离 worst），再修复重复/缺失。"""
        jobs = self.inst.I
        new = list(x)
        for i in range(jobs):
            if self.rng.random() < 0.5:
                new[i] = x_best[i] if self.rng.random() < 0.5 else x_worst[i]
        return self._repair(new, jobs)

    @staticmethod
    def _repair(order, jobs) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for x in order:
            if 0 <= x < jobs and x not in seen:
                out.append(x)
                seen.add(x)
        out.extend(x for x in range(jobs) if x not in seen)
        return out
