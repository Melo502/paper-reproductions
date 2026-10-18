"""结果表渲染：给定各算法的前沿集合，生成 HV/GD/IGD、覆盖率 C、Wilcoxon p 值的 Markdown 表。

被 run_experiments.py（控制器/算法对比）与 run_ablation.py（消融）共用。
"""
from __future__ import annotations

import numpy as np

from ehfsp_hmc import fast_non_dominated_sort
from metrics import gd, hv, igd, normalize_objs, set_coverage, wilcoxon_ranksum


def nondominated(objs):
    """取 objs 的第一非支配前沿。"""
    _, fronts = fast_non_dominated_sort(objs)
    return objs[fronts[0]] if fronts else objs


def render(title: str, labels: dict, fronts: dict, instances: list, seeds: list) -> str:
    """渲染一个实验的完整结果表。

    labels : {算法键: 显示名}
    fronts : {算法键: {实例: [每个种子的前沿 ndarray]}}
    """
    names = list(labels)
    L = [f"\n## {title}\n"]
    L.append("| 实例 | " + " | ".join(labels[n] for n in names) + " |")
    L.append("|" + "---|" * (len(names) + 1))

    hv_table = {n: [] for n in names}
    for inst in instances:
        # 归一化参考集：本实验内所有算法的所有前沿
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
                      for i in instances for s in range(len(seeds))]
                row.append(f"{np.mean(cs):.3f}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n### Wilcoxon rank-sum p 值（行 vs 列，基于 HV，各种子×实例）\n")
    L.append("| p | " + " | ".join(labels[n] for n in names) + " |")
    L.append("|" + "---|" * (len(names) + 1))
    for n1 in names:
        row = [f"**{labels[n1]}**"]
        for n2 in names:
            row.append("—" if n1 == n2 else
                       f"{wilcoxon_ranksum(hv_table[n1], hv_table[n2]):.2e}")
        L.append("| " + " | ".join(row) + " |")
    return "\n".join(L) + "\n"
