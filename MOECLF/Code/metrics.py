"""多目标评价指标 C / GD / IGD / HV (式39-46)。输入均为已归一化的目标矩阵。"""
from __future__ import annotations

import numpy as np

from ehfsp_hmc import dominates


def normalize_objs(objs: np.ndarray, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0,1]。"""
    rng = fmax - fmin
    rng[rng == 0] = 1.0
    return (objs - fmin) / rng


def set_coverage(U: np.ndarray, V: np.ndarray) -> float:
    """式(40): U 支配 V 的比例。"""
    if len(V) == 0:
        return 0.0
    cnt = sum(1 for v in V if any(dominates(u, v) for u in U))
    return cnt / len(V)


def _min_dists(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """E 中每个解到 P 的最小欧氏距离。"""
    if len(P) == 0:
        return np.full(len(E), np.inf)
    d = np.linalg.norm(E[:, None, :] - P[None, :, :], axis=-1)
    return d.min(axis=-1)


def gd(E: np.ndarray, P: np.ndarray) -> float:
    """式(42): 收敛性。"""
    d = _min_dists(E, P)
    return float(np.sqrt(np.sum(d ** 2) / len(E))) if len(E) else float("inf")


def igd(P: np.ndarray, E: np.ndarray) -> float:
    """式(44): 收敛 + 多样性。"""
    d = _min_dists(P, E)
    return float(d.mean()) if len(P) else float("inf")


def hv(E: np.ndarray, ref=(1.0, 1.0)) -> float:
    """式(46): 超体积（二维最小化，nadir 参考点）。"""
    pts = [(x, y) for x, y in E if x <= ref[0] and y <= ref[1]]
    pts.sort(key=lambda p: p[0])
    vol = 0.0
    y_best = ref[1]
    for x, y in pts:
        if y < y_best:
            vol += (ref[0] - x) * (y_best - y)
            y_best = y
    return vol


def wilcoxon_ranksum(a, b, alternative: str = "two-sided") -> float:
    """Wilcoxon rank-sum test（Mann-Whitney U），论文 Table 13 所用检验。返回 p 值。"""
    from scipy import stats

    return float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)
