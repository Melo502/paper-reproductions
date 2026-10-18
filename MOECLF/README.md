# MOECLF 复现

对论文 **Multi-objective evolutionary co-learning framework for energy-efficient hybrid flow-shop scheduling problem with human-machine collaboration** 的阅读笔记与代码复现，是 [`paper-reproductions`](../) 合集中的一个子项目。

## 📄 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | Multi-objective evolutionary co-learning framework for energy-efficient hybrid flow-shop scheduling problem with human-machine collaboration |
| 作者 | Wu J., Liu Y., Zhang Y.（上海理工大学管理学院） |
| 期刊 | Swarm and Evolutionary Computation, 95 (2025) 101932 |
| DOI | [10.1016/j.swevo.2025.101932](https://doi.org/10.1016/j.swevo.2025.101932) |
| 领域 | 调度优化 / 多目标进化 / 强化学习增强 EA |
| 官方代码 | 未找到，本复现为独立实现 |

## 🎯 复现目标

论文研究 **人机协作的节能混合流水车间调度问题（EHFSP-HMC）**，提出 **多目标进化协同学习框架（MOECLF）**：用 PPO 与 Q-learning 双智能体在线选择低层启发式（LLH），驱动多目标进化搜索，通过"进化—学习"闭环增强搜索能力。

本复现依据论文公式与流程图**独立实现**（非作者代码），复现以下实验：

| 实验 | 对应论文 | 内容 |
|------|----------|------|
| 实验 1 | 消融实验 | v1 基线 → +MSI → +VNS → +EP → +PPO 逐级增益 |
| 实验 2 | 控制器对比 | Random / Q-only / PPO+Q |
| 实验 3 | 算法对比 | MODJA / NSGA-II / MOEQLF / MOECLF |
| 实验 2、3 预训练版 | Table 5/6 协议 | 跨 10 实例预训练 → 冻结策略测试 |
| **实验 A**（额外） | 原文无 | Oracle 完美选择器上限诊断 |

规模与论文一致：`Np=200 / Ngen=1000`，5 个随机种子 × 6 个代表实例。

## 📁 目录结构

```
MOECLF/
├── README.md                          # 本文件
├── Code/                              # 复现代码
│   ├── REPORT.md                      # ★ 完整复现报告（结果、分析、局限）
│   ├── ehfsp_hmc.py                   # 问题定义：实例加载、编解码、非支配排序、拥挤距离
│   ├── moeclf.py                      # 主算法：MSI 初始化 + PPO/Q 协同 + 精英保留
│   ├── agents.py                      # PPO（MobileNetV2 + FSM 输入）与 Q-learning
│   ├── operators.py                   # 10 个低层启发式 LLH1–LLH10
│   ├── baselines.py                   # 对比基线：NSGA-II、MODJA
│   ├── metrics.py                     # 指标 C / GD / IGD / HV 与 Wilcoxon 检验
│   ├── run.py                         # 入口：smoke / train / test
│   ├── run_experiments.py             # 三组对照实验（并行 + 断点续跑）
│   ├── run_pretrain.py                # 跨 episode 预训练版
│   ├── experiment_a_oracle.py         # 实验 A：Oracle 上限诊断
│   ├── EHFSP-HMC.json                 # 官方算例（90 个实例）
│   ├── requirements.txt
│   └── results/                       # 6 份结果表（Np=200 / Ngen=1000）
└── Notion/                            # 精读笔记
    ├── 阅读笔记.md
    └── DRL增强EA笔记/
        ├── DRL增强EA学习笔记.md
        └── images/
```

## 🚀 环境与依赖

| 项目 | 配置 |
|------|------|
| 操作系统 | macOS 26.6.2（Apple Silicon, arm64） |
| Python | 3.12.14 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.14.0 |
| GPU | 无（CPU 上 14 进程并行） |

```bash
cd Code
pip install -r requirements.txt
```

设备自动选择（CUDA → MPS → CPU），也可用 `DEVICE=cpu` 显式指定。

## 📊 数据与算例

`Code/EHFSP-HMC.json` 为论文官方算例，共 90 个实例，覆盖 J15/J30/J45 三档作业数与 2/4/6 阶段。实验选用 6 个代表实例：

```
J15-S2-1  J15-S4-1  J30-S2-1  J30-S4-1  J45-S4-1  J45-S6-1
```

疲劳影响因子取主实验档位（`delta_idx=0`，即 `0.3/ln2`）。数据已随仓库提供，无需额外下载。

## ▶️ 运行命令

```bash
cd Code

# 全链路冒烟测试（秒级）
python run.py smoke

# 三组对照实验（可加 --workers 指定并行度）
python run_experiments.py

# 预训练版（跨 10 个实例预训练 → 冻结策略测试）
python run_pretrain.py

# 实验 A：Oracle 上限诊断
python experiment_a_oracle.py
```

结果写入 `Code/results/`，中间结果缓存在 `Code/results/_cache/`，中断后重跑会自动跳过已完成任务。

实测耗时（14 进程 / CPU）：

| 阶段 | 任务数 | 耗时 |
|------|-------:|-----:|
| 实验 1 消融 | 150 | 6067 s |
| 实验 2 控制器 | 30 | 1272 s |
| 实验 3 算法 | 60 | 1237 s |
| 预训练 Phase A | 10 ep × 2 组 | 6405 s |
| 实验 A Oracle | 30 | 约 260 s |

## 📈 复现结果

HV 均值（6 实例 × 5 种子），完整数据与逐实例表格见 [`Code/REPORT.md`](./Code/REPORT.md)。

### 结论速览

| 论文结论 | 复现结果 | 是否一致 |
|----------|----------|----------|
| 精英保留（EP）显著提升性能 | v3→v4：HV 0.393 → 0.858，p = 3.0e-11 | ✅ 一致 |
| MSI 初始化有贡献 | v1→v2：HV 0.318 → 0.410，p = 4.2e-4 | ✅ 一致 |
| VNS 有贡献 | v2→v3：HV −0.017，p = 0.45 | ❌ 不显著 |
| MOECLF/MOEQLF 优于 MODJA、NSGA-II | 覆盖率 C 最高 0.998，p ≈ 3e-11 | ✅ 一致 |
| PPO 控制器带来增益 | 4 组实验一致显示无增益 | ❌ 未能复现 |

### 实验 1：消融

| 配置 | HV 均值 | 相对上一版 | p |
|------|--------:|-----------:|---:|
| v1 基线 | 0.3177 | — | — |
| v2 +MSI | 0.4102 | +0.0925 | 4.22e-04 ✅ |
| v3 +VNS | 0.3933 | −0.0169 | 4.46e-01 ❌ |
| v4 +EP (= MOEQLF) | **0.8577** | +0.4644 | 3.02e-11 ✅ |
| MOECLF +PPO | 0.8643 | +0.0066 | 6.84e-01 ❌ |

**EP 是唯一的主要贡献**（+0.46），MSI 中等贡献，VNS 无贡献，PPO 无增益。

### 实验 2：控制器对比

| 配置 | HV 均值 | vs Random 的 p |
|------|--------:|---------------:|
| **Random** | **0.7182** | — |
| Q-only | 0.6557 | 7.96e-03（更差） |
| PPO+Q | 0.6747 | 3.03e-02（更差） |

**两个学习型控制器都打不过随机**，与论文预期方向相反。

### 实验 2、3：预训练版

| 配置 | HV 均值 | 说明 |
|------|--------:|------|
| Random | 0.7215 | 基线 |
| Q-only(预训练) | 0.6610 | 与不预训练几乎一致（Q 表本就收敛） |
| PPO+Q(预训练) | 0.6210 | 差距从 −0.044 扩大到 −0.101 |
| MOEQLF(预训练) | **0.8280** | — |
| MOECLF(预训练) | 0.7897 | 显著低于 MOEQLF，p = 0.047 |

预训练把 PPO 样本量放大 10 倍后表现更差，**"样本不足"这一解释不成立**。

### 实验 3：算法对比

| 配置 | HV 均值 | C（支配 NSGA-II 的比例） |
|------|--------:|-------------------------:|
| MODJA | 0.4960 | 0.134 |
| NSGA-II | 0.5540 | — |
| MOEQLF | 0.8032 | 0.955 |
| MOECLF | **0.8120** | 0.956 |

MOECLF/MOEQLF 显著优于两个基线（p ≈ 3e-11），但 MOECLF 与 MOEQLF 之间不显著（p = 0.663）——**PPO 模块在算法对比中同样没有贡献**。

### 实验 A：Oracle 上限诊断（额外实验）

为回答"PPO 没赢是因为学得不好，还是这个位置本来没价值"，额外实现 Oracle：对 PPO 负责的 2 个边界个体**枚举全部 10 个 LLH** 并取 reward 最大者，代表该位置的理论上限。

| 对比 | HV 平均差 | 前者胜出 | p |
|------|----------:|---------:|---:|
| Oracle vs Q-only | +0.0293 | 5/6 | 0.219 |
| **Oracle vs Random** | **−0.0319** | 2/6 | 0.219 |
| Oracle vs PPO+Q | +0.0106 | 4/6 | 0.562 |

三点结论：

1. **2/200 边界位置不是瓶颈**——即使完美选择，也只比 Q-only 高 0.029（不显著）。
2. **Oracle 比纯 Random 还低**——不只 PPO，Q-learning 在该位置同样没学到东西。
3. **奖励函数梯度稀疏**——即使取每步最大可达 reward，`reward=+2`（真正的支配改进）仅占 0.7%–2.6%；J45-S4-1 上平均最优 reward 为 **−0.135**，51.2% 的决策即便选最优也是"变差"。

## ⚠️ 与论文的差异

| 项目 | 原文 | 本复现 | 影响 |
|------|------|--------|------|
| 预训练测试集 | 必须是 unseen 实例 | 含 `J45-S6-1`，它在训练集内 | 见下 |
| 运行设备 | 服务器 GPU | 本地 CPU | 不改变算法语义 |
| 统计样本量 | 未明确 | 6 实例 × 5 种子 | 检验功效偏小 |

**训练/测试集重叠**：`run_pretrain.py` 的 `TRAIN_INSTANCES = J45-S6-1..10`，而测试集也含 `J45-S6-1`。按当前结论这不影响判断（该实例上 PPO+Q 的 HV 为 0.642，仍低于 Random 的 0.830），但**不应基于预训练版数字论证正向结论**。若要修正，从测试集剔除该实例即可，Phase A 无需重跑。

**其他局限**：

- **n = 6 功效弱**：p = 0.219 只说明"无证据显示有增益"，不能断言"绝对无增益"。
- **归一化参考集各实验独立构造**：同一配置在不同表中的绝对值不同（如 J15-S2-1 的 Random：实验 2 为 0.762，实验 A 为 0.7487）。**跨表比较无意义**，只能引用同表内的对比。
- **实验 B（PPO 全权控制）未完成**：让 PPO 接管全部 200 个个体以区分"PPO 学不动"与"位置无价值"。该实验未产出结果，相关代码已在整理时移除，本仓库不包含其实现——基准测试显示 PPO 单次前向约 12 ms，单 run 约需 11 小时，30 个 run 需 37 小时，CPU 上不可行。

**对论文核心创新点的判断**：论文的创新是"PPO + Q-learning 双智能体协同"。复现在四条独立路径上一致不支持该模块的增益（常规 p=0.68、低于 Random p=0.030、预训练后更差 p=6.9e-4、预训练算法对比显著更差 p=0.047），且上限实验排除了"样本不足"与"位置有潜力"两种解释。瓶颈更可能在结构设计（PPO 仅作用于 2/200 个体）与奖励函数的稀疏性。

## 📝 笔记导航

### 阅读笔记

围绕论文本身展开，包含：

- **基本信息**：标题、作者、期刊、关键词、DOI
- **核心内容**：9 个小节 —— 研究问题、核心思想、研究方法、创新点、实验与结果、主要结论、局限性、个人思考、关联主题

### DRL 增强 EA 学习笔记

以 MOECLF 为载体，梳理"RL 如何增强进化算法"的完整链路：

| 章节 | 内容 |
|------|------|
| 0 | 一页纸速览 |
| 1 | 背景：RL 为什么能帮 EA |
| 2 | 问题回顾：EHFSP-HMC |
| 3 | 总体框架：一代进化怎么跑 |
| 4 | EA 端的支撑模块 |
| 5 | 动作空间：10 个 LLH |
| 6 | 状态表示：LLH 状态与 FSM |
| 7 | 两类智能体：谁管谁、怎么学 |
| 8 | 奖励函数：用支配关系说话 |
| 9 | 训练与测试 |
| 10 | 闭环协同：到底怎么"增强" |
| 11 | 实验证据 |
| 12 | 批判性阅读：哪些结论要打折扣 |
| 13 | 复现用伪代码 |
| 14 | 自测题 |
| 15 | 符号与缩写速查 |

## 🔑 关键概念速查

- **EHFSP**：混合流水车间调度问题（Hybrid Flow-Shop Scheduling Problem）
- **HMC**：人机协作（Human-Machine Collaboration）
- **MOECLF**：多目标进化协同学习框架
- **MOEQLF**：去掉 PPO、仅用 Q-learning 的版本（消融中的 v4）
- **LLH**：低层启发式（Low-Level Heuristic），作为 RL 的动作空间
- **FSM**：可行解矩阵（Feasible Solution Matrix），用于状态表示
- **BI**：边界个体（Boundary Individual），拥挤距离最大的个体，由 PPO 负责
- **EP**：精英保留（Elitist Preservation），环境选择
- **MHV / HV**：超体积指标（HyperVolume），衡量解集收敛性与多样性
- **Oracle**：本复现额外实现的上限诊断组件，穷举 LLH 取最优 reward

## 📚 参考资料

- 论文原文：[DOI](https://doi.org/10.1016/j.swevo.2025.101932)
- 官方代码：未找到，本复现为独立实现
- 完整复现报告：[`Code/REPORT.md`](./Code/REPORT.md)

## 📄 说明

- 本目录为**学术复现**，非原作者代码，实现依据论文公式与流程图独立完成。
- 笔记中的图表用于学术交流与学习，版权归原论文作者及出版方所有，如涉版权问题请联系删除。
- 本仓库不包含论文 PDF，请通过官方渠道获取。
- 实验中间结果（`Code/results/_cache/`、预训练权重）体积较大，未入库；按上面的命令可重新生成。

## 引用

```bibtex
@article{wu2025multi,
  title   = {Multi-objective evolutionary co-learning framework for energy-efficient
             hybrid flow-shop scheduling problem with human-machine collaboration},
  author  = {Wu, Jiawei and Liu, Yong and Zhang, Yani},
  journal = {Swarm and Evolutionary Computation},
  volume  = {95},
  pages   = {101932},
  year    = {2025}
}
```
