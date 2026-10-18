# MOECLF 复现

对论文 **Multi-objective evolutionary co-learning framework for energy-efficient hybrid flow-shop scheduling problem with human-machine collaboration** 的阅读笔记与复现工作。

> Wu J., Liu Y., Zhang Y. *Multi-objective evolutionary co-learning framework for energy-efficient hybrid flow-shop scheduling problem with human-machine collaboration*. **Swarm and Evolutionary Computation**, 95 (2025) 101932. (上海理工大学管理学院)

## 项目简介

论文研究 **人机协作的节能混合流水车间调度问题（EHFSP-HMC）**，提出 **多目标进化协同学习框架（MOECLF）**：用强化学习（PPO 等）在线选择低层启发式（LLH），驱动多目标进化算法求解，并通过"进化—学习"闭环持续增强搜索能力。

本目录用于整理该论文的精读笔记、关键图表复刻与后续代码复现，是 [`paper-reproductions`](../) 合集中的一个子项目。

## 目录结构

```
MOECLF/
├── Code/                          # 复现代码（待补充）
├── Notion/
│   ├── 阅读笔记.md                 # 论文精读笔记：基本信息、核心内容、阅读总结
│   └── DRL增强EA笔记/
│       ├── DRL增强EA学习笔记.md    # 以 MOECLF 为例的 DRL 增强 EA 系统学习笔记
│       └── images/                # 笔记插图（Pareto/BI、奖励分区、FSM、消融实验等）
├── README.md
└── .gitignore
```

## 笔记导航

### 阅读笔记

围绕论文本身展开，包含：

- **基本信息**：标题、作者、期刊、关键词
- **核心内容**：问题建模（EHFSP-HMC）、MOECLF 框架、实验设置与结论
- **阅读总结**：方法价值、局限与可借鉴点
- **阅读原则**：如何读这类多目标进化 + RL 交叉论文

### DRL 增强 EA 学习笔记

以 MOECLF 为载体，梳理"RL 如何增强进化算法"的完整链路：

| 章节 | 内容 |
| --- | --- |
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

## 关键概念速查

- **EHFSP**：混合流水车间调度问题（Hybrid Flow-Shop Scheduling Problem）
- **HMC**：人机协作（Human-Machine Collaboration）
- **MOECLF**：多目标进化协同学习框架
- **LLH**：低层启发式（Low-Level Heuristic），作为 RL 的动作空间
- **FSM**：可行解矩阵（Feasible Solution Matrix），用于状态表示
- **BI**：边界交点（Boundary Intersection），用于分解式多目标选择
- **MHV**：超体积指标（HyperVolume），衡量解集收敛性与多样性

## 说明

笔记中的图表用于学术交流与学习，版权归原论文作者及出版方所有。若涉及版权问题请联系删除。

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
