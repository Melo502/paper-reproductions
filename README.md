# Paper Reproductions

论文复现合集。每篇论文一个独立子目录，内含精读笔记、图表复刻与复现代码。

## 已收录

| 论文 | 主题 | 期刊 / 年份 | 目录 |
| --- | --- | --- | --- |
| **MOECLF** — Multi-objective evolutionary co-learning framework for energy-efficient hybrid flow-shop scheduling problem with human-machine collaboration | 多目标进化协同学习；人机协作节能混合流水车间调度（EHFSP-HMC） | Swarm and Evolutionary Computation, 95 (2025) 101932 | [`MOECLF/`](MOECLF/) |

## 目录约定

新增一篇论文时，按以下结构建目录：

```
<论文简称>/
├── README.md          # 论文信息、复现进度、如何运行
├── Notion/            # 精读笔记与插图
│   └── images/
├── Code/              # 复现代码
├── data/              # 数据（已在 .gitignore 中忽略）
└── results/           # 实验输出（已在 .gitignore 中忽略）
```

- 目录名用论文方法的简称（大写），如 `MOECLF`、`NSGA3`
- 每个子目录的 README 需写明：论文标题、作者、期刊年份、复现进度
- 图片统一放 `Notion/images/`，命名 `fig<编号>_<含义>.png`
- 数据与实验输出不入库，只提交代码、笔记与配置

## 说明

笔记中的图表来自原论文，用于学术交流与学习，版权归原作者及出版方所有。如涉版权问题请联系删除。
