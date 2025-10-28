# 高级搜索机制实验 - 快速启动指南

## 概述

本指南帮助你快速运行完整的消融实验和基准测试,验证frontier-based探索和协同机制的有效性。

---

## 前置准备

### 1. 检查代码提交

```bash
# 确认在reg_search分支
git branch

# 查看最近提交(应该看到4周的工作)
git log --oneline -4

# 预期输出:
# 95f4b49 feat: implement ablation study and benchmark framework (Week 4)
# 8133707 feat: implement comprehensive evaluation metrics system (Week 3)
# 8dc62da feat: implement coordination mechanisms for multi-agent search (Week 2)
# 689e615 feat: implement frontier-based intrinsic reward mechanism (Week 1)
```

### 2. 安装依赖

```bash
# 核心依赖
pip install numpy pandas scipy pyyaml

# 可视化(推荐)
pip install matplotlib seaborn

# TensorBoard支持(推荐)
pip install tensorboard
```

### 3. 验证文件结构

```bash
# 检查关键文件
ls marl_framework/mapping/frontier_detection.py  # Week 1
ls marl_framework/utils/coordination.py          # Week 2
ls marl_framework/utils/metrics.py               # Week 3
ls scripts/ablation_study.py                     # Week 4
ls scripts/benchmark_runner.py                   # Week 4
ls scripts/result_analyzer.py                    # Week 4
```

---

## 快速测试(推荐先运行)

### Option 1: 单元测试

```bash
# 测试frontier检测
python -c "from marl_framework.mapping.frontier_detection import test_frontier_detector; test_frontier_detector()"

# 测试协同机制
python -c "from marl_framework.utils.coordination import test_coordination_manager; test_coordination_manager()"

# 测试评估指标
python -c "from marl_framework.utils.metrics import test_metrics_system; test_metrics_system()"
```

### Option 2: 配置验证

```bash
# 验证消融实验配置生成
python scripts/ablation_study.py --setup_only

# 验证基准测试场景生成
python scripts/benchmark_runner.py --setup_only
```

---

## 实验流程

### 阶段1: 小规模消融实验(推荐开始)

**目标**: 验证框架可用性,快速迭代

#### Step 1.1: 运行3组关键对比实验

```bash
# 只运行3组最关键的实验(约3小时)
python scripts/ablation_study.py \
    --base_config marl_framework/configs/params_advanced_search.yaml \
    --output_dir experiments/ablation_quick \
    --run_experiments intrinsic_baseline intrinsic_frontier intrinsic_full
```

**预期输出**:
```
experiments/ablation_quick/
├── configs/
│   ├── intrinsic_baseline.yaml
│   ├── intrinsic_frontier.yaml
│   └── intrinsic_full.yaml
└── logs/
    ├── intrinsic_baseline.log
    ├── intrinsic_frontier.log
    └── intrinsic_full.log
```

#### Step 1.2: 分析初步结果

```bash
python scripts/result_analyzer.py \
    --log_dir experiments/ablation_quick/logs \
    --output_dir experiments/analysis_quick \
    --baseline intrinsic_baseline \
    --experiments intrinsic_baseline intrinsic_frontier intrinsic_full
```

**预期输出**:
```
experiments/analysis_quick/
├── ablation_metrics.csv          # 原始数据
├── statistical_tests.csv         # 统计结果
└── plots/
    ├── ablation_discovery_rate.png
    ├── ablation_search_efficiency.png
    └── learning_curves_reward.png
```

#### Step 1.3: 检查结果

```bash
# 查看统计检验结果
cat experiments/analysis_quick/statistical_tests.csv

# 期望: frontier显著优于baseline (p < 0.05)
```

---

### 阶段2: 完整消融实验

**目标**: 系统性验证所有组件

#### Step 2.1: 运行完整消融实验(14组)

```bash
# 运行所有14组消融实验(约14小时)
python scripts/ablation_study.py \
    --base_config marl_framework/configs/params_advanced_search.yaml \
    --output_dir experiments/ablation_full
```

**实验组**:
- 内在奖励: baseline, coverage, frontier, curiosity, full (5组)
- 协同机制: baseline, overlap, division, collab, full (5组)
- 通信条件: full_comm, limited, sparse, no_comm (4组)

#### Step 2.2: 全面分析

```bash
python scripts/result_analyzer.py \
    --log_dir experiments/ablation_full/logs \
    --output_dir experiments/analysis_full \
    --baseline intrinsic_baseline \
    --experiments intrinsic_baseline intrinsic_coverage intrinsic_frontier \
                 intrinsic_curiosity intrinsic_full \
                 coord_baseline coord_overlap coord_division coord_collab coord_full \
                 full_comm limited_comm sparse_comm no_comm
```

---

### 阶段3: 基准测试

**目标**: 多场景验证和baseline对比

#### Step 3.1: 运行基准测试(12场景)

```bash
# 并行运行12组场景测试(约6小时,2 workers)
python scripts/benchmark_runner.py \
    --base_config marl_framework/configs/params_advanced_search.yaml \
    --output_dir experiments/benchmark \
    --max_workers 2
```

**场景**:
- 规模: small, medium, large, xlarge (4组)
- 密度: sparse, normal, dense, very_dense (4组)
- 复杂度: simple, moderate, complex, extreme (4组)

#### Step 3.2: Baseline对比

```bash
# 如果已经运行了基准测试但跳过了baseline
# 可以单独运行baseline对比
python -c "
from scripts.benchmark_runner import BaselineComparator
import glob

scenarios = [f.stem for f in Path('experiments/benchmark/scenarios').glob('*.yaml')]
comparator = BaselineComparator(
    'experiments/benchmark/scenarios',
    'experiments/benchmark/baselines'
)
comparator.run_all_baselines(scenarios)
"
```

---

## 监控实验进度

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=experiments/ablation_full/logs

# 访问 http://localhost:6006
# 查看实时训练曲线
```

### 日志查看

```bash
# 查看某个实验的日志
tail -f experiments/ablation_full/logs/intrinsic_frontier.log

# 检查是否有错误
grep -i error experiments/ablation_full/logs/*.log
```

---

## 结果解读

### 关键指标说明

| 指标 | 含义 | 期望值 |
|------|------|--------|
| `discovery_rate` | 目标发现率 | 越高越好 (0-1) |
| `completion_time` | 任务完成时间 | 越低越好 |
| `search_efficiency` | 搜索效率 | 越高越好 |
| `coordination_efficiency` | 协同效率 | 越高越好 |
| `path_redundancy` | 路径冗余度 | 越低越好 |
| `load_balance` | 负载均衡 | 越接近1越好 |

### 成功标准

✅ **Frontier vs Baseline**:
- discovery_rate提升 > 10%
- search_efficiency提升 > 15%
- p-value < 0.05, Cohen's d > 0.5

✅ **完整协同 vs 单一机制**:
- coordination_efficiency提升 > 20%
- load_balance改善 > 10%
- p-value < 0.05

✅ **可扩展性**:
- 性能退化 < 20% (agents数量翻倍时)
- 保持显著优于baseline

---

## 常见问题

### Q1: 实验运行失败

```bash
# 检查配置文件
python -c "import yaml; yaml.safe_load(open('marl_framework/configs/params_advanced_search.yaml'))"

# 检查Python环境
python -c "import numpy, pandas, scipy; print('Dependencies OK')"

# 查看详细错误
cat experiments/ablation_full/logs/FAILED_EXPERIMENT.log
```

### Q2: 内存不足

```bash
# 减少并行worker数量
python scripts/benchmark_runner.py --max_workers 1

# 或分批运行
python scripts/ablation_study.py --run_experiments intrinsic_baseline intrinsic_frontier
python scripts/ablation_study.py --run_experiments coord_baseline coord_full
```

### Q3: 训练时间过长

```bash
# 减少训练轮数(仅用于快速测试)
# 修改 params_advanced_search.yaml:
# training:
#   num_episodes: 100  # 原来1000

# 或只在小场景测试
python scripts/benchmark_runner.py --run_experiments scale_small density_sparse
```

### Q4: TensorBoard日志解析失败

```bash
# 手动检查event文件
ls experiments/ablation_full/logs/*/events.out.tfevents.*

# 如果缺失,说明实验可能未正确运行
# 重新运行该实验
```

---

## 生成论文结果

### 1. 提取关键数据

```bash
# 消融实验结果表
python scripts/result_analyzer.py \
    --log_dir experiments/ablation_full/logs \
    --output_dir paper_results \
    --baseline intrinsic_baseline \
    --experiments intrinsic_baseline intrinsic_frontier intrinsic_full

# 输出: paper_results/ablation_metrics.csv
# 可直接导入LaTeX表格
```

### 2. 生成图表

所有图表已自动生成在 `experiments/analysis_full/plots/`:
- `ablation_*.png`: 消融实验对比(Figure 1-3)
- `learning_curves_*.png`: 学习曲线(Figure 4)
- `performance_heatmap.png`: 性能热力图(Figure 5)

### 3. 统计显著性

```bash
# 统计检验结果
cat experiments/analysis_full/statistical_tests.csv

# 包含:
# - p-values (显著性)
# - Cohen's d (效应大小)
# - 均值和标准差
```

---

## 下一步

### 如果初步结果良好:
1. ✅ 运行完整消融实验
2. ✅ 运行完整基准测试
3. ✅ 生成所有图表和统计结果
4. 📝 开始撰写论文实验章节

### 如果结果不理想:
1. 🔧 调整超参数 (params_advanced_search.yaml)
2. 🔧 增加训练轮数
3. 🔧 检查reward设计
4. 🔄 重新运行实验

### 论文写作:
- 参考 `WEEK4_SUMMARY.md` 中的论文章节结构
- 使用生成的图表和统计结果
- 强调frontier-based探索和协同机制的创新性

---

## 时间预估

| 任务 | 时间 | 说明 |
|------|------|------|
| 快速测试 | 3 hours | 3组关键对比 |
| 完整消融实验 | 14 hours | 14组实验 |
| 基准测试 | 6 hours | 12场景,2 workers |
| Baseline对比 | 8 hours | 12场景×3 baselines |
| 结果分析 | 2 hours | 统计分析+可视化 |
| **总计** | **33 hours** | 约1.5天(连续运行) |

**建议**: 使用服务器或高性能工作站,开启多worker并行训练。

---

## 支持

遇到问题?
1. 查看 `WEEK4_SUMMARY.md` 详细文档
2. 检查各周的实现总结 (FRONTIER_SUMMARY.md, COORDINATION_SUMMARY.md)
3. 运行单元测试验证各模块

祝实验顺利! 🚀
