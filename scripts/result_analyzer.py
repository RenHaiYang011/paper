"""
结果分析工具 - 统计分析和可视化

功能:
1. 从TensorBoard日志提取指标
2. 统计显著性检验(t-test, ANOVA)
3. 生成对比图表
4. 自动生成实验报告

分析维度:
- 任务完成率对比
- 搜索效率对比
- 协同效能对比
- 学习曲线对比
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 可选的绘图库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not available, plotting disabled")

# 可选的TensorBoard日志读取
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logger.warning("tensorboard not available, TB log parsing disabled")


class TensorBoardLogParser:
    """
    TensorBoard日志解析器
    
    从event文件提取训练指标
    """
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: TensorBoard日志目录
        """
        self.log_dir = Path(log_dir)
        
        if not HAS_TENSORBOARD:
            raise ImportError("tensorboard package required for log parsing")
    
    def parse_single_run(self, run_path: str) -> Dict[str, List[Tuple[int, float]]]:
        """
        解析单次运行的日志
        
        Args:
            run_path: 运行日志路径
        
        Returns:
            标签 -> [(step, value), ...]
        """
        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()
        
        metrics = {}
        
        # 获取所有标量标签
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            events = ea.Scalars(tag)
            metrics[tag] = [(e.step, e.value) for e in events]
        
        return metrics
    
    def parse_experiment(self, experiment_name: str) -> Dict:
        """
        解析整个实验的日志
        
        Args:
            experiment_name: 实验名称
        
        Returns:
            解析后的指标
        """
        experiment_path = self.log_dir / experiment_name
        
        if not experiment_path.exists():
            logger.error(f"Experiment not found: {experiment_path}")
            return {}
        
        # 查找event文件
        event_files = list(experiment_path.glob('events.out.tfevents.*'))
        
        if not event_files:
            logger.warning(f"No event files found in {experiment_path}")
            return {}
        
        # 解析最新的event文件
        latest_file = max(event_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"Parsing: {latest_file}")
        return self.parse_single_run(str(latest_file))


class MetricsExtractor:
    """
    指标提取器
    
    从日志或结果文件提取关键指标
    """
    
    def __init__(self, result_dir: str):
        """
        Args:
            result_dir: 结果目录
        """
        self.result_dir = Path(result_dir)
    
    def extract_from_tb_logs(
        self,
        log_dir: str,
        experiment_names: List[str]
    ) -> pd.DataFrame:
        """
        从TensorBoard日志提取指标
        
        Args:
            log_dir: 日志目录
            experiment_names: 实验名称列表
        
        Returns:
            指标DataFrame
        """
        if not HAS_TENSORBOARD:
            logger.error("TensorBoard parsing not available")
            return pd.DataFrame()
        
        parser = TensorBoardLogParser(log_dir)
        
        all_metrics = []
        
        for exp_name in experiment_names:
            logger.info(f"Extracting metrics: {exp_name}")
            
            metrics = parser.parse_experiment(exp_name)
            
            if not metrics:
                logger.warning(f"No metrics found for: {exp_name}")
                continue
            
            # 提取关键指标的最终值
            exp_metrics = {'experiment': exp_name}
            
            # 搜索核心指标
            if 'metrics/search/discovery_rate' in metrics:
                values = [v for _, v in metrics['metrics/search/discovery_rate']]
                exp_metrics['discovery_rate'] = values[-1] if values else 0.0
            
            if 'metrics/search/completion_time' in metrics:
                values = [v for _, v in metrics['metrics/search/completion_time']]
                exp_metrics['completion_time'] = values[-1] if values else 999.0
            
            # 效率指标
            if 'metrics/efficiency/coverage' in metrics:
                values = [v for _, v in metrics['metrics/efficiency/coverage']]
                exp_metrics['final_coverage'] = values[-1] if values else 0.0
            
            if 'metrics/efficiency/path_redundancy' in metrics:
                values = [v for _, v in metrics['metrics/efficiency/path_redundancy']]
                exp_metrics['path_redundancy'] = values[-1] if values else 1.0
            
            if 'metrics/efficiency/search_efficiency' in metrics:
                values = [v for _, v in metrics['metrics/efficiency/search_efficiency']]
                exp_metrics['search_efficiency'] = values[-1] if values else 0.0
            
            # 协同指标
            if 'metrics/coordination/efficiency' in metrics:
                values = [v for _, v in metrics['metrics/coordination/efficiency']]
                exp_metrics['coordination_efficiency'] = values[-1] if values else 0.0
            
            if 'metrics/coordination/load_balance' in metrics:
                values = [v for _, v in metrics['metrics/coordination/load_balance']]
                exp_metrics['load_balance'] = values[-1] if values else 0.0
            
            if 'metrics/coordination/speedup' in metrics:
                values = [v for _, v in metrics['metrics/coordination/speedup']]
                exp_metrics['speedup'] = values[-1] if values else 1.0
            
            # 奖励指标
            if 'reward/episode_reward' in metrics:
                values = [v for _, v in metrics['reward/episode_reward']]
                exp_metrics['avg_episode_reward'] = np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)
            
            all_metrics.append(exp_metrics)
        
        return pd.DataFrame(all_metrics)
    
    def extract_from_json(
        self,
        json_dir: str,
        experiment_names: List[str]
    ) -> pd.DataFrame:
        """
        从JSON结果文件提取指标
        
        Args:
            json_dir: JSON目录
            experiment_names: 实验名称列表
        
        Returns:
            指标DataFrame
        """
        json_dir = Path(json_dir)
        all_metrics = []
        
        for exp_name in experiment_names:
            json_path = json_dir / f'{exp_name}_metrics.json'
            
            if not json_path.exists():
                logger.warning(f"Metrics file not found: {json_path}")
                continue
            
            with open(json_path, 'r') as f:
                metrics = json.load(f)
            
            metrics['experiment'] = exp_name
            all_metrics.append(metrics)
        
        return pd.DataFrame(all_metrics)


class StatisticalAnalyzer:
    """
    统计分析器
    
    进行显著性检验和统计分析
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 显著性水平
        """
        self.alpha = alpha
    
    def paired_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> Dict:
        """
        配对t检验
        
        Args:
            group1: 组1数据
            group2: 组2数据
            group1_name: 组1名称
            group2_name: 组2名称
        
        Returns:
            检验结果
        """
        statistic, pvalue = stats.ttest_rel(group1, group2)
        
        result = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'group1_std': np.std(group1, ddof=1),
            'group2_std': np.std(group2, ddof=1),
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'effect_size': (np.mean(group1) - np.mean(group2)) / np.std(group1 - group2, ddof=1)
        }
        
        return result
    
    def independent_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> Dict:
        """
        独立样本t检验
        
        Args:
            group1: 组1数据
            group2: 组2数据
            group1_name: 组1名称
            group2_name: 组2名称
        
        Returns:
            检验结果
        """
        statistic, pvalue = stats.ttest_ind(group1, group2)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        result = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'group1_std': np.std(group1, ddof=1),
            'group2_std': np.std(group2, ddof=1),
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'cohens_d': cohens_d
        }
        
        return result
    
    def anova(
        self,
        groups: List[np.ndarray],
        group_names: List[str]
    ) -> Dict:
        """
        方差分析(ANOVA)
        
        Args:
            groups: 多组数据
            group_names: 组名称
        
        Returns:
            检验结果
        """
        statistic, pvalue = stats.f_oneway(*groups)
        
        result = {
            'group_names': group_names,
            'group_means': [np.mean(g) for g in groups],
            'group_stds': [np.std(g, ddof=1) for g in groups],
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha
        }
        
        return result
    
    def compare_ablation_groups(
        self,
        metrics_df: pd.DataFrame,
        metric_name: str,
        baseline_name: str
    ) -> List[Dict]:
        """
        对比消融实验各组与baseline
        
        Args:
            metrics_df: 指标DataFrame
            metric_name: 指标名称
            baseline_name: baseline实验名称
        
        Returns:
            对比结果列表
        """
        results = []
        
        baseline_data = metrics_df[metrics_df['experiment'] == baseline_name][metric_name].values
        
        for exp_name in metrics_df['experiment'].unique():
            if exp_name == baseline_name:
                continue
            
            exp_data = metrics_df[metrics_df['experiment'] == exp_name][metric_name].values
            
            if len(exp_data) == 0:
                continue
            
            # t检验
            test_result = self.independent_t_test(
                exp_data,
                baseline_data,
                group1_name=exp_name,
                group2_name=baseline_name
            )
            
            test_result['metric'] = metric_name
            results.append(test_result)
        
        return results


class ResultVisualizer:
    """
    结果可视化器
    
    生成各种对比图表
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PLOTTING:
            logger.warning("Plotting libraries not available")
    
    def plot_ablation_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric_name: str,
        title: str = None,
        ylabel: str = None
    ):
        """
        绘制消融实验对比图
        
        Args:
            metrics_df: 指标DataFrame
            metric_name: 指标名称
            title: 图表标题
            ylabel: y轴标签
        """
        if not HAS_PLOTTING:
            logger.warning("Plotting not available")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 按实验分组
        experiments = metrics_df['experiment'].unique()
        values = [metrics_df[metrics_df['experiment'] == exp][metric_name].values for exp in experiments]
        
        # 绘制箱线图
        bp = plt.boxplot(values, labels=experiments, patch_artist=True)
        
        # 美化
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel or metric_name)
        plt.title(title or f'Ablation Study: {metric_name}')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / f'ablation_{metric_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
    
    def plot_learning_curves(
        self,
        log_dir: str,
        experiment_names: List[str],
        metric_tag: str = 'reward/episode_reward'
    ):
        """
        绘制学习曲线对比
        
        Args:
            log_dir: 日志目录
            experiment_names: 实验名称列表
            metric_tag: 指标标签
        """
        if not HAS_PLOTTING or not HAS_TENSORBOARD:
            logger.warning("Plotting or TensorBoard not available")
            return
        
        parser = TensorBoardLogParser(log_dir)
        
        plt.figure(figsize=(12, 6))
        
        for exp_name in experiment_names:
            metrics = parser.parse_experiment(exp_name)
            
            if metric_tag in metrics:
                steps, values = zip(*metrics[metric_tag])
                
                # 平滑
                window = min(100, len(values) // 10)
                if window > 1:
                    values_smooth = pd.Series(values).rolling(window=window, center=True).mean()
                else:
                    values_smooth = values
                
                plt.plot(steps, values_smooth, label=exp_name, alpha=0.8)
        
        plt.xlabel('Training Steps')
        plt.ylabel(metric_tag)
        plt.title(f'Learning Curves: {metric_tag}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / f'learning_curves_{metric_tag.replace("/", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved learning curves: {output_path}")
    
    def plot_performance_heatmap(
        self,
        metrics_df: pd.DataFrame,
        metrics: List[str]
    ):
        """
        绘制性能热力图
        
        Args:
            metrics_df: 指标DataFrame
            metrics: 指标名称列表
        """
        if not HAS_PLOTTING:
            logger.warning("Plotting not available")
            return
        
        # 准备数据
        data = metrics_df[['experiment'] + metrics].set_index('experiment')
        
        # 标准化
        data_normalized = (data - data.min()) / (data.max() - data.min())
        
        plt.figure(figsize=(10, len(data_normalized) * 0.5))
        sns.heatmap(
            data_normalized,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Normalized Score'}
        )
        plt.title('Performance Heatmap (Normalized)')
        plt.tight_layout()
        
        output_path = self.output_dir / 'performance_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap: {output_path}")


class ResultAnalyzer:
    """
    完整的结果分析流程
    
    整合指标提取、统计分析、可视化
    """
    
    def __init__(
        self,
        log_dir: str,
        output_dir: str = 'experiments/analysis'
    ):
        """
        Args:
            log_dir: 日志目录
            output_dir: 输出目录
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = MetricsExtractor(str(self.output_dir))
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer(str(self.output_dir / 'plots'))
    
    def analyze_ablation_study(
        self,
        experiment_names: List[str],
        baseline_name: str,
        key_metrics: List[str] = None
    ):
        """
        分析消融实验
        
        Args:
            experiment_names: 实验名称列表
            baseline_name: baseline实验名称
            key_metrics: 关键指标列表
        """
        logger.info("=== Analyzing Ablation Study ===\n")
        
        # 1. 提取指标
        logger.info("Extracting metrics from TensorBoard logs...")
        metrics_df = self.extractor.extract_from_tb_logs(
            str(self.log_dir),
            experiment_names
        )
        
        if metrics_df.empty:
            logger.error("No metrics extracted")
            return
        
        # 保存原始数据
        metrics_path = self.output_dir / 'ablation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics: {metrics_path}")
        
        # 2. 统计分析
        if key_metrics is None:
            key_metrics = [
                'discovery_rate',
                'completion_time',
                'search_efficiency',
                'coordination_efficiency'
            ]
        
        logger.info("\nPerforming statistical tests...")
        all_test_results = []
        
        for metric in key_metrics:
            if metric not in metrics_df.columns:
                continue
            
            logger.info(f"\n--- Metric: {metric} ---")
            
            test_results = self.analyzer.compare_ablation_groups(
                metrics_df,
                metric,
                baseline_name
            )
            
            for result in test_results:
                logger.info(f"{result['group1_name']} vs {result['group2_name']}:")
                logger.info(f"  Mean: {result['group1_mean']:.4f} vs {result['group2_mean']:.4f}")
                logger.info(f"  p-value: {result['pvalue']:.4f}")
                logger.info(f"  Significant: {'✓' if result['significant'] else '✗'}")
                logger.info(f"  Cohen's d: {result['cohens_d']:.4f}")
            
            all_test_results.extend(test_results)
        
        # 保存统计结果
        test_results_df = pd.DataFrame(all_test_results)
        test_path = self.output_dir / 'statistical_tests.csv'
        test_results_df.to_csv(test_path, index=False)
        logger.info(f"\nSaved statistical tests: {test_path}")
        
        # 3. 可视化
        logger.info("\nGenerating visualizations...")
        
        for metric in key_metrics:
            if metric in metrics_df.columns:
                self.visualizer.plot_ablation_comparison(
                    metrics_df,
                    metric,
                    title=f'Ablation Study: {metric}',
                    ylabel=metric.replace('_', ' ').title()
                )
        
        # 学习曲线
        self.visualizer.plot_learning_curves(
            str(self.log_dir),
            experiment_names
        )
        
        # 性能热力图
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        if available_metrics:
            self.visualizer.plot_performance_heatmap(metrics_df, available_metrics)
        
        logger.info("\n=== Analysis Completed ===")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/ablation/logs',
        help='TensorBoard log directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='intrinsic_baseline',
        help='Baseline experiment name'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        help='Experiment names to analyze'
    )
    
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(
        log_dir=args.log_dir,
        output_dir=args.output_dir
    )
    
    if args.experiments:
        analyzer.analyze_ablation_study(
            experiment_names=args.experiments,
            baseline_name=args.baseline
        )
    else:
        logger.error("Please specify experiments to analyze with --experiments")


if __name__ == "__main__":
    main()
