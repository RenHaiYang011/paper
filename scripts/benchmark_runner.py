"""
基准测试运行器 - 多场景评估和对比

功能:
1. 在不同场景下评估模型性能
2. 与baseline算法对比(Random, Lawn-mower, IG)
3. 并行运行多个实验
4. 自动生成对比报告

场景设计:
- 小规模: 4 agents, 10 targets, 200 budget
- 中规模: 6 agents, 15 targets, 300 budget
- 大规模: 8 agents, 20 targets, 400 budget
- 高密度: 标准规模但目标密度高
- 稀疏场景: 标准规模但目标稀疏
"""

import os
import yaml
import copy
import json
import subprocess
import time
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    场景配置生成器
    
    生成不同规模和难度的测试场景
    """
    
    def __init__(self, base_config_path: str):
        """
        Args:
            base_config_path: 基础配置文件路径
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_scale_scenarios(self) -> Dict[str, dict]:
        """
        生成不同规模场景
        
        Returns:
            场景名称 -> 配置字典
        """
        scenarios = {}
        
        scale_configs = {
            'small': {
                'num_agents': 4,
                'num_targets': 10,
                'mission_budget': 200,
                'map_size': 50
            },
            'medium': {
                'num_agents': 6,
                'num_targets': 15,
                'mission_budget': 300,
                'map_size': 60
            },
            'large': {
                'num_agents': 8,
                'num_targets': 20,
                'mission_budget': 400,
                'map_size': 70
            },
            'xlarge': {
                'num_agents': 10,
                'num_targets': 25,
                'mission_budget': 500,
                'map_size': 80
            }
        }
        
        for name, params in scale_configs.items():
            config = copy.deepcopy(self.base_config)
            config['experiment']['num_agents'] = params['num_agents']
            config['experiment']['num_targets'] = params['num_targets']
            config['experiment']['mission_budget'] = params['mission_budget']
            config['experiment']['map']['size'] = params['map_size']
            config['experiment']['title'] = f'benchmark_scale_{name}'
            scenarios[f'scale_{name}'] = config
        
        return scenarios
    
    def generate_density_scenarios(self) -> Dict[str, dict]:
        """
        生成不同密度场景
        
        Returns:
            场景名称 -> 配置字典
        """
        scenarios = {}
        
        density_configs = {
            'sparse': {
                'num_targets': 8,
                'obstacle_density': 0.05
            },
            'normal': {
                'num_targets': 15,
                'obstacle_density': 0.1
            },
            'dense': {
                'num_targets': 25,
                'obstacle_density': 0.15
            },
            'very_dense': {
                'num_targets': 35,
                'obstacle_density': 0.2
            }
        }
        
        for name, params in density_configs.items():
            config = copy.deepcopy(self.base_config)
            config['experiment']['num_targets'] = params['num_targets']
            config['experiment']['map']['obstacle_density'] = params['obstacle_density']
            config['experiment']['title'] = f'benchmark_density_{name}'
            scenarios[f'density_{name}'] = config
        
        return scenarios
    
    def generate_complexity_scenarios(self) -> Dict[str, dict]:
        """
        生成不同复杂度场景
        
        Returns:
            场景名称 -> 配置字典
        """
        scenarios = {}
        
        # 1. 简单场景: 开阔地形
        simple = copy.deepcopy(self.base_config)
        simple['experiment']['map']['obstacle_density'] = 0.05
        simple['experiment']['map']['use_complex_obstacles'] = False
        simple['experiment']['title'] = 'benchmark_complexity_simple'
        scenarios['complexity_simple'] = simple
        
        # 2. 中等场景: 一般障碍
        moderate = copy.deepcopy(self.base_config)
        moderate['experiment']['map']['obstacle_density'] = 0.1
        moderate['experiment']['map']['use_complex_obstacles'] = True
        moderate['experiment']['title'] = 'benchmark_complexity_moderate'
        scenarios['complexity_moderate'] = moderate
        
        # 3. 复杂场景: 密集障碍
        complex = copy.deepcopy(self.base_config)
        complex['experiment']['map']['obstacle_density'] = 0.15
        complex['experiment']['map']['use_complex_obstacles'] = True
        complex['experiment']['title'] = 'benchmark_complexity_complex'
        scenarios['complexity_complex'] = complex
        
        # 4. 极端场景: 非常密集
        extreme = copy.deepcopy(self.base_config)
        extreme['experiment']['map']['obstacle_density'] = 0.25
        extreme['experiment']['map']['use_complex_obstacles'] = True
        extreme['experiment']['title'] = 'benchmark_complexity_extreme'
        scenarios['complexity_extreme'] = extreme
        
        return scenarios
    
    def generate_all_scenarios(self) -> Dict[str, dict]:
        """生成所有场景配置"""
        all_scenarios = {}
        
        all_scenarios.update(self.generate_scale_scenarios())
        all_scenarios.update(self.generate_density_scenarios())
        all_scenarios.update(self.generate_complexity_scenarios())
        
        return all_scenarios
    
    def save_scenarios(self, output_dir: str) -> List[str]:
        """
        保存所有场景配置
        
        Args:
            output_dir: 输出目录
        
        Returns:
            场景名称列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_scenarios = self.generate_all_scenarios()
        
        for name, config in all_scenarios.items():
            output_path = os.path.join(output_dir, f'{name}.yaml')
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved scenario: {output_path}")
        
        return list(all_scenarios.keys())


class BaselineComparator:
    """
    基线算法对比器
    
    运行baseline算法并收集结果用于对比
    """
    
    def __init__(self, scenario_dir: str, output_dir: str):
        """
        Args:
            scenario_dir: 场景配置目录
            output_dir: 输出目录
        """
        self.scenario_dir = Path(scenario_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_results = {}
    
    def run_random_baseline(self, scenario_name: str) -> Dict:
        """
        运行随机基线
        
        Args:
            scenario_name: 场景名称
        
        Returns:
            评估结果
        """
        logger.info(f"Running Random baseline: {scenario_name}")
        
        config_path = self.scenario_dir / f'{scenario_name}.yaml'
        
        cmd = [
            'python',
            'marl_framework/random_baseline.py',
            '--params', str(config_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                # 解析输出获取metrics
                # 这里简化处理,实际需要解析stdout
                return {'success': True, 'method': 'Random'}
            else:
                logger.error(f"Random baseline failed: {scenario_name}")
                return {'success': False, 'method': 'Random'}
        
        except Exception as e:
            logger.error(f"Error running Random baseline: {e}")
            return {'success': False, 'method': 'Random'}
    
    def run_lawnmower_baseline(self, scenario_name: str) -> Dict:
        """
        运行Lawn-mower基线
        
        Args:
            scenario_name: 场景名称
        
        Returns:
            评估结果
        """
        logger.info(f"Running Lawn-mower baseline: {scenario_name}")
        
        config_path = self.scenario_dir / f'{scenario_name}.yaml'
        
        cmd = [
            'python',
            'marl_framework/lawn_mower.py',
            '--params', str(config_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                return {'success': True, 'method': 'Lawn-mower'}
            else:
                logger.error(f"Lawn-mower baseline failed: {scenario_name}")
                return {'success': False, 'method': 'Lawn-mower'}
        
        except Exception as e:
            logger.error(f"Error running Lawn-mower baseline: {e}")
            return {'success': False, 'method': 'Lawn-mower'}
    
    def run_ig_baseline(self, scenario_name: str) -> Dict:
        """
        运行Information Gain基线
        
        Args:
            scenario_name: 场景名称
        
        Returns:
            评估结果
        """
        logger.info(f"Running IG baseline: {scenario_name}")
        
        config_path = self.scenario_dir / f'{scenario_name}.yaml'
        
        cmd = [
            'python',
            'marl_framework/IG_baseline.py',
            '--params', str(config_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                return {'success': True, 'method': 'IG'}
            else:
                logger.error(f"IG baseline failed: {scenario_name}")
                return {'success': False, 'method': 'IG'}
        
        except Exception as e:
            logger.error(f"Error running IG baseline: {e}")
            return {'success': False, 'method': 'IG'}
    
    def run_all_baselines(self, scenario_names: List[str]):
        """
        运行所有baseline算法
        
        Args:
            scenario_names: 场景名称列表
        """
        for scenario in scenario_names:
            logger.info(f"\n=== Running baselines for: {scenario} ===")
            
            results = {}
            
            # Random
            results['random'] = self.run_random_baseline(scenario)
            
            # Lawn-mower
            results['lawnmower'] = self.run_lawnmower_baseline(scenario)
            
            # Information Gain
            results['ig'] = self.run_ig_baseline(scenario)
            
            self.baseline_results[scenario] = results
            
            # 保存结果
            result_path = self.output_dir / f'{scenario}_baselines.json'
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        logger.info("\n=== Baseline comparison completed ===")


class ParallelBenchmarkRunner:
    """
    并行基准测试运行器
    
    支持多进程并行运行测试
    """
    
    def __init__(
        self,
        scenario_dir: str,
        log_dir: str,
        max_workers: int = 2
    ):
        """
        Args:
            scenario_dir: 场景配置目录
            log_dir: 日志目录
            max_workers: 最大并行worker数
        """
        self.scenario_dir = Path(scenario_dir)
        self.log_dir = Path(log_dir)
        self.max_workers = max_workers
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def run_single_benchmark(
        self,
        scenario_name: str,
        python_script: str = 'marl_framework/main.py'
    ) -> Tuple[str, bool, str]:
        """
        运行单个基准测试
        
        Args:
            scenario_name: 场景名称
            python_script: 训练脚本
        
        Returns:
            (scenario_name, success, log_path)
        """
        config_path = self.scenario_dir / f'{scenario_name}.yaml'
        log_path = self.log_dir / f'{scenario_name}.log'
        
        logger.info(f"[Worker] Running benchmark: {scenario_name}")
        
        cmd = [
            'python',
            python_script,
            '--params', str(config_path)
        ]
        
        try:
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1小时超时
                )
                
                success = result.returncode == 0
                
                if success:
                    logger.info(f"✓ [Worker] Completed: {scenario_name}")
                else:
                    logger.error(f"✗ [Worker] Failed: {scenario_name}")
                
                return scenario_name, success, str(log_path)
        
        except subprocess.TimeoutExpired:
            logger.error(f"✗ [Worker] Timeout: {scenario_name}")
            return scenario_name, False, str(log_path)
        
        except Exception as e:
            logger.error(f"✗ [Worker] Error: {scenario_name}, {e}")
            return scenario_name, False, str(log_path)
    
    def run_parallel_benchmarks(
        self,
        scenario_names: List[str],
        python_script: str = 'marl_framework/main.py'
    ):
        """
        并行运行多个基准测试
        
        Args:
            scenario_names: 场景名称列表
            python_script: 训练脚本
        """
        logger.info(f"Starting {len(scenario_names)} benchmarks with {self.max_workers} workers...")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.run_single_benchmark, name, python_script): name
                for name in scenario_names
            }
            
            # 收集结果
            for future in as_completed(futures):
                scenario_name, success, log_path = future.result()
                self.results[scenario_name] = {
                    'success': success,
                    'log_path': log_path
                }
        
        elapsed = time.time() - start_time
        
        # 统计
        successful = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - successful
        
        logger.info(f"\n=== Benchmark Summary ===")
        logger.info(f"Total time: {elapsed/3600:.2f} hours")
        logger.info(f"Successful: {successful}/{len(scenario_names)}")
        logger.info(f"Failed: {failed}/{len(scenario_names)}")
        
        if failed > 0:
            logger.info("\nFailed benchmarks:")
            for name, result in self.results.items():
                if not result['success']:
                    logger.info(f"  - {name}: {result['log_path']}")


class BenchmarkStudy:
    """
    完整的基准测试流程
    
    整合场景生成、baseline对比、并行运行
    """
    
    def __init__(
        self,
        base_config_path: str,
        output_dir: str = 'experiments/benchmark',
        max_workers: int = 2
    ):
        """
        Args:
            base_config_path: 基础配置文件
            output_dir: 输出目录
            max_workers: 并行worker数
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        self.scenario_dir = self.output_dir / 'scenarios'
        self.log_dir = self.output_dir / 'logs'
        self.baseline_dir = self.output_dir / 'baselines'
        
        for d in [self.scenario_dir, self.log_dir, self.baseline_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.generator = ScenarioGenerator(base_config_path)
        self.comparator = BaselineComparator(
            str(self.scenario_dir),
            str(self.baseline_dir)
        )
        self.runner = ParallelBenchmarkRunner(
            str(self.scenario_dir),
            str(self.log_dir),
            max_workers=max_workers
        )
    
    def setup_scenarios(self) -> List[str]:
        """设置场景"""
        logger.info("Generating benchmark scenarios...")
        scenario_names = self.generator.save_scenarios(str(self.scenario_dir))
        logger.info(f"Generated {len(scenario_names)} scenarios")
        return scenario_names
    
    def run_benchmarks(
        self,
        scenario_names: List[str],
        python_script: str = 'marl_framework/main.py'
    ):
        """运行基准测试"""
        logger.info(f"Running benchmarks on {len(scenario_names)} scenarios...")
        self.runner.run_parallel_benchmarks(scenario_names, python_script)
    
    def run_baselines(self, scenario_names: List[str]):
        """运行baseline对比"""
        logger.info("Running baseline comparisons...")
        self.comparator.run_all_baselines(scenario_names)
    
    def run_full_study(
        self,
        python_script: str = 'marl_framework/main.py',
        include_baselines: bool = True
    ):
        """运行完整基准测试"""
        logger.info("=== Starting Full Benchmark Study ===\n")
        
        # 1. 生成场景
        scenario_names = self.setup_scenarios()
        
        # 2. 运行我们的模型
        self.run_benchmarks(scenario_names, python_script)
        
        # 3. 运行baseline对比(可选)
        if include_baselines:
            self.run_baselines(scenario_names)
        
        logger.info("\n=== Benchmark Study Completed ===")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmark study')
    parser.add_argument(
        '--base_config',
        type=str,
        default='marl_framework/configs/params_advanced_search.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/benchmark',
        help='Output directory'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=2,
        help='Maximum parallel workers'
    )
    parser.add_argument(
        '--setup_only',
        action='store_true',
        help='Only generate scenarios'
    )
    parser.add_argument(
        '--no_baselines',
        action='store_true',
        help='Skip baseline comparisons'
    )
    
    args = parser.parse_args()
    
    study = BenchmarkStudy(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    if args.setup_only:
        scenario_names = study.setup_scenarios()
        print(f"\n✓ Generated {len(scenario_names)} scenarios")
        print(f"  Scenario dir: {study.scenario_dir}")
        print("\nScenario groups:")
        print("  - Scale scenarios: 4 configs (small/medium/large/xlarge)")
        print("  - Density scenarios: 4 configs (sparse/normal/dense/very_dense)")
        print("  - Complexity scenarios: 4 configs (simple/moderate/complex/extreme)")
        print(f"\nTotal: {len(scenario_names)} scenarios")
    else:
        study.run_full_study(
            include_baselines=not args.no_baselines
        )


if __name__ == "__main__":
    main()
