"""
消融实验框架 - 系统性验证各组件贡献

功能:
1. 自动生成不同配置的实验组
2. 批量运行消融实验
3. 收集和分析实验结果

消融实验设计:
- 实验1: 内在奖励消融 (baseline, coverage, frontier, full)
- 实验2: 协同机制消融 (no_penalty, no_division, no_collab, full)
- 实验3: 通信条件分析 (full_comm, limited, sparse, no_comm)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationConfigGenerator:
    """
    消融实验配置生成器
    
    基于基础配置生成不同的实验配置
    """
    
    def __init__(self, base_config_path: str):
        """
        Args:
            base_config_path: 基础配置文件路径
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        self.experiment_configs = {}
    
    def generate_intrinsic_reward_ablation(self) -> Dict[str, dict]:
        """
        生成内在奖励消融实验配置
        
        Returns:
            实验名称 -> 配置字典
        """
        configs = {}
        
        # 1. Baseline: 无内在奖励
        baseline = copy.deepcopy(self.base_config)
        baseline['experiment']['intrinsic_rewards']['enable'] = False
        baseline['experiment']['title'] = 'ablation_intrinsic_baseline'
        configs['intrinsic_baseline'] = baseline
        
        # 2. Coverage-driven only
        coverage_only = copy.deepcopy(self.base_config)
        coverage_only['experiment']['intrinsic_rewards']['enable'] = True
        coverage_only['experiment']['intrinsic_rewards']['coverage_exploration_weight'] = 0.8
        coverage_only['experiment']['intrinsic_rewards']['frontier_reward_weight'] = 0.0
        coverage_only['experiment']['intrinsic_rewards']['curiosity_weight'] = 0.0
        coverage_only['experiment']['title'] = 'ablation_intrinsic_coverage'
        configs['intrinsic_coverage'] = coverage_only
        
        # 3. Frontier-driven only (我们的核心创新)
        frontier_only = copy.deepcopy(self.base_config)
        frontier_only['experiment']['intrinsic_rewards']['enable'] = True
        frontier_only['experiment']['intrinsic_rewards']['coverage_exploration_weight'] = 0.0
        frontier_only['experiment']['intrinsic_rewards']['frontier_reward_weight'] = 1.0
        frontier_only['experiment']['intrinsic_rewards']['curiosity_weight'] = 0.0
        frontier_only['experiment']['title'] = 'ablation_intrinsic_frontier'
        configs['intrinsic_frontier'] = frontier_only
        
        # 4. Curiosity-driven only
        curiosity_only = copy.deepcopy(self.base_config)
        curiosity_only['experiment']['intrinsic_rewards']['enable'] = True
        curiosity_only['experiment']['intrinsic_rewards']['coverage_exploration_weight'] = 0.0
        curiosity_only['experiment']['intrinsic_rewards']['frontier_reward_weight'] = 0.0
        curiosity_only['experiment']['intrinsic_rewards']['curiosity_weight'] = 0.5
        curiosity_only['experiment']['title'] = 'ablation_intrinsic_curiosity'
        configs['intrinsic_curiosity'] = curiosity_only
        
        # 5. Full: 所有内在奖励
        full = copy.deepcopy(self.base_config)
        full['experiment']['intrinsic_rewards']['enable'] = True
        full['experiment']['title'] = 'ablation_intrinsic_full'
        configs['intrinsic_full'] = full
        
        return configs
    
    def generate_coordination_ablation(self) -> Dict[str, dict]:
        """
        生成协同机制消融实验配置
        
        Returns:
            实验名称 -> 配置字典
        """
        configs = {}
        
        # 1. Baseline: 无协同机制
        baseline = copy.deepcopy(self.base_config)
        baseline['experiment']['coordination']['enable'] = False
        baseline['experiment']['title'] = 'ablation_coord_baseline'
        configs['coord_baseline'] = baseline
        
        # 2. Overlap penalty only
        overlap_only = copy.deepcopy(self.base_config)
        overlap_only['experiment']['coordination']['enable'] = True
        overlap_only['experiment']['coordination']['overlap_penalty_weight'] = 1.5
        overlap_only['experiment']['coordination']['division_reward_weight'] = 0.0
        overlap_only['experiment']['coordination']['joint_discovery_weight'] = 0.0
        overlap_only['experiment']['title'] = 'ablation_coord_overlap'
        configs['coord_overlap'] = overlap_only
        
        # 3. Division reward only
        division_only = copy.deepcopy(self.base_config)
        division_only['experiment']['coordination']['enable'] = True
        division_only['experiment']['coordination']['overlap_penalty_weight'] = 0.0
        division_only['experiment']['coordination']['division_reward_weight'] = 0.8
        division_only['experiment']['coordination']['joint_discovery_weight'] = 0.0
        division_only['experiment']['title'] = 'ablation_coord_division'
        configs['coord_division'] = division_only
        
        # 4. Collaboration reward only
        collab_only = copy.deepcopy(self.base_config)
        collab_only['experiment']['coordination']['enable'] = True
        collab_only['experiment']['coordination']['overlap_penalty_weight'] = 0.0
        collab_only['experiment']['coordination']['division_reward_weight'] = 0.0
        collab_only['experiment']['coordination']['joint_discovery_weight'] = 2.0
        collab_only['experiment']['title'] = 'ablation_coord_collab'
        configs['coord_collab'] = collab_only
        
        # 5. Full: 所有协同机制
        full = copy.deepcopy(self.base_config)
        full['experiment']['coordination']['enable'] = True
        full['experiment']['title'] = 'ablation_coord_full'
        configs['coord_full'] = full
        
        return configs
    
    def generate_communication_ablation(self) -> Dict[str, dict]:
        """
        生成通信条件消融实验配置
        
        Returns:
            实验名称 -> 配置字典
        """
        configs = {}
        
        comm_ranges = {
            'full_comm': 25.0,
            'limited_comm': 15.0,
            'sparse_comm': 10.0,
            'no_comm': 0.0
        }
        
        for name, comm_range in comm_ranges.items():
            config = copy.deepcopy(self.base_config)
            config['experiment']['uav']['communication_range'] = comm_range
            config['experiment']['title'] = f'ablation_comm_{name}'
            configs[name] = config
        
        return configs
    
    def generate_all_ablations(self) -> Dict[str, dict]:
        """生成所有消融实验配置"""
        all_configs = {}
        
        # 内在奖励消融
        all_configs.update(self.generate_intrinsic_reward_ablation())
        
        # 协同机制消融
        all_configs.update(self.generate_coordination_ablation())
        
        # 通信条件消融
        all_configs.update(self.generate_communication_ablation())
        
        return all_configs
    
    def save_configs(self, output_dir: str):
        """
        保存所有配置文件
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_configs = self.generate_all_ablations()
        
        for name, config in all_configs.items():
            output_path = os.path.join(output_dir, f'{name}.yaml')
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved config: {output_path}")
        
        return list(all_configs.keys())


class ExperimentScheduler:
    """
    实验调度器
    
    管理多个实验的运行、监控和结果收集
    """
    
    def __init__(
        self,
        config_dir: str,
        log_dir: str,
        num_parallel: int = 1
    ):
        """
        Args:
            config_dir: 配置文件目录
            log_dir: 日志输出目录
            num_parallel: 并行运行的实验数量
        """
        self.config_dir = Path(config_dir)
        self.log_dir = Path(log_dir)
        self.num_parallel = num_parallel
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.running_experiments = {}
        self.completed_experiments = {}
        self.failed_experiments = {}
    
    def run_experiment(
        self,
        config_name: str,
        python_script: str = 'marl_framework/main.py',
        timeout: int = None
    ) -> Tuple[bool, str]:
        """
        运行单个实验
        
        Args:
            config_name: 配置文件名(不含.yaml)
            python_script: Python训练脚本路径
            timeout: 超时时间(秒)
        
        Returns:
            success: 是否成功
            log_path: 日志文件路径
        """
        config_path = self.config_dir / f'{config_name}.yaml'
        log_path = self.log_dir / f'{config_name}.log'
        
        if not config_path.exists():
            logger.error(f"Config not found: {config_path}")
            return False, str(log_path)
        
        logger.info(f"Starting experiment: {config_name}")
        
        # 构建命令
        cmd = [
            'python',
            python_script,
            '--params', str(config_path)
        ]
        
        try:
            # 运行实验
            with open(log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                self.running_experiments[config_name] = process
                
                # 等待完成
                returncode = process.wait(timeout=timeout)
                
                if returncode == 0:
                    logger.info(f"✓ Experiment completed: {config_name}")
                    self.completed_experiments[config_name] = str(log_path)
                    return True, str(log_path)
                else:
                    logger.error(f"✗ Experiment failed: {config_name} (code={returncode})")
                    self.failed_experiments[config_name] = str(log_path)
                    return False, str(log_path)
        
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Experiment timeout: {config_name}")
            process.kill()
            self.failed_experiments[config_name] = str(log_path)
            return False, str(log_path)
        
        except Exception as e:
            logger.error(f"✗ Experiment error: {config_name}, {e}")
            self.failed_experiments[config_name] = str(log_path)
            return False, str(log_path)
        
        finally:
            if config_name in self.running_experiments:
                del self.running_experiments[config_name]
    
    def run_all_experiments(
        self,
        experiment_names: List[str],
        python_script: str = 'marl_framework/main.py'
    ):
        """
        顺序运行所有实验
        
        Args:
            experiment_names: 实验名称列表
            python_script: Python训练脚本
        """
        logger.info(f"Starting {len(experiment_names)} experiments...")
        
        start_time = time.time()
        
        for i, name in enumerate(experiment_names):
            logger.info(f"\n[{i+1}/{len(experiment_names)}] Running: {name}")
            success, log_path = self.run_experiment(name, python_script)
            
            if not success:
                logger.warning(f"Experiment failed: {name}, check log: {log_path}")
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n=== Experiment Summary ===")
        logger.info(f"Total time: {elapsed/3600:.2f} hours")
        logger.info(f"Completed: {len(self.completed_experiments)}")
        logger.info(f"Failed: {len(self.failed_experiments)}")
        
        if self.failed_experiments:
            logger.info("\nFailed experiments:")
            for name, log in self.failed_experiments.items():
                logger.info(f"  - {name}: {log}")
    
    def get_summary(self) -> Dict:
        """获取实验摘要"""
        return {
            'completed': list(self.completed_experiments.keys()),
            'failed': list(self.failed_experiments.keys()),
            'total': len(self.completed_experiments) + len(self.failed_experiments)
        }


class AblationStudy:
    """
    完整的消融实验流程
    
    整合配置生成、实验运行、结果分析
    """
    
    def __init__(
        self,
        base_config_path: str,
        output_dir: str = 'experiments/ablation'
    ):
        """
        Args:
            base_config_path: 基础配置文件
            output_dir: 实验输出目录
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        
        self.config_dir = self.output_dir / 'configs'
        self.log_dir = self.output_dir / 'logs'
        self.result_dir = self.output_dir / 'results'
        
        # 创建目录
        for d in [self.config_dir, self.log_dir, self.result_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.generator = AblationConfigGenerator(base_config_path)
        self.scheduler = ExperimentScheduler(
            str(self.config_dir),
            str(self.log_dir)
        )
    
    def setup_experiments(self) -> List[str]:
        """设置实验配置"""
        logger.info("Generating ablation configurations...")
        experiment_names = self.generator.save_configs(str(self.config_dir))
        logger.info(f"Generated {len(experiment_names)} experiment configs")
        return experiment_names
    
    def run_experiments(
        self,
        experiment_names: List[str] = None,
        python_script: str = 'marl_framework/main.py'
    ):
        """运行实验"""
        if experiment_names is None:
            # 运行所有实验
            experiment_names = [
                f.stem for f in self.config_dir.glob('*.yaml')
            ]
        
        logger.info(f"Running {len(experiment_names)} experiments...")
        self.scheduler.run_all_experiments(experiment_names, python_script)
        
        # 保存摘要
        summary = self.scheduler.get_summary()
        summary_path = self.result_dir / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_path}")
    
    def run_full_study(self, python_script: str = 'marl_framework/main.py'):
        """运行完整的消融实验"""
        logger.info("=== Starting Full Ablation Study ===\n")
        
        # 1. 生成配置
        experiment_names = self.setup_experiments()
        
        # 2. 运行实验
        self.run_experiments(experiment_names, python_script)
        
        logger.info("\n=== Ablation Study Completed ===")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """主函数 - 运行消融实验"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument(
        '--base_config',
        type=str,
        default='marl_framework/configs/params_advanced_search.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/ablation',
        help='Output directory for experiments'
    )
    parser.add_argument(
        '--setup_only',
        action='store_true',
        help='Only generate configs without running'
    )
    parser.add_argument(
        '--run_experiments',
        type=str,
        nargs='+',
        help='Run specific experiments by name'
    )
    
    args = parser.parse_args()
    
    # 创建消融实验
    study = AblationStudy(
        base_config_path=args.base_config,
        output_dir=args.output_dir
    )
    
    if args.setup_only:
        # 只生成配置
        experiment_names = study.setup_experiments()
        print(f"\n✓ Generated {len(experiment_names)} configurations")
        print(f"  Config dir: {study.config_dir}")
        print("\nExperiment groups:")
        print("  - Intrinsic rewards ablation: 5 configs")
        print("  - Coordination ablation: 5 configs")
        print("  - Communication ablation: 4 configs")
        print(f"\nTotal: {len(experiment_names)} experiments")
        
    elif args.run_experiments:
        # 运行指定实验
        study.run_experiments(args.run_experiments)
        
    else:
        # 运行完整消融实验
        study.run_full_study()


if __name__ == "__main__":
    main()
