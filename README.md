## Requirements

matplotlib==3.5.1
   
numpy==1.22.2
   
opencv-python==4.5.5.62
   
scipy==1.8.1
   
torch==1.13.0+cu117
   


# 启动训练
## 快速测试
cd /home/renhaiyang/paper_v2/ipp-marl && LD_PRELOAD=$(find /home/renhaiyang/anaconda3/envs/paper_ipp_marl/ -name "libstdc++.so.6") python -m marl_framework.main 
cd /home/renhaiyang/paper_v2/ipp-marl && LD_PRELOAD=$(find /home/renhaiyang/anaconda3/envs/paper_ipp_marl/ -name "libstdc++.so.6") python -m marl_framework.main --config marl_framework/params_test.yaml

# 查看结果
tensorboard --logdir /home/renhaiyang/paper_v2/ipp-marl/marl_framework/log --port 6006 --host 127.0.0.1


## 保存路径视频
cd ipp-marl/marl_framework python /home/renhaiyang/paper_v2/ipp-marl/marl_framework/utils/video_creator.py


## 自动化实验（批量运行）

项目中提供了简易的实验脚本，用于网格搜索 `coverage_weight`, `distance_weight`, `n_agents` 并记录可复现的运行元数据。

- 生成并运行实验（在仓库根目录下运行）：

```bash
python scripts/experiment_runner.py --coverage 0.0 0.05 --distance 0.0 0.01 --agents 2 4 --repeats 3
```

- 结果与元数据：
	- 每次运行会在 `runs/` 下创建 `run_<id>/params.yaml`（该 config）和 `runs.csv`（包含 run_id、seed、参数、logdir）。
	- TensorBoard 日志写入到 `marl_framework/log/run_<id>`（由 runner 创建）。

- 收集 TensorBoard scalars 到 CSV：

```bash
python scripts/collect_tb_scalars.py --runs_csv runs/runs.csv --out runs/scalars.csv
```

- 绘制 scalar 曲线：

```bash
python scripts/plot_results.py --scalars_csv runs/scalars.csv --out_dir runs/plots
```


信息驱动（Information-driven）策略会估计每个可能行动的预期熵减少（或直接测算执行后熵减少），选择最大化熵减少的动作 → 导致 UAV 去“最不确定/信息量最大的区域”采样。
带权熵（weightings）可以把不同类别或目标的权重放大/缩小（例如更关注特定目标类别的熵）。


覆盖率（coverage）奖励：鼓励观察到新区域（把“见过的格子数”也作为 reward），会促使更多广域覆盖而不是在同一区域精细采样。
距离/能耗惩罚：会抑制走过远距离去采样的冲动，促成更节能或更短路径的行为。
足迹重叠/碰撞惩罚：用于减少重复观测与避免碰撞，从而提升多机协作效率。
总体行为是这些项的权重平衡：熵减少高但距离大可能被 distance_weight 抑制；覆盖奖励会鼓励遍历尚未观测区域

熵驱动 → “跳跃式”前往高不确定区（信息密集采样）；覆盖驱动 → 更均匀分布的扫掠；两者混合 → 在保持覆盖的同时优先采样高不确定区。

当地图整体熵很低（已知足够多），信息驱动信号弱，agent 可能无明显偏好；
观测噪声大时，单次观测带来的熵减少小，IG 信号变弱 → 需要提高 coverage 权重或改变传感器配置；
计算时对 p 做了数值裁剪（避免 log(0)），通常不必修改，除非你有特殊数值需求。c


