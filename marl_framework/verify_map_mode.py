"""
验证当前使用的地图生成模式
"""
import sys
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加 marl_framework 到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapping.grid_maps import GridMap
from mapping import ground_truths_region_based
from mapping import ground_truths

# 加载配置
config_path = Path("configs/params_fast.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

print("="*70)
print("验证地图生成模式")
print("="*70)

# 检查配置
map_type = params.get("sensor", {}).get("simulation", {}).get("map_type", "random_field")
print(f"\n配置文件中的 map_type: '{map_type}'")

# 创建 GridMap
grid_map = GridMap(params)
print(f"\n网格地图尺寸: {grid_map.y_dim} × {grid_map.x_dim} pixels")

# 直接调用地图生成函数（不需要创建 Simulation）
print(f"\n生成 episode 0 的 simulated_map...")

if map_type == "region_based":
    simulated_map = ground_truths_region_based.generate_region_based_map(
        params,
        grid_map.y_dim,
        grid_map.x_dim,
        episode=0
    )
else:
    cluster_radius = params['sensor']['simulation']['cluster_radius']
    simulated_map = ground_truths.gaussian_random_field(
        lambda k: k ** (-cluster_radius),
        grid_map.y_dim,
        grid_map.x_dim,
        episode=0
    )

print(f"生成的地图形状: {simulated_map.shape}")
print(f"总体目标密度: {simulated_map.mean() * 100:.1f}%")

# 分析各个搜索区域
print("\n各搜索区域的目标密度:")
for region in params['search_regions']['regions']:
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    
    # 转换为像素坐标
    x_min_px = int(x_min / grid_map.resolution_x)
    x_max_px = int(x_max / grid_map.resolution_x)
    y_min_px = int(y_min / grid_map.resolution_y)
    y_max_px = int(y_max / grid_map.resolution_y)
    
    # 提取该区域
    region_map = simulated_map[y_min_px:y_max_px, x_min_px:x_max_px]
    actual_density = region_map.mean() * 100
    expected = region.get('target_probability', 0.5) * 100
    
    match_status = "✅ 匹配" if abs(actual_density - expected) < 15 else "❌ 不匹配"
    
    print(f"\n  {region['name']}:")
    print(f"    像素范围: [{x_min_px}, {y_min_px}] 到 [{x_max_px}, {y_max_px}]")
    print(f"    期望密度: {expected:.1f}%")
    print(f"    实际密度: {actual_density:.1f}%")
    print(f"    状态: {match_status}")

# 可视化
print("\n生成可视化图...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

im = ax.imshow(simulated_map, cmap='coolwarm', origin='lower', 
               extent=[0, 50, 0, 50], vmin=0, vmax=1)
ax.set_title(f'当前使用的地图生成模式: {map_type}', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')

# 绘制搜索区域边界
for region in params['search_regions']['regions']:
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor='lime', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    
    # 添加标签
    label = f"{region['name']}\n期望:{region.get('target_probability', 0.5)*100:.0f}%"
    ax.text(x_min + 1, y_max - 2, label, 
            color='lime', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.colorbar(im, ax=ax, label='Target Presence')

# 保存
save_path = Path("test_plots/current_map_mode.png")
save_path.parent.mkdir(exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存到: {save_path}")

# 判断结论
print("\n" + "="*70)
print("结论:")
print("="*70)

if map_type == "region_based":
    # 检查是否真的按区域生成了
    all_match = True
    for region in params['search_regions']['regions']:
        coords = region['coordinates'][0]
        x_min, y_min, x_max, y_max = coords
        x_min_px = int(x_min / grid_map.resolution_x)
        x_max_px = int(x_max / grid_map.resolution_x)
        y_min_px = int(y_min / grid_map.resolution_y)
        y_max_px = int(y_max / grid_map.resolution_y)
        region_map = simulated_map[y_min_px:y_max_px, x_min_px:x_max_px]
        actual_density = region_map.mean() * 100
        expected = region.get('target_probability', 0.5) * 100
        if abs(actual_density - expected) > 15:
            all_match = False
            break
    
    if all_match:
        print("✅ 正在使用 region_based 模式，目标分布与配置匹配！")
        print("   红蓝色区域应该与绿色虚线框对齐。")
    else:
        print("⚠️ 配置显示 region_based，但实际分布不匹配！")
        print("   可能是缓存问题或旧的模型/图片。")
else:
    print("❌ 当前使用 random_field 模式")
    print("   目标分布是随机的，与搜索区域配置无关。")
    print("   红蓝色区域会与绿色虚线框错位。")

print("\n如果你看到的图片与上面保存的图片不同，")
print("说明你看到的是旧的训练结果。需要重新训练生成新图片。")
