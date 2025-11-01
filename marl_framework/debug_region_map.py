"""
调试 region_based 地图生成，验证坐标转换是否正确
"""
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加 marl_framework 到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapping.grid_maps import GridMap
from mapping import ground_truths_region_based

# 加载配置
config_path = Path("configs/params_fast.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

print("="*70)
print("调试 region_based 地图生成")
print("="*70)
 
# 创建 GridMap
grid_map = GridMap(params)

print(f"网格尺寸: {grid_map.y_dim} × {grid_map.x_dim} pixels")
print(f"分辨率: x={grid_map.resolution_x:.4f}, y={grid_map.resolution_y:.4f} 米/像素")

# 生成地图（使用训练时相同的episode）
# Step 150 对应 episode ≈ 10800/8 ≈ 1350
training_episode = 1350  # 对应 Step 150 的大概 episode

print(f"生成 episode {training_episode} 的地图（对应训练 Step 150）...")

simulated_map = ground_truths_region_based.generate_region_based_map(
    params,
    grid_map.y_dim,
    grid_map.x_dim,
    episode=training_episode  # 使用训练时的episode
)

print(f"\n生成的地图形状: {simulated_map.shape}")
print(f"总体目标密度: {simulated_map.mean() * 100:.1f}%")

# 手动验证每个区域
print("\n验证区域生成:")
for i, region in enumerate(params['search_regions']['regions'], 1):
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    expected_prob = region.get('target_probability', 0.5)
    
    # 转换为像素坐标 (使用相同的逻辑)
    x_min_px = int(x_min / grid_map.resolution_x)
    x_max_px = int(x_max / grid_map.resolution_x)
    y_min_px = int(y_min / grid_map.resolution_y)
    y_max_px = int(y_max / grid_map.resolution_y)
    
    # 提取该区域
    region_map = simulated_map[y_min_px:y_max_px, x_min_px:x_max_px]
    actual_density = region_map.mean()
    
    print(f"\n{i}. {region['name']}:")
    print(f"   世界坐标: [{x_min}, {y_min}] → [{x_max}, {y_max}] 米")
    print(f"   像素坐标: [{x_min_px}, {y_min_px}] → [{x_max_px}, {y_max_px}]")
    print(f"   期望密度: {expected_prob*100:.1f}%")
    print(f"   实际密度: {actual_density*100:.1f}%")
    print(f"   区域大小: {region_map.shape}")
    
    # 检查区域外是否有目标（应该为0）
    before_x = simulated_map[y_min_px:y_max_px, max(0, x_min_px-10):x_min_px].mean() if x_min_px > 10 else 0
    after_x = simulated_map[y_min_px:y_max_px, x_max_px:min(simulated_map.shape[1], x_max_px+10)].mean() if x_max_px < simulated_map.shape[1]-10 else 0
    print(f"   左侧10像素密度: {before_x*100:.1f}% (应该接近0%)")
    print(f"   右侧10像素密度: {after_x*100:.1f}% (应该接近0%)")

# 可视化验证
print("\n生成验证图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：完整地图
im1 = ax1.imshow(simulated_map, cmap='coolwarm', origin='lower', 
                 extent=[0, 50, 0, 50], vmin=0, vmax=1)
ax1.set_title(f'Episode {training_episode} Map (对应 Step 150)', fontsize=14, fontweight='bold')
ax1.set_xlabel('X (meters)')
ax1.set_ylabel('Y (meters)')

# 绘制搜索区域边界
colors = ['lime', 'cyan', 'magenta']
for i, region in enumerate(params['search_regions']['regions']):
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor=colors[i], linewidth=3, linestyle='--')
    ax1.add_patch(rect)
    
    # 添加标签
    label = f"{region['name']}\n期望:{region.get('target_probability', 0.5)*100:.0f}%"
    ax1.text(x_min + 1, y_max - 2, label, 
            color=colors[i], fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.colorbar(im1, ax=ax1, label='Target Presence')

# 右图：像素级验证
ax2.imshow(simulated_map, cmap='coolwarm', origin='lower', vmin=0, vmax=1)
ax2.set_title('Pixel-Level Verification', fontsize=14, fontweight='bold')
ax2.set_xlabel('X pixels')
ax2.set_ylabel('Y pixels')

# 在像素坐标系中绘制区域
for i, region in enumerate(params['search_regions']['regions']):
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    
    x_min_px = int(x_min / grid_map.resolution_x)
    x_max_px = int(x_max / grid_map.resolution_x)
    y_min_px = int(y_min / grid_map.resolution_y)
    y_max_px = int(y_max / grid_map.resolution_y)
    
    rect = plt.Rectangle((x_min_px, y_min_px), x_max_px - x_min_px, y_max_px - y_min_px,
                         fill=False, edgecolor=colors[i], linewidth=2, linestyle='-')
    ax2.add_patch(rect)

plt.colorbar(im1, ax=ax2, label='Target Presence')

plt.tight_layout()

# 保存
save_path = Path("test_plots/region_debug.png")
save_path.parent.mkdir(exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 调试图已保存到: {save_path}")

# 最终检查
print("\n" + "="*70)
print("诊断结果:")
print("="*70)

# 检查是否有区域重叠
non_zero_total = np.sum(simulated_map > 0)
expected_total = 0
for region in params['search_regions']['regions']:
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    x_min_px = int(x_min / grid_map.resolution_x)
    x_max_px = int(x_max / grid_map.resolution_x)
    y_min_px = int(y_min / grid_map.resolution_y)
    y_max_px = int(y_max / grid_map.resolution_y)
    
    region_map = simulated_map[y_min_px:y_max_px, x_min_px:x_max_px]
    expected_total += np.sum(region_map > 0)

print(f"地图中非零像素总数: {non_zero_total}")
print(f"区域内非零像素总数: {expected_total}")

if non_zero_total == expected_total:
    print("✅ 目标分布正确：所有目标都在配置的搜索区域内")
else:
    print(f"❌ 目标分布异常：有 {non_zero_total - expected_total} 个目标在区域外")

# 检查区域是否有目标
for i, region in enumerate(params['search_regions']['regions'], 1):
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    x_min_px = int(x_min / grid_map.resolution_x)
    x_max_px = int(x_max / grid_map.resolution_x)
    y_min_px = int(y_min / grid_map.resolution_y)
    y_max_px = int(y_max / grid_map.resolution_y)
    
    region_map = simulated_map[y_min_px:y_max_px, x_min_px:x_max_px]
    has_targets = np.any(region_map > 0)
    
    if has_targets:
        print(f"✅ {region['name']}: 有目标分布")
    else:
        print(f"❌ {region['name']}: 没有目标！")

print("\n如果区域内有目标但可视化错位，可能是绘图坐标系问题。")
print("请对比此调试图与训练生成的轨迹图。")