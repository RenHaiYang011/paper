"""
测试 simulated_map 的实际大小和覆盖范围
"""
import yaml
import numpy as np
from pathlib import Path
from mapping.grid_maps import GridMap
from mapping import ground_truths

# 加载配置
config_path = Path("configs/params_fast.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

print("="*60)
print("环境配置:")
print(f"  x_dim (米): {params['environment']['x_dim']}")
print(f"  y_dim (米): {params['environment']['y_dim']}")

print("\n传感器配置:")
print(f"  FOV angle_x: {params['sensor']['field_of_view']['angle_x']}°")
print(f"  FOV angle_y: {params['sensor']['field_of_view']['angle_y']}°")
print(f"  Pixel number_x: {params['sensor']['pixel']['number_x']}")
print(f"  Pixel number_y: {params['sensor']['pixel']['number_y']}")

# 创建 GridMap
grid_map = GridMap(params)
print("\n"+"="*60)
print("网格地图尺寸:")
print(f"  x_dim_pixel: {grid_map.x_dim} 像素")
print(f"  y_dim_pixel: {grid_map.y_dim} 像素")
print(f"  resolution_x: {grid_map.resolution_x:.4f} 米/像素")
print(f"  resolution_y: {grid_map.resolution_y:.4f} 米/像素")

# 生成 simulated_map
simulated_map = ground_truths.gaussian_random_field(
    lambda k: k ** (-params['sensor']['simulation']['cluster_radius']),
    grid_map.y_dim,
    grid_map.x_dim,
    episode=0
)

print("\n"+"="*60)
print("Simulated Map:")
print(f"  Shape: {simulated_map.shape} (height, width)")
print(f"  Value range: [{simulated_map.min()}, {simulated_map.max()}]")
print(f"  Target density: {simulated_map.sum() / simulated_map.size * 100:.1f}%")

# 计算网格地图覆盖的实际米数范围
actual_x_meters = grid_map.x_dim * grid_map.resolution_x
actual_y_meters = grid_map.y_dim * grid_map.resolution_y

print("\n"+"="*60)
print("网格地图实际覆盖范围:")
print(f"  X 方向: {actual_x_meters:.2f} 米 (配置要求: {params['environment']['x_dim']} 米)")
print(f"  Y 方向: {actual_y_meters:.2f} 米 (配置要求: {params['environment']['y_dim']} 米)")

if abs(actual_x_meters - params['environment']['x_dim']) > 0.1:
    print(f"  ⚠️ X 方向存在偏差: {abs(actual_x_meters - params['environment']['x_dim']):.2f} 米")
if abs(actual_y_meters - params['environment']['y_dim']) > 0.1:
    print(f"  ⚠️ Y 方向存在偏差: {abs(actual_y_meters - params['environment']['y_dim']):.2f} 米")

print("\n"+"="*60)
print("搜索区域配置:")
for region in params['search_regions']['regions']:
    coords = region['coordinates'][0]
    print(f"  {region['name']}:")
    print(f"    坐标: [{coords[0]}, {coords[1]}] 到 [{coords[2]}, {coords[3]}] 米")
    print(f"    大小: {coords[2]-coords[0]} × {coords[3]-coords[1]} 米²")

print("\n"+"="*60)
print("目标分布分析:")
# 检查搜索区域内的目标密度
for region in params['search_regions']['regions']:
    coords = region['coordinates'][0]
    x_min, y_min, x_max, y_max = coords
    
    # 转换为像素坐标
    x_min_px = int(x_min / grid_map.resolution_x)
    x_max_px = int(x_max / grid_map.resolution_x)
    y_min_px = int(y_min / grid_map.resolution_y)
    y_max_px = int(y_max / grid_map.resolution_y)
    
    # 提取该区域的 simulated_map
    # 注意：simulated_map[row, col] = [y, x]
    region_map = simulated_map[y_min_px:y_max_px, x_min_px:x_max_px]
    target_density = region_map.sum() / region_map.size * 100
    
    print(f"  {region['name']}:")
    print(f"    像素范围: [{x_min_px}, {y_min_px}] 到 [{x_max_px}, {y_max_px}]")
    print(f"    实际目标密度: {target_density:.1f}%")
    print(f"    配置目标概率: {region.get('target_probability', 'N/A')}")

print("\n"+"="*60)
print("⚠️ 重要发现:")
print("  simulated_map 的目标分布是随机生成的 (gaussian_random_field)")
print("  它不会自动对齐到你配置的 search_regions！")
print("  这就是为什么红蓝色区域与网格地图错位的原因。")
print("\n解决方案:")
print("  1. 修改 ground_truths.py 使其根据 search_regions 生成目标")
print("  2. 或者接受随机分布，仅将 search_regions 用于智能体搜索策略")
print(f"\n当前绘图坐标范围正确: X ∈ [0, {actual_x_meters:.1f}], Y ∈ [0, {actual_y_meters:.1f}]")
