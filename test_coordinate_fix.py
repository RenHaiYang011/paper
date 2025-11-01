#!/usr/bin/env python3
"""
测试坐标修复效果 - 验证地图生成和可视化的坐标对齐
"""

import os
import sys
import yaml
import numpy as np
import math

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'marl_framework'))

from marl_framework.mapping.ground_truths_region_based import generate_region_based_map

def test_coordinate_alignment():
    """测试坐标对齐修复"""
    
    # 加载配置
    config_path = "marl_framework/configs/params_fast.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    print("=== 坐标对齐测试 ===")
    
    # 1. 计算分辨率（复制 grid_maps.py 的逻辑）
    min_altitude = params["experiment"]["constraints"]["min_altitude"]
    angle_x = params["sensor"]["field_of_view"]["angle_x"] 
    angle_y = params["sensor"]["field_of_view"]["angle_y"]
    number_x = params["sensor"]["pixel"]["number_x"]
    number_y = params["sensor"]["pixel"]["number_y"]
    
    res_x = (2 * min_altitude * math.tan(math.radians(angle_x) * 0.5)) / number_x
    res_y = (2 * min_altitude * math.tan(math.radians(angle_y) * 0.5)) / number_y
    
    print(f"传感器参数:")
    print(f"  - min_altitude: {min_altitude} m")
    print(f"  - field_of_view: {angle_x}° × {angle_y}°")
    print(f"  - pixel_numbers: {number_x} × {number_y}")
    print(f"分辨率计算:")
    print(f"  - res_x: {res_x:.6f} m/pixel")
    print(f"  - res_y: {res_y:.6f} m/pixel")
    
    # 2. 计算网格尺寸（复制 grid_maps.py 的逻辑）
    x_dim_m = params["environment"]["x_dim"]
    y_dim_m = params["environment"]["y_dim"]
    
    x_dim_pixels = int(x_dim_m / res_x)
    y_dim_pixels = int(y_dim_m / res_y)
    
    print(f"网格计算:")
    print(f"  - 世界尺寸: {x_dim_m} × {y_dim_m} 米")
    print(f"  - 像素尺寸: {x_dim_pixels} × {y_dim_pixels} 像素")
    
    # 3. 计算实际覆盖范围
    actual_x_coverage = x_dim_pixels * res_x
    actual_y_coverage = y_dim_pixels * res_y
    
    print(f"实际覆盖:")
    print(f"  - X覆盖: {actual_x_coverage:.4f} 米 (误差: {abs(actual_x_coverage - x_dim_m):.4f})")
    print(f"  - Y覆盖: {actual_y_coverage:.4f} 米 (误差: {abs(actual_y_coverage - y_dim_m):.4f})")
    
    # 4. 生成测试地图
    print(f"\n生成测试地图...")
    test_episode = 9999
    simulated_map = generate_region_based_map(params, y_dim_pixels, x_dim_pixels, test_episode)
    
    print(f"地图形状: {simulated_map.shape}")
    print(f"地图值范围: [{simulated_map.min():.3f}, {simulated_map.max():.3f}]")
    
    # 5. 验证搜索区域坐标转换
    print(f"\n验证搜索区域坐标转换:")
    for region in params["search_regions"]["regions"]:
        name = region["name"] 
        # coordinates 是一个列表，包含 [x_min, y_min, x_max, y_max]
        coords = region["coordinates"][0]  # 取第一个坐标组
        x_min, y_min, x_max, y_max = coords
        
        # 转换为像素坐标
        x_min_pixel = int(x_min / res_x)
        x_max_pixel = int(x_max / res_x)
        y_min_pixel = int(y_min / res_y)
        y_max_pixel = int(y_max / res_y)
        
        print(f"  {name}:")
        print(f"    世界坐标: [{x_min}, {x_max}] × [{y_min}, {y_max}]")
        print(f"    像素坐标: [{x_min_pixel}, {x_max_pixel}] × [{y_min_pixel}, {y_max_pixel}]")
        
        # 验证区域内目标密度
        region_map = simulated_map[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
        target_ratio = np.mean(region_map > 0.5)
        expected_prob = region["target_probability"]
        
        print(f"    目标密度: {target_ratio:.3f} (期望: {expected_prob:.3f}, 误差: {abs(target_ratio - expected_prob):.3f})")
    
    print(f"\n=== 测试完成 ===")
    print(f"关键发现:")
    print(f"1. 实际覆盖范围是 {actual_x_coverage:.4f}×{actual_y_coverage:.4f} 米，不是 50×50 米")
    print(f"2. 网格分辨率是 {res_x:.6f} 米/像素")
    print(f"3. 这就是之前可视化错位的原因！")
    
    return {
        'resolution': (res_x, res_y),
        'grid_size': (x_dim_pixels, y_dim_pixels),
        'actual_coverage': (actual_x_coverage, actual_y_coverage),
        'simulated_map': simulated_map
    }

if __name__ == "__main__":
    test_coordinate_alignment()