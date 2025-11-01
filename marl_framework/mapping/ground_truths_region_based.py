"""
基于搜索区域配置生成目标分布的地面真值生成器
这个版本会根据 params 中的 search_regions 配置生成目标
"""
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def generate_region_based_map(params: Dict, y_dim: int, x_dim: int, episode: int) -> np.array:
    """
    根据 search_regions 配置生成目标分布地图
    
    Args:
        params: 配置参数字典
        y_dim: 地图高度（像素）
        x_dim: 地图宽度（像素）
        episode: episode编号（用于随机种子）
    
    Returns:
        np.array: shape (y_dim, x_dim)，值为 0 或 1
    """
    np.random.seed(episode)
    
    # 初始化全零地图
    field = np.zeros((y_dim, x_dim), dtype=np.float32)
    
    # 计算分辨率（米/像素）
    env_x_dim = params['environment']['x_dim']
    env_y_dim = params['environment']['y_dim']
    res_x = env_x_dim / x_dim
    res_y = env_y_dim / y_dim
    
    # 如果没有配置搜索区域，返回随机地图
    if 'search_regions' not in params or 'regions' not in params['search_regions']:
        logger.warning("No search_regions configured, generating random map")
        field = np.random.binomial(1, 0.5, (y_dim, x_dim)).astype(np.float32)
        return field
    
    # 遍历每个搜索区域
    for region in params['search_regions']['regions']:
        # 获取区域参数
        coords = region['coordinates'][0]  # [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = coords
        target_prob = region.get('target_probability', 0.5)
        
        # 转换为像素坐标
        x_min_px = int(x_min / res_x)
        x_max_px = int(x_max / res_x)
        y_min_px = int(y_min / res_y)
        y_max_px = int(y_max / res_y)
        
        # 确保坐标在有效范围内
        x_min_px = max(0, min(x_min_px, x_dim - 1))
        x_max_px = max(0, min(x_max_px, x_dim))
        y_min_px = max(0, min(y_min_px, y_dim - 1))
        y_max_px = max(0, min(y_max_px, y_dim))
        
        # 在该区域内生成目标
        # 使用 binomial 分布生成 0/1 值
        region_height = y_max_px - y_min_px
        region_width = x_max_px - x_min_px
        
        if region_height > 0 and region_width > 0:
            # 生成随机目标分布
            region_targets = np.random.binomial(1, target_prob, 
                                               (region_height, region_width))
            
            # 填充到地图中
            field[y_min_px:y_max_px, x_min_px:x_max_px] = region_targets
            
            # 只在第一个episode或每100个episode时记录详细日志
            if episode == 0 or episode % 100 == 0:
                logger.info(f"Episode {episode} - Generated {region['name']}: "
                           f"pixels [{x_min_px}, {y_min_px}] to [{x_max_px}, {y_max_px}], "
                           f"target_prob={target_prob:.2f}, "
                           f"actual_density={region_targets.mean():.2f}")
            else:
                # 使用 debug 级别，正常训练时不显示
                logger.debug(f"Episode {episode} - Generated {region['name']}: density={region_targets.mean():.2f}")
    
    return field


def gaussian_random_field_original(pk, x_dim: int, y_dim: int, episode: int) -> np.array:
    """
    原始的随机场生成函数（保留用于兼容性）
    生成的目标分布与 search_regions 无关
    """
    import math
    
    def fft_indices(n):
        a = list(range(0, math.floor(n / 2) + 1))
        b = reversed(range(1, math.floor(n / 2)))
        b = [-i for i in b]
        return a + b
    
    def pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(pk(np.sqrt(kx ** 2 + ky ** 2)))

    np.random.seed(episode)
    noise = np.fft.fft2(np.random.normal(size=(y_dim, x_dim)))
    amplitude = np.zeros((y_dim, x_dim))

    for i, kx in enumerate(fft_indices(y_dim)):
        for j, ky in enumerate(fft_indices(x_dim)):
            amplitude[i, j] = pk2(kx, ky)

    random_field = np.fft.ifft2(noise * amplitude).real
    normalized_random_field = (random_field - np.min(random_field)) / (
        np.max(random_field) - np.min(random_field)
    )

    # Make field binary
    normalized_random_field[normalized_random_field >= 0.5] = 1
    normalized_random_field[normalized_random_field < 0.5] = 0

    field = np.zeros((y_dim, x_dim))
    np.random.seed(episode)
    environment_type_idx = 0
    split_idx = np.random.randint(4)

    # ... (原来的随机分布逻辑)
    
    return field
