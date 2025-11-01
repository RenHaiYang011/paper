import matplotlib
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle
import cv2
import os
import logging
from typing import Optional, List, Dict
# Plot each face
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
matplotlib.use("Agg")
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from marl_framework.constants import REPO_DIR

LOG_PLOTS_DIR = os.path.join(REPO_DIR, "log", "plots")
RES_PLOTS_DIR = os.path.join(REPO_DIR, "res", "plots")
os.makedirs(LOG_PLOTS_DIR, exist_ok=True)
os.makedirs(RES_PLOTS_DIR, exist_ok=True)


# ==================== 3D Shape Drawing Helper Functions ====================

def plot_cube(ax, x, y, z, size=1.0, color='red', alpha=0.8):
    """
    Draw a 3D cube at position (x, y, z) to represent a target
    
    Args:
        ax: matplotlib 3D axis
        x, y, z: center position
        size: cube size
        color: cube color
        alpha: transparency
    """
    # Define cube vertices relative to center
    r = size / 2
    vertices = np.array([
        [x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r],  # bottom
        [x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r]   # top
    ])
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # top
    ]
    
  
    
    # Use contrasting edge color based on face color
    if color == 'lime' or color == 'green':
        edge_color = 'darkgreen'
    elif color == 'yellow':
        edge_color = 'darkorange'
    else:
        edge_color = 'black'
    
    cube = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                           edgecolor=edge_color, linewidths=2.0)
    cube.set_zsort('max')  # 确保立方体在上层显示
    ax.add_collection3d(cube)


def plot_pyramid(ax, x, y, z, height=10, base_size=3.0, color='gray', alpha=0.7):
    """
    Draw a 3D pyramid/cone at position (x, y, z) to represent an obstacle
    
    Args:
        ax: matplotlib 3D axis
        x, y, z: base center position (通常z=0，从地面开始)
        height: pyramid height (向上延伸的高度)
        base_size: base square size
        color: pyramid color
        alpha: transparency
    """
    # Define pyramid vertices
    r = base_size / 2
    vertices = np.array([
        [x-r, y-r, z],          # base corner 1 (底面角1)
        [x+r, y-r, z],          # base corner 2 (底面角2)
        [x+r, y+r, z],          # base corner 3 (底面角3)
        [x-r, y+r, z],          # base corner 4 (底面角4)
        [x, y, z+height]        # apex (顶点)
    ])
    
    # Define the 5 faces of the pyramid (4 triangular + 1 square base)
    faces = [
        [vertices[0], vertices[1], vertices[4]],  # front triangle (前面三角形)
        [vertices[1], vertices[2], vertices[4]],  # right triangle (右面三角形)
        [vertices[2], vertices[3], vertices[4]],  # back triangle (后面三角形)
        [vertices[3], vertices[0], vertices[4]],  # left triangle (左面三角形)
        [vertices[0], vertices[1], vertices[2], vertices[3]]  # base square (底面正方形)
    ]
    
    # Plot each face
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Use contrasting edge color for visibility
    if color == 'yellow':
        edge_color = 'darkorange'
    else:
        edge_color = 'black'
    
    # 创建3D多边形集合，增强边缘线宽度让立体感更强
    pyramid = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                              edgecolor=edge_color, linewidths=3.0)
    pyramid.set_zsort('min')  # 使用min排序，让障碍物显示清晰
    ax.add_collection3d(pyramid)
    
    # 在障碍物底部周围添加一个灰色基座，增强"立在地面"的视觉效果
    # 基座比障碍物底面稍大一圈，颜色稍暗，增强3D立体感
    base_r = base_size / 2 * 1.15  # 基座比障碍物底面大15%
    base_z = max(0, z - 0.1)  # 基座稍微低一点，但不能低于0
    base_vertices = np.array([
        [x-base_r, y-base_r, base_z],
        [x+base_r, y-base_r, base_z],
        [x+base_r, y+base_r, base_z],
        [x-base_r, y+base_r, base_z]
    ])
    base_face = [base_vertices]
    base_patch = Poly3DCollection(base_face, alpha=0.4, facecolor='gray',
                                 edgecolor='darkgray', linewidths=1.5)
    base_patch.set_zsort('max')
    ax.add_collection3d(base_patch)
    
    # 在障碍物四个角画从基座到底面的垂直支撑线，增强立体感
    r = base_size / 2
    for corner_x, corner_y in [[x-r, y-r], [x+r, y-r], [x+r, y+r], [x-r, y+r]]:
        ax.plot([corner_x, corner_x], [corner_y, corner_y], [base_z, z], 
               color=edge_color, alpha=0.5, linewidth=2.0, linestyle='-')


# ==================== Main Plotting Functions ====================


def plot_trajectories(
    agent_positions,
    n_agents,
    writer,
    training_step_index,
    t_collision,
    budget,
    simulated_map,
    obstacles=None,
):



    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["c", "g", "m", "orange", "k", "w", "y", "b"]

    # Use actual simulated_map shape instead of hardcoded (493, 493)
    map_height, map_width = simulated_map.shape
    
    # CRITICAL FIX: Calculate exact world coverage based on grid map resolution
    # The grid map resolution is calculated from sensor parameters in grid_maps.py:
    # res = (2 * min_altitude * tan(angle/2)) / pixel_number
    # Using default params from params_fast.yaml:
    # - min_altitude = 5m, angle_x = angle_y = 60°, number_x = number_y = 57
    # - res = (2 * 5 * tan(30°)) / 57 = (10 * 0.577) / 57 ≈ 0.1013 m/pixel
    # - Grid size: 50m / 0.1013 ≈ 493 pixels
    # - Actual coverage: 493 * 0.1013 ≈ 49.94m (NOT 50.0m!)
    

    
    # Calculate exact resolution to match grid_maps.py calculations
    min_altitude = 5  # Default from params_fast.yaml
    angle_degrees = 60  # Default field_of_view
    pixel_number = 57   # Default pixel number
    
    # Exact resolution calculation matching GridMap.res_x property
    resolution = (2 * min_altitude * math.tan(math.radians(angle_degrees) * 0.5)) / pixel_number
    
    # Calculate actual world coverage to match grid_maps.py exactly
    actual_x_coverage = map_width * resolution
    actual_y_coverage = map_height * resolution
    
    print(f"PLOT DEBUG: Map size: {map_width}×{map_height}, Resolution: {resolution:.6f} m/pixel")
    print(f"PLOT DEBUG: Actual coverage: {actual_x_coverage:.4f}×{actual_y_coverage:.4f} meters")
    
    # Create coordinate meshgrid using EXACT coverage (not assumed 50m)
    # CRITICAL: meshgrid must match plot_surface X,Y ordering
    # simulated_map is indexed as [row, col] = [y, x]
    # For plot_surface(X, Y, Z): X varies along columns, Y varies along rows
    x_coords = np.linspace(0, actual_x_coverage, map_width)   # columns -> X
    y_coords = np.linspace(0, actual_y_coverage, map_height)  # rows -> Y
    X, Y = np.meshgrid(x_coords, y_coords)                    # X: columns, Y: rows
    
    # Plot ground surface with proper coordinate alignment - 地图在 z=0（地面）
    # 障碍物将从 z=0.5 开始，稍微悬浮避免 z-fighting
    map_z_level = 0  # 地图在 z=0 平面（地面）
    surface = ax.plot_surface(
        X,  # X coordinates (columns, 0-50)
        Y,  # Y coordinates (rows, 0-50)
        np.full_like(simulated_map, map_z_level),  # 地图在 z=0 平面
        facecolors=cm.coolwarm(simulated_map),
        zorder=1,  # 地图在底层
        alpha=1.0,  # 完全不透明，清晰显示红蓝色区域对比
        shade=False,  # 减少阴影效果，让颜色更清晰
    )
    
    # Plot obstacles AFTER map to ensure proper layering
    # 障碍物绘制在地图之后，确保显示在上层
    # CRITICAL: 障碍物的 x,y 坐标必须与地图坐标系统一致
    if obstacles is not None and len(obstacles) > 0:
        print(f"PLOT DEBUG: 绘制 {len(obstacles)} 个障碍物")
        print(f"PLOT DEBUG: 地图坐标范围: X[0, {actual_x_coverage:.2f}], Y[0, {actual_y_coverage:.2f}]")
        
        for i, obs in enumerate(obstacles):
            # 障碍物坐标是世界坐标（米），直接使用
            obs_x, obs_y, obs_z = obs['x'], obs['y'], obs['z']
            obs_height = obs.get('height', 10)
            
            # 验证障碍物是否在地图范围内
            if 0 <= obs_x <= actual_x_coverage and 0 <= obs_y <= actual_y_coverage:
                print(f"PLOT DEBUG: 障碍物 {i+1}: 世界坐标({obs_x:.1f}, {obs_y:.1f}, {obs_z:.1f}) 高度: {obs_height}m ✓ 在地图范围内")
                
                # 障碍物从 z=0.5 开始向上延伸（稍微悬浮避免与地图 z=0 重叠）
                obstacle_base_z = 0.5
                
                # 直接使用世界坐标绘制，与地图坐标系统一致
                # 增大 base_size 让障碍物更明显醒目
                plot_pyramid(ax, obs_x, obs_y, obstacle_base_z, height=obs_height, 
                            base_size=5.5, color='yellow', alpha=0.9)
            else:
                print(f"PLOT DEBUG: 障碍物 {i+1}: 坐标({obs_x}, {obs_y}) ✗ 超出地图范围!")
    else:
        print(f"PLOT DEBUG: 没有障碍物数据")
    
    # Plot agent trajectories
    for agent_id in range(n_agents):
        x = []
        y = []
        z = []
        for positions in agent_positions:
            # Use world coordinates directly
            x.append(positions[agent_id][0])
            y.append(positions[agent_id][1])
            z.append(positions[agent_id][2])

        # Plot trajectory line
        ax.plot(x, y, z, color=colors[agent_id], linestyle="-", 
               linewidth=4, zorder=100, label=f'Agent {agent_id+1}')
        
        # Mark start position with sphere
        if len(x) > 0:
            ax.scatter(x[0], y[0], z[0], color=colors[agent_id], 
                      s=100, marker='o', zorder=101, edgecolors='black', linewidths=2)
    
    # Set viewing angle for better 3D effect - 调整视角让障碍物真正看起来立在地图上
    ax.view_init(elev=30, azim=45)  # 稍微提高仰角，调整方位角，获得更好的3D立体感

    # Use dynamic limits based on actual world coordinate coverage
    ax.set_xlim(0, actual_x_coverage)
    ax.set_ylim(0, actual_y_coverage)
    
    # Get altitude range from agent positions and obstacles
    all_altitudes = [pos[agent_id][2] for agent_id in range(n_agents) for pos in agent_positions]
    
    # Include obstacle heights in altitude calculation
    if obstacles is not None and len(obstacles) > 0:
        obstacle_heights = [obs['z'] + obs.get('height', 10) for obs in obstacles]
        all_altitudes.extend(obstacle_heights)
        print(f"PLOT DEBUG: 障碍物最高点: {max(obstacle_heights):.1f}米")
    
    # 定义地图所在的z平面
    map_z_level = 0  # 地图在 z=0
    
    if all_altitudes:
        min_alt = min(all_altitudes)
        max_alt = max(all_altitudes)
        # Z轴范围：从地图平面 z=0 开始
        z_min = 0  # 从地面开始
        z_max = max(max_alt + 8, 25)  # 确保最高障碍物+padding都可见
        ax.set_zlim(z_min, z_max)
        print(f"PLOT DEBUG: Z轴范围: [{z_min}, {z_max}]")
        # Set z ticks dynamically
        z_step = max(5, int((z_max - z_min) / 5))
        z_ticks = list(range(int(z_min), int(z_max) + 1, z_step))
        if z_ticks:
            ax.set_zticks(z_ticks)
    else:
        ax.set_zlim(map_z_level, 30)  # 从地图平面开始
        ax.set_zticks([0, 5, 10, 15, 20, 25, 30])
    
    # Set x/y ticks to match actual world coordinates
    x_tick_step = actual_x_coverage / 5  # 5 intervals
    y_tick_step = actual_y_coverage / 5  # 5 intervals
    x_ticks = [i * x_tick_step for i in range(6)]  # 0, 10, 20, 30, 40, 50 (approx)
    y_ticks = [i * y_tick_step for i in range(6)]  # 0, 10, 20, 30, 40, 50 (approx)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
    
    # Add axis labels
    ax.set_xlabel('X Position (m)', fontsize=10, labelpad=8)
    ax.set_ylabel('Y Position (m)', fontsize=10, labelpad=8)
    ax.set_zlabel('Altitude (m)', fontsize=10, labelpad=8)
    
    # Add title
    ax.set_title(f'UAV Search Trajectories - Step {training_step_index}', 
                fontsize=12, fontweight='bold', pad=15)
    
    # Add legend - 只保留智能体轨迹的图例
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Add custom legend for obstacles only - 移除目标立方体图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='darkorange', label='Obstacle (Yellow Pyramid)', alpha=0.85),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # Save to disk and TensorBoard
    # Try to add to TensorBoard (if writer provided). Ignore writer errors.
    try:
        writer.add_figure(f"Agent trajectories", fig, training_step_index, close=False)
    except Exception:
        pass

    # Always save a copy to the project's log and res folders for later inspection
    out_name = f"coma_pathes_3d_{training_step_index}.png"
    out_path_log = os.path.join(LOG_PLOTS_DIR, out_name)
    out_path_res = os.path.join(RES_PLOTS_DIR, out_name)
    try:
        fig.savefig(out_path_log, dpi=150, bbox_inches='tight')
    except Exception:
        # best effort, continue
        pass
    try:
        fig.savefig(out_path_res, dpi=150, bbox_inches='tight')
    except Exception:
        pass

    # Close the figure to free memory. If we added to tensorboard with close=False, close here.
    plt.close(fig)
    # ax = fig.gca(projection='3d')
    #
    # for agent_id in range(n_agents):
    #     x = []
    #     y = []
    #     z = []
    #     for positions in agent_positions:
    #         x.append(positions[agent_id][0])
    #         y.append(positions[agent_id][1])
    #         z.append(positions[agent_id][2])
    #
    #     ax.plot(y, x, z, color=colors[agent_id], linestyle="-", linewidth=10)

    # ax.savefig(os.path.join(LOG_PLOTS_DIR, f"ig_pathes_3d_{training_step_index}.png"))
    # writer.add_figure(f"Agent trajectories - 3D", ax.gcf(), training_step_index, close=True)


def plot_performance(budget, entropies):
    x = list(range(0, budget + 2))
    y = entropies

    plt.plot(x, y)
    np.savetxt(os.path.join(RES_PLOTS_DIR, "learned_new.txt"), y, delimiter=",")
    plt.savefig(os.path.join(RES_PLOTS_DIR, "lawnmower_comparison_uncertainty_reduction.png"))


# ==================== Region Search Visualization Functions ====================

def plot_search_regions_on_map(
    simulated_map: np.ndarray,
    search_region_manager,
    spacing: float,
    save_path: Optional[str] = None,
    title: str = "Search Regions",
):
    """
    Plot search regions with priorities overlaid on the map
    
    Args:
        simulated_map: The ground truth map
        search_region_manager: SearchRegionManager instance
        spacing: Grid spacing
        save_path: Optional path to save the figure
        title: Plot title
    """
    if search_region_manager is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot base map
    ax.imshow(simulated_map, cmap='gray', alpha=0.5)
    
    # Color map for priorities
    colors = plt.cm.RdYlGn_r  # Red (high) to Green (low)
    
    # Plot each region
    for region in search_region_manager.regions:
        if region.region_type == 'rectangle':
            coords = region.coordinates[0]
            # Convert world coords to grid indices
            x_min = coords[0] / spacing
            y_min = coords[1] / spacing
            width = (coords[2] - coords[0]) / spacing
            height = (coords[3] - coords[1]) / spacing
            
            # Draw rectangle
            color = colors(region.priority)
            rect = Rectangle(
                (y_min, x_min), height, width,
                linewidth=3, edgecolor=color, facecolor=color, alpha=0.3
            )
            ax.add_patch(rect)
            
            # Add label
            center_x = x_min + width / 2
            center_y = y_min + height / 2
            ax.text(
                center_y, center_x, 
                f"{region.name}\nP:{region.priority:.2f}\nD:{region.search_density}",
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Y (Grid Units)', fontsize=12)
    ax.set_ylabel('X (Grid Units)', fontsize=12)
    
    # Add colorbar for priority
    sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Priority', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return fig


def plot_region_coverage_heatmap(
    search_region_manager,
    training_step: int,
    writer=None,
    save_dir: str = LOG_PLOTS_DIR,
):
    """
    Plot heatmap showing visit counts for each region
    
    Args:
        search_region_manager: SearchRegionManager instance
        training_step: Current training step
        writer: TensorBoard writer
        save_dir: Directory to save plots
    """
    if search_region_manager is None:
        return
    
    n_regions = len(search_region_manager.regions)
    if n_regions == 0:
        return
    
    fig, axes = plt.subplots(1, n_regions, figsize=(6 * n_regions, 5))
    if n_regions == 1:
        axes = [axes]
    
    for idx, region in enumerate(search_region_manager.regions):
        ax = axes[idx]
        
        # Plot visit count map
        im = ax.imshow(region.visit_count_map, cmap='hot', interpolation='nearest')
        ax.set_title(
            f"{region.name}\nCoverage: {region.current_coverage:.2%}\nRequired: {region.min_coverage:.2%}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Y (Grid)', fontsize=10)
        ax.set_ylabel('X (Grid)', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Visit Count', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.suptitle(f'Region Coverage Heatmap - Step {training_step}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to file
    save_path = os.path.join(save_dir, f'region_coverage_heatmap_{training_step}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Add to TensorBoard
    if writer is not None:
        try:
            writer.add_figure('RegionSearch/Coverage_Heatmap', fig, training_step, close=False)
        except Exception:
            pass
    
    plt.close(fig)


def plot_search_progress(
    search_stats_history: List[Dict],
    training_step: int,
    writer=None,
    save_dir: str = LOG_PLOTS_DIR,
):
    """
    Plot search progress over time
    
    Args:
        search_stats_history: List of search statistics over time
        training_step: Current training step
        writer: TensorBoard writer
        save_dir: Directory to save plots
    """
    if not search_stats_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    steps = [s['step'] for s in search_stats_history]
    global_completion = [s['global_completion'] for s in search_stats_history]
    total_visits = [s['total_visits'] for s in search_stats_history]
    covered_cells = [s['covered_cells'] for s in search_stats_history]
    
    # Plot 1: Global completion over time
    axes[0, 0].plot(steps, global_completion, linewidth=2, color='blue', marker='o', markersize=4)
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[0, 0].set_title('Global Search Completion', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Training Step', fontsize=10)
    axes[0, 0].set_ylabel('Completion (%)', fontsize=10)
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Total visits over time
    axes[0, 1].plot(steps, total_visits, linewidth=2, color='green', marker='s', markersize=4)
    axes[0, 1].set_title('Total Visits', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Training Step', fontsize=10)
    axes[0, 1].set_ylabel('Visit Count', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Covered cells over time
    axes[1, 0].plot(steps, covered_cells, linewidth=2, color='orange', marker='^', markersize=4)
    axes[1, 0].set_title('Covered Cells', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Training Step', fontsize=10)
    axes[1, 0].set_ylabel('Cell Count', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Per-region coverage
    if search_stats_history[-1]['regions']:
        region_names = [r['name'] for r in search_stats_history[-1]['regions']]
        region_coverages = [r['coverage'] for r in search_stats_history[-1]['regions']]
        region_required = [r['required_coverage'] for r in search_stats_history[-1]['regions']]
        
        x = np.arange(len(region_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, region_coverages, width, label='Current', color='skyblue')
        axes[1, 1].bar(x + width/2, region_required, width, label='Required', color='lightcoral')
        axes[1, 1].set_title('Region Coverage Status', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Coverage (%)', fontsize=10)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(region_names, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Search Progress - Step {training_step}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to file
    save_path = os.path.join(save_dir, f'search_progress_{training_step}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Add to TensorBoard
    if writer is not None:
        try:
            writer.add_figure('RegionSearch/Progress', fig, training_step, close=False)
        except Exception:
            pass
    
    plt.close(fig)


def plot_trajectories_with_regions(
    agent_positions,
    n_agents,
    search_region_manager,
    spacing: float,
    writer=None,
    training_step_index: int = 0,
    simulated_map: Optional[np.ndarray] = None,
):
    """
    Plot 3D trajectories with search regions overlaid
    
    Args:
        agent_positions: List of agent positions over time
        n_agents: Number of agents
        search_region_manager: SearchRegionManager instance
        spacing: Grid spacing
        writer: TensorBoard writer
        training_step_index: Current training step
        simulated_map: Optional ground truth map
    """
    if search_region_manager is None:
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot base map if provided
    if simulated_map is not None:
        map_height, map_width = simulated_map.shape
        Y, X = np.meshgrid(range(0, map_width), range(0, map_height))
        ax.plot_surface(
            Y, X, np.zeros_like(simulated_map),
            facecolors=cm.coolwarm(simulated_map),
            alpha=0.5, zorder=1
        )
    
    # Plot search regions as 3D boxes
    colors_priority = plt.cm.RdYlGn_r
    for region in search_region_manager.regions:
        if region.region_type == 'rectangle':
            coords = region.coordinates[0]
            x_min, y_min = coords[0] / spacing, coords[1] / spacing
            x_max, y_max = coords[2] / spacing, coords[3] / spacing
            
            # Draw vertical lines at corners
            color = colors_priority(region.priority)
            z_max = 30  # Visual height
            for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
                ax.plot([y, y], [x, x], [0, z_max], color=color, linewidth=2, alpha=0.5)
    
    # Plot agent trajectories
    agent_colors = ["c", "g", "m", "orange", "k", "w", "b", "y"]
    resolution = 0.1014
    
    for agent_id in range(n_agents):
        x, y, z = [], [], []
        for positions in agent_positions:
            x.append(positions[agent_id][0] / resolution)
            y.append(positions[agent_id][1] / resolution)
            z.append(positions[agent_id][2])
        
        ax.plot(y, x, z, color=agent_colors[agent_id], linestyle='-', linewidth=6, zorder=100)
    
    ax.view_init(40, 50)
    ax.set_xlabel('Y', fontsize=12)
    ax.set_ylabel('X', fontsize=12)
    ax.set_zlabel('Altitude', fontsize=12)
    ax.set_title(f'Trajectories with Search Regions - Step {training_step_index}', fontsize=14, fontweight='bold')
    
    # Save and add to TensorBoard
    save_path = os.path.join(LOG_PLOTS_DIR, f'trajectories_with_regions_{training_step_index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if writer is not None:
        try:
            writer.add_figure('RegionSearch/Trajectories_With_Regions', fig, training_step_index, close=False)
        except Exception:
            pass
    
    plt.close(fig)

