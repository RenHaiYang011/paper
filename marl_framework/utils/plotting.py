import matplotlib
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle
import cv2
import os
import logging
from typing import Optional, List, Dict

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from marl_framework.constants import REPO_DIR

LOG_PLOTS_DIR = os.path.join(REPO_DIR, "log", "plots")
RES_PLOTS_DIR = os.path.join(REPO_DIR, "res", "plots")
os.makedirs(LOG_PLOTS_DIR, exist_ok=True)
os.makedirs(RES_PLOTS_DIR, exist_ok=True)


def plot_trajectories(
    agent_positions,
    n_agents,
    writer,
    training_step_index,
    t_collision,
    budget,
    simulated_map,
):

    # colors = ["b", "g", "c", "r", "k", "w", "m", "y"]
    # plt.figure()
    #
    # simulated_map = cv2.resize(
    #     simulated_map,
    #     (51, 51),
    #     interpolation=cv2.INTER_AREA,
    # )
    # plt.imshow(simulated_map)
    # plt.colorbar()
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
    #     plt.plot(y, x, color=colors[agent_id], linestyle="-", linewidth=10)
    #     plt.plot(y[0], x[0], color=colors[agent_id], marker="o", markersize=14)
    #
    # save example plot to project log directory
    # plt.savefig(os.path.join(LOG_PLOTS_DIR, f"coma_pathes_3d_{training_step_index}.png"))
    # writer.add_figure(f"Agent trajectories", plt.gcf(), training_step_index, close=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ax = plt.gca(projection='3d')

    colors = ["c", "g", "m", "orange", "k", "w", "m", "y"]

    resolution = 0.1014
    # plt.figure()

    # simulated_map = cv2.resize(
    #     simulated_map,
    #     (51, 51),
    #     interpolation=cv2.INTER_AREA,
    # )

    # Use actual simulated_map shape instead of hardcoded (493, 493)
    map_height, map_width = simulated_map.shape
    
    # Create coordinate meshgrid in world coordinates (0-50 range)
    # Scale the meshgrid to match the actual world coordinate system
    world_x = np.linspace(0, 50, map_width)
    world_y = np.linspace(0, 50, map_height)
    Y, X = np.meshgrid(world_x, world_y)
    
    ax.plot_surface(
        Y,
        X,
        np.zeros_like(simulated_map),
        facecolors=cm.coolwarm(simulated_map),
        zorder=1,
    )
    # plt.colorbar()

    for agent_id in range(n_agents):
        x = []
        y = []
        z = []
        for positions in agent_positions:
            # Use world coordinates directly (no resolution division needed)
            x.append(positions[agent_id][0])
            y.append(positions[agent_id][1])
            z.append(positions[agent_id][2])

        ax.plot(y, x, z, color=colors[agent_id], linestyle="-", linewidth=6, zorder=100)
        # ax.plot(y[0], x[0], color=colors[agent_id], marker="o", markersize=14)
    ax.view_init(40, 50)

    # Use dynamic limits based on world coordinate system (0-50)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    
    # Get altitude range from agent positions
    all_altitudes = [pos[agent_id][2] for agent_id in range(n_agents) for pos in agent_positions]
    if all_altitudes:
        min_alt = min(all_altitudes)
        max_alt = max(all_altitudes)
        # Add some padding
        z_min = max(0, min_alt - 5)
        z_max = max_alt + 5
        ax.set_zlim(z_min, z_max)
        # Set z ticks dynamically
        z_step = max(5, int((z_max - z_min) / 3))
        z_ticks = list(range(int(z_min), int(z_max) + 1, z_step))
        if z_ticks:
            ax.set_zticks(z_ticks)
    else:
        ax.set_zlim(0, 30)  # fallback
        ax.set_zticks([0, 10, 20, 30])
    
    # Set x/y ticks to match world coordinates (0-50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels([0, 10, 20, 30, 40, 50])

    # Save to disk (uncomment if you prefer files)
    # fig.savefig(os.path.join(LOG_PLOTS_DIR, f"coma_pathes_3d_{training_step_index}.png"))
    # Add to TensorBoard if writer supplied
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
        fig.savefig(out_path_log, dpi=150)
    except Exception:
        # best effort, continue
        pass
    try:
        fig.savefig(out_path_res, dpi=150)
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

