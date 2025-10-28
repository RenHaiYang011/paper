import matplotlib
from matplotlib import cm
import cv2
import os
import logging

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
    Y, X = np.meshgrid(range(0, map_width), range(0, map_height))
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
            x.append(positions[agent_id][0] / resolution)
            y.append(positions[agent_id][1] / resolution)
            z.append(positions[agent_id][2])

        ax.plot(y, x, z, color=colors[agent_id], linestyle="-", linewidth=6, zorder=100)
        # ax.plot(y[0], x[0], color=colors[agent_id], marker="o", markersize=14)
    ax.view_init(40, 50)

    # Use dynamic limits based on actual map and altitude range
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    
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
    
    # Set x/y ticks dynamically
    x_step = map_width // 5
    y_step = map_height // 5
    ax.set_xticks([i * x_step for i in range(6)])
    ax.set_xticklabels([0, 10, 20, 30, 40, 50])
    ax.set_yticks([i * y_step for i in range(6)])
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
