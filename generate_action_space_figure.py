"""
Generate 27-action space visualization for UAV discrete action space
For paper illustration
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Use moderate fonts and English labels to avoid Chinese font issues
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

def generate_27_actions_3d():
    """Generate 3D visualization of 27-action space"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # ==================== Subplot 1: 3D Cube View ====================
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Generate 27 action coordinates
    d = 1  # Grid spacing (normalized to 1)
    actions = []
    action_labels = []
    
    idx = 0
    for dz in [d, 0, -d]:  # z: up, middle, down
        for dy in [-d, 0, d]:  # y: south, middle, north
            for dx in [-d, 0, d]:  # x: west, middle, east
                actions.append([dx, dy, dz])
                action_labels.append(f'a{idx}')
                idx += 1
    
    actions = np.array(actions)
    
    # Draw all action points
    ax1.scatter(actions[:, 0], actions[:, 1], actions[:, 2], 
                c='lightblue', s=150, alpha=0.6, edgecolors='blue', linewidth=2)
    
    # Highlight center point (current position)
    center_idx = 13  # a13 = (0, 0, 0)
    ax1.scatter([0], [0], [0], c='red', s=400, marker='*', 
                edgecolors='darkred', linewidth=3, label='Current (a13)', zorder=10)
    
    # Highlight 6 main directions
    main_directions = {
        4: ('Up', 'green'),
        22: ('Down', 'purple'),
        10: ('West', 'orange'),
        16: ('East', 'cyan'),
        12: ('North', 'yellow'),
        14: ('South', 'pink')
    }
    
    for idx, (label, color) in main_directions.items():
        pos = actions[idx]
        ax1.scatter([pos[0]], [pos[1]], [pos[2]], 
                   c=color, s=250, marker='o', alpha=0.8,
                   edgecolors='black', linewidth=2)
        # Add arrow from center to main direction
        ax1.quiver(0, 0, 0, pos[0], pos[1], pos[2], 
                  color=color, arrow_length_ratio=0.3, linewidth=3, alpha=0.7)
    
    # Draw cube edges
    edges = [
        # Upper layer (z=1) 4 edges
        [[-1,-1,1], [1,-1,1]], [[1,-1,1], [1,1,1]], [[1,1,1], [-1,1,1]], [[-1,1,1], [-1,-1,1]],
        # Lower layer (z=-1) 4 edges
        [[-1,-1,-1], [1,-1,-1]], [[1,-1,-1], [1,1,-1]], [[1,1,-1], [-1,1,-1]], [[-1,1,-1], [-1,-1,-1]],
        # Connecting edges 4 edges
        [[-1,-1,-1], [-1,-1,1]], [[1,-1,-1], [1,-1,1]], [[1,1,-1], [1,1,1]], [[-1,1,-1], [-1,1,1]]
    ]
    
    for edge in edges:
        points = np.array(edge)
        ax1.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                  'gray', linewidth=1.5, alpha=0.3, linestyle='--')
    
    ax1.set_xlabel('X (East)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (North)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (Up)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 27-Action Space - 3D Cube View', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.view_init(elev=20, azim=45)
    
    # ==================== Subplot 2: Three-Layer Grid Layout ====================
    ax2 = fig.add_subplot(222)
    ax2.axis('off')
    
    # Draw three layers
    layers = ['Upper (z=z0+d)', 'Middle (z=z0)', 'Lower (z=z0-d)']
    layer_colors = ['lightgreen', 'lightblue', 'lightyellow']
    
    for layer_idx, (layer_name, layer_color) in enumerate(zip(layers, layer_colors)):
        # Calculate starting position for each layer
        x_offset = layer_idx * 4
        y_base = 8
        
        # Draw 3x3 grid
        for i in range(3):
            for j in range(3):
                x = x_offset + j
                y = y_base - i
                
                # Calculate action index
                action_idx = layer_idx * 9 + i * 3 + j
                
                # Draw square
                rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                    facecolor=layer_color, 
                                    edgecolor='black', linewidth=2)
                ax2.add_patch(rect)
                
                # Add action label (optimized to avoid overlap)
                if action_idx == 13:  # Center point
                    ax2.text(x, y, f'a{action_idx}', ha='center', va='center', 
                            fontsize=8, fontweight='bold', color='red')
                    ax2.text(x, y-0.25, '*', ha='center', va='center', 
                            fontsize=12, fontweight='bold', color='red')
                elif action_idx in main_directions:
                    direction_name = main_directions[action_idx][0]
                    ax2.text(x, y+0.15, f'a{action_idx}', 
                            ha='center', va='center', fontsize=7, fontweight='bold')
                    ax2.text(x, y-0.15, f'{direction_name}', 
                            ha='center', va='center', fontsize=6, style='italic')
                else:
                    ax2.text(x, y, f'a{action_idx}', 
                            ha='center', va='center', fontsize=8)
        
        # Add layer title
        ax2.text(x_offset + 1, y_base + 1, layer_name, 
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim([-1, 13])
    ax2.set_ylim([4, 10])
    ax2.set_aspect('equal')
    ax2.set_title('(b) 27-Action Space - Three-Layer Grid', fontsize=12, fontweight='bold', pad=15)
    
    # ==================== Subplot 3: Top View (Middle Layer) ====================
    ax3 = fig.add_subplot(223)
    
    # Middle layer 9 actions
    middle_layer = actions[9:18]
    
    # Draw grid points (optimized spacing to avoid overlap)
    for i, pos in enumerate(middle_layer):
        idx = i + 9
        if idx == 13:  # Center point
            ax3.scatter(pos[0], pos[1], c='red', s=600, marker='*', 
                       edgecolors='darkred', linewidth=3, zorder=10)
            ax3.text(pos[0], pos[1]-0.5, 'Current\n(a13)', 
                    ha='center', va='top', fontsize=7, fontweight='bold', color='red')
        else:
            ax3.scatter(pos[0], pos[1], c='lightblue', s=400, 
                       edgecolors='blue', linewidth=2)
            ax3.text(pos[0], pos[1]+0.35, f'a{idx}', 
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Draw grid lines
    for x in [-d, 0, d]:
        ax3.axvline(x, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    for y in [-d, 0, d]:
        ax3.axhline(y, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    
    # Add direction arrows
    arrow_props = dict(arrowstyle='->', lw=3, color='green')
    ax3.annotate('', xy=(1.5, 0), xytext=(1.2, 0), arrowprops=arrow_props)
    ax3.text(1.5, -0.15, 'East (X)', ha='center', fontsize=10, fontweight='bold')
    
    ax3.annotate('', xy=(0, 1.5), xytext=(0, 1.2), arrowprops=arrow_props)
    ax3.text(-0.15, 1.5, 'North (Y)', ha='right', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('X Direction (m)', fontsize=11)
    ax3.set_ylabel('Y Direction (m)', fontsize=11)
    ax3.set_title('(c) Top View (Horizontal Movement)', fontsize=12, fontweight='bold')
    ax3.set_xlim([-1.8, 1.8])
    ax3.set_ylim([-1.8, 1.8])
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # ==================== Subplot 4: Side View (X-Z Plane) ====================
    ax4 = fig.add_subplot(224)
    
    # X-Z plane 9 actions (Y=0) - optimized to avoid overlap
    xz_actions = actions[[1, 4, 7, 10, 13, 16, 19, 22, 25]]
    xz_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25]
    
    for i, pos in enumerate(xz_actions):
        idx = xz_indices[i]
        if idx == 13:  # Center point
            ax4.scatter(pos[0], pos[2], c='red', s=600, marker='*', 
                       edgecolors='darkred', linewidth=3, zorder=10)
            ax4.text(pos[0], pos[2]-0.5, 'Current\n(a13)', 
                    ha='center', va='top', fontsize=7, fontweight='bold', color='red')
        elif idx == 4:  # Up
            ax4.scatter(pos[0], pos[2], c='green', s=400, 
                       edgecolors='darkgreen', linewidth=2)
            ax4.text(pos[0]-0.4, pos[2]+0.15, f'a{idx}\n(Up)', 
                    ha='center', va='center', fontsize=7, fontweight='bold', color='green')
        elif idx == 22:  # Down
            ax4.scatter(pos[0], pos[2], c='purple', s=400, 
                       edgecolors='darkviolet', linewidth=2)
            ax4.text(pos[0]-0.4, pos[2]-0.15, f'a{idx}\n(Down)', 
                    ha='center', va='center', fontsize=7, fontweight='bold', color='purple')
        else:
            ax4.scatter(pos[0], pos[2], c='lightblue', s=350, 
                       edgecolors='blue', linewidth=2)
            ax4.text(pos[0], pos[2]+0.35, f'a{idx}', 
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Draw grid lines
    for x in [-d, 0, d]:
        ax4.axvline(x, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    for z in [-d, 0, d]:
        ax4.axhline(z, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    
    ax4.set_xlabel('X Direction (m)', fontsize=11)
    ax4.set_ylabel('Z Direction (m)', fontsize=11)
    ax4.set_title('(d) Side View (X-Z Plane, Y=0)', fontsize=12, fontweight='bold')
    ax4.set_xlim([-1.8, 1.8])
    ax4.set_ylim([-1.8, 1.8])
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_simple_3d_cube():
    """Generate simplified 3D cube figure (suitable for main text)"""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate 27 actions
    d = 1
    actions = []
    for dz in [d, 0, -d]:
        for dy in [-d, 0, d]:
            for dx in [-d, 0, d]:
                actions.append([dx, dy, dz])
    actions = np.array(actions)
    
    # Draw all points
    ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2], 
              c='lightblue', s=150, alpha=0.6, edgecolors='blue', linewidth=2)
    
    # Center point
    ax.scatter([0], [0], [0], c='red', s=500, marker='*', 
              edgecolors='darkred', linewidth=3, label='Current UAV Position', zorder=10)
    
    # 6 main directions
    directions = [
        ([0, 0, d], 'green', 'Up'),
        ([0, 0, -d], 'purple', 'Down'),
        ([-d, 0, 0], 'orange', 'West'),
        ([d, 0, 0], 'cyan', 'East'),
        ([0, -d, 0], 'pink', 'South'),
        ([0, d, 0], 'yellow', 'North')
    ]
    
    for pos, color, label in directions:
        ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                  c=color, s=250, marker='o', alpha=0.9,
                  edgecolors='black', linewidth=2)
        # Arrows
        ax.quiver(0, 0, 0, pos[0], pos[1], pos[2], 
                 color=color, arrow_length_ratio=0.2, linewidth=3, alpha=0.8)
        # Labels
        ax.text(pos[0]*1.3, pos[1]*1.3, pos[2]*1.3, label, 
               fontsize=11, fontweight='bold', color=color)
    
    # Cube frame
    edges = [
        [[-1,-1,1], [1,-1,1]], [[1,-1,1], [1,1,1]], [[1,1,1], [-1,1,1]], [[-1,1,1], [-1,-1,1]],
        [[-1,-1,-1], [1,-1,-1]], [[1,-1,-1], [1,1,-1]], [[1,1,-1], [-1,1,-1]], [[-1,1,-1], [-1,-1,-1]],
        [[-1,-1,-1], [-1,-1,1]], [[1,-1,-1], [1,-1,1]], [[1,1,-1], [1,1,1]], [[-1,1,-1], [-1,1,1]]
    ]
    
    for edge in edges:
        points = np.array(edge)
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                 'gray', linewidth=1.5, alpha=0.4, linestyle='--')
    
    ax.set_xlabel('X Direction (East)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Direction (North)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z Direction (Up)', fontsize=11, fontweight='bold')
    ax.set_title('UAV 27-Action Space (3x3x3 Neighborhood)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.view_init(elev=25, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text description
    text_str = '* Center: Current position (usually masked)\n' \
               '* Cube vertices: 26 reachable positions\n' \
               '* Grid spacing: d = 3m or 5m'
    ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Generate complete version with 4 subplots
    print("Generating complete 27-action space visualization...")
    fig1 = generate_27_actions_3d()
    fig1.savefig('27_action_space_complete.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 27_action_space_complete.png")
    
    # Generate simplified single figure
    print("\nGenerating simplified 3D cube figure...")
    fig2 = generate_simple_3d_cube()
    fig2.savefig('27_action_space_simple.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 27_action_space_simple.png")
    
    print("\n✓ All figures generated successfully!")
    print("\nUsage recommendations:")
    print("  - Paper main text: Use 27_action_space_simple.png (clean and clear)")
    print("  - Appendix/Details: Use 27_action_space_complete.png (comprehensive)")
    
    plt.show()
