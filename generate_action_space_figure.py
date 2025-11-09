"""
生成27动作空间可视化图
用于论文中展示无人机的离散动作空间结构
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_27_actions_3d():
    """生成3D立方体邻域的27个动作可视化"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # ==================== 子图1: 3D立方体视图 ====================
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 生成27个动作的坐标
    d = 1  # 网格间距（归一化为1）
    actions = []
    action_labels = []
    
    idx = 0
    for dz in [d, 0, -d]:  # z方向: 上, 中, 下
        for dy in [-d, 0, d]:  # y方向: 南, 中, 北
            for dx in [-d, 0, d]:  # x方向: 西, 中, 东
                actions.append([dx, dy, dz])
                action_labels.append(f'a{idx}')
                idx += 1
    
    actions = np.array(actions)
    
    # 绘制所有动作点
    ax1.scatter(actions[:, 0], actions[:, 1], actions[:, 2], 
                c='lightblue', s=100, alpha=0.6, edgecolors='blue', linewidth=2)
    
    # 突出显示中心点（当前位置）
    center_idx = 13  # a13 = (0, 0, 0)
    ax1.scatter([0], [0], [0], c='red', s=300, marker='*', 
                edgecolors='darkred', linewidth=2, label='当前位置 (a₁₃)', zorder=10)
    
    # 突出显示6个主要方向
    main_directions = {
        4: ('上', 'green'),
        22: ('下', 'purple'),
        10: ('西', 'orange'),
        16: ('东', 'cyan'),
        12: ('北', 'yellow'),
        14: ('南', 'pink')
    }
    
    for idx, (label, color) in main_directions.items():
        pos = actions[idx]
        ax1.scatter([pos[0]], [pos[1]], [pos[2]], 
                   c=color, s=200, marker='o', alpha=0.8,
                   edgecolors='black', linewidth=1.5)
        # 添加箭头从中心指向主方向
        ax1.quiver(0, 0, 0, pos[0], pos[1], pos[2], 
                  color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
    
    # 绘制立方体边框
    # 定义立方体的12条边
    edges = [
        # 上层 (z=1) 的4条边
        [[-1,-1,1], [1,-1,1]], [[1,-1,1], [1,1,1]], [[1,1,1], [-1,1,1]], [[-1,1,1], [-1,-1,1]],
        # 下层 (z=-1) 的4条边
        [[-1,-1,-1], [1,-1,-1]], [[1,-1,-1], [1,1,-1]], [[1,1,-1], [-1,1,-1]], [[-1,1,-1], [-1,-1,-1]],
        # 连接上下层的4条边
        [[-1,-1,-1], [-1,-1,1]], [[1,-1,-1], [1,-1,1]], [[1,1,-1], [1,1,1]], [[-1,1,-1], [-1,1,1]]
    ]
    
    for edge in edges:
        points = np.array(edge)
        ax1.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                  'gray', linewidth=1, alpha=0.3, linestyle='--')
    
    ax1.set_xlabel('X (东 →)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (北 →)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (上 ↑)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 27动作空间 - 3D立方体视图', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.view_init(elev=20, azim=45)
    
    # ==================== 子图2: 三层平面展开图 ====================
    ax2 = fig.add_subplot(222)
    ax2.axis('off')
    
    # 绘制三层网格
    layers = ['上层 (z=z₀+d)', '中层 (z=z₀)', '下层 (z=z₀-d)']
    layer_colors = ['lightgreen', 'lightblue', 'lightyellow']
    
    for layer_idx, (layer_name, layer_color) in enumerate(zip(layers, layer_colors)):
        # 计算每层的起始位置
        x_offset = layer_idx * 4
        y_base = 8
        
        # 绘制3x3网格
        for i in range(3):
            for j in range(3):
                x = x_offset + j
                y = y_base - i
                
                # 计算动作索引
                action_idx = layer_idx * 9 + i * 3 + j
                
                # 绘制方块
                rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                    facecolor=layer_color, 
                                    edgecolor='black', linewidth=1.5)
                ax2.add_patch(rect)
                
                # 添加动作标签
                if action_idx == 13:  # 中心点
                    ax2.text(x, y, f'a₁₃\n●', ha='center', va='center', 
                            fontsize=10, fontweight='bold', color='red')
                elif action_idx in main_directions:
                    direction_name = main_directions[action_idx][0]
                    ax2.text(x, y, f'a₁{action_idx}\n{direction_name}', 
                            ha='center', va='center', fontsize=9, fontweight='bold')
                else:
                    ax2.text(x, y, f'a₁{action_idx}' if action_idx >= 10 else f'a₀{action_idx}', 
                            ha='center', va='center', fontsize=8)
        
        # 添加层标题
        ax2.text(x_offset + 1, y_base + 1, layer_name, 
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim([-1, 13])
    ax2.set_ylim([4, 10])
    ax2.set_aspect('equal')
    ax2.set_title('(b) 27动作空间 - 三层平面展开图', fontsize=12, fontweight='bold', pad=20)
    
    # ==================== 子图3: 俯视图 (中层) ====================
    ax3 = fig.add_subplot(223)
    
    # 中层的9个动作
    middle_layer = actions[9:18]
    
    # 绘制网格点
    for i, pos in enumerate(middle_layer):
        idx = i + 9
        if idx == 13:  # 中心点
            ax3.scatter(pos[0], pos[1], c='red', s=500, marker='*', 
                       edgecolors='darkred', linewidth=2, zorder=10)
            ax3.text(pos[0], pos[1]-0.3, '当前位置\n(a₁₃)', 
                    ha='center', va='top', fontsize=10, fontweight='bold', color='red')
        else:
            ax3.scatter(pos[0], pos[1], c='lightblue', s=300, 
                       edgecolors='blue', linewidth=2)
            ax3.text(pos[0], pos[1]+0.25, f'a₁{idx}', 
                    ha='center', va='bottom', fontsize=9)
    
    # 绘制网格线
    for x in [-d, 0, d]:
        ax3.axvline(x, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    for y in [-d, 0, d]:
        ax3.axhline(y, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    
    # 添加方向箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='green')
    ax3.annotate('', xy=(1.5, 0), xytext=(1.2, 0), arrowprops=arrow_props)
    ax3.text(1.5, -0.15, '东 (X)', ha='center', fontsize=10, fontweight='bold')
    
    ax3.annotate('', xy=(0, 1.5), xytext=(0, 1.2), arrowprops=arrow_props)
    ax3.text(-0.15, 1.5, '北 (Y)', ha='right', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('X 方向 (米)', fontsize=10)
    ax3.set_ylabel('Y 方向 (米)', fontsize=10)
    ax3.set_title('(c) 中层俯视图 (水平移动)', fontsize=12, fontweight='bold')
    ax3.set_xlim([-1.8, 1.8])
    ax3.set_ylim([-1.8, 1.8])
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # ==================== 子图4: 侧视图 (X-Z平面) ====================
    ax4 = fig.add_subplot(224)
    
    # X-Z平面的9个动作 (Y=0)
    xz_actions = actions[[1, 4, 7, 10, 13, 16, 19, 22, 25]]
    xz_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25]
    
    for i, pos in enumerate(xz_actions):
        idx = xz_indices[i]
        if idx == 13:  # 中心点
            ax4.scatter(pos[0], pos[2], c='red', s=500, marker='*', 
                       edgecolors='darkred', linewidth=2, zorder=10)
            ax4.text(pos[0], pos[2]-0.3, '当前位置\n(a₁₃)', 
                    ha='center', va='top', fontsize=10, fontweight='bold', color='red')
        elif idx == 4:  # 向上
            ax4.scatter(pos[0], pos[2], c='green', s=300, 
                       edgecolors='darkgreen', linewidth=2)
            ax4.text(pos[0], pos[2]+0.2, f'a₀{idx}\n(上)', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
        elif idx == 22:  # 向下
            ax4.scatter(pos[0], pos[2], c='purple', s=300, 
                       edgecolors='darkviolet', linewidth=2)
            ax4.text(pos[0], pos[2]-0.2, f'a₂₂\n(下)', 
                    ha='center', va='top', fontsize=9, fontweight='bold', color='purple')
        else:
            ax4.scatter(pos[0], pos[2], c='lightblue', s=250, 
                       edgecolors='blue', linewidth=2)
            label = f'a₀{idx}' if idx < 10 else f'a₁{idx}'
            ax4.text(pos[0], pos[2]+0.2, label, 
                    ha='center', va='bottom', fontsize=8)
    
    # 绘制网格线
    for x in [-d, 0, d]:
        ax4.axvline(x, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    for z in [-d, 0, d]:
        ax4.axhline(z, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    
    ax4.set_xlabel('X 方向 (米)', fontsize=10)
    ax4.set_ylabel('Z 方向 (米)', fontsize=10)
    ax4.set_title('(d) 侧视图 (X-Z平面, Y=0)', fontsize=12, fontweight='bold')
    ax4.set_xlim([-1.8, 1.8])
    ax4.set_ylim([-1.8, 1.8])
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_simple_3d_cube():
    """生成简洁版的3D立方体图 (适合论文正文)"""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 生成27个动作
    d = 1
    actions = []
    for dz in [d, 0, -d]:
        for dy in [-d, 0, d]:
            for dx in [-d, 0, d]:
                actions.append([dx, dy, dz])
    actions = np.array(actions)
    
    # 绘制所有点
    ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2], 
              c='lightblue', s=150, alpha=0.6, edgecolors='blue', linewidth=2)
    
    # 中心点
    ax.scatter([0], [0], [0], c='red', s=400, marker='*', 
              edgecolors='darkred', linewidth=3, label='当前无人机位置', zorder=10)
    
    # 6个主要方向
    directions = [
        ([0, 0, d], 'green', '上'),
        ([0, 0, -d], 'purple', '下'),
        ([-d, 0, 0], 'orange', '西'),
        ([d, 0, 0], 'cyan', '东'),
        ([0, -d, 0], 'pink', '南'),
        ([0, d, 0], 'yellow', '北')
    ]
    
    for pos, color, label in directions:
        ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                  c=color, s=250, marker='o', alpha=0.9,
                  edgecolors='black', linewidth=2)
        # 箭头
        ax.quiver(0, 0, 0, pos[0], pos[1], pos[2], 
                 color=color, arrow_length_ratio=0.2, linewidth=3, alpha=0.8)
        # 标签
        ax.text(pos[0]*1.3, pos[1]*1.3, pos[2]*1.3, label, 
               fontsize=12, fontweight='bold', color=color)
    
    # 立方体框架
    edges = [
        [[-1,-1,1], [1,-1,1]], [[1,-1,1], [1,1,1]], [[1,1,1], [-1,1,1]], [[-1,1,1], [-1,-1,1]],
        [[-1,-1,-1], [1,-1,-1]], [[1,-1,-1], [1,1,-1]], [[1,1,-1], [-1,1,-1]], [[-1,1,-1], [-1,-1,-1]],
        [[-1,-1,-1], [-1,-1,1]], [[1,-1,-1], [1,-1,1]], [[1,1,-1], [1,1,1]], [[-1,1,-1], [-1,1,1]]
    ]
    
    for edge in edges:
        points = np.array(edge)
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                 'gray', linewidth=1.5, alpha=0.4, linestyle='--')
    
    ax.set_xlabel('X 方向 (东)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y 方向 (北)', fontsize=13, fontweight='bold')
    ax.set_zlabel('Z 方向 (上)', fontsize=13, fontweight='bold')
    ax.set_title('无人机27动作空间 (3×3×3邻域)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.view_init(elev=25, azim=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加文字说明
    text_str = '• 中心: 当前位置 (通常屏蔽)\n' \
               '• 立方体顶点: 26个可达位置\n' \
               '• 网格间距: d = 3m 或 5m'
    ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # 生成完整版四子图
    print("生成完整版27动作空间可视化图...")
    fig1 = generate_27_actions_3d()
    fig1.savefig('27_action_space_complete.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 27_action_space_complete.png")
    
    # 生成简洁版单图
    print("\n生成简洁版3D立方体图...")
    fig2 = generate_simple_3d_cube()
    fig2.savefig('27_action_space_simple.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 27_action_space_simple.png")
    
    print("\n✓ 所有图片生成完成!")
    print("\n使用建议:")
    print("  - 论文正文: 使用 27_action_space_simple.png (简洁清晰)")
    print("  - 附录/详细说明: 使用 27_action_space_complete.png (完整详细)")
    
    plt.show()
