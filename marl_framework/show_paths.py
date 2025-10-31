import os

# 模拟constants.py中的路径计算逻辑
REPO_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep  # marl_framework文件夹

LOG_DIR = os.path.join(REPO_DIR, "log")
EXPERIMENTS_FOLDER = os.path.join(REPO_DIR, "res")

print("📁 文件存储位置:")
print(f"marl_framework目录: {REPO_DIR}")
print(f"日志存储路径: {LOG_DIR}")
print(f"结果存储路径: {EXPERIMENTS_FOLDER}")
print()
print("📂 目录结构:")
print("E:/code/paper_code/paper/marl_framework/")
print("├── log/           # 训练日志文件")
print("├── res/           # 训练结果文件")
print("├── configs/       # 配置文件")
print("├── actor/         # Actor网络")
print("├── critic/        # Critic网络")
print("├── missions/      # 任务模块")
print("└── ...")