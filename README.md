## Requirements

matplotlib==3.5.1
   
numpy==1.22.2
   
opencv-python==4.5.5.62
   
scipy==1.8.1
   
torch==1.13.0+cu117

# 日志结果结构
marl_framework/
├── log/                              # 当前训练（会覆盖）
│   ├── best_model.pth
│   └── events.out.tfevents.*
│
└── training_history/                 # 自动备份（永久保存）
    ├── baseline/
    │   ├── best_model.pth
    │   ├── events.out.tfevents.*
    │   ├── params_backup.yaml
    │   └── metadata.txt
    ├── collision_2.0/
    └── batch64/

# 训练
./train_with_backup.sh [实验名] 

# 在另一个终端监控GPU
watch -n 1 nvidia-smi

# 训练完成后查看历史
./manage_training_history.sh list


# 例： 调整collision_weight  或者直接修改文件内容
nano params.yaml  # 修改collision_weight: 3.0
./train_with_backup.sh exp_collision_3.0

# 对比实验
./manage_training_history.sh list
tensorboard --logdir training_history/

# 启动TensorBoard查看所有训练
cd ~/paper_v2/paper/marl_framework
tensorboard --logdir training_history/ --host 0.0.0.0 --port 6006

# 在本地浏览器访问
http://服务器IP:6006