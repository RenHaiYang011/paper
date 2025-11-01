#!/usr/bin/env python3
"""
测试脚本：验证日志和结果文件存储路径
"""

import os
import sys

# 添加marl_framework到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'marl_framework'))

import marl_framework.constants as constants

def test_paths():
    print("=== 路径配置测试 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {os.path.dirname(__file__)}")
    print()
    
    print("=== 项目路径配置 ===")
    print(f"PROJECT_ROOT (项目主目录): {constants.PROJECT_ROOT}")
    print(f"REPO_DIR (marl_framework目录): {constants.REPO_DIR}")
    print()
    
    print("=== 日志和结果路径 ===")
    print(f"LOG_DIR (日志目录): {constants.LOG_DIR}")
    print(f"EXPERIMENTS_FOLDER (结果目录): {constants.EXPERIMENTS_FOLDER}")
    print()
    
    print("=== 路径验证 ===")
    
    # 检查项目根目录
    if os.path.exists(constants.PROJECT_ROOT):
        print(f"✅ 项目根目录存在: {constants.PROJECT_ROOT}")
    else:
        print(f"❌ 项目根目录不存在: {constants.PROJECT_ROOT}")
    
    # 创建并检查日志目录
    try:
        os.makedirs(constants.LOG_DIR, exist_ok=True)
        print(f"✅ 日志目录已创建/存在: {constants.LOG_DIR}")
    except Exception as e:
        print(f"❌ 无法创建日志目录: {e}")
    
    # 创建并检查结果目录
    try:
        os.makedirs(constants.EXPERIMENTS_FOLDER, exist_ok=True)
        print(f"✅ 结果目录已创建/存在: {constants.EXPERIMENTS_FOLDER}")
    except Exception as e:
        print(f"❌ 无法创建结果目录: {e}")
    
    print()
    print("=== 预期目录结构 ===")
    print("paper/")
    print("├── log/                    # 日志文件")
    print("├── res/                    # 结果文件")
    print("├── marl_framework/         # 代码目录")
    print("├── README.md")
    print("└── ...")
    print()
    
    # 创建测试文件验证写入权限
    test_log_file = os.path.join(constants.LOG_DIR, "test_log.txt")
    test_res_file = os.path.join(constants.EXPERIMENTS_FOLDER, "test_result.txt")
    
    try:
        with open(test_log_file, 'w') as f:
            f.write("测试日志文件")
        print(f"✅ 日志目录写入测试成功: {test_log_file}")
        os.remove(test_log_file)  # 清理测试文件
    except Exception as e:
        print(f"❌ 日志目录写入测试失败: {e}")
    
    try:
        with open(test_res_file, 'w') as f:
            f.write("测试结果文件")
        print(f"✅ 结果目录写入测试成功: {test_res_file}")
        os.remove(test_res_file)  # 清理测试文件
    except Exception as e:
        print(f"❌ 结果目录写入测试失败: {e}")

if __name__ == "__main__":
    test_paths()