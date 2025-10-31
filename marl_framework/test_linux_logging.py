#!/usr/bin/env python3
"""
测试Linux上的日志功能
"""

import os
import sys
import time

# 添加marl_framework到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import constants
    from logger import setup_logger
    
    print("=== Linux日志功能测试 ===")
    print(f"当前目录: {current_dir}")
    print(f"日志目录: {constants.LOG_DIR}")
    print(f"结果目录: {constants.EXPERIMENTS_FOLDER}")
    
    # 测试日志设置
    logger = setup_logger()
    
    print(f"\n📝 测试日志写入...")
    
    # 写入一系列测试日志
    logger.info("🚀 开始训练测试")
    logger.info("📊 当前步数: 1/100")
    logger.info("🎯 当前奖励: 10.5")
    logger.warning("⚠️ 这是一个警告信息")
    logger.error("❌ 这是一个错误信息")
    
    print("📁 检查日志文件...")
    
    # 检查log目录中的文件
    log_files = []
    if os.path.exists(constants.LOG_DIR):
        for file in os.listdir(constants.LOG_DIR):
            if file.startswith("log_") and file.endswith(".log"):
                log_files.append(file)
                file_path = os.path.join(constants.LOG_DIR, file)
                file_size = os.path.getsize(file_path)
                print(f"  ✅ 发现日志文件: {file} (大小: {file_size} 字节)")
                
                # 显示文件内容的前几行
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content:
                            lines = content.strip().split('\n')
                            print(f"     📄 文件内容 ({len(lines)} 行):")
                            for i, line in enumerate(lines[:3]):  # 显示前3行
                                print(f"     {i+1}: {line}")
                            if len(lines) > 3:
                                print(f"     ... (还有 {len(lines)-3} 行)")
                        else:
                            print("     ⚠️ 文件为空")
                except Exception as e:
                    print(f"     ❌ 读取文件失败: {e}")
    else:
        print(f"❌ 日志目录不存在: {constants.LOG_DIR}")
    
    if not log_files:
        print("❌ 没有找到日志文件")
        
        # 尝试手动创建测试文件
        print("\n🔧 尝试手动创建测试文件...")
        test_file = os.path.join(constants.LOG_DIR, "test.log")
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("测试日志内容\n")
                f.flush()
                os.fsync(f.fileno())  # 强制同步到磁盘
            
            if os.path.exists(test_file):
                print(f"✅ 测试文件创建成功: {test_file}")
                print(f"📏 文件大小: {os.path.getsize(test_file)} 字节")
            else:
                print(f"❌ 测试文件创建失败")
        except Exception as e:
            print(f"❌ 创建测试文件时出错: {e}")
    else:
        print(f"✅ 日志功能正常，找到 {len(log_files)} 个日志文件")
    
    print(f"\n💡 在Linux上监控日志的命令:")
    if log_files:
        latest_log = os.path.join(constants.LOG_DIR, log_files[-1])
        print(f"tail -f {latest_log}")
    else:
        print(f"tail -f {constants.LOG_DIR}/log_*.log")
    
    print(f"\n📋 目录权限检查:")
    print(f"日志目录权限: {oct(os.stat(constants.LOG_DIR).st_mode)[-3:]}")
    
except Exception as e:
    print(f"❌ 测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()