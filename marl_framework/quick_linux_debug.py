#!/usr/bin/env python3
"""
快速诊断Linux日志问题的脚本
"""

import os
import sys
import time
import logging

def quick_linux_log_test():
    """快速测试Linux日志功能"""
    print("🔧 Linux日志快速诊断")
    print("=" * 40)
    
    # 1. 检查当前目录
    current_dir = os.getcwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 2. 检查是否在正确的目录
    if not os.path.exists('constants.py'):
        print("❌ 错误：请在marl_framework目录中运行此脚本")
        return False
    
    # 3. 导入模块
    try:
        import constants
        from logger import FlushingFileHandler
        print("✅ 模块导入成功")
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # 4. 检查目录创建
    log_dir = constants.LOG_DIR
    res_dir = constants.EXPERIMENTS_FOLDER
    
    print(f"📁 日志目录: {log_dir}")
    print(f"📁 结果目录: {res_dir}")
    
    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    # 5. 测试原生Python日志
    print("\n🧪 测试原生Python文件写入...")
    test_file = os.path.join(log_dir, "native_test.log")
    try:
        with open(test_file, 'w') as f:
            f.write(f"原生测试 {time.time()}\n")
            f.flush()
            if hasattr(os, 'fsync'):
                os.fsync(f.fileno())
        
        # 检查文件是否存在且有内容
        if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
            print("✅ 原生文件写入成功")
        else:
            print("❌ 原生文件写入失败")
            return False
    except Exception as e:
        print(f"❌ 原生文件写入异常: {e}")
        return False
    
    # 6. 测试自定义FlushingFileHandler
    print("\n🧪 测试FlushingFileHandler...")
    try:
        log_file = os.path.join(log_dir, f"flushing_test_{int(time.time())}.log")
        handler = FlushingFileHandler(log_file)
        
        # 创建临时logger测试
        test_logger = logging.getLogger("flush_test")
        test_logger.setLevel(logging.INFO)
        test_logger.addHandler(handler)
        
        # 发送测试消息
        test_logger.info("FlushingFileHandler测试消息 1")
        test_logger.info("FlushingFileHandler测试消息 2")
        test_logger.warning("FlushingFileHandler警告消息")
        
        # 立即检查文件
        time.sleep(0.1)  # 给一点时间让写入完成
        
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file)
            print(f"✅ FlushingFileHandler测试成功 ({file_size} 字节)")
            
            # 显示内容
            with open(log_file, 'r') as f:
                content = f.read()
                if content:
                    print(f"📄 文件内容预览:")
                    for line in content.strip().split('\n')[:3]:
                        print(f"   {line}")
                else:
                    print("⚠️ 文件存在但为空")
        else:
            print("❌ FlushingFileHandler测试失败 - 文件未创建")
            return False
        
        # 清理
        test_logger.removeHandler(handler)
        handler.close()
        
    except Exception as e:
        print(f"❌ FlushingFileHandler测试异常: {e}")
        return False
    
    # 7. 测试完整的logger setup
    print("\n🧪 测试完整的logger设置...")
    try:
        from logger import setup_logger
        
        # 重新设置logger
        logger = setup_logger("full_test")
        
        # 发送消息
        logger.info("完整logger测试开始")
        logger.info(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.warning("这是警告消息")
        logger.error("这是错误消息")
        logger.info("完整logger测试结束")
        
        # 检查日志文件
        log_files = [f for f in os.listdir(log_dir) if f.startswith('log_') and f.endswith('.log')]
        
        if log_files:
            print(f"✅ 找到 {len(log_files)} 个日志文件")
            
            # 检查最新的文件
            latest_file = max([os.path.join(log_dir, f) for f in log_files], 
                             key=os.path.getmtime)
            file_size = os.path.getsize(latest_file)
            print(f"📄 最新日志文件: {os.path.basename(latest_file)} ({file_size} 字节)")
            
            if file_size > 0:
                print("✅ 完整logger测试成功")
                
                # 显示最后几行
                with open(latest_file, 'r') as f:
                    lines = f.readlines()
                    print("📋 最后几行内容:")
                    for line in lines[-3:]:
                        print(f"   {line.strip()}")
                        
                return True
            else:
                print("❌ 日志文件为空")
                return False
        else:
            print("❌ 没有创建日志文件")
            return False
            
    except Exception as e:
        print(f"❌ 完整logger测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def provide_linux_tips():
    """提供Linux使用提示"""
    print("\n💡 Linux训练监控提示:")
    print("1. 实时查看日志:")
    print("   tail -f log/log_*.log")
    print("")
    print("2. 检查文件实时更新:")
    print("   watch -n 2 'ls -la log/'")
    print("")
    print("3. 监控训练进度:")
    print("   watch -n 10 'cat res/training_progress.json'")
    print("")
    print("4. 检查Python进程:")
    print("   ps aux | grep python")

if __name__ == "__main__":
    success = quick_linux_log_test()
    
    if success:
        print("\n🎉 所有测试通过！日志系统应该正常工作")
    else:
        print("\n❌ 测试失败，需要进一步调试")
        print("\n🔧 建议检查项目:")
        print("1. 确保在marl_framework目录中运行")
        print("2. 检查Python环境和权限")
        print("3. 确保磁盘空间充足")
        print("4. 检查是否有防火墙/权限限制")
    
    provide_linux_tips()