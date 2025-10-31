#!/usr/bin/env python3
"""
Linux日志调试脚本 - Python版本
用于诊断MARL Framework在Linux上的日志问题
"""

import os
import sys
import time
import platform
import subprocess
from pathlib import Path

def check_platform():
    """检查平台信息"""
    print("=== 系统环境检查 ===")
    print(f"🖥️  操作系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📁 当前目录: {os.getcwd()}")
    print(f"👤 用户权限: UID={os.getuid() if hasattr(os, 'getuid') else 'N/A'}")

def check_directories():
    """检查目录状态"""
    print("\n=== 目录状态检查 ===")
    
    dirs_to_check = ['log', 'res']
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        print(f"\n📁 检查 {dir_name}/ 目录:")
        
        if dir_path.exists():
            print(f"   ✅ 目录存在")
            
            # 检查权限
            if hasattr(os, 'access'):
                readable = os.access(dir_path, os.R_OK)
                writable = os.access(dir_path, os.W_OK)
                print(f"   📋 权限: 读取={readable}, 写入={writable}")
            
            # 列出文件
            files = list(dir_path.iterdir())
            print(f"   📄 文件数量: {len(files)}")
            
            if files:
                print("   📋 文件列表:")
                for file in files[:10]:  # 只显示前10个文件
                    if file.is_file():
                        size = file.stat().st_size
                        print(f"     - {file.name} ({size} 字节)")
                    else:
                        print(f"     - {file.name}/ (目录)")
                
                if len(files) > 10:
                    print(f"     ... 还有 {len(files) - 10} 个文件")
        else:
            print(f"   ❌ 目录不存在")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ 目录创建成功")
            except Exception as e:
                print(f"   ❌ 目录创建失败: {e}")

def test_python_imports():
    """测试Python模块导入"""
    print("\n=== Python模块导入测试 ===")
    
    modules_to_test = ['constants', 'logger']
    
    for module_name in modules_to_test:
        print(f"\n📦 测试 {module_name} 模块:")
        try:
            if module_name == 'constants':
                import constants
                print("   ✅ 导入成功")
                print(f"   📁 LOG_DIR: {getattr(constants, 'LOG_DIR', 'undefined')}")
                print(f"   📁 EXPERIMENTS_FOLDER: {getattr(constants, 'EXPERIMENTS_FOLDER', 'undefined')}")
                
            elif module_name == 'logger':
                from logger import setup_logger, FlushingFileHandler
                print("   ✅ 导入成功")
                print("   📝 可用组件: setup_logger, FlushingFileHandler")
                
        except ImportError as e:
            print(f"   ❌ 导入失败: {e}")
        except Exception as e:
            print(f"   ⚠️  导入警告: {e}")

def test_file_writing():
    """测试文件写入功能"""
    print("\n=== 文件写入测试 ===")
    
    test_file = Path("log/debug_test.log")
    
    # 确保log目录存在
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 测试普通文件写入
        print("📝 测试普通文件写入...")
        with open(test_file, 'w') as f:
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("这是一个测试消息\n")
        
        # 检查文件是否被创建
        if test_file.exists():
            size = test_file.stat().st_size
            print(f"   ✅ 文件创建成功 ({size} 字节)")
            
            # 读取内容验证
            content = test_file.read_text()
            print(f"   📄 文件内容预览: {content[:50]}...")
        else:
            print("   ❌ 文件创建失败")
        
    except Exception as e:
        print(f"   ❌ 写入测试失败: {e}")
    
    # 测试实时刷新
    print("\n📝 测试实时刷新...")
    try:
        with open(test_file, 'a') as f:
            for i in range(3):
                f.write(f"实时消息 {i+1}: {time.strftime('%H:%M:%S')}\n")
                f.flush()
                if hasattr(os, 'fsync'):
                    os.fsync(f.fileno())
                print(f"   ✅ 消息 {i+1} 写入完成")
                time.sleep(1)
        
        print("   ✅ 实时刷新测试完成")
        
    except Exception as e:
        print(f"   ❌ 实时刷新测试失败: {e}")

def test_logger_functionality():
    """测试日志记录器功能"""
    print("\n=== 日志记录器功能测试 ===")
    
    try:
        # 导入并设置日志记录器
        from logger import setup_logger
        import constants
        
        # 设置路径（如果需要）
        if hasattr(constants, 'setup_paths'):
            constants.setup_paths()
        
        print("📝 设置日志记录器...")
        logger = setup_logger("test_logger")
        
        print("📝 测试不同级别的日志消息...")
        logger.info("这是一个INFO消息")
        logger.warning("这是一个WARNING消息")
        logger.error("这是一个ERROR消息")
        
        print("   ✅ 日志消息发送完成")
        
        # 检查日志文件是否被创建和写入
        log_dir = Path(constants.LOG_DIR)
        log_files = list(log_dir.glob("log_*.log"))
        
        if log_files:
            print(f"   ✅ 找到 {len(log_files)} 个日志文件")
            for log_file in log_files[:3]:  # 只检查前3个
                size = log_file.stat().st_size
                print(f"     - {log_file.name}: {size} 字节")
                
                if size > 0:
                    # 读取最后几行
                    try:
                        content = log_file.read_text().strip().split('\n')
                        print(f"       最后一行: {content[-1][:80]}...")
                    except Exception as e:
                        print(f"       读取失败: {e}")
        else:
            print("   ❌ 没有找到日志文件")
            
    except Exception as e:
        print(f"   ❌ 日志记录器测试失败: {e}")

def provide_monitoring_commands():
    """提供监控命令建议"""
    print("\n=== Linux训练监控建议 ===")
    
    print("💡 实时监控命令:")
    print("1. 实时查看最新日志:")
    print("   tail -f log/log_*.log")
    print("")
    print("2. 查看训练进度:")
    print("   watch -n 10 'cat res/training_progress.json 2>/dev/null || echo \"进度文件尚未创建\"'")
    print("")
    print("3. 监控目录变化:")
    print("   watch -n 5 'ls -la log/ res/'")
    print("")
    print("4. 检查磁盘空间:")
    print("   df -h .")
    print("")
    print("5. 检查进程:")
    print("   ps aux | grep python")

def main():
    """主函数"""
    print("🔧 MARL Framework Linux日志调试工具")
    print("=" * 50)
    
    check_platform()
    check_directories()
    test_python_imports()
    test_file_writing()
    test_logger_functionality()
    provide_monitoring_commands()
    
    print("\n✅ 调试完成!")
    print("\n📋 如果仍有问题，请检查:")
    print("1. 文件系统权限")
    print("2. 磁盘空间")
    print("3. Python环境配置")
    print("4. 防火墙/安全策略")

if __name__ == "__main__":
    main()