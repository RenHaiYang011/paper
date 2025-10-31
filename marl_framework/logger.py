import logging
import os
import time

import constants


def setup_logger() -> logging.Logger:
    # 如果LOG_DIR还没有设置，使用默认路径
    log_dir = constants.LOG_DIR if constants.LOG_DIR is not None else os.path.join(constants.REPO_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure the root marl_framework logger
    logger = logging.getLogger("marl_framework")
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    # 定期刷新日志的自定义处理器 - 强制实时写入
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()  # 每次写入后立即刷新
            # 在Linux下强制同步到磁盘
            try:
                os.fsync(self.stream.fileno())
            except (OSError, AttributeError):
                pass

    # File handler with immediate flushing
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_filename = f"log_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # 创建实时刷新的文件处理器
    file_handler = FlushingFileHandler(
        filename=log_file_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 立即写入一条测试日志确保文件创建
    logger.info(f"📝 Log file created: {log_file_path}")
    logger.info(f"🐧 Running on: {os.name} system")
    logger.info(f"📁 Log directory: {log_dir}")
    
    # Also configure child loggers to use the same handlers
    # This ensures modules using getLogger(__name__) will also write to our files
    for name in ["marl_framework.missions.coma_mission", 
                 "marl_framework.missions.mission_factories",
                 "marl_framework.constants"]:
        child_logger = logging.getLogger(name)
        child_logger.setLevel(logging.DEBUG)
        child_logger.propagate = True  # Let it propagate to parent

    return logger
