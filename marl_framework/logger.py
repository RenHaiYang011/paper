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

    # File handler
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_filename = f"log_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(
        filename=log_file_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 确保日志实时写入（强制刷新缓冲）
    logger.info(f"📝 Log file created: {log_file_path}")
    
    # 定期刷新日志的自定义处理器
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()  # 每次写入后立即刷新
    
    # 添加实时刷新的文件处理器
    realtime_handler = FlushingFileHandler(
        filename=log_file_path, mode="a", encoding="utf-8"
    )
    realtime_handler.setLevel(logging.INFO)
    realtime_handler.setFormatter(file_formatter)
    
    # 移除原有的file_handler，使用实时刷新的handler
    logger.removeHandler(file_handler)
    logger.addHandler(realtime_handler)
    
    # Also configure child loggers to use the same handlers
    # This ensures modules using getLogger(__name__) will also write to our files
    marl_logger = logging.getLogger("marl_framework")
    for name in ["marl_framework.missions.coma_mission", 
                 "marl_framework.missions.mission_factories",
                 "marl_framework.constants"]:
        child_logger = logging.getLogger(name)
        child_logger.setLevel(logging.DEBUG)
        child_logger.propagate = True  # Let it propagate to parent

    return logger
