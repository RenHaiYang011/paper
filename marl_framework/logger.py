import logging
import os
import time

import constants


def setup_logger() -> logging.Logger:
    os.makedirs(constants.LOG_DIR, exist_ok=True)
    
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
    log_file_path = os.path.join(constants.LOG_DIR, log_filename)

    file_handler = logging.FileHandler(
        filename=log_file_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
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
