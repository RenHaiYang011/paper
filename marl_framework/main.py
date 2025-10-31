import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import constants
from logger import setup_logger
from missions.mission_factories import MissionFactory
from params import load_params

def main():
    # 首先加载参数和设置路径
    params = load_params(constants.CONFIG_FILE_PATH)
    
    # Setup paths based on configuration
    constants.setup_paths(params)
    
    # 然后设置logger（只设置一次，使用正确的路径）
    logger = setup_logger()
    
    # 记录环境信息
    constants.log_env_variables()
    
    logger.info(f"🚀 MARL Framework starting...")
    logger.info(f"📁 Directories configured:")
    logger.info(f"  - Log directory: {constants.LOG_DIR}")
    logger.info(f"  - Results directory: {constants.EXPERIMENTS_FOLDER}")
    logger.info(f"🐧 Operating system: {os.name}")
    logger.info(f"🐍 Python version: {sys.version}")

    # Set device
    if torch.cuda.is_available() and params["networks"]["device"] == "cuda":
        # Use first GPU (cuda:0) - you have 4 GPUs available
        constants.DEVICE = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # GPU optimization settings
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels for better performance
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        torch.backends.cudnn.enabled = True
        
        # Print GPU info
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        constants.DEVICE = torch.device("cpu")
        logger.info("Using CPU")
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Training will be slow.")

    logger.info(
        "\n-------------------------------------- START PIPELINE --------------------------------------\n"
    )

    mission_factory = MissionFactory(params)
    mission = mission_factory.create_mission()
    # 增加训练时间 
    start_time = time.time()
    mission.execute()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Mission execution took: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes).")

    logger.info(
        "\n-------------------------------------- STOP PIPELINE --------------------------------------\n"
    )


if __name__ == "__main__":
    main()
