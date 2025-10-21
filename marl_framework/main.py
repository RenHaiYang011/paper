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
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH) 

    # Set device
    if torch.cuda.is_available() and params["networks"]["device"] == "cuda":
        constants.DEVICE = torch.device("cuda")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # Speed-up for convs with stable input sizes
        torch.backends.cudnn.benchmark = True
        logger.info("Using GPU")
    else:
        constants.DEVICE = torch.device("cpu")
        logger.info("Using CPU")

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
    logger = setup_logger()
    main()
