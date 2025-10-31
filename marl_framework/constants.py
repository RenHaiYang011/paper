import logging
import os
import torch

# Use the same logger name as setup_logger
logger = logging.getLogger("marl_framework")

DEVICE = torch.device("cpu")


def load_from_env(env_var_name: str, data_type: callable, default=None):
    if env_var_name in os.environ and os.environ[env_var_name] != "":
        value = os.environ[env_var_name]
        if data_type == bool:
            if value.lower() == "true":
                value = True
            else:
                value = False
        else:
            value = data_type(value)
        return value
    elif env_var_name not in os.environ and default is None:
        raise ValueError(
            f"Could not find environment variable '{env_var_name}'. "
            f"Please check .env file or provide a default value when calling load_from_env()."
        )
    return default


PLOT_LABEL_FONT_SIZE = 30
PLOT_LEGEND_FONT_SIZE = 20
PLOT_TICKS_SIZE = 20
PLOT_LINE_WIDTH = 5

# 获取marl_framework文件夹路径作为基准
REPO_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep  # marl_framework文件夹

CONFIG_FILE_PATH = load_from_env("CONFIG_FILE_PATH", str, "configs/params.yaml")
CONFIG_FILE_PATH = os.path.join(REPO_DIR, CONFIG_FILE_PATH)

CHECKPOINTS_DIR = load_from_env("CHECKPOINTS_DIR", str, "checkpoints")
CHECKPOINTS_DIR = os.path.join(REPO_DIR, CHECKPOINTS_DIR)

TRAIN_DATA_DIR = load_from_env("TRAIN_DATA_DIR", str, "generated_train_data")
TRAIN_DATA_DIR = os.path.join(REPO_DIR, TRAIN_DATA_DIR)

# 默认路径，可以通过setup_paths()函数重新配置
EXPERIMENTS_FOLDER = load_from_env("EXPERIMENT_FILE_PATH", str, "res")
EXPERIMENTS_FOLDER = os.path.join(REPO_DIR, EXPERIMENTS_FOLDER)

LOG_DIR = load_from_env("LOG_DIR", str, "log")
LOG_DIR = os.path.join(REPO_DIR, LOG_DIR)
LOG_LEVEL = logging.DEBUG

DATASETS_DIR = load_from_env("DATASETS_DIR", str, "datasets")
DATASETS_DIR = os.path.join(REPO_DIR, DATASETS_DIR)


def setup_paths(params):
    """
    根据配置参数设置日志和结果存储路径
    
    Args:
        params: 配置参数字典
    """
    global LOG_DIR, EXPERIMENTS_FOLDER
    
    # 获取路径配置，如果没有则使用默认值
    paths_config = params.get("paths", {})
    log_dir_config = paths_config.get("log_dir", "log")
    results_dir_config = paths_config.get("results_dir", "res")
    
    # 处理日志路径
    if os.path.isabs(log_dir_config):
        # 绝对路径
        LOG_DIR = log_dir_config
    else:
        # 相对路径，相对于marl_framework目录
        LOG_DIR = os.path.join(REPO_DIR, log_dir_config)
    
    # 处理结果路径
    if os.path.isabs(results_dir_config):
        # 绝对路径
        EXPERIMENTS_FOLDER = results_dir_config
    else:
        # 相对路径，相对于marl_framework目录
        EXPERIMENTS_FOLDER = os.path.join(REPO_DIR, results_dir_config)
    
    # 确保目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
    
    logger.info(f"📁 Paths configured:")
    logger.info(f"  - Log directory: {LOG_DIR}")
    logger.info(f"  - Results directory: {EXPERIMENTS_FOLDER}")


class SensorType:
    RGB_CAMERA = "rgb_camera"


class SensorParams:
    CAMERA = ["field_of_view"]
    RGB_CAMERA = ["encoding"]


class SensorModelType:
    ALTITUDE_DEPENDENT = "altitude_dependent"


class SensorModelParams:
    ALTITUDE_DEPENDENT = ["coeff_a", "coeff_b"]


class SensorSimulationType:
    RANDOM_FIELD = "random_field"


class SensorSimulationParams:
    RANDOM_FIELD = ["cluster_radius"]


SENSOR_TYPES = ["rgb_camera"]
SENSOR_MODELS = ["altitude_dependent"]

SENSOR_SIMULATIONS = ["random_field"]


class MissionType:
    COMA = "COMA"
    reduced = "reduced"
    random = "random"
    lawnmower = "lawnmower"
    DeepQ = "DeepQ"


class MissionParams:
    STATIC_MISSION = ["min_altitude", "max_altitude", "budget"]
    COMA = ["spacing", "n_agents"]


MISSION_TYPES = ["COMA", "reduced", "random", "lawnmower", "DeepQ"]

UAV_PARAMS = ["max_v", "max_a", "sampling_time"]
ENV_PARAMS = ["x_dim", "y_dim"]


class EvaluationMeasure:
    NUM_WAYPOINTS = "num_waypoints"
    PATHS = "paths"
    RMSE = "rmse"
    WRMSE = "wrmse"
    MLL = "mll"
    WMLL = "wmll"
    RUN_TIME = "run_time"


def log_env_variables():
    env_variables = {
        "REPO_DIR": REPO_DIR,
        "CONFIG_FILE_PATH": CONFIG_FILE_PATH,
        "CHECKPOINTS_DIR": CHECKPOINTS_DIR,
        "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
        "EXPERIMENTS_FOLDER": EXPERIMENTS_FOLDER,
        "LOG_DIR": LOG_DIR,
    }

    logger.info(
        "\n-------------------------------------- LOG ENV-VARIABLES --------------------------------------\n"
    )
    for env_var in env_variables.keys():
        logger.info(
            f"{env_var}: {env_variables[env_var]} | type: {type(env_variables[env_var])}"
        )
