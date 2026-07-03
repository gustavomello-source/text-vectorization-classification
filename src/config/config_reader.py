import configparser

import torch

from src.utils.logger import log


class ConfigReader:
    """Reads INI configuration files and sets up the CUDA/torch environment."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_config_files(file_paths: list[str]) -> list[configparser.ConfigParser]:
        log.info("-------------- Reading config files: %s", file_paths)
        try:
            configs = []
            for file_path in file_paths:
                config = configparser.ConfigParser()
                config.read(file_path)
                configs.append(config)
            log.info("-------------- Successfully read config files")
            return configs
        except Exception as e:
            log.error("Error reading config files: %s", e)
            return []

    @staticmethod
    def setup_cuda_environment():
        """Setup optimal CUDA environment variables"""
        torch.manual_seed(1)

        if torch.cuda.is_available():
            log.info(
                "-------------- Setting up CUDA environment with device %s",
                torch.cuda.get_device_name(0),
            )
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

        else:
            log.info("-------------- CUDA not available, using CPU")
