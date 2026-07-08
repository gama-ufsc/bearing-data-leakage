import logging
from logging import INFO, Logger
from typing import List


def setup_logger(
    name: str, log_file: str, format_str: str, level: int = INFO
) -> List[Logger]:
    """To setup as many loggers as you want"""

    formatter = logging.Formatter(format_str)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
