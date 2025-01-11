import logging
from logging import getLogger, INFO, Formatter, StreamHandler
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

logger = None
def get_logger():
    """Singleton for logger with handlers"""
    global logger
    if logger is None:
        log_dir = Path("/app/logs")
        log_dir.mkdir(exist_ok=True)
        logger = getLogger(__name__)
        formatter = Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        rotated_handler = TimedRotatingFileHandler(str(log_dir)+"/font_logs.txt", when='midnight')
        rotated_handler.setFormatter(formatter)
        rotated_handler.setLevel(INFO)
        logger.addHandler(rotated_handler)
        logger.setLevel(INFO)
        console_handler = StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

