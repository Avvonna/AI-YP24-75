import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file: str = "logs/app.log") -> logging.Logger:
    """Настройка логирования с ротацией файла"""

    base_dir = Path(__file__).resolve().parent.parent.parent
    log_path = base_dir / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Формат логов
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Обработчик логов в файл с ротацией
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Обработчик логов в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Главный логгер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
