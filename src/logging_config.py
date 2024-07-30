import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logging(
        name: str,
        log_file: str = "../.gitignore/log/main.log"
):
    """
    Set up logging configuration.
    """
    # Ensure the logs directory exists
    # if not os.path.exists('.gitignore/log'):
    #     os.makedirs('../.gitignore/log')

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a file handler
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=5*1024*1024,
        backupCount=2,
        encoding=None,
        delay=0
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if already present
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
