import logging
import sys
import io
import os
from config import local_path, LOG_LEVEL


def setup_logger(name):
    logger = logging.getLogger(name)
    assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    logger.setLevel(getattr(logging, LOG_LEVEL))  # Set logger level

    c_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    )
    f_handler = logging.FileHandler(
        os.path.join(os.path.expanduser(local_path), "app.log")
    )

    c_handler.setLevel(level=getattr(logging, LOG_LEVEL))
    f_handler.setLevel(logging.WARNING)

    c_format = logging.Formatter(
        '{"time": "%(asctime)s", "name": %(name)r, "level": "%(levelname)s", "message": %(message)r}',
        "%m-%d %H:%M:%S",
    )
    f_format = logging.Formatter(
        '{"time": "%(asctime)s", "name": %(name)r, "level": "%(levelname)s", "message": %(message)r}',
        "%m-%d %H:%M:%S",
    )

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info(f"log level {os.path.basename(__file__)}: {LOG_LEVEL}")

    return logger
