import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from logging import Formatter
from logging.handlers import RotatingFileHandler


def init_logger(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_stream_format = "[%(levelname)s]: %(message)s"

    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(Formatter(log_stream_format))

    info_file_handler = RotatingFileHandler(os.path.join(log_dir, 'exp_info.log'))
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(Formatter(log_file_format))

    warning_file_handler = RotatingFileHandler(os.path.join(log_dir, 'exp_error.log'))
    warning_file_handler.setLevel(logging.WARNING)
    warning_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(stream_handler)
    main_logger.addHandler(info_file_handler)
    main_logger.addHandler(warning_file_handler)
