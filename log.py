"""
Some of following code from https://stackoverflow.com/a/42328068/10389584
Please acknowledge if you use it.
"""

import logging
import datetime

logger = logging.getLogger()


def setup_file_logger(log_file):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log(message):
    # outputs to Jupyter console
    print('{} {}'.format(datetime.datetime.now(), message))
    # outputs to file
    logger.info(message)