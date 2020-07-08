""" Package Initialization file. """
import os
import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

# Create the Handler for logging data to a file
logger_handler = RotatingFileHandler(os.path.join(os.path.dirname(__file__), '../logs/tn.log'), maxBytes=1024, backupCount=5)
logger_handler.setLevel(logging.INFO)

#Create the Handler for logging data to console.
console_handler = StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)
console_handler.setFormatter(logger_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

root_logger.addHandler(logger_handler)
root_logger.addHandler(console_handler)