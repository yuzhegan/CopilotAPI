"""Logging utility for the application."""

import logging
import os
import sys

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create logger
logger = logging.getLogger("copilot_api")
logger.setLevel(getattr(logging, log_level))

# Add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)

# Add file handler if LOG_FILE is specified
log_file = os.getenv("LOG_FILE")
if log_file:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
