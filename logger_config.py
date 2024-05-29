import logging
import sys

def add_file_handler(logger, log_file, log_level=logging.DEBUG):
    # Create a file handler
    handler = logging.FileHandler(log_file)

    handler.setLevel(log_level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

def remove_file_handler(logger, log_file):
    for handler in logger.handlers:
        if handler.baseFilename == log_file:
            handler.close()
            logger.removeHandler(handler)

def add_stream_handler(logger, log_level=logging.DEBUG):
    # Create a stream handler
    handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(log_level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

logging.Logger.root.level = 10

# azure logger
azure_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
azure_logger.setLevel(logging.WARNING)

# Set up local logger
local_logger = logging.getLogger('local_logger')
local_logger.setLevel(logging.DEBUG)

# local_logger.handlers = logging.root.handlers
local_logger.handlers = []

# add_file_handler(local_logger, 'logs.log', logging.DEBUG)
add_stream_handler(local_logger, logging.DEBUG)

####### use as follows to log to different files in different functions #######
# import logger_config

# def some_function():
#     # Get the logger
#     logger = logging.getLogger('local_logger')

#     # Add a new file handler
#     logger_config.add_file_handler(logger, 'some_function.log')

#     # Do some stuff...
#     logger.debug('This is a debug message from some_function')

#     # Remove the file handler
#     logger_config.remove_file_handler(logger, 'some_function.log')