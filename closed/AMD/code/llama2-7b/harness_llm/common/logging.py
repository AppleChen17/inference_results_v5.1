import logging

def set_level(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

_logging_initialized = False

if not _logging_initialized:
    set_level()
    _logging_initialized = True

def get_logger(file: str):  
    return logging.getLogger(file)