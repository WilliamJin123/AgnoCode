import logging
import logging.config
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },

    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "level": "DEBUG",
            "rich_tracebacks": True, # Beautiful traceback formatting
            "markup": True,          # Allow "[bold]text[/]" inside log messages!
            "show_path": False       # Cleaner output (hides file path column)
        },
         "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "app.log",
            "maxBytes": 10 * 1024 * 1024,   # 10 MiB per file
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    
    "root": {
        "handlers": ["console", "file"],
        "level": "WARNING", 
    },
    
    "loggers": {
       "key_manager": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

def configure_logging():
    logging.config.dictConfig(LOGGING_CONFIG)