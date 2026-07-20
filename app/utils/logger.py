import logging
import re
import json
from typing import Any

AADHAAR_REGEX = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that sanitizes log messages by masking Aadhaar numbers
    and preventing logging of face embeddings (long float lists).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize_string(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.sanitize_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.sanitize_value(arg) for arg in record.args)
            else:
                record.args = self.sanitize_value(record.args)

        return True

    def sanitize_string(self, text: str) -> str:
        return AADHAAR_REGEX.sub("XXXX-XXXX-XXXX", text)

    def sanitize_value(self, val: Any) -> Any:
        if isinstance(val, str):
            return self.sanitize_string(val)
        if isinstance(val, (list, tuple)) and len(val) > 10 and all(isinstance(x, (int, float)) for x in val[:5]):
            return f"[EMBEDDING ARRAY OF SIZE {len(val)}]"
        return val

class StructuredJSONFormatter(logging.Formatter):
    """
    Formatter that outputs log records as JSON objects.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logger(name: str = "aadhaar_api") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = StructuredJSONFormatter('%(asctime)s')
    ch.setFormatter(formatter)

    ch.addFilter(SensitiveDataFilter())

    logger.addHandler(ch)
    logger.propagate = False

    return logger

logger = setup_logger()
