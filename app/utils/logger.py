import logging
import re
import json
from typing import Any

# Regex to match 12-digit Aadhaar numbers (with or without spaces/dashes)
AADHAAR_REGEX = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that sanitizes log messages by masking Aadhaar numbers
    and preventing logging of face embeddings (long float lists).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize the message if it's a string
        if isinstance(record.msg, str):
            record.msg = self.sanitize_string(record.msg)
        
        # Sanitize any arguments passed to the log
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.sanitize_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.sanitize_value(arg) for arg in record.args)
            else:
                record.args = self.sanitize_value(record.args)
                
        return True

    def sanitize_string(self, text: str) -> str:
        # Mask Aadhaar numbers with XXXX-XXXX-XXXX
        return AADHAAR_REGEX.sub("XXXX-XXXX-XXXX", text)

    def sanitize_value(self, val: Any) -> Any:
        if isinstance(val, str):
            return self.sanitize_string(val)
        # Check if list of floats (like face embeddings of length 512)
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
    
    # If logger is already configured, don't duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Apply JSON formatter
    formatter = StructuredJSONFormatter('%(asctime)s')
    ch.setFormatter(formatter)
    
    # Add sensitivity filter to mask Aadhaar and embeddings
    ch.addFilter(SensitiveDataFilter())
    
    logger.addHandler(ch)
    # Prevent propagation to root logger to avoid double logging
    logger.propagate = False
    
    return logger

# Default logger instance
logger = setup_logger()
