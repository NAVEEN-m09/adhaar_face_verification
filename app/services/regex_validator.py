import re
from typing import Optional, Tuple
from app.utils.logger import logger

# Verhoeff Algorithm Tables for Aadhaar validation
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

class RegexValidator:
    """
    Validates and extracts Aadhaar numbers from OCR texts.
    Performs verification against user input and validates the Verhoeff checksum.
    """
    
    @staticmethod
    def validate_verhoeff(number: str) -> bool:
        """
        Validates a number using the Verhoeff algorithm.
        """
        if not number.isdigit() or len(number) != 12:
            return False
        
        c = 0
        for i, digit in enumerate(reversed(number)):
            c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(digit)]]
        return c == 0

    def extract_aadhaar_number(self, text_lines: list[str]) -> Optional[str]:
        """
        Extracts a valid 12-digit Aadhaar number from lines of text.
        Returns the clean 12-digit string if found, otherwise None.
        """
        # Formats to look for:
        # 1. 12 digits with spaces: \b\d{4}\s\d{4}\s\d{4}\b
        # 2. 12 digits with dashes: \b\d{4}-\d{4}-\d{4}\b
        # 3. 12 digits continuous: \b\d{12}\b
        
        patterns = [
            re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b'),
            re.compile(r'\b\d{4}-\d{4}-\d{4}\b'),
            re.compile(r'\b\d{12}\b')
        ]

        # First pass: try exact regex on each line
        for line in text_lines:
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    matched_str = match.group(0)
                    # Clean it
                    clean_num = re.sub(r'[-\s]', '', matched_str)
                    if self.validate_verhoeff(clean_num):
                        logger.info("RegexValidator: Extracted valid Verhoeff-checksummed Aadhaar number.")
                        return clean_num
                    else:
                        logger.warning("RegexValidator: Found Aadhaar-like number, but failed Verhoeff checksum validation.")

        # Second pass: strip all whitespaces from each line and look for 12 consecutive digits
        # This handles cases where OCR split spaces strangely
        for line in text_lines:
            cleaned_line = re.sub(r'[-\s]', '', line)
            match = re.search(r'\d{12}', cleaned_line)
            if match:
                clean_num = match.group(0)
                if self.validate_verhoeff(clean_num):
                    logger.info("RegexValidator: Extracted valid Aadhaar number after merging/cleaning line.")
                    return clean_num

        return None

    def verify_match(self, extracted: Optional[str], provided: str) -> Tuple[bool, str]:
        """
        Compares the extracted Aadhaar number with the user-provided one.
        Returns:
            - matched: boolean indicating if they match
            - status_msg: message explaining the match status
        """
        # Clean the provided number
        clean_provided = re.sub(r'[-\s]', '', provided)
        
        if len(clean_provided) != 12 or not clean_provided.isdigit():
            return False, "Provided Aadhaar number is invalid. Must be 12 digits."
            
        if not self.validate_verhoeff(clean_provided):
            return False, "Provided Aadhaar number failed Verhoeff checksum validation."

        if not extracted:
            return False, "Aadhaar number could not be extracted from card."

        if extracted == clean_provided:
            return True, "Matched"
        else:
            return False, "Not Matched"
