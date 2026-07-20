import base64
import hashlib
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from cryptography.fernet import Fernet
import jwt
from app.config import settings
from app.utils.logger import logger


# Generate secure Fernet instance safely from config string
def get_fernet_cipher() -> Fernet:
    """
    Safely generates a valid 32-byte URL-safe base64 key using SHA-256 hash of ENCRYPTION_KEY
    to prevent ValueError crashes.
    """
    key_bytes = settings.ENCRYPTION_KEY.encode()
    hashed = hashlib.sha256(key_bytes).digest()
    fernet_key = base64.urlsafe_b64encode(hashed)
    return Fernet(fernet_key)

cipher = get_fernet_cipher()

# Text Cryptography
def encrypt_text(text: str) -> str:
    if not text:
        return ""
    try:
        encrypted = cipher.encrypt(text.encode("utf-8"))
        return encrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encrypt text: {str(e)}")
        raise e

def decrypt_text(encrypted_text: str) -> str:
    if not encrypted_text:
        return ""
    try:
        decrypted = cipher.decrypt(encrypted_text.encode("utf-8"))
        return decrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decrypt text: {str(e)}")
        raise e

# File Cryptography
def encrypt_file(file_bytes: bytes) -> bytes:
    try:
        return cipher.encrypt(file_bytes)
    except Exception as e:
        logger.error(f"Failed to encrypt file bytes: {str(e)}")
        raise e

def decrypt_file(encrypted_bytes: bytes) -> bytes:
    try:
        return cipher.decrypt(encrypted_bytes)
    except Exception as e:
        logger.error(f"Failed to decrypt file bytes: {str(e)}")
        raise e

# Password Hashing via native PBKDF2-SHA256 (independent of passlib/bcrypt conflicts)
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return f"{salt.hex()}${key.hex()}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        salt_hex, key_hex = hashed_password.split("$")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt, 100000)
        return new_key == key
    except Exception:
        return False

# JWT Authentication Tokens
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None
