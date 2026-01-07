"""
Utility functions for the email spam detection system.
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_subject_and_body(subject, body):
    """
    Combine email subject and body into a single text string.
    
    Args:
        subject: Email subject (can be None or empty)
        body: Email body text
        
    Returns:
        Combined text string
    """
    subject = subject or ""
    body = body or ""
    combined = f"{subject} {body}".strip()
    return combined


def truncate_text(text, max_length=200):
    """
    Truncate text to max_length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def safe_decode(bytes_data, encoding='utf-8', errors='ignore'):
    """
    Safely decode bytes to string, handling encoding errors.
    
    Args:
        bytes_data: Bytes to decode
        encoding: Encoding to use
        errors: Error handling strategy
        
    Returns:
        Decoded string
    """
    if isinstance(bytes_data, str):
        return bytes_data
    
    try:
        return bytes_data.decode(encoding, errors=errors)
    except (UnicodeDecodeError, AttributeError):
        try:
            return bytes_data.decode('latin-1', errors=errors)
        except:
            return str(bytes_data)


def format_probability(prob):
    """
    Format probability as percentage string.
    
    Args:
        prob: Probability (0.0 to 1.0)
        
    Returns:
        Formatted string like "95.2%"
    """
    return f"{prob * 100:.1f}%"


