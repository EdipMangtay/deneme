"""
Email anonymization module - removes sensitive information.
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def anonymize_email(text):
    """
    Anonymize sensitive information in email text.
    Removes: emails, URLs, phone numbers, collapses whitespace.
    
    Args:
        text: Email text to anonymize
        
    Returns:
        Anonymized text
    """
    if not text:
        return ""
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '[PHONE]', text)
    text = re.sub(r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '[PHONE]', text)
    
    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def anonymize_email_data(email_data):
    """
    Anonymize email data dictionary.
    
    Args:
        email_data: dict with keys: subject, text, body, etc.
        
    Returns:
        Anonymized email data dict
    """
    anonymized = email_data.copy()
    
    if 'subject' in anonymized:
        anonymized['subject'] = anonymize_email(anonymized['subject'])
    
    if 'text' in anonymized:
        anonymized['text'] = anonymize_email(anonymized['text'])
    
    if 'body' in anonymized:
        anonymized['body'] = anonymize_email(anonymized['body'])
    
    return anonymized


