"""
Email parsing module - extracts text from email messages.
"""
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import logging
from .utils import safe_decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_mime_header(header_value):
    """
    Decode MIME header value.
    
    Args:
        header_value: Header value (can be bytes or string)
        
    Returns:
        Decoded string
    """
    if not header_value:
        return ""
    
    try:
        decoded_parts = decode_header(header_value)
        decoded_string = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    decoded_string += part.decode(encoding, errors='ignore')
                else:
                    decoded_string += safe_decode(part)
            else:
                decoded_string += str(part)
        return decoded_string
    except Exception as e:
        logger.warning(f"Error decoding header: {e}")
        return str(header_value)


def extract_text_from_html(html_content):
    """
    Extract plain text from HTML email body.
    
    Args:
        html_content: HTML string
        
    Returns:
        Plain text extracted from HTML
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text and clean up
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        logger.warning(f"Error parsing HTML: {e}")
        return html_content


def parse_email_message(msg_bytes):
    """
    Parse email message bytes into structured data.
    
    Args:
        msg_bytes: Raw email message bytes
        
    Returns:
        dict with keys: subject, date, body, text, is_html
    """
    try:
        msg = email.message_from_bytes(msg_bytes)
    except Exception as e:
        logger.error(f"Error parsing email message: {e}")
        return None
    
    # Extract subject
    subject = decode_mime_header(msg.get('Subject', ''))
    
    # Extract date
    date = msg.get('Date', '')
    
    # Extract body
    body_text = ""
    body_html = ""
    is_html = False
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
            
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    if content_type == "text/plain":
                        body_text = safe_decode(payload)
                    elif content_type == "text/html":
                        body_html = safe_decode(payload)
                        is_html = True
            except Exception as e:
                logger.warning(f"Error extracting part: {e}")
                continue
    else:
        # Single part message
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                if content_type == "text/html":
                    body_html = safe_decode(payload)
                    is_html = True
                else:
                    body_text = safe_decode(payload)
        except Exception as e:
            logger.warning(f"Error extracting payload: {e}")
    
    # Prefer plain text, but use HTML if that's all we have
    if body_text:
        text = body_text
    elif body_html:
        text = extract_text_from_html(body_html)
        is_html = True
    else:
        text = ""
    
    return {
        'subject': subject or "",
        'date': date or "",
        'body': text,
        'text': text,  # Alias for consistency
        'is_html': is_html
    }



