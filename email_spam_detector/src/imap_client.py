"""
IMAP client for fetching emails from Gmail.
"""
import imaplib
import email
import os
import sys
import argparse
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from .email_parser import parse_email_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IMAPClient:
    """
    IMAP client for connecting to Gmail and fetching emails.
    """
    
    def __init__(self, email_address, app_password, imap_server="imap.gmail.com"):
        """
        Initialize IMAP client.
        
        Args:
            email_address: Gmail email address
            app_password: Gmail App Password
            imap_server: IMAP server (default: imap.gmail.com)
        """
        self.email_address = email_address
        self.app_password = app_password
        self.imap_server = imap_server
        self.imap = None
        self.connected = False
    
    def connect(self):
        """Connect to IMAP server."""
        try:
            logger.info(f"Connecting to {self.imap_server}...")
            self.imap = imaplib.IMAP4_SSL(self.imap_server)
            self.imap.login(self.email_address, self.app_password)
            self.connected = True
            logger.info("Connected successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to IMAP server: {e}")
            self.connected = False
            raise
    
    def disconnect(self):
        """Disconnect from IMAP server."""
        if self.imap and self.connected:
            try:
                # Only close if a mailbox is selected
                try:
                    self.imap.close()
                except:
                    pass  # Ignore if no mailbox is selected
                self.imap.logout()
                self.connected = False
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
                # Force logout even if close fails
                try:
                    self.imap.logout()
                except:
                    pass
                self.connected = False
    
    def list_folders(self):
        """
        List all available folders.
        
        Returns:
            List of folder names
        """
        if not self.connected:
            self.connect()
        
        try:
            status, folders = self.imap.list()
            if status == 'OK':
                folder_names = []
                for folder in folders:
                    # Parse folder name from response
                    folder_str = folder.decode('utf-8')
                    # Extract folder name (format: '(\HasNoChildren) "/" "INBOX"')
                    parts = folder_str.split('"')
                    if len(parts) >= 3:
                        folder_name = parts[-2]
                        folder_names.append(folder_name)
                return folder_names
            return []
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []
    
    def find_spam_folder(self):
        """
        Find spam folder by searching for folders containing "Spam".
        
        Returns:
            Spam folder name or None
        """
        folders = self.list_folders()
        for folder in folders:
            if 'spam' in folder.lower():
                logger.info(f"Found spam folder: {folder}")
                return folder
        logger.warning("Spam folder not found")
        return None
    
    def fetch_emails(self, folder="INBOX", limit=20):
        """
        Fetch emails from specified folder.
        
        Args:
            folder: Folder name (default: INBOX)
            limit: Maximum number of emails to fetch
            
        Returns:
            List of email data dictionaries
        """
        if not self.connected:
            self.connect()
        
        emails = []
        
        try:
            # Select folder
            status, messages = self.imap.select(folder)
            if status != 'OK':
                logger.error(f"Error selecting folder {folder}: {status}")
                return emails
            
            # Search for all emails
            status, message_ids = self.imap.search(None, 'ALL')
            if status != 'OK':
                logger.error(f"Error searching emails: {status}")
                return emails
            
            # Get message IDs
            message_id_list = message_ids[0].split()
            
            # Fetch most recent emails (reverse order)
            message_id_list = message_id_list[-limit:] if len(message_id_list) > limit else message_id_list
            message_id_list.reverse()  # Most recent first
            
            logger.info(f"Fetching {len(message_id_list)} emails from {folder}...")
            
            for msg_id in message_id_list:
                try:
                    # Fetch email
                    status, msg_data = self.imap.fetch(msg_id, '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    # Parse email
                    email_data = parse_email_message(msg_data[0][1])
                    if email_data:
                        email_data['folder'] = folder
                        email_data['message_id'] = msg_id.decode('utf-8')
                        emails.append(email_data)
                
                except Exception as e:
                    logger.warning(f"Error fetching email {msg_id}: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(emails)} emails from {folder}")
            return emails
        
        except Exception as e:
            logger.error(f"Error fetching emails from {folder}: {e}")
            return emails
    
    def fetch_inbox_and_spam(self, inbox_limit=20, spam_limit=20, spam_folder=None):
        """
        Fetch emails from both INBOX and Spam folders.
        
        Args:
            inbox_limit: Number of emails to fetch from INBOX
            spam_limit: Number of emails to fetch from Spam folder
            spam_folder: Spam folder name (auto-detected if None)
            
        Returns:
            dict with keys: 'inbox' (label=0) and 'spam' (label=1)
        """
        if not self.connected:
            self.connect()
        
        result = {
            'inbox': [],
            'spam': []
        }
        
        # Fetch inbox emails
        inbox_emails = self.fetch_emails("INBOX", inbox_limit)
        for email_data in inbox_emails:
            email_data['label'] = 0  # NOT SPAM
        result['inbox'] = inbox_emails
        
        # Find and fetch spam folder
        if spam_folder is None:
            spam_folder = self.find_spam_folder()
            if spam_folder is None:
                spam_folder = "[Gmail]/Spam"  # Default Gmail spam folder
        
        spam_emails = self.fetch_emails(spam_folder, spam_limit)
        for email_data in spam_emails:
            email_data['label'] = 1  # SPAM
        result['spam'] = spam_emails
        
        return result
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def main():
    """CLI entry point for IMAP client."""
    parser = argparse.ArgumentParser(description='Gmail IMAP Client CLI')
    parser.add_argument(
        '--list-folders',
        action='store_true',
        help='List all available folders'
    )
    parser.add_argument(
        '--fetch-inbox',
        type=int,
        metavar='N',
        help='Fetch N emails from INBOX'
    )
    parser.add_argument(
        '--fetch-spam',
        type=int,
        metavar='N',
        help='Fetch N emails from Spam folder'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    # Try to find .env file in parent directory (email_spam_detector/)
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    env_path = os.path.abspath(env_path)
    
    # Try multiple locations and manual parsing
    env_loaded = False
    for path in [env_path, '.env', os.path.join('email_spam_detector', '.env')]:
        if os.path.exists(path):
            try:
                load_dotenv(path, override=True)
                # Also manually parse to ensure it works
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                env_loaded = True
                break
            except Exception as e:
                logger.warning(f"Error loading .env from {path}: {e}")
                continue
    
    if not env_loaded:
        # Try default load_dotenv
        load_dotenv(override=True)
    
    email_address = os.getenv('GMAIL_EMAIL')
    app_password = os.getenv('GMAIL_APP_PASSWORD')
    imap_server = os.getenv('IMAP_SERVER', 'imap.gmail.com')
    spam_folder = os.getenv('SPAM_FOLDER')
    
    # Debug output
    if not email_address or not app_password:
        print("ERROR: GMAIL_EMAIL and GMAIL_APP_PASSWORD must be set in .env file")
        print(f"   Looking for .env at: {env_path}")
        print(f"   .env exists: {os.path.exists(env_path)}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   GMAIL_EMAIL found: {bool(email_address)}")
        print(f"   GMAIL_APP_PASSWORD found: {bool(app_password)}")
        # Try to read .env file directly
        if os.path.exists(env_path):
            print(f"\n   Reading .env file directly:")
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if 'GMAIL_EMAIL' in line or 'GMAIL_APP_PASSWORD' in line:
                            # Mask password
                            if 'GMAIL_APP_PASSWORD' in line:
                                parts = line.split('=', 1)
                                if len(parts) == 2:
                                    print(f"   Line {i}: {parts[0]}={parts[1][:4]}...")
                                else:
                                    print(f"   Line {i}: {line.strip()}")
                            else:
                                print(f"   Line {i}: {line.strip()}")
            except Exception as e:
                print(f"   Error reading file: {e}")
        sys.exit(1)
    
    try:
        with IMAPClient(email_address, app_password, imap_server) as client:
            if args.list_folders:
                print("\nListing Gmail folders...\n")
                folders = client.list_folders()
                spam_folder_detected = client.find_spam_folder()
                
                print(f"Found {len(folders)} folders:\n")
                for folder in folders:
                    marker = " <- SPAM FOLDER" if spam_folder_detected and folder == spam_folder_detected else ""
                    print(f"  - {folder}{marker}")
                print()
                
                if spam_folder_detected:
                    print(f"SUCCESS: Spam folder detected: {spam_folder_detected}\n")
                else:
                    print("WARNING: Spam folder not found\n")
            
            elif args.fetch_inbox is not None:
                print(f"\nFetching {args.fetch_inbox} emails from INBOX...\n")
                emails = client.fetch_emails("INBOX", args.fetch_inbox)
                
                print(f"SUCCESS: Fetched {len(emails)} emails:\n")
                for i, email_data in enumerate(emails, 1):
                    subject = email_data.get('subject', 'No Subject')
                    date = email_data.get('date', 'Unknown date')
                    snippet = email_data.get('text', '')[:100].replace('\n', ' ')
                    # Encode to handle non-ASCII characters for Windows terminal
                    try:
                        subject_safe = subject.encode('ascii', 'ignore').decode('ascii') or '[Non-ASCII Subject]'
                        snippet_safe = snippet.encode('ascii', 'ignore').decode('ascii') or '[Non-ASCII Content]'
                    except:
                        subject_safe = '[Subject]'
                        snippet_safe = '[Content]'
                    print(f"{i}. {subject_safe}")
                    print(f"   Date: {date}")
                    print(f"   Snippet: {snippet_safe}...")
                    print()
            
            elif args.fetch_spam is not None:
                # Auto-detect spam folder if not set
                if spam_folder is None:
                    spam_folder = client.find_spam_folder()
                    if spam_folder is None:
                        print("ERROR: Spam folder not found. Please set SPAM_FOLDER in .env file")
                        sys.exit(1)
                
                print(f"\nFetching {args.fetch_spam} emails from Spam folder ({spam_folder})...\n")
                emails = client.fetch_emails(spam_folder, args.fetch_spam)
                
                print(f"SUCCESS: Fetched {len(emails)} emails:\n")
                for i, email_data in enumerate(emails, 1):
                    subject = email_data.get('subject', 'No Subject')
                    date = email_data.get('date', 'Unknown date')
                    snippet = email_data.get('text', '')[:100].replace('\n', ' ')
                    # Encode to handle non-ASCII characters for Windows terminal
                    try:
                        subject_safe = subject.encode('ascii', 'ignore').decode('ascii') or '[Non-ASCII Subject]'
                        snippet_safe = snippet.encode('ascii', 'ignore').decode('ascii') or '[Non-ASCII Content]'
                    except:
                        subject_safe = '[Subject]'
                        snippet_safe = '[Content]'
                    print(f"{i}. {subject_safe}")
                    print(f"   Date: {date}")
                    print(f"   Snippet: {snippet_safe}...")
                    print()
            
            else:
                parser.print_help()
    
    except Exception as e:
        print(f"\nERROR: {e}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()

