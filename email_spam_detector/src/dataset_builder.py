"""
Dataset builder - fetches emails from Gmail and builds dataset CSV.
"""
import os
import sys
import pandas as pd
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

from .imap_client import IMAPClient
from .anonymize import anonymize_email_data
from .utils import combine_subject_and_body

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataset(inbox_limit=50, spam_limit=50, output_path=None, append=False):
    """
    Build dataset by fetching emails from Gmail.
    
    Args:
        inbox_limit: Number of emails to fetch from INBOX
        spam_limit: Number of emails to fetch from Spam folder
        output_path: Path to output CSV file
        append: If True, append to existing dataset; if False, overwrite
    """
    # Load environment variables
    load_dotenv()
    
    email_address = os.getenv('GMAIL_EMAIL')
    app_password = os.getenv('GMAIL_APP_PASSWORD')
    spam_folder = os.getenv('SPAM_FOLDER')
    
    if not email_address or not app_password:
        raise ValueError(
            "GMAIL_EMAIL and GMAIL_APP_PASSWORD must be set in .env file"
        )
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'dataset.csv'
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Fetching emails: {inbox_limit} from INBOX, {spam_limit} from Spam")
    
    # Fetch emails
    with IMAPClient(email_address, app_password) as client:
        emails_data = client.fetch_inbox_and_spam(
            inbox_limit=inbox_limit,
            spam_limit=spam_limit,
            spam_folder=spam_folder
        )
    
    # Combine and process emails
    all_emails = emails_data['inbox'] + emails_data['spam']
    logger.info(f"Fetched {len(all_emails)} total emails")
    
    # Build dataset rows
    dataset_rows = []
    for email_data in all_emails:
        # Anonymize
        anonymized = anonymize_email_data(email_data)
        
        # Combine subject and body
        combined_text = combine_subject_and_body(
            anonymized.get('subject', ''),
            anonymized.get('text', '')
        )
        
        # Skip empty emails
        if not combined_text.strip():
            continue
        
        row = {
            'subject': anonymized.get('subject', ''),
            'text': combined_text,
            'label': email_data.get('label', 0),
            'source_folder': email_data.get('folder', ''),
            'date': email_data.get('date', '')
        }
        dataset_rows.append(row)
    
    # Create DataFrame
    df_new = pd.DataFrame(dataset_rows)
    
    # Load existing dataset if appending
    if append and os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates based on text
            df = df.drop_duplicates(subset=['text'], keep='last')
            logger.info(f"Appended to existing dataset. Total rows: {len(df)}")
        except Exception as e:
            logger.warning(f"Error loading existing dataset: {e}. Creating new dataset.")
            df = df_new
    else:
        df = df_new
    
    # Save dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build dataset from Gmail emails")
    parser.add_argument(
        '--inbox',
        type=int,
        default=50,
        help='Number of emails to fetch from INBOX (default: 50)'
    )
    parser.add_argument(
        '--spam',
        type=int,
        default=50,
        help='Number of emails to fetch from Spam folder (default: 50)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: data/dataset.csv)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing dataset instead of overwriting'
    )
    
    args = parser.parse_args()
    
    try:
        build_dataset(
            inbox_limit=args.inbox,
            spam_limit=args.spam,
            output_path=args.output,
            append=args.append
        )
        logger.info("Dataset building completed successfully!")
    except Exception as e:
        logger.error(f"Error building dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



