"""
Fine-tune model with Gmail data and test.
Complete pipeline: fetch emails -> combine datasets -> fine-tune -> test
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from src.dataset_builder import build_dataset
from src.train_with_all_data import combine_all_datasets
from src.train_or_prepare import train_model
from src.ml_adapter import MLAdapter
from src.imap_client import IMAPClient
from src.utils import combine_subject_and_body
from src.anonymize import anonymize_email_data
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def fetch_gmail_emails(inbox_limit=500, spam_limit=500):
    """Fetch emails from Gmail."""
    logger.info("=" * 70)
    logger.info("STEP 1: Fetching emails from Gmail")
    logger.info("=" * 70)
    
    email_address = os.getenv('GMAIL_EMAIL')
    app_password = os.getenv('GMAIL_APP_PASSWORD')
    spam_folder = os.getenv('SPAM_FOLDER')
    
    if not email_address or not app_password:
        raise ValueError("GMAIL_EMAIL and GMAIL_APP_PASSWORD must be set in .env file")
    
    # Build dataset from Gmail
    gmail_dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'gmail_dataset.csv')
    build_dataset(
        inbox_limit=inbox_limit,
        spam_limit=spam_limit,
        output_path=gmail_dataset_path,
        append=False
    )
    
    logger.info(f"✅ Fetched Gmail emails and saved to {gmail_dataset_path}")
    return gmail_dataset_path


def combine_all_datasets_with_gmail():
    """Combine w1998, abdallah, kucev, and Gmail datasets."""
    logger.info("=" * 70)
    logger.info("STEP 2: Combining all datasets")
    logger.info("=" * 70)
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    datasets = []
    
    # 1. w1998.csv
    w1998_path = os.path.join(base_dir, '_w1998.csv')
    if os.path.exists(w1998_path):
        logger.info(f"Loading {w1998_path}...")
        df_w1998 = pd.read_csv(w1998_path)
        df_w1998 = df_w1998.rename(columns={'spam': 'label'})
        df_w1998 = df_w1998[['text', 'label']].copy()
        df_w1998.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_w1998)} emails from w1998")
        datasets.append(df_w1998)
    
    # 2. abdallah.csv
    abdallah_path = os.path.join(base_dir, 'abdallah.csv')
    if os.path.exists(abdallah_path):
        logger.info(f"Loading {abdallah_path}...")
        df_abdallah = pd.read_csv(abdallah_path)
        df_abdallah = df_abdallah.rename(columns={'Message': 'text', 'Category': 'label'})
        df_abdallah['label'] = df_abdallah['label'].map({'ham': 0, 'spam': 1})
        df_abdallah = df_abdallah[['text', 'label']].copy()
        df_abdallah.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_abdallah)} emails from abdallah")
        datasets.append(df_abdallah)
    
    # 3. kucev.csv
    kucev_path = os.path.join(base_dir, 'kucev.csv')
    if os.path.exists(kucev_path):
        logger.info(f"Loading {kucev_path}...")
        df_kucev = pd.read_csv(kucev_path)
        df_kucev['text'] = df_kucev['title'].fillna('') + ' ' + df_kucev['text'].fillna('')
        df_kucev = df_kucev.rename(columns={'type': 'label'})
        df_kucev['label'] = df_kucev['label'].map({'not spam': 0, 'spam': 1})
        df_kucev = df_kucev[['text', 'label']].copy()
        df_kucev.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_kucev)} emails from kucev")
        datasets.append(df_kucev)
    
    # 4. Gmail dataset
    gmail_path = os.path.join(os.path.dirname(__file__), 'data', 'gmail_dataset.csv')
    if os.path.exists(gmail_path):
        logger.info(f"Loading {gmail_path}...")
        df_gmail = pd.read_csv(gmail_path)
        # Gmail dataset already has 'text' and 'label' columns
        if 'text' in df_gmail.columns and 'label' in df_gmail.columns:
            df_gmail = df_gmail[['text', 'label']].copy()
            df_gmail.dropna(inplace=True)
            logger.info(f"  Loaded {len(df_gmail)} emails from Gmail")
            datasets.append(df_gmail)
    
    if not datasets:
        raise ValueError("No datasets found!")
    
    # Combine all
    combined_df = pd.concat(datasets, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    combined_df['label'] = combined_df['label'].astype(int)
    combined_df = combined_df[combined_df['label'].isin([0, 1])]
    
    logger.info(f"\n✅ Combined dataset:")
    logger.info(f"  Total emails: {len(combined_df)}")
    logger.info(f"  Label distribution:\n{combined_df['label'].value_counts()}")
    
    # Save combined dataset
    combined_path = os.path.join(os.path.dirname(__file__), 'data', 'final_combined_dataset.csv')
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"  Saved to: {combined_path}")
    
    return combined_path


def fine_tune_model(dataset_path, checkpoint_dir=None):
    """Fine-tune model from existing checkpoint."""
    logger.info("=" * 70)
    logger.info("STEP 3: Fine-tuning model")
    logger.info("=" * 70)
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'checkpoint')
    
    # Check if checkpoint exists
    nested_checkpoint = os.path.join(checkpoint_dir, 'checkpoint', 'checkpoint-958')
    if not os.path.exists(nested_checkpoint):
        logger.warning("No existing checkpoint found. Training from scratch...")
        train_model(dataset_path)
        return
    
    logger.info(f"Fine-tuning from checkpoint: {nested_checkpoint}")
    logger.info("This will continue training from the existing model...")
    
    # Train (will continue from checkpoint if model.py supports it)
    train_model(dataset_path, output_dir=checkpoint_dir)
    
    logger.info("✅ Fine-tuning completed!")


def test_model(test_limit=50):
    """Test model on Gmail emails."""
    logger.info("=" * 70)
    logger.info("STEP 4: Testing model on Gmail emails")
    logger.info("=" * 70)
    
    email_address = os.getenv('GMAIL_EMAIL')
    app_password = os.getenv('GMAIL_APP_PASSWORD')
    spam_folder = os.getenv('SPAM_FOLDER')
    
    # Load model
    logger.info("Loading model...")
    adapter = MLAdapter()
    adapter.load_model()
    logger.info("✅ Model loaded")
    
    # Fetch test emails
    logger.info(f"Fetching {test_limit} test emails (INBOX + SPAM)...")
    with IMAPClient(email_address, app_password) as client:
        emails_data = client.fetch_inbox_and_spam(
            inbox_limit=test_limit,
            spam_limit=test_limit,
            spam_folder=spam_folder
        )
    
    all_emails = emails_data['inbox'] + emails_data['spam']
    logger.info(f"Fetched {len(all_emails)} test emails")
    
    # Test
    stats = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'true_positive': 0,
        'true_negative': 0,
        'false_positive': 0,
        'false_negative': 0
    }
    
    results = []
    
    logger.info("Classifying emails...")
    for i, email_data in enumerate(all_emails):
        if i % 10 == 0:
            logger.info(f"  Processing {i}/{len(all_emails)}...")
        
        text = combine_subject_and_body(
            email_data.get('subject', ''),
            email_data.get('text', '')
        )
        
        if not text.strip():
            continue
        
        ground_truth = email_data.get('label', 0)
        is_actually_spam = ground_truth == 1
        
        try:
            prediction = adapter.predict_email(text)
            predicted_spam = prediction['label'] == 1
            
            stats['total'] += 1
            if predicted_spam == is_actually_spam:
                stats['correct'] += 1
                if predicted_spam:
                    stats['true_positive'] += 1
                else:
                    stats['true_negative'] += 1
            else:
                stats['incorrect'] += 1
                if predicted_spam:
                    stats['false_positive'] += 1
                else:
                    stats['false_negative'] += 1
            
            results.append({
                'subject': email_data.get('subject', 'No Subject')[:50],
                'predicted': 'SPAM' if predicted_spam else 'NOT SPAM',
                'actual': 'SPAM' if is_actually_spam else 'NOT SPAM',
                'correct': predicted_spam == is_actually_spam,
                'confidence': prediction['probability']
            })
        except Exception as e:
            logger.warning(f"Error classifying email: {e}")
    
    # Calculate metrics
    if stats['total'] > 0:
        accuracy = stats['correct'] / stats['total']
        precision = stats['true_positive'] / (stats['true_positive'] + stats['false_positive']) if (stats['true_positive'] + stats['false_positive']) > 0 else 0
        recall = stats['true_positive'] / (stats['true_positive'] + stats['false_negative']) if (stats['true_positive'] + stats['false_negative']) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        accuracy = precision = recall = f1_score = 0
    
    # Print results
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total emails tested: {stats['total']}")
    logger.info(f"Correct: {stats['correct']}")
    logger.info(f"Incorrect: {stats['incorrect']}")
    logger.info("")
    logger.info("METRICS:")
    logger.info(f"  Accuracy:  {accuracy * 100:.2f}%")
    logger.info(f"  Precision: {precision * 100:.2f}%")
    logger.info(f"  Recall:    {recall * 100:.2f}%")
    logger.info(f"  F1 Score:  {f1_score * 100:.2f}%")
    logger.info("")
    logger.info("CONFUSION MATRIX:")
    logger.info(f"  True Negative (NOT SPAM -> NOT SPAM):  {stats['true_negative']}")
    logger.info(f"  False Positive (NOT SPAM -> SPAM):     {stats['false_positive']}")
    logger.info(f"  False Negative (SPAM -> NOT SPAM):     {stats['false_negative']}")
    logger.info(f"  True Positive (SPAM -> SPAM):           {stats['true_positive']}")
    logger.info("")
    logger.info("SAMPLE RESULTS (first 10):")
    for i, result in enumerate(results[:10], 1):
        status = "✓" if result['correct'] else "✗"
        logger.info(f"  {i}. {status} {result['subject']}")
        logger.info(f"     Predicted: {result['predicted']} ({result['confidence']*100:.1f}%) | Actual: {result['actual']}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'stats': stats,
        'results': results
    }


def main():
    """Main pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 FINE-TUNING AND TESTING PIPELINE")
    logger.info("=" * 70)
    logger.info("\nThis will:")
    logger.info("  1. Fetch 1000 emails from Gmail (500 inbox + 500 spam)")
    logger.info("  2. Combine with w1998, abdallah, kucev datasets")
    logger.info("  3. Fine-tune model from existing checkpoint")
    logger.info("  4. Test model and show results")
    logger.info("\n" + "=" * 70 + "\n")
    
    try:
        # Step 1: Fetch Gmail emails
        gmail_dataset = fetch_gmail_emails(inbox_limit=500, spam_limit=500)
        
        # Step 2: Combine all datasets
        combined_dataset = combine_all_datasets_with_gmail()
        
        # Step 3: Fine-tune
        fine_tune_model(combined_dataset)
        
        # Step 4: Test
        test_results = test_model(test_limit=50)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"\nFinal Test Accuracy: {test_results['accuracy'] * 100:.2f}%")
        logger.info(f"Precision: {test_results['precision'] * 100:.2f}%")
        logger.info(f"Recall: {test_results['recall'] * 100:.2f}%")
        logger.info(f"F1 Score: {test_results['f1_score'] * 100:.2f}%")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



