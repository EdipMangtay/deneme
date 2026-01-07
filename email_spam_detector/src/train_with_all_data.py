"""
Train model with all existing CSV files (w1998, abdallah, kucev).
Combines all datasets and trains a unified model.
"""
import os
import sys
import pandas as pd
import argparse
import logging

# Add parent directory to path to import model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model import Model
# Import train_model function
sys.path.insert(0, os.path.dirname(__file__))
from train_or_prepare import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_all_datasets(base_dir=None):
    """
    Combine all CSV files (w1998, abdallah, kucev) into one dataset.
    
    Returns:
        Combined DataFrame with columns: text, label
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    
    datasets = []
    
    # 1. _w1998.csv: text, spam (0/1)
    w1998_path = os.path.join(base_dir, '_w1998.csv')
    if os.path.exists(w1998_path):
        logger.info(f"Loading {w1998_path}...")
        df_w1998 = pd.read_csv(w1998_path)
        df_w1998 = df_w1998.rename(columns={'spam': 'label'})
        df_w1998 = df_w1998[['text', 'label']].copy()
        df_w1998.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_w1998)} emails from w1998")
        datasets.append(df_w1998)
    
    # 2. abdallah.csv: Category, Message (ham/spam)
    abdallah_path = os.path.join(base_dir, 'abdallah.csv')
    if os.path.exists(abdallah_path):
        logger.info(f"Loading {abdallah_path}...")
        df_abdallah = pd.read_csv(abdallah_path)
        df_abdallah = df_abdallah.rename(columns={'Message': 'text', 'Category': 'label'})
        # Convert ham/spam to 0/1
        df_abdallah['label'] = df_abdallah['label'].map({'ham': 0, 'spam': 1})
        df_abdallah = df_abdallah[['text', 'label']].copy()
        df_abdallah.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_abdallah)} emails from abdallah")
        datasets.append(df_abdallah)
    
    # 3. kucev.csv: title, text, type (not spam/spam)
    kucev_path = os.path.join(base_dir, 'kucev.csv')
    if os.path.exists(kucev_path):
        logger.info(f"Loading {kucev_path}...")
        df_kucev = pd.read_csv(kucev_path)
        # Combine title and text
        df_kucev['text'] = df_kucev['title'].fillna('') + ' ' + df_kucev['text'].fillna('')
        df_kucev = df_kucev.rename(columns={'type': 'label'})
        # Convert not spam/spam to 0/1
        df_kucev['label'] = df_kucev['label'].map({'not spam': 0, 'spam': 1})
        df_kucev = df_kucev[['text', 'label']].copy()
        df_kucev.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_kucev)} emails from kucev")
        datasets.append(df_kucev)
    
    if not datasets:
        raise ValueError("No datasets found! Please ensure CSV files exist in parent directory.")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    
    # Ensure labels are 0/1
    combined_df['label'] = combined_df['label'].astype(int)
    combined_df = combined_df[combined_df['label'].isin([0, 1])]
    
    logger.info(f"\nCombined dataset:")
    logger.info(f"  Total emails: {len(combined_df)}")
    logger.info(f"  Label distribution:\n{combined_df['label'].value_counts()}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Train model with all existing CSV datasets")
    parser.add_argument(
        '--model',
        type=str,
        default='distilbert/distilbert-base-uncased',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for checkpoint (default: artifacts/checkpoint)'
    )
    parser.add_argument(
        '--save-combined',
        type=str,
        default=None,
        help='Save combined dataset to CSV file'
    )
    
    args = parser.parse_args()
    
    # Combine all datasets
    combined_df = combine_all_datasets()
    
    # Save combined dataset if requested
    if args.save_combined:
        combined_df.to_csv(args.save_combined, index=False)
        logger.info(f"Combined dataset saved to {args.save_combined}")
    
    # Save to temporary file for training
    temp_dataset = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_dataset.csv')
    os.makedirs(os.path.dirname(temp_dataset), exist_ok=True)
    combined_df.to_csv(temp_dataset, index=False)
    
    # Train model
    logger.info(f"\nStarting training with {len(combined_df)} emails...")
    train_model(temp_dataset, args.model, args.output)
    
    logger.info("\n✅ Model training completed!")
    logger.info(f"Model saved to: {args.output or 'artifacts/checkpoint'}")


if __name__ == "__main__":
    main()

