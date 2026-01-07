"""
Training script that uses the existing Model class.
This script trains the model and saves it to artifacts/checkpoint.
"""
import os
import sys
import pandas as pd
import argparse
import logging

# Add parent directory to path to import model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(dataset_path, model_name="distilbert/distilbert-base-uncased", output_dir=None):
    """
    Train the spam detection model using existing Model class.
    
    Args:
        dataset_path: Path to CSV file with columns: text, label (or subject, text, label)
        model_name: HuggingFace model name
        output_dir: Directory to save checkpoint (default: artifacts/checkpoint)
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', 'artifacts', 'checkpoint'
        )
    
    # Convert to absolute path
    output_dir = os.path.abspath(output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Handle different column names
    if 'text' in df.columns and 'label' in df.columns:
        # Standard format
        df_processed = df[['text', 'label']].copy()
        df_processed = df_processed.rename(columns={'text': 'sentence', 'label': 'category'})
    elif 'subject' in df.columns and 'text' in df.columns and 'label' in df.columns:
        # Combine subject and text
        df_processed = df.copy()
        df_processed['sentence'] = df_processed['subject'].fillna('') + ' ' + df_processed['text'].fillna('')
        df_processed = df_processed[['sentence', 'label']].copy()
        df_processed = df_processed.rename(columns={'label': 'category'})
    else:
        raise ValueError(
            f"Dataset must have columns: 'text' and 'label', or 'subject', 'text', and 'label'.\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Remove NaN values
    df_processed.dropna(inplace=True)
    
    # Ensure labels are 0/1
    if df_processed['category'].dtype == 'object':
        # Convert string labels to 0/1
        unique_labels = df_processed['category'].unique()
        if len(unique_labels) != 2:
            raise ValueError(f"Expected 2 unique labels, found: {unique_labels}")
        label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        df_processed['category'] = df_processed['category'].map(label_map)
        logger.info(f"Label mapping: {label_map}")
    
    # Ensure labels are 0 and 1
    unique_labels = sorted(df_processed['category'].unique())
    if set(unique_labels) != {0, 1}:
        raise ValueError(f"Labels must be 0 and 1, found: {unique_labels}")
    
    labels = [0, 1]
    
    logger.info(f"Dataset shape: {df_processed.shape}")
    logger.info(f"Label distribution:\n{df_processed['category'].value_counts()}")
    
    # Initialize and train model
    logger.info(f"Initializing model: {model_name}")
    model = Model(model_name)
    
    # Temporarily change output_dir for training
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        logger.info(f"Training model (this may take several minutes)...")
        model.train_evaluate(labels, df_processed)
        logger.info("Training completed successfully!")
    finally:
        os.chdir(original_cwd)
    
    logger.info(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train spam detection model")
    default_dataset = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=default_dataset,
        help='Path to dataset CSV file'
    )
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        logger.info("Please build the dataset first using: python -m src.dataset_builder")
        sys.exit(1)
    
    train_model(args.dataset, args.model, args.output)


if __name__ == "__main__":
    main()

