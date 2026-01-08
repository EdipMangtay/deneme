"""
Train model from scratch with all combined data.
"""
import os
import sys
import shutil
import logging
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from src.train_with_all_data import combine_all_datasets
from src.train_or_prepare import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backup_existing_checkpoint():
    """Backup existing checkpoint before training from scratch."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'checkpoint')
    if os.path.exists(checkpoint_dir):
        backup_dir = os.path.join(
            os.path.dirname(__file__), 
            'artifacts', 
            f'checkpoint_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        logger.info(f"Backing up existing checkpoint to {backup_dir}...")
        shutil.copytree(checkpoint_dir, backup_dir)
        logger.info("✅ Backup completed")
        return backup_dir
    return None


def train_from_scratch():
    """Train model from scratch with all combined data."""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 TRAINING FROM SCRATCH")
    logger.info("=" * 70)
    
    # Backup existing checkpoint
    backup_dir = backup_existing_checkpoint()
    
    # Remove existing checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'checkpoint')
    if os.path.exists(checkpoint_dir):
        logger.info("Removing existing checkpoint for fresh training...")
        shutil.rmtree(checkpoint_dir)
        logger.info("✅ Old checkpoint removed")
    
    # Combine all datasets (including Gmail data if exists)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Combining all datasets")
    logger.info("=" * 70)
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    datasets = []
    
    # 1. w1998.csv
    w1998_path = os.path.join(base_dir, '_w1998.csv')
    if os.path.exists(w1998_path):
        import pandas as pd
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
        import pandas as pd
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
        import pandas as pd
        logger.info(f"Loading {kucev_path}...")
        df_kucev = pd.read_csv(kucev_path)
        df_kucev['text'] = df_kucev['title'].fillna('') + ' ' + df_kucev['text'].fillna('')
        df_kucev = df_kucev.rename(columns={'type': 'label'})
        df_kucev['label'] = df_kucev['label'].map({'not spam': 0, 'spam': 1})
        df_kucev = df_kucev[['text', 'label']].copy()
        df_kucev.dropna(inplace=True)
        logger.info(f"  Loaded {len(df_kucev)} emails from kucev")
        datasets.append(df_kucev)
    
    # 4. Gmail dataset (if exists)
    gmail_path = os.path.join(os.path.dirname(__file__), 'data', 'gmail_dataset.csv')
    if os.path.exists(gmail_path):
        import pandas as pd
        logger.info(f"Loading {gmail_path}...")
        df_gmail = pd.read_csv(gmail_path)
        if 'text' in df_gmail.columns and 'label' in df_gmail.columns:
            df_gmail = df_gmail[['text', 'label']].copy()
            df_gmail.dropna(inplace=True)
            logger.info(f"  Loaded {len(df_gmail)} emails from Gmail")
            datasets.append(df_gmail)
    
    if not datasets:
        raise ValueError("No datasets found!")
    
    # Combine all
    import pandas as pd
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
    
    # Train from scratch
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Training model from scratch")
    logger.info("=" * 70)
    logger.info("This will take approximately 10-15 minutes...")
    
    train_model(combined_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING FROM SCRATCH COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"\nModel saved to: {checkpoint_dir}")
    if backup_dir:
        logger.info(f"Previous checkpoint backed up to: {backup_dir}")


if __name__ == "__main__":
    try:
        train_from_scratch()
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


