"""
ML Adapter - Wraps the existing Model class for inference
Reuses the exact training logic from model.py
"""
import os
import sys
import logging

# Lazy imports for ML libraries (only load when needed)
torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None

def _import_ml_libs():
    """Lazy import ML libraries."""
    global torch, AutoModelForSequenceClassification, AutoTokenizer
    if torch is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"ML libraries not installed. Please run: pip install torch transformers\n"
                f"Original error: {e}"
            )

# Add parent directory to path to import model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from model import Model
except ImportError:
    Model = None  # Will be handled in __init__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLAdapter:
    """
    Adapter that wraps the existing Model class and provides inference capabilities.
    Loads trained model from checkpoint directory.
    """
    
    def __init__(self, checkpoint_dir=None, model_name="distilbert/distilbert-base-uncased"):
        """
        Initialize ML adapter.
        
        Args:
            checkpoint_dir: Path to saved model checkpoint (default: artifacts/checkpoint)
            model_name: HuggingFace model name (must match training)
        """
        self.model_name = model_name
        # Device will be set when torch is imported
        self.device = "cpu"  # Default, will be updated when torch loads
        self.model = None
        self.tokenizer = None
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(__file__), '..', 'artifacts', 'checkpoint'
            )
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        
        # Standard labels for binary classification
        self.label_map = {0: "NOT SPAM", 1: "SPAM"}
        self.id_to_label = {0: 0, 1: 1}  # For binary: 0=not spam, 1=spam
        
    def load_model(self):
        """Load trained model and tokenizer from checkpoint."""
        # Import ML libraries when needed
        _import_ml_libs()
        
        # Update device after torch is imported
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            if not os.path.exists(self.checkpoint_dir):
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {self.checkpoint_dir}\n"
                    "Please train the model first using: python -m src.train_or_prepare"
                )
            
            logger.info(f"Loading model from {self.checkpoint_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Find the best checkpoint
            # Structure can be:
            # 1. artifacts/checkpoint/checkpoint-XXX/ (direct)
            # 2. artifacts/checkpoint/checkpoint/checkpoint-XXX/ (nested, from training)
            
            model_path = None
            
            # First, check for nested structure (checkpoint/checkpoint-XXX/)
            nested_checkpoint_dir = os.path.join(self.checkpoint_dir, 'checkpoint')
            if os.path.exists(nested_checkpoint_dir):
                checkpoint_paths = [
                    d for d in os.listdir(nested_checkpoint_dir) 
                    if os.path.isdir(os.path.join(nested_checkpoint_dir, d)) and d.startswith('checkpoint-')
                ]
                if checkpoint_paths:
                    checkpoint_nums = [int(c.split('-')[1]) for c in checkpoint_paths]
                    latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
                    model_path = os.path.join(nested_checkpoint_dir, latest_checkpoint)
                    logger.info(f"Found nested checkpoint: {model_path}")
            
            # If not found, check direct structure (checkpoint-XXX/)
            if model_path is None:
                checkpoint_paths = [
                    d for d in os.listdir(self.checkpoint_dir) 
                    if os.path.isdir(os.path.join(self.checkpoint_dir, d)) and d.startswith('checkpoint-')
                ]
                if checkpoint_paths:
                    checkpoint_nums = [int(c.split('-')[1]) for c in checkpoint_paths]
                    latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
                    model_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
                    logger.info(f"Found direct checkpoint: {model_path}")
            
            # If still not found, try loading directly from checkpoint dir
            if model_path is None:
                model_path = self.checkpoint_dir
                logger.info(f"Trying to load directly from: {model_path}")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                ignore_mismatched_sizes=True
            ).to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_email(self, text):
        """
        Predict if email is spam or not.
        
        Args:
            text: Email text (subject + body combined)
            
        Returns:
            dict with keys:
                - label: int (0=NOT SPAM, 1=SPAM)
                - label_name: str ("NOT SPAM" or "SPAM")
                - probability: float (confidence score)
                - probabilities: dict with both class probabilities
        """
        # Import ML libraries when needed
        _import_ml_libs()
        
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Tokenize input (matching training: max_length=50, truncation, padding)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=50,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_id = probabilities.argmax().item()
        confidence = probabilities[0][predicted_id].item()
        
        # Get probabilities for both classes
        prob_not_spam = probabilities[0][0].item()
        prob_spam = probabilities[0][1].item()
        
        return {
            "label": predicted_id,
            "label_name": self.label_map[predicted_id],
            "probability": confidence,
            "probabilities": {
                "NOT SPAM": prob_not_spam,
                "SPAM": prob_spam
            }
        }
    
    def get_tokenizer(self):
        """Get the tokenizer (needed for SHAP explanations)."""
        if self.tokenizer is None:
            self.load_model()
        return self.tokenizer
    
    def get_model(self):
        """Get the model (needed for SHAP explanations)."""
        if self.model is None:
            self.load_model()
        return self.model

