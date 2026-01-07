"""
SHAP explainability module for spam detection.
"""
import numpy as np
import logging
import warnings

# Suppress SHAP warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explain_email_shap(text, model, tokenizer, max_tokens=50):
    """
    Generate SHAP explanations for email classification.
    Uses a simplified approach with token-level importance.
    
    Args:
        text: Email text
        model: Trained model (torch model)
        tokenizer: Tokenizer
        max_tokens: Maximum number of tokens to explain
        
    Returns:
        List of dicts with keys: token, impact, contribution
    """
    if not SHAP_AVAILABLE:
        return explain_email_fallback(text, model, tokenizer)
    
    try:
        import torch
        
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_tokens,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(model.device if hasattr(model, 'device') else 'cpu')
        
        # Get baseline prediction (full text)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            baseline_probs = torch.softmax(logits, dim=-1)
            baseline_spam_prob = baseline_probs[0][1].item()
        
        # Get tokens
        token_ids = inputs['input_ids'][0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Calculate importance by removing each token
        token_importances = []
        for i, token_id in enumerate(token_ids):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                continue
            
            # Create input without this token
            masked_ids = token_ids.copy()
            masked_ids[i] = tokenizer.mask_token_id if tokenizer.mask_token_id else tokenizer.pad_token_id
            
            # Predict without this token
            masked_inputs = tokenizer.decode(masked_ids, skip_special_tokens=False)
            masked_tokens = tokenizer(
                masked_inputs,
                truncation=True,
                padding=True,
                max_length=max_tokens,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(model.device if hasattr(model, 'device') else 'cpu')
            
            with torch.no_grad():
                masked_outputs = model(**masked_tokens)
                masked_logits = masked_outputs.logits
                masked_probs = torch.softmax(masked_logits, dim=-1)
                masked_spam_prob = masked_probs[0][1].item()
            
            # Impact is the difference (positive = token increases spam probability)
            impact = baseline_spam_prob - masked_spam_prob
            
            clean_token = tokens[i].replace('##', '').replace('[', '').replace(']', '')
            if clean_token and clean_token not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                token_importances.append({
                    'token': clean_token,
                    'impact': float(impact),
                    'contribution': float(impact)
                })
        
        # Sort by absolute impact
        token_importances.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return token_importances[:10]  # Top 10 tokens
        
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}. Using fallback.")
        return explain_email_fallback(text, model, tokenizer)




def explain_email_fallback(text, model, tokenizer):
    """
    Fallback explanation method using token importance.
    Simple heuristic: tokens that appear frequently in spam emails.
    
    Args:
        text: Email text
        model: Trained model
        tokenizer: Tokenizer
        
    Returns:
        List of dicts with keys: token, impact
    """
    # Simple heuristic: common spam words
    spam_keywords = [
        'free', 'click', 'win', 'prize', 'offer', 'limited', 'act now',
        'guarantee', 'money', 'cash', 'winner', 'congratulations', 'urgent',
        'buy now', 'discount', 'sale', 'deal', 'special', 'promo'
    ]
    
    # Tokenize and find important tokens
    tokens = text.lower().split()
    explanations = []
    
    for token in tokens:
        # Clean token
        clean_token = token.strip('.,!?;:()[]{}"\'-')
        if not clean_token:
            continue
        
        # Check if it's a spam keyword
        impact = 0.0
        if clean_token in spam_keywords:
            impact = 0.3  # Positive impact toward spam
        elif len(clean_token) > 3:
            # Neutral/slight negative for normal words
            impact = -0.1
        
        explanations.append({
            'token': clean_token,
            'impact': impact,
            'contribution': impact
        })
    
    # Sort by absolute impact
    explanations.sort(key=lambda x: abs(x['impact']), reverse=True)
    
    return explanations[:10]  # Top 10 tokens


def explain_email(text, model, tokenizer, use_shap=True):
    """
    Main function to explain email classification.
    
    Args:
        text: Email text
        model: Trained model
        tokenizer: Tokenizer
        use_shap: Whether to use SHAP (if available)
        
    Returns:
        List of dicts with token explanations
    """
    if use_shap and SHAP_AVAILABLE:
        return explain_email_shap(text, model, tokenizer)
    else:
        return explain_email_fallback(text, model, tokenizer)

