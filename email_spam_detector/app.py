"""
Flask application for Email Spam Detection System.
"""
import os
import sys
import time
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.imap_client import IMAPClient
from src.utils import combine_subject_and_body, truncate_text, format_probability
from src.anonymize import anonymize_email_data

# Lazy import ML modules (only when needed)
MLAdapter = None
explain_email = None

def _import_ml_modules():
    """Lazy import ML modules."""
    global MLAdapter, explain_email
    if MLAdapter is None:
        try:
            from src.ml_adapter import MLAdapter
            from src.explain import explain_email
        except ImportError as e:
            logger.warning(f"ML modules not available: {e}")
            # Create dummy functions
            class DummyMLAdapter:
                def __init__(self): pass
                def load_model(self): raise ImportError("ML libraries not installed")
                def predict_email(self, text): raise ImportError("ML libraries not installed")
            MLAdapter = DummyMLAdapter
            explain_email = lambda *args, **kwargs: []

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ML adapter (lazy loading)
ml_adapter = None


def get_ml_adapter():
    """Get or initialize ML adapter."""
    global ml_adapter
    _import_ml_modules()  # Ensure ML modules are imported
    
    if ml_adapter is None:
        try:
            ml_adapter = MLAdapter()
            ml_adapter.load_model()
        except Exception as e:
            logger.error(f"Error initializing ML adapter: {e}")
            raise
    return ml_adapter


@app.route('/')
def index():
    """Pipeline page - automated flow."""
    return render_template('pipeline.html')

@app.route('/manual')
def manual():
    """Manual control page."""
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def scan_inbox():
    """Scan inbox and classify emails with ground truth comparison."""
    try:
        # Get parameters
        data = request.get_json() or {}
        inbox_limit = data.get('inbox_limit', 20)
        spam_limit = data.get('spam_limit', 20)  # Also fetch spam for comparison
        
        # Get credentials
        email_address = os.getenv('GMAIL_EMAIL')
        app_password = os.getenv('GMAIL_APP_PASSWORD')
        
        if not email_address or not app_password:
            return jsonify({
                'success': False,
                'error': 'Gmail credentials not configured. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD in .env file.'
            }), 400
        
        # Get ML adapter (required for classification)
        adapter = None
        try:
            adapter = get_ml_adapter()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Model not loaded. Please train the model first: {str(e)}'
            }), 500
        
        # Fetch emails from both INBOX and SPAM for comparison
        logger.info(f"Scanning: {inbox_limit} inbox, {spam_limit} spam emails")
        with IMAPClient(email_address, app_password) as client:
            emails_data = client.fetch_inbox_and_spam(
                inbox_limit=inbox_limit,
                spam_limit=spam_limit,
                spam_folder=os.getenv('SPAM_FOLDER')
            )
        
        # Classify emails and compare with ground truth
        results = []
        all_emails = emails_data['inbox'] + emails_data['spam']
        
        # Statistics for report
        stats = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'true_positive': 0,  # Predicted spam, actually spam
            'true_negative': 0,  # Predicted not spam, actually not spam
            'false_positive': 0,  # Predicted spam, actually not spam
            'false_negative': 0   # Predicted not spam, actually spam
        }
        
        for email_data in all_emails:
            # Combine subject and body
            text = combine_subject_and_body(
                email_data.get('subject', ''),
                email_data.get('text', '')
            )
            
            if not text.strip():
                continue
            
            # Ground truth label (from folder: 0=INBOX/not spam, 1=SPAM/spam)
            ground_truth = email_data.get('label', 0)
            is_actually_spam = ground_truth == 1
            
            # Anonymize for display
            anonymized = anonymize_email_data(email_data)
            
            # Predict
            try:
                prediction = adapter.predict_email(text)
                predicted_spam = prediction['label'] == 1
                
                # Get explanation
                try:
                    _import_ml_modules()
                    model = adapter.get_model()
                    tokenizer = adapter.get_tokenizer()
                    explanations = explain_email(text, model, tokenizer)
                    top_tokens = explanations[:5]
                except Exception as e:
                    logger.warning(f"Error generating explanation: {e}")
                    top_tokens = []
                
                # Update statistics
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
                
                # Determine correctness
                is_correct = predicted_spam == is_actually_spam
                
                result = {
                    'subject': anonymized.get('subject', 'No Subject'),
                    'snippet': truncate_text(anonymized.get('text', ''), 150),
                    'prediction': prediction['label_name'],
                    'is_spam': predicted_spam,
                    'ground_truth': 'SPAM' if is_actually_spam else 'NOT SPAM',
                    'is_correct': is_correct,
                    'probability': prediction['probability'],
                    'probability_formatted': format_probability(prediction['probability']),
                    'date': email_data.get('date', ''),
                    'folder': email_data.get('folder', ''),
                    'explanations': [
                        {
                            'token': exp['token'],
                            'impact': exp['impact'],
                            'impact_formatted': f"{exp['impact']:+.2f}"
                        }
                        for exp in top_tokens
                    ]
                }
            except Exception as e:
                logger.warning(f"Error classifying email: {e}")
                result = {
                    'subject': anonymized.get('subject', 'No Subject'),
                    'snippet': truncate_text(anonymized.get('text', ''), 150),
                    'prediction': 'ERROR',
                    'is_spam': False,
                    'ground_truth': 'SPAM' if is_actually_spam else 'NOT SPAM',
                    'is_correct': False,
                    'probability': 0.0,
                    'probability_formatted': 'N/A',
                    'date': email_data.get('date', ''),
                    'folder': email_data.get('folder', ''),
                    'explanations': []
                }
            
            results.append(result)
        
        # Sort by date (most recent first)
        results.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate metrics
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            precision = stats['true_positive'] / (stats['true_positive'] + stats['false_positive']) if (stats['true_positive'] + stats['false_positive']) > 0 else 0
            recall = stats['true_positive'] / (stats['true_positive'] + stats['false_negative']) if (stats['true_positive'] + stats['false_negative']) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            accuracy = precision = recall = f1_score = 0
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total': stats['total'],
                'correct': stats['correct'],
                'incorrect': stats['incorrect'],
                'true_positive': stats['true_positive'],
                'true_negative': stats['true_negative'],
                'false_positive': stats['false_positive'],
                'false_negative': stats['false_negative']
            }
        })
    
    except Exception as e:
        logger.error(f"Error scanning inbox: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    """Test IMAP connection and list folders."""
    try:
        # Get credentials from .env
        email_address = os.getenv('GMAIL_EMAIL')
        app_password = os.getenv('GMAIL_APP_PASSWORD')
        imap_server = os.getenv('IMAP_SERVER', 'imap.gmail.com')
        
        if not email_address or not app_password:
            return jsonify({
                'success': False,
                'error': 'Gmail credentials not configured. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD in .env file.'
            }), 400
        
        # Test connection
        logger.info("Testing IMAP connection...")
        with IMAPClient(email_address, app_password, imap_server) as client:
            folders = client.list_folders()
            spam_folder = client.find_spam_folder()
        
        logger.info(f"Connection successful. Found {len(folders)} folders.")
        print(f"\n✅ IMAP Connection Successful!")
        print(f"📧 Email: {email_address}")
        print(f"📁 Found {len(folders)} folders:")
        for folder in folders:
            marker = " (SPAM)" if spam_folder and folder == spam_folder else ""
            print(f"   - {folder}{marker}")
        print()
        
        return jsonify({
            'success': True,
            'message': f'Connection successful! Found {len(folders)} folders.',
            'folders': folders,
            'spam_folder': spam_folder,
            'email': email_address
        })
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"IMAP connection failed: {error_msg}")
        print(f"\n❌ IMAP Connection Failed: {error_msg}\n")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/api/fetch-inbox', methods=['POST'])
def fetch_inbox():
    """Fetch emails from INBOX only (no classification)."""
    try:
        # Get credentials from .env
        email_address = os.getenv('GMAIL_EMAIL')
        app_password = os.getenv('GMAIL_APP_PASSWORD')
        imap_server = os.getenv('IMAP_SERVER', 'imap.gmail.com')
        fetch_limit = int(os.getenv('FETCH_LIMIT', 20))
        
        if not email_address or not app_password:
            return jsonify({
                'success': False,
                'error': 'Gmail credentials not configured. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD in .env file.'
            }), 400
        
        # Get parameters
        data = request.get_json() or {}
        limit = data.get('limit', fetch_limit)
        
        # Fetch emails
        logger.info(f"Fetching {limit} emails from INBOX...")
        with IMAPClient(email_address, app_password, imap_server) as client:
            emails = client.fetch_emails("INBOX", limit)
        
        # Format results (no classification)
        results = []
        for email_data in emails:
            anonymized = anonymize_email_data(email_data)
            results.append({
                'subject': anonymized.get('subject', 'No Subject'),
                'snippet': truncate_text(anonymized.get('text', ''), 160),
                'date': email_data.get('date', ''),
                'folder': email_data.get('folder', 'INBOX'),
                'is_classified': False
            })
        
        # Sort by date (most recent first)
        results.sort(key=lambda x: x['date'], reverse=True)
        
        logger.info(f"Fetched {len(results)} emails from INBOX")
        print(f"\n✅ Fetched {len(results)} emails from INBOX\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching INBOX: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/fetch-spam', methods=['POST'])
def fetch_spam():
    """Fetch emails from Spam folder only (no classification)."""
    try:
        # Get credentials from .env
        email_address = os.getenv('GMAIL_EMAIL')
        app_password = os.getenv('GMAIL_APP_PASSWORD')
        imap_server = os.getenv('IMAP_SERVER', 'imap.gmail.com')
        spam_folder = os.getenv('SPAM_FOLDER')
        fetch_limit = int(os.getenv('FETCH_LIMIT', 20))
        
        if not email_address or not app_password:
            return jsonify({
                'success': False,
                'error': 'Gmail credentials not configured. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD in .env file.'
            }), 400
        
        # Get parameters
        data = request.get_json() or {}
        limit = data.get('limit', fetch_limit)
        
        # Fetch emails
        logger.info(f"Fetching {limit} emails from Spam folder...")
        with IMAPClient(email_address, app_password, imap_server) as client:
            # Auto-detect spam folder if not specified
            if spam_folder is None:
                spam_folder = client.find_spam_folder()
                if spam_folder is None:
                    return jsonify({
                        'success': False,
                        'error': 'Spam folder not found. Please set SPAM_FOLDER in .env file.'
                    }), 400
            
            emails = client.fetch_emails(spam_folder, limit)
        
        # Format results (no classification)
        results = []
        for email_data in emails:
            anonymized = anonymize_email_data(email_data)
            results.append({
                'subject': anonymized.get('subject', 'No Subject'),
                'snippet': truncate_text(anonymized.get('text', ''), 160),
                'date': email_data.get('date', ''),
                'folder': email_data.get('folder', spam_folder),
                'is_classified': False
            })
        
        # Sort by date (most recent first)
        results.sort(key=lambda x: x['date'], reverse=True)
        
        logger.info(f"Fetched {len(results)} emails from Spam folder")
        print(f"\n✅ Fetched {len(results)} emails from Spam folder\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching Spam folder: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    try:
        adapter = get_ml_adapter()
        model_loaded = adapter.model is not None
        
        return jsonify({
            'success': True,
            'model_loaded': model_loaded,
            'model_name': adapter.model_name if model_loaded else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'model_loaded': False,
            'error': str(e)
        })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info("Starting Email Spam Detection System...")
    logger.info(f"Server will run on http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

