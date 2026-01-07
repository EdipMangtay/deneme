# Email Spam Detection System

A complete, local email spam detection system with AI-powered classification, explainability (SHAP), and a modern web interface. This system scans your Gmail inbox, classifies emails as spam or not spam, and provides explanations for each decision.

## Features

- ✅ **Customized Dataset** (+3): Build your own dataset from Gmail emails with automatic anonymization
- ✅ **User Interface** (+2): Modern, responsive Flask web UI with real-time email scanning
- ✅ **SHAP Explainability** (+2): Understand why emails are classified as spam with token-level explanations

## Project Structure

```
email_spam_detector/
├── app.py                 # Flask application
├── requirements.txt        # Python dependencies
├── README.md             # This file
├── .env.example          # Environment variables template
├── src/
│   ├── __init__.py
│   ├── ml_adapter.py     # ML model wrapper for inference
│   ├── train_or_prepare.py  # Training script
│   ├── imap_client.py    # Gmail IMAP client
│   ├── email_parser.py   # Email parsing utilities
│   ├── anonymize.py      # Email anonymization
│   ├── dataset_builder.py # Dataset building from Gmail
│   ├── explain.py        # SHAP explainability
│   └── utils.py          # Utility functions
├── artifacts/
│   └── checkpoint/       # Saved model checkpoints
├── data/
│   └── dataset.csv       # Training dataset
├── templates/
│   └── index.html        # Web UI template
└── static/
    └── style.css         # Web UI styles
```

## Prerequisites

- Python 3.8 or higher
- Gmail account with App Password enabled
- Windows (tested on Windows 10/11)

## Setup Instructions

### 1. Create Virtual Environment

```powershell
# Navigate to project directory
cd email_spam_detector

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Enable Gmail App Password

1. Go to your Google Account: https://myaccount.google.com/
2. Navigate to **Security** → **2-Step Verification** (must be enabled)
3. Scroll down to **App passwords**
4. Select **Mail** and **Other (Custom name)**
5. Enter "Email Spam Detector" as the name
6. Click **Generate**
7. Copy the 16-character password (you'll need this for `.env`)

### 4. Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```

2. Edit `.env` and fill in your credentials:
   ```
   GMAIL_EMAIL=your-email@gmail.com
   GMAIL_APP_PASSWORD=your-16-char-app-password
   ```

### 5. Build Dataset

Fetch emails from your Gmail account to build the training dataset:

```powershell
python -m src.dataset_builder --inbox 50 --spam 50
```

This command will:
- Fetch 50 emails from your INBOX (labeled as NOT SPAM)
- Fetch 50 emails from your Spam folder (labeled as SPAM)
- Anonymize sensitive information (emails, URLs, phone numbers)
- Save to `data/dataset.csv`

**Options:**
- `--inbox N`: Number of emails from INBOX (default: 50)
- `--spam N`: Number of emails from Spam folder (default: 50)
- `--output PATH`: Custom output path (default: `data/dataset.csv`)
- `--append`: Append to existing dataset instead of overwriting

### 6. Train the Model

Train the spam detection model using your dataset:

```powershell
python -m src.train_or_prepare
```

This will:
- Load the dataset from `data/dataset.csv`
- Train a DistilBERT model (fast and efficient)
- Save the trained model to `artifacts/checkpoint/`
- Training takes approximately 5-15 minutes depending on your hardware

**Options:**
- `--dataset PATH`: Path to dataset CSV (default: `data/dataset.csv`)
- `--model MODEL_NAME`: HuggingFace model name (default: `distilbert/distilbert-base-uncased`)
- `--output PATH`: Output directory for checkpoint

### 7. Run the Web Application

Start the Flask web server:

```powershell
python app.py
```

The application will be available at: **http://localhost:5000**

## Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Click **"Scan Inbox"** to fetch and classify emails from your Gmail INBOX
3. View results:
   - **Green cards**: NOT SPAM emails
   - **Red cards**: SPAM emails
   - Each card shows:
     - Subject and snippet
     - Prediction and confidence score
     - **"Why this decision?"** section with top tokens that influenced the classification

### Building Dataset (CLI)

```powershell
# Build dataset with default settings (50 inbox, 50 spam)
python -m src.dataset_builder

# Build with custom limits
python -m src.dataset_builder --inbox 100 --spam 100

# Append to existing dataset
python -m src.dataset_builder --inbox 20 --spam 20 --append
```

### Training Model (CLI)

```powershell
# Train with default settings
python -m src.train_or_prepare

# Train with custom dataset
python -m src.train_or_prepare --dataset data/my_dataset.csv

# Train with different model
python -m src.train_or_prepare --model distilbert/distilbert-base-uncased
```

## Dataset Description

The dataset is built from your personal Gmail emails and includes the following columns:

- **subject**: Email subject line (anonymized)
- **text**: Combined subject and body text (anonymized)
- **label**: Binary label (0 = NOT SPAM, 1 = SPAM)
- **source_folder**: Source Gmail folder (INBOX or Spam)
- **date**: Email date

### Anonymization Process

To protect your privacy, the following information is automatically removed or replaced:

1. **Email addresses**: Replaced with `[EMAIL]`
2. **URLs**: Replaced with `[URL]`
3. **Phone numbers**: Replaced with `[PHONE]`
4. **Whitespace**: Collapsed to single spaces

All processing is done locally on your machine. No data is sent to external servers.

## Model Architecture

The system uses a **DistilBERT** transformer model fine-tuned for binary spam classification:

- **Base Model**: `distilbert/distilbert-base-uncased`
- **Task**: Binary classification (NOT SPAM / SPAM)
- **Input**: Email text (subject + body), truncated to 50 tokens
- **Output**: Probability distribution over 2 classes

The model is trained using the existing `Model` class from the original codebase, ensuring compatibility with your training pipeline.

## Explainability (SHAP)

The system provides token-level explanations using SHAP (SHapley Additive exPlanations):

- **Top 5 tokens** that influenced the classification decision
- **Impact scores** showing how each token pushed toward SPAM (positive) or NOT SPAM (negative)
- **Visual indicators**: Red badges for spam-indicating tokens, green for ham-indicating tokens

If SHAP is not available, the system falls back to a keyword-based explanation method.

## Troubleshooting

### "Model not loaded" Error

**Solution**: Train the model first:
```powershell
python -m src.train_or_prepare
```

### "Gmail credentials not configured" Error

**Solution**: 
1. Ensure `.env` file exists in the project root
2. Verify `GMAIL_EMAIL` and `GMAIL_APP_PASSWORD` are set correctly
3. Make sure you're using an App Password (not your regular Gmail password)

### "Checkpoint directory not found" Error

**Solution**: The model hasn't been trained yet. Run:
```powershell
python -m src.train_or_prepare
```

### IMAP Connection Errors

**Possible causes:**
- Incorrect email or App Password
- 2-Step Verification not enabled
- App Password not generated correctly
- Network/firewall issues

**Solution**: 
1. Verify your App Password is correct (16 characters, no spaces)
2. Ensure 2-Step Verification is enabled
3. Check your internet connection

### "Spam folder not found" Warning

**Solution**: The system will auto-detect the spam folder. If it fails, you can manually set `SPAM_FOLDER` in `.env`:
```
SPAM_FOLDER=[Gmail]/Spam
```

### Training Takes Too Long

**Solution**: 
- Use a smaller dataset for faster training
- Consider using CPU if CUDA is causing issues
- Reduce `num_train_epochs` in `model.py` (not recommended for production)

### SHAP Not Working

**Solution**: Install SHAP explicitly:
```powershell
pip install shap
```

The system will automatically fall back to keyword-based explanations if SHAP is unavailable.

## Technical Details

### Dependencies

- **Flask**: Web framework
- **transformers**: HuggingFace transformer models
- **torch**: PyTorch for model inference
- **pandas**: Data manipulation
- **beautifulsoup4**: HTML email parsing
- **shap**: Explainability (optional)
- **python-dotenv**: Environment variable management

### Model Inference

The `MLAdapter` class wraps the original `Model` class and provides:
- Model loading from checkpoint
- Text preprocessing (tokenization)
- Prediction with probability scores
- Tokenizer access for SHAP explanations

### Email Processing Pipeline

1. **Fetch**: IMAP client connects to Gmail and fetches emails
2. **Parse**: Extract subject, body, and metadata from raw email bytes
3. **Anonymize**: Remove sensitive information
4. **Classify**: Run through trained model
5. **Explain**: Generate SHAP explanations
6. **Display**: Show results in web UI

## License

This project uses the existing ML codebase and extends it with a complete application framework.

## Credits

- ML Model: Based on HuggingFace Transformers
- UI Design: Modern, responsive web interface
- Explainability: SHAP library

---

**Note**: This system processes emails locally. No data leaves your machine. All model training and inference happens on your local hardware.


