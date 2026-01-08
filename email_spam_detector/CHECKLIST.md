# Setup Checklist

Follow these steps to get the Email Spam Detection System running locally:

## ✅ Pre-Setup

- [ ] Python 3.8+ installed
- [ ] Gmail account with 2-Step Verification enabled
- [ ] Gmail App Password generated (16 characters)

## ✅ Step 1: Environment Setup

```powershell
# Navigate to project
cd email_spam_detector

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## ✅ Step 2: Configure Gmail

1. Go to: https://myaccount.google.com/security
2. Enable 2-Step Verification (if not already enabled)
3. Go to App Passwords
4. Generate new App Password for "Email Spam Detector"
5. Copy the 16-character password

## ✅ Step 3: Configure Environment Variables

```powershell
# Copy example file
copy .env.example .env

# Edit .env file and add:
# GMAIL_EMAIL=your-email@gmail.com
# GMAIL_APP_PASSWORD=your-16-char-password
```

## ✅ Step 4: Build Dataset

```powershell
python -m src.dataset_builder --inbox 50 --spam 50
```

Expected output:
- Fetches 50 emails from INBOX
- Fetches 50 emails from Spam folder
- Saves to `data/dataset.csv`
- Shows label distribution

## ✅ Step 5: Train Model

```powershell
python -m src.train_or_prepare
```

Expected output:
- Loads dataset
- Trains DistilBERT model
- Saves checkpoint to `artifacts/checkpoint/`
- Takes 5-15 minutes

## ✅ Step 6: Run Web Application

```powershell
python app.py
```

Expected output:
```
Starting Email Spam Detection System...
Server will run on http://localhost:5000
 * Running on http://0.0.0.0:5000
```

## ✅ Step 7: Test the System

1. Open browser: http://localhost:5000
2. Click "Scan Inbox" button
3. Wait for results to load
4. Verify:
   - Emails are displayed as cards
   - Spam emails show in red
   - Not spam emails show in green
   - "Why this decision?" section shows tokens

## 🐛 Troubleshooting

### "Model not loaded"
- Solution: Run `python -m src.train_or_prepare` first

### "Gmail credentials not configured"
- Solution: Check `.env` file exists and has correct values

### "Checkpoint directory not found"
- Solution: Train the model first

### IMAP connection errors
- Solution: Verify App Password is correct (16 chars, no spaces)
- Ensure 2-Step Verification is enabled

### Training takes too long
- Solution: Use smaller dataset or wait (normal for first training)

## 📝 Quick Commands Reference

```powershell
# Build dataset
python -m src.dataset_builder --inbox 50 --spam 50

# Train model
python -m src.train_or_prepare

# Run app
python app.py

# Build dataset (append mode)
python -m src.dataset_builder --inbox 20 --spam 20 --append
```

## 🎯 Success Criteria

- ✅ Dataset built with emails from Gmail
- ✅ Model trained and saved
- ✅ Web UI accessible at http://localhost:5000
- ✅ Can scan inbox and see classifications
- ✅ Explanations shown for each email

---

**Note**: All processing happens locally. No data leaves your machine.



