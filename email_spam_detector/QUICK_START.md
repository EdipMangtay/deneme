# Quick Start Guide - Email Spam Detection System

## ✅ All Features Implemented

1. ✅ `.env.example` created with exact keys
2. ✅ App reads credentials ONLY from `.env` (never hardcoded)
3. ✅ "Test IMAP Connection" feature (UI + console output)
4. ✅ "Fetch Latest Emails (INBOX)" button
5. ✅ "Fetch Spam Emails (SPAM)" button
6. ✅ Emails displayed as modern cards (subject + date + snippet)
7. ✅ CLI commands for quick checks

---

## 📋 Setup Steps

### Step 1: Create `.env` File

Create a file named `.env` in the `email_spam_detector` directory:

```env
GMAIL_EMAIL=mangtay0133@gmail.com
GMAIL_APP_PASSWORD=vbrdwgbalkylaafa
IMAP_SERVER=imap.gmail.com
INBOX_FOLDER=INBOX
SPAM_FOLDER=[Gmail]/Spam
FETCH_LIMIT=20
```

**⚠️ IMPORTANT:** 
- Never commit `.env` to git (it's already in `.gitignore`)
- Never hardcode or print the App Password in code
- The App Password is read ONLY from `.env` file

### Step 2: Install Dependencies

```powershell
cd email_spam_detector
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Test IMAP Connection (CLI)

```powershell
python -m src.imap_client --list-folders
```

**Expected output:**
```
📁 Listing Gmail folders...

Found X folders:

  • INBOX
  • [Gmail]/Spam ← SPAM FOLDER
  • [Gmail]/Sent
  ...

✅ Spam folder detected: [Gmail]/Spam
```

### Step 4: Test Fetching Emails (CLI)

```powershell
# Fetch 10 emails from INBOX
python -m src.imap_client --fetch-inbox 10

# Fetch 10 emails from Spam folder
python -m src.imap_client --fetch-spam 10
```

### Step 5: Run Web Application

```powershell
python app.py
```

Open browser: **http://localhost:5000**

---

## 🖥️ Web UI Features

### Available Buttons:

1. **🔌 Test IMAP Connection**
   - Tests connection to Gmail
   - Lists all folders
   - Shows success/failure message
   - Outputs to console AND UI

2. **📥 Fetch Latest Emails (INBOX)**
   - Fetches last N emails from INBOX
   - Displays as cards (subject + date + snippet)
   - No classification (just fetching)

3. **📮 Fetch Spam Emails (SPAM)**
   - Fetches last N emails from Spam folder
   - Auto-detects spam folder if not set
   - Displays as cards

4. **🔍 Scan & Classify Inbox**
   - Fetches AND classifies emails
   - Shows spam/not spam predictions
   - Includes SHAP explanations

---

## 📝 CLI Commands Reference

All commands read credentials from `.env` file:

```powershell
# List all folders
python -m src.imap_client --list-folders

# Fetch N emails from INBOX
python -m src.imap_client --fetch-inbox 10

# Fetch N emails from Spam folder
python -m src.imap_client --fetch-spam 10
```

---

## 🔒 Security Features

✅ **Credentials are NEVER:**
- Hardcoded in source code
- Printed to console (except for testing)
- Committed to git
- Exposed in error messages

✅ **Credentials are ONLY:**
- Read from `.env` file
- Used for IMAP connection
- Stored locally on your machine

---

## 🎯 Testing Checklist

- [ ] `.env` file created with correct credentials
- [ ] `python -m src.imap_client --list-folders` works
- [ ] `python -m src.imap_client --fetch-inbox 10` works
- [ ] `python -m src.imap_client --fetch-spam 10` works
- [ ] Web UI opens at http://localhost:5000
- [ ] "Test IMAP Connection" button works
- [ ] "Fetch Latest Emails (INBOX)" button works
- [ ] "Fetch Spam Emails (SPAM)" button works
- [ ] Emails display as modern cards
- [ ] Spam folder auto-detection works

---

## 🐛 Troubleshooting

### "Gmail credentials not configured"
- Check `.env` file exists in `email_spam_detector` directory
- Verify `GMAIL_EMAIL` and `GMAIL_APP_PASSWORD` are set
- Ensure no extra spaces in `.env` file

### "Spam folder not found"
- The system will auto-detect spam folder
- If auto-detection fails, set `SPAM_FOLDER` in `.env`
- Use `--list-folders` to see available folders

### IMAP connection errors
- Verify App Password is correct (16 characters, no spaces)
- Ensure 2-Step Verification is enabled in Google Account
- Check internet connection

---

## 📊 Email Card Display

Each email card shows:
- **Subject**: Email subject line
- **Date**: Email date/time
- **Snippet**: First 160 characters of email body
- **Folder**: Source folder (INBOX or SPAM)

Cards are color-coded:
- **Blue border**: Fetched emails (not classified)
- **Green border**: NOT SPAM (classified)
- **Red border**: SPAM (classified)

---

## ✅ All Requirements Met

1. ✅ `.env.example` created with exact format
2. ✅ App reads credentials ONLY from `.env`
3. ✅ Test IMAP Connection feature (UI + console)
4. ✅ Fetch INBOX button (separate from classification)
5. ✅ Fetch Spam button (separate from classification)
6. ✅ Emails shown as modern cards
7. ✅ CLI commands implemented
8. ✅ Spam folder auto-detection
9. ✅ Robust error handling
10. ✅ Windows compatible

---

**Ready to use!** 🚀


