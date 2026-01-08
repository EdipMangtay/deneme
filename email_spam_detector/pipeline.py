"""
Automated Pipeline - One command to rule them all!
Starts the web server and opens browser automatically.
"""
import os
import sys
import webbrowser
import time
import threading
from flask import Flask
from app import app

def open_browser():
    """Open browser after a short delay."""
    time.sleep(1.5)  # Wait for server to start
    url = "http://localhost:5000"
    print(f"\n🌐 Opening browser: {url}\n")
    webbrowser.open(url)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("🚀 Email Spam Detection System - Starting Pipeline")
    print("=" * 60)
    print(f"\n📡 Server starting on http://localhost:{port}")
    print("⏳ Opening browser in 2 seconds...\n")
    
    # Start browser in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)



