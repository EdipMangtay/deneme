"""
Email Spam Detection - Simple Pipeline
Just fetch emails and classify them using existing trained model.
"""
import os
import webbrowser
import time
import threading

def start_server():
    """Start Flask server."""
    print("=" * 70)
    print("🚀 Email Spam Detection System")
    print("=" * 70)
    print("\n📡 Starting web server on http://localhost:5000")
    print("⏳ Opening browser in 2 seconds...\n")
    
    def open_browser():
        time.sleep(2)
        url = "http://localhost:5000"
        print(f"🌐 Opening browser: {url}\n")
        webbrowser.open(url)
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run app
    from app import app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    start_server()
