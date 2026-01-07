"""
Complete Pipeline - One command to rule them all!
1. Trains model with all existing CSV files (w1998, abdallah, kucev)
2. Starts web server
3. Opens browser automatically
"""
import os
import sys
import subprocess
import webbrowser
import time
import threading

def train_model_first():
    """Train model with all existing datasets."""
    print("=" * 70)
    print("🤖 STEP 1: Training Model with All Datasets")
    print("=" * 70)
    print("\n📊 Combining datasets:")
    print("   - _w1998.csv")
    print("   - abdallah.csv")
    print("   - kucev.csv")
    print("\n⏳ This will take 5-15 minutes...\n")
    
    # Run training
    train_script = os.path.join(os.path.dirname(__file__), 'src', 'train_with_all_data.py')
    result = subprocess.run(
        [sys.executable, '-m', 'src.train_with_all_data'],
        cwd=os.path.dirname(__file__),
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✅ Model training completed successfully!\n")
        return True
    else:
        print("\n⚠️  Training had issues, but continuing...\n")
        return False

def start_server():
    """Start Flask server."""
    print("=" * 70)
    print("🌐 STEP 2: Starting Web Server")
    print("=" * 70)
    print("\n📡 Server starting on http://localhost:5000")
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

def main():
    """Main pipeline execution."""
    print("\n" + "=" * 70)
    print("🚀 Email Spam Detection - Complete Pipeline")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Train model with w1998, abdallah, and kucev datasets")
    print("  2. Start web server")
    print("  3. Open browser automatically")
    print("\n" + "=" * 70 + "\n")
    
    # Check if model already exists
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'checkpoint')
    model_exists = os.path.exists(checkpoint_dir) and any(
        d.startswith('checkpoint-') for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    )
    
    if not model_exists:
        print("📦 Model not found. Training new model...\n")
        train_model_first()
    else:
        print("✅ Model already exists. Skipping training.\n")
        print("💡 To retrain, delete artifacts/checkpoint folder first.\n")
    
    # Start server
    start_server()

if __name__ == '__main__':
    main()


