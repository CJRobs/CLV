#!/usr/bin/env python3
"""
Local runner for CLV Analysis Dashboard
Run this script to start the Streamlit dashboard locally without Docker
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements if missing"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 Starting CLV Analysis Dashboard (Local Mode)")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard.py").exists():
        print("❌ Error: dashboard.py not found in current directory")
        print("   Please run this script from the CLV project directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("⚠️  Missing required packages. Installing...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please run:")
            print("   pip install -r requirements.txt")
            sys.exit(1)
    
    # Create necessary directories
    os.makedirs("dashboard_results", exist_ok=True)
    os.makedirs("clv_results_kaggle", exist_ok=True)
    
    print("✅ All requirements satisfied")
    print("🎯 Starting Streamlit dashboard...")
    print("")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()