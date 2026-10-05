"""
Run this script once to:
  1. Generate the 5,000-sample dataset
  2. Train both XGBoost models
  3. Launch the Streamlit app

Usage:
    python setup_and_run.py
"""
import subprocess, sys, os

PYTHON = sys.executable
BASE   = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  AI Food Safety & Quality Intelligence System — Setup")
print("=" * 60)

# Step 1: Generate dataset
print("\n[1/3] Generating 5,000-sample dataset...")
subprocess.run([PYTHON, os.path.join(BASE, "generate_dataset.py")], check=True)

# Step 2: Train models
print("\n[2/3] Training XGBoost models (this may take ~2 min)...")
subprocess.run([PYTHON, os.path.join(BASE, "train_model.py")], check=True)

# Step 3: Launch Streamlit
print("\n[3/3] Launching Streamlit UI...")
print("\n  ✅  Open your browser at:  http://localhost:8501\n")
subprocess.run([
    PYTHON, "-m", "streamlit", "run",
    os.path.join(BASE, "app.py"),
    "--server.port", "8501",
    "--server.headless", "false",
], check=True)
