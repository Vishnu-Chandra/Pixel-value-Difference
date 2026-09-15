"""
AI-EPVD Main Entrypoint / Server Runner
Usage:
    python main.py --server
    python backend/app.py
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    from backend.app import app
    print("Starting AI-EPVD Flask Server on http://127.0.0.1:5000 ...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
