#!/usr/bin/env python
"""Run the Streamlit frontend."""

import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])