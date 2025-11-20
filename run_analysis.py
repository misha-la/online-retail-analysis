#!/usr/bin/env python3
"""
Runner script for Online Retail Analysis
Execute from project root directory
"""

import sys
import os

# Add scripts directory to Python path
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.insert(0, script_dir)

# Change to project root for file operations
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Import and run main analysis
from main import run_full_analysis

if __name__ == "__main__":
    run_full_analysis()
