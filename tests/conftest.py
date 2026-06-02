# tests/conftest.py
import sys
import os

# This adds the current project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))