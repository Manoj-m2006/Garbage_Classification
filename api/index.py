import sys
import os

# Ensure root directory is on Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
