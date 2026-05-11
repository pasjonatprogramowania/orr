"""
conftest.py – konfiguracja pytest.
Dodaje root projektu do sys.path żeby importy działały.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
