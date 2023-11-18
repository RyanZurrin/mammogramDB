# omama/odm/__init__.py

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
lib_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(lib_dir))