from pathlib import Path

PYTHON_DIR = Path(__file__).absolute().parent
PROJECT_DIR = PYTHON_DIR.parent

CREDENTIALS_DIR = PROJECT_DIR / "credentials"
OUTPUT_DIR = PROJECT_DIR / "output"
CACHE_DIR = PROJECT_DIR / "cache"

SEED = 1998
