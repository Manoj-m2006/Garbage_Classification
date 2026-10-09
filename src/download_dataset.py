"""
download_dataset.py
-------------------
Automated dataset download and verification helper script for Garbage Classification project.
Handles automated retrieval from Kaggle / public mirror if local dataset is missing.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATASET_DIR = Path("dataset")
DATASET_RESIZED_DIR = DATASET_DIR / "dataset-resized"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Public fallback mirror URL for Kaggle Garbage Classification dataset (resized zip)
DATASET_URL = "https://github.com/garythung/garbage-classification/raw/master/data/dataset-resized.zip"


def check_dataset_exists(base_dir: Path = DATASET_DIR) -> bool:
    """
    Checks if all 6 garbage classification class directories exist and contain images.
    """
    if not base_dir.exists():
        return False

    for cls in CLASSES:
        # Check direct subdirectory or nested in dataset-resized
        direct = base_dir / cls
        nested = DATASET_RESIZED_DIR / cls
        if direct.is_dir() and any(direct.glob("*.*")):
            continue
        elif nested.is_dir() and any(nested.glob("*.*")):
            continue
        else:
            return False
    return True


def download_and_extract(dataset_dir: Path = DATASET_DIR, url: str = DATASET_URL):
    """
    Downloads the garbage classification zip dataset if missing and extracts it.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / "dataset-resized.zip"

    if check_dataset_exists(dataset_dir):
        print("[INFO] Garbage classification dataset is already present and verified.")
        return

    print(f"[INFO] Dataset missing or incomplete. Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("[INFO] Download complete. Extracting dataset archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("[INFO] Extraction complete.")
        
        # Clean up zip file
        if zip_path.exists():
            os.remove(zip_path)
    except Exception as e:
        print(f"[ERROR] Automated download failed: {e}")
        print("[MANUAL ACTION REQUIRED] Please place the 'dataset-resized' directory inside 'dataset/'.")
        sys.exit(1)


if __name__ == "__main__":
    download_and_extract()
