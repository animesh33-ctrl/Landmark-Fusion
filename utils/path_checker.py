
import os
import sys


def validate_paths(required_paths: list[str]) -> bool:
    all_ok = True
    for p in required_paths:
        if not os.path.exists(p):
            print(f"Dataset path not found: {p}")
            all_ok = False
    return all_ok


def validate_and_exit(required_paths: list[str]) -> None:
    if not validate_paths(required_paths):
        print("\n[ERROR] One or more dataset paths are missing. "
              "Please check the paths in config/config.py and ensure "
              "the datasets are placed correctly.")
        sys.exit(1)
    else:
        print("[OK] All dataset paths verified successfully.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.config import REQUIRED_DATASET_PATHS
    validate_and_exit(REQUIRED_DATASET_PATHS)
