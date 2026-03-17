import os
import sys
import argparse

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import REQUIRED_DATASET_PATHS, DEVICE, SEED
from utils.path_checker import validate_and_exit


def main():
    parser = argparse.ArgumentParser(
        description="Indian Sign Language Recognition & Sentence Translation"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "check_paths", "train_word",
                 "train_sentence", "inference"],
        help="Which part of the pipeline to run.",
    )
    parser.add_argument(
        "--sentence_model", type=str, default="transformer",
        choices=["transformer", "lstm"],
        help="Sequence model architecture for sentence recognition.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  INDIAN SIGN LANGUAGE RECOGNITION SYSTEM")
    print("=" * 65)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Device  : {DEVICE}")
    print(f"  Seed    : {SEED}")
    print()

    print("[Step 0] Validating dataset paths …")
    validate_and_exit(REQUIRED_DATASET_PATHS)
    print()

    if args.mode == "check_paths":
        print("Path check complete. Exiting.")
        return
    
    if args.mode in ("all", "train_word"):
        print("\n[Step 1] Training CNN Word Recognition Model …\n")
        from training.train_word_model import train_word_model
        train_word_model()
        
    if args.mode in ("all", "train_sentence"):
        print(f"\n[Step 2] Training {args.sentence_model.upper()} "
              f"Sentence Model …\n")
        from training.train_sentence_model import train_sentence_model
        train_sentence_model(model_type=args.sentence_model)

    if args.mode in ("all", "inference"):
        print("\n[Step 3] Launching Real‑Time Inference …\n")
        from inference.realtime_prediction import run_realtime
        run_realtime(sentence_model_type=args.sentence_model)

    print("\n[Done] Pipeline complete.")


if __name__ == "__main__":
    main()
