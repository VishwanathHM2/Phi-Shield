#!/usr/bin/env python3
"""
run_pipeline.py
Master script to run the full PHI de-identification pipeline end-to-end.
"""
import sys, subprocess, argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent

def run(cmd, cwd=None):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=cwd or BASE, check=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="PHI De-identification Pipeline")
    parser.add_argument("--skip-data",  action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--eval-only",  action="store_true", help="Only run evaluation")
    parser.add_argument("--ui",         action="store_true", help="Launch Streamlit UI")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_data and not args.eval_only:
        run([py, "data_generation/generate_dataset_from_reports.py"])

    if not args.skip_train and not args.eval_only:
        run([py, "training/train.py"])

    run([py, "training/evaluate.py"])

    if args.ui:
        run(["streamlit", "run", "ui/streamlit_app.py"])

if __name__ == "__main__":
    main()
