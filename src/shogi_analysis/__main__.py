"""Run with: python -m shogi_analysis --data-dir data/raw --output-dir outputs/tables"""
import argparse
from pathlib import Path
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Shogi Yearbook text analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    counts = run(args.data_dir, args.output_dir)
    print("Extracted utterances by year:", counts)
    print("CSV outputs:", args.output_dir)


if __name__ == "__main__":
    main()
