"""Command-line entry point for generating synthetic document datasets."""
import argparse
from pathlib import Path

from .dataset import generate_dataset


def main():
    """Parse args and generate dataset."""
    parser = argparse.ArgumentParser(description="Generate a synthetic document dataset")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    out_dir = generate_dataset(Path(args.config))
    print(f"Wrote dataset to: {out_dir}")


if __name__ == "__main__":
    main()
