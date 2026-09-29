"""Print official dataset access guidance (datasets require licence acceptance)."""

from __future__ import annotations

import argparse

DATASETS = {
    "faceforensics": "https://github.com/ondyari/FaceForensics",
    "celeb_df": "https://github.com/yuezunli/celeb-deepfakeforensics",
    "ai_genbench": "https://github.com/MI-BioLab/AI-GenBench",
    "genimage": "https://github.com/GenImage-Dataset/GenImage",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()
    print(f"Official access page for {args.dataset}: {DATASETS[args.dataset]}")
    print("Accept the dataset's terms, download it manually, and follow DATA.md.")


if __name__ == "__main__":
    main()

