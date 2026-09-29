#!/usr/bin/env bash
set -euo pipefail

# Example only: organise source videos before running this script so labels and
# methods remain explicit. Dataset downloads are intentionally not automated.
python -m src.data.extract_frames --input data/raw/faceforensics/original --output data/frames/real --num-frames 30
python -m src.data.detect_faces --input data/frames/real --output data/processed/faceforensics/real --label 0 --method real --dataset faceforensics --manifest data/real_manifest.csv

echo "Prepare each manipulation method likewise, combine manifests, then run:"
echo "python -m src.data.splits --manifest data/manifest.csv --output-dir data/splits"

