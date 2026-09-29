#!/usr/bin/env bash
set -euo pipefail

for model in meso xception freq_cnn hybrid; do
  python -m src.train --config "config/${model}.yaml"
done

