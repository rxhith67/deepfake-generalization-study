#!/usr/bin/env bash
set -euo pipefail

for model in meso xception freq_cnn hybrid; do
  checkpoint="outputs/checkpoints/${model}_best.pt"
  python -m src.eval.cross_generation \
    --checkpoint "$checkpoint" \
    --in-domain-csv data/splits/test.csv \
    --cross-csv data/splits/diffusion_test.csv \
    --output "outputs/tables/${model}_cross_generation.json"
  python -m src.interpret.gradcam \
    --checkpoint "$checkpoint" \
    --csv data/splits/test.csv \
    --output "outputs/figures/${model}_gradcam.png"
done

python -m src.eval.robustness \
  --checkpoint \
    outputs/checkpoints/meso_best.pt \
    outputs/checkpoints/xception_best.pt \
    outputs/checkpoints/freq_cnn_best.pt \
    outputs/checkpoints/hybrid_best.pt \
  --csv data/splits/test.csv \
  --qualities 100 90 60 40 20 10 \
  --output-csv outputs/tables/compression_all_models.csv \
  --output-figure outputs/figures/compression_all_models.png
