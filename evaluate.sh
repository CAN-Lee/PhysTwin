#!/bin/bash

# Default to experiments directory for backward compatibility
PREDICTION_DIR="${1:-experiments}"

echo "Evaluating predictions from: $PREDICTION_DIR"

# Run evaluation scripts with prediction directory argument
python evaluate_chamfer.py --prediction_dir "$PREDICTION_DIR"
python evaluate_track.py --prediction_path "$PREDICTION_DIR"
python gaussian_splatting/evaluate_render.py --prediction_dir "$PREDICTION_DIR"