#!/bin/bash

# Default values
prediction_dir="${1:-./experiments}"  # First argument or default to ./experiments

# Auto-generate output directory based on prediction_dir
if [[ "$prediction_dir" == *"physics_net"* ]]; then
    output_dir="./gaussian_output_dynamic_physics_net"
else
    output_dir="./gaussian_output_dynamic"
fi

# views=("0" "1" "2")
views=("0")

# If second argument is provided, use it as scene list (comma-separated)
# Otherwise, auto-detect scenes from prediction_dir
if [ -n "$2" ]; then
    # Manual scene list provided
    IFS=',' read -ra scenes <<< "$2"
else
    # Auto-detect: find all directories in prediction_dir that have inference.pkl
    echo "Auto-detecting scenes from $prediction_dir..."
    scenes=()
    if [ -d "$prediction_dir" ]; then
        for scene_dir in "$prediction_dir"/*; do
            if [ -d "$scene_dir" ] && [ -f "$scene_dir/inference.pkl" ]; then
                scene_name=$(basename "$scene_dir")
                # Also check if gaussian_data exists for this scene
                if [ -d "./data/gaussian_data/$scene_name" ]; then
                    scenes+=("$scene_name")
                    echo "  Found: $scene_name"
                else
                    echo "  Skipping $scene_name (no gaussian_data found)"
                fi
            fi
        done
    fi
    
    if [ ${#scenes[@]} -eq 0 ]; then
        echo "Error: No valid scenes found in $prediction_dir"
        echo "Please ensure inference.pkl exists in subdirectories and corresponding gaussian_data exists."
        exit 1
    fi
fi

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

echo "Using prediction directory: $prediction_dir"
echo "Output directory: $output_dir"
echo "Processing scenes: ${scenes[*]}"

for scene_name in "${scenes[@]}"; do

    python gs_render_dynamics.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ./gaussian_output/${scene_name}/${exp_name} \
        --name ${scene_name} \
        --prediction_dir ${prediction_dir} \
        --output_dir ${output_dir}

    for view_name in "${views[@]}"; do
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done