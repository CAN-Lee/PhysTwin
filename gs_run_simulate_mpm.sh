#!/bin/bash
#
# Render Gaussian Splatting dynamic videos driven by Hybrid Physics-Neural MPM
# simulator's particle trajectories.
#
# Prerequisite: run script_inference_mpm.py first to generate inference.pkl files.
#
# Usage:
#   bash gs_run_simulate_mpm.sh                          # all scenes, default dirs
#   bash gs_run_simulate_mpm.sh ./output_3/mpm_inference # custom prediction dir
#   bash gs_run_simulate_mpm.sh ./output_3/mpm_inference "double_lift_sloth,double_lift_zebra"
#

prediction_dir="${1:-./output_3/mpm_inference}"
output_dir="./gaussian_output_dynamic_mpm"

views=("0")

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

# If second argument is provided, use it as scene list (comma-separated)
# Otherwise, auto-detect scenes from prediction_dir
if [ -n "$2" ]; then
    IFS=',' read -ra scenes <<< "$2"
else
    echo "Auto-detecting scenes from $prediction_dir..."
    scenes=()
    if [ -d "$prediction_dir" ]; then
        for scene_dir in "$prediction_dir"/*; do
            if [ -d "$scene_dir" ] && [ -f "$scene_dir/inference.pkl" ]; then
                scene_name=$(basename "$scene_dir")
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
        echo "Please run script_inference_mpm.py first to generate inference.pkl files."
        exit 1
    fi
fi

echo ""
echo "=== Hybrid MPM Gaussian Rendering ==="
echo "Prediction directory: $prediction_dir"
echo "Output directory:     $output_dir"
echo "Processing scenes:    ${scenes[*]}"
echo ""

for scene_name in "${scenes[@]}"; do
    echo ">>> Rendering: $scene_name"

    python gs_render_dynamics.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ./gaussian_output/${scene_name}/${exp_name} \
        --name ${scene_name} \
        --prediction_dir ${prediction_dir} \
        --output_dir ${output_dir}

    for view_name in "${views[@]}"; do
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

    # Overlay sim particles + hand + GT onto the rendered frames
    echo "    Overlaying points on: $scene_name"
    python overlay_points_on_render.py \
        --case_name ${scene_name} \
        --inference_dir ${prediction_dir} \
        --gs_render_dir ${output_dir} \
        --output_dir ${output_dir}_overlay \
        --bg_mode gs

    # Also render overlay on original RGB for comparison
    python overlay_points_on_render.py \
        --case_name ${scene_name} \
        --inference_dir ${prediction_dir} \
        --gs_render_dir ${output_dir} \
        --output_dir ${output_dir}_overlay_rgb \
        --bg_mode rgb

    echo "<<< Done: $scene_name"
done

echo ""
echo "=== All rendering complete ==="
echo "Videos saved to:"
echo "  Gaussian render: $output_dir/{scene_name}/0.mp4"
echo "  Overlay (GS bg): ${output_dir}_overlay/{scene_name}/overlay.mp4"
echo "  Overlay (RGB bg): ${output_dir}_overlay_rgb/{scene_name}/overlay.mp4"
