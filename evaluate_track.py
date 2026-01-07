import pickle
import glob
import csv
import json
import numpy as np
import argparse
import os
from scipy.spatial import KDTree

base_path = "./data/different_types"
output_file = "results/final_track.csv"


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracking error for predictions")
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="experiments",
        help="Directory containing prediction results (default: experiments). Use experiments_physics_net for PhysicsNet results."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file path (default: results/final_track.csv or results/final_track_physics_net.csv if using experiments_physics_net)"
    )
    args = parser.parse_args()
    
    prediction_path = args.prediction_path
    if args.output_file is None:
        # Auto-generate output filename based on prediction_path
        if "physics_net" in prediction_path:
            output_file = "results/final_track_physics_net.csv"
        else:
            output_file = "results/final_track.csv"
    else:
        output_file = args.output_file
    
    print(f"Using prediction path: {prediction_path}")
    print(f"Output file: {output_file}")

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(
        [
            "Case Name",
            "Train Track Error",
            "Test Track Error",
        ]
    )

    dir_names = glob.glob(f"{base_path}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        # if case_name != "single_lift_dinosor":
        #     continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        frame_len = split["frame_len"]
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        # Check if inference.pkl exists
        inference_path = f"{prediction_path}/{case_name}/inference.pkl"
        if not os.path.exists(inference_path):
            print(f"  Skipping {case_name} (inference.pkl not found in {prediction_path})")
            continue

        with open(inference_path, "rb") as f:
            vertices = pickle.load(f)

        with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "rb") as f:
            gt_track_3d = pickle.load(f)

        # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
        mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

        kdtree = KDTree(vertices[0])
        dis, idx = kdtree.query(gt_track_3d[0][mask])

        train_track_error = evaluate_prediction(
            1, train_frame, vertices, gt_track_3d, idx, mask
        )
        test_track_error = evaluate_prediction(
            train_frame, test_frame, vertices, gt_track_3d, idx, mask
        )
        writer.writerow([case_name, train_track_error, test_track_error])
    file.close()
