import torch
import os
import argparse
import pickle
from deform_fields import MeshSequenceTrainer, render_mesh_sequence

def main():
    parser = argparse.ArgumentParser(description="Train MeshSequence deformation field.")
    parser.add_argument("--case_dir", type=str, default=None, help="Directory containing final_data.pkl and shape/object.glb")
    parser.add_argument("--data_path", type=str, default=None, help="Path to final_data.pkl (overrides case_dir)")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to canonical mesh (.obj/.ply/.glb, overrides case_dir)")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of SE(3) clusters")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_temporal", type=float, default=0.1, help="Temporal smoothness weight")
    parser.add_argument("--lambda_arap", type=float, default=0.1, help="ARAP regularization weight")
    parser.add_argument("--lambda_rot_reg", type=float, default=0.01, help="Rotation regularization weight")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for logs and checkpoints (defaults to deform_fields/logs/...)")
    parser.add_argument("--output_name", type=str, default="mesh_sequence_results.pkl", help="Name of the output results file")
    
    args = parser.parse_args()
    
    # 1. Resolve Paths from case_dir
    data_path = args.data_path
    mesh_path = args.mesh_path
    log_dir = args.log_dir
    
    if args.case_dir:
        if data_path is None:
            data_path = os.path.join(args.case_dir, "final_data.pkl")
        
        if mesh_path is None:
            # Try potential mesh paths
            potential_meshes = [
                os.path.join(args.case_dir, "shape", "object.glb"),
                os.path.join(args.case_dir, "shape", "object.ply"),
                os.path.join(args.case_dir, "shape", "object.obj"),
                os.path.join(args.case_dir, "object.glb"),
            ]
            for p in potential_meshes:
                if os.path.exists(p):
                    mesh_path = p
                    break
        
        if log_dir is None:
            case_name = os.path.basename(args.case_dir.rstrip("/"))
            log_dir = os.path.join("deform_fields", "logs", "mesh_sequence", case_name)

    # Defaults if not provided and no case_dir
    if data_path is None:
        raise ValueError("Either --case_dir or --data_path must be provided.")
    if log_dir is None:
        log_dir = "deform_fields/logs/mesh_sequence/default"

    print(f"--- Configuration ---")
    print(f"Data Path: {data_path}")
    print(f"Mesh Path: {mesh_path}")
    print(f"Log Dir:   {log_dir}")
    print(f"----------------------")

    cfg = {
        'data_path': data_path,
        'mesh_path': mesh_path,
        'num_clusters': args.num_clusters,
        'lr': args.lr,
        'lambda_temporal': args.lambda_temporal,
        'lambda_arap': args.lambda_arap,
        'lambda_rot_reg': args.lambda_rot_reg,
        'log_dir': log_dir,
        'save_freq': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    trainer = MeshSequenceTrainer(cfg)
    trainer.train(num_epochs=args.num_epochs)
    
    output_path = os.path.join(log_dir, args.output_name)
    trainer.save_results(output_path)
    print(f"Training complete. Results saved to {output_path}")

    # 4. Render Video
    print("Starting video rendering...")
    with open(output_path, 'rb') as f:
        results = pickle.load(f)
    
    video_path = os.path.join(log_dir, "mesh_sequence_visualization.mp4")
    render_mesh_sequence(
        results['vertices'], 
        results['faces'], 
        video_path,
        fps=30
    )

if __name__ == "__main__":
    main()
