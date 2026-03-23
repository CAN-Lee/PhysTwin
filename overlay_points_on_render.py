"""
Overlay simulated particles, hand/controller points, and GT points onto
rendered Gaussian images or original RGB video frames.

Produces per-frame overlay PNGs and a composite MP4 video.

Usage:
    # Overlay on Gaussian-rendered images (default)
    python overlay_points_on_render.py --case_name double_lift_sloth

    # Overlay on original RGB video (camera 0)
    python overlay_points_on_render.py --case_name double_lift_sloth --bg_mode rgb

    # Only specific point types
    python overlay_points_on_render.py --case_name double_lift_sloth --no_gt

    # Custom output
    python overlay_points_on_render.py --case_name double_lift_sloth --output_dir ./my_overlay
"""

import os
import argparse
import pickle
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import subprocess

SERIF_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def load_camera(data_root, case_name, cam_idx=0):
    """Load camera intrinsics and extrinsics from the original data."""
    with open(os.path.join(data_root, case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    with open(os.path.join(data_root, case_name, "metadata.json"), "r") as f:
        meta = json.load(f)

    K = np.array(meta["intrinsics"][cam_idx], dtype=np.float64)
    W, H = meta["WH"]
    c2w = np.array(c2ws[cam_idx], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    return K, w2c, W, H


def project_points(pts_3d, K, w2c):
    """Project Nx3 world-space points to Nx2 pixel coords. Returns (uv, mask)."""
    if len(pts_3d) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    pts = np.asarray(pts_3d, dtype=np.float64)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_pts = (R @ pts.T).T + t  # (N, 3)

    depth = cam_pts[:, 2]
    valid = depth > 0.01

    uv = np.zeros((len(pts), 2), dtype=np.float32)
    if valid.any():
        proj = (K @ cam_pts[valid].T).T  # (M, 3)
        uv[valid, 0] = proj[:, 0] / proj[:, 2]
        uv[valid, 1] = proj[:, 1] / proj[:, 2]

    return uv, valid


def draw_points(img, uv, mask, color, radius=2, alpha=0.6, subsample=1):
    """Draw projected points on image with transparency."""
    W, H = img.shape[1], img.shape[0]
    overlay = img.copy()
    pts = uv[mask]
    if subsample > 1:
        idx = np.random.RandomState(42).choice(len(pts), min(len(pts), len(pts) // subsample), replace=False)
        pts = pts[idx]
    for u, v in pts:
        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= u_i < W and 0 <= v_i < H:
            cv2.circle(overlay, (u_i, v_i), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def load_rgb_frame(data_root, case_name, cam_idx, frame_idx):
    """Load a frame from the original RGB video."""
    video_dir = os.path.join(data_root, case_name, "color", str(cam_idx))
    if os.path.isdir(video_dir):
        # Frames are stored as individual images
        candidates = [
            os.path.join(video_dir, f"{frame_idx}.png"),
            os.path.join(video_dir, f"{frame_idx:05d}.png"),
            os.path.join(video_dir, f"{frame_idx}.jpg"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return cv2.imread(c)

    # Try video file
    video_path = os.path.join(data_root, case_name, "color", f"{cam_idx}.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    return None


def load_gs_rendered_frame(gs_dir, case_name, view_idx, frame_idx):
    """Load a frame from the Gaussian-rendered output."""
    path = os.path.join(gs_dir, case_name, str(view_idx), f"{frame_idx:05d}.png")
    if os.path.exists(path):
        return cv2.imread(path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Overlay sim/hand/GT points on rendered frames")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data/different_types")
    parser.add_argument("--inference_dir", type=str, default="./output_3/mpm_inference",
                        help="Dir containing {case_name}/inference.pkl")
    parser.add_argument("--gs_render_dir", type=str, default="./gaussian_output_dynamic_mpm",
                        help="Dir containing Gaussian-rendered frames")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (auto: _overlay for gs, _overlay_rgb for rgb, _overlay_white for white)")
    parser.add_argument("--cam_idx", type=int, default=0, help="Camera index from calibrate.pkl")
    parser.add_argument("--bg_mode", type=str, default="gs", choices=["gs", "rgb", "white"],
                        help="Background: 'gs'=Gaussian render, 'rgb'=original video, 'white'=blank")
    parser.add_argument("--no_sim", action="store_true", help="Skip simulated particles")
    parser.add_argument("--no_hand", action="store_true", help="Skip hand/controller points")
    parser.add_argument("--no_gt", action="store_true", help="Skip GT surface points")
    parser.add_argument("--sim_subsample", type=int, default=3,
                        help="Subsample ratio for sim particles (1=all)")
    parser.add_argument("--gt_subsample", type=int, default=3,
                        help="Subsample ratio for GT points (1=all)")
    parser.add_argument("--fps", type=int, default=15, help="Match original img2video.py default")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = {"gs": "_overlay", "rgb": "_overlay_rgb", "white": "_overlay_white"}
        args.output_dir = args.gs_render_dir + suffix[args.bg_mode]

    case_name = args.case_name

    # 1. Load camera
    K, w2c, W, H = load_camera(args.data_root, case_name, args.cam_idx)
    print(f"Camera {args.cam_idx}: {W}x{H}, fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    # 2. Load point data
    # Simulated particles
    sim_traj = None
    if not args.no_sim:
        inf_path = os.path.join(args.inference_dir, case_name, "inference.pkl")
        if os.path.exists(inf_path):
            with open(inf_path, "rb") as f:
                sim_traj = pickle.load(f)
            print(f"Simulated particles: {sim_traj.shape}")
        else:
            print(f"WARNING: {inf_path} not found, skipping sim particles")

    # GT + Controller from final_data.pkl
    data_path = os.path.join(args.data_root, case_name, "final_data.pkl")
    gt_tracks = None
    ctrl_pts = None
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if not args.no_gt:
        gt_tracks = np.array(data["object_points"], dtype=np.float32)
        print(f"GT surface points: {gt_tracks.shape}")

    if not args.no_hand:
        ctrl_pts = np.array(data["controller_points"], dtype=np.float32)
        print(f"Controller/hand points: {ctrl_pts.shape}")

    # Determine number of frames
    T = sim_traj.shape[0] if sim_traj is not None else gt_tracks.shape[0]
    if gt_tracks is not None:
        T = min(T, gt_tracks.shape[0])
    if ctrl_pts is not None:
        T = min(T, ctrl_pts.shape[0])
    print(f"Rendering {T} frames")

    # 3. Output setup
    out_frame_dir = os.path.join(args.output_dir, case_name, "overlay_frames")
    os.makedirs(out_frame_dir, exist_ok=True)

    # 4. Process each frame
    for t in tqdm(range(T), desc="Overlaying"):
        # Load background
        if args.bg_mode == "gs":
            bg = load_gs_rendered_frame(args.gs_render_dir, case_name, 0, t)
            if bg is not None:
                bg = cv2.resize(bg, (W, H))
        elif args.bg_mode == "rgb":
            bg = load_rgb_frame(args.data_root, case_name, args.cam_idx, t)
        else:
            bg = None

        if bg is None:
            bg = np.ones((H, W, 3), dtype=np.uint8) * 255

        frame = bg.copy()

        # GT points (green, draw first = bottom layer)
        if gt_tracks is not None:
            gt_valid = np.linalg.norm(gt_tracks[t], axis=-1) > 1e-5
            uv, mask = project_points(gt_tracks[t][gt_valid], K, w2c)
            draw_points(frame, uv, mask, color=(0, 200, 0), radius=3, alpha=0.45,
                        subsample=args.gt_subsample)

        # Simulated particles (blue, middle layer)
        if sim_traj is not None:
            uv, mask = project_points(sim_traj[t], K, w2c)
            draw_points(frame, uv, mask, color=(255, 100, 0), radius=3, alpha=0.55,
                        subsample=args.sim_subsample)

        # Controller/hand points (red, top layer - most visible)
        if ctrl_pts is not None:
            uv, mask = project_points(ctrl_pts[t], K, w2c)
            draw_points(frame, uv, mask, color=(0, 0, 255), radius=5, alpha=0.9, subsample=1)

        # Legend (white background box + serif font text via PIL)
        legend_entries = []
        if sim_traj is not None:
            legend_entries.append(("Predicted Particles", (255, 100, 0)))
        if gt_tracks is not None:
            legend_entries.append(("GT Points", (0, 200, 0)))
        if ctrl_pts is not None:
            legend_entries.append(("Action", (0, 0, 255)))

        if legend_entries:
            font_size = 15
            row_h = 26
            box_h = len(legend_entries) * row_h + 12
            box_w = 210
            cv2.rectangle(frame, (4, 4), (4 + box_w, 4 + box_h), (255, 255, 255), -1)
            cv2.rectangle(frame, (4, 4), (4 + box_w, 4 + box_h), (120, 120, 120), 1)

            # Draw colored dots with OpenCV (anti-aliased circles)
            legend_y = 24
            for _, color in legend_entries:
                cv2.circle(frame, (20, legend_y), 6, color, -1, lineType=cv2.LINE_AA)
                legend_y += row_h

            # Draw text labels with PIL (DejaVu Serif ≈ Times New Roman)
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype(SERIF_FONT_PATH, font_size)
            except OSError:
                font = ImageFont.load_default()
            legend_y = 14
            for label, _ in legend_entries:
                draw.text((34, legend_y), label, font=font, fill=(0, 0, 0))
                legend_y += row_h
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(out_frame_dir, f"{t:05d}.png"), frame)

    # 5. Encode video
    video_path = os.path.join(args.output_dir, case_name, "overlay.mp4")
    input_pattern = os.path.join(os.path.abspath(out_frame_dir), "%05d.png")
    ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-r", str(args.fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        os.path.abspath(video_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Overlay video saved: {video_path}")

    # 6. Also make a side-by-side with the Gaussian render if available
    if args.bg_mode != "gs":
        return  # Skip side-by-side if not using GS background

    gs_video = os.path.join(args.gs_render_dir, case_name, "0.mp4")
    if os.path.exists(gs_video):
        sbs_path = os.path.join(args.output_dir, case_name, "side_by_side.mp4")
        cmd_sbs = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", os.path.abspath(gs_video),
            "-i", os.path.abspath(video_path),
            "-filter_complex", f"[0:v]scale=-2:480[left];[1:v]scale=-2:480[right];[left][right]hstack,fps={args.fps}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            os.path.abspath(sbs_path),
        ]
        try:
            subprocess.run(cmd_sbs, check=True)
            print(f"Side-by-side video saved: {sbs_path}")
        except subprocess.CalledProcessError:
            print("Side-by-side encoding failed (non-critical)")


if __name__ == "__main__":
    main()
