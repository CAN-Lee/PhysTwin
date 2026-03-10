import os

# Check if we're in headless environment (before importing open3d)
def _check_headless():
    """Check if running in a headless environment (no display available)"""
    if os.name == 'posix':
        display = os.environ.get('DISPLAY')
        if not display or display.strip() == '':
            return True
    return False

# Force Open3D to use OSMesa for headless rendering (must be set before importing open3d)
if _check_headless():
    # Try to use OSMesa for software rendering
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    # Additional environment variables for OSMesa
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import open3d as o3d
import numpy as np
import torch
import time
import cv2
import subprocess
import tempfile
import shutil
from .config import cfg
from . import logger
# Optional imports for headless environments
try:
    import pyrender
except (ImportError, OSError):
    pyrender = None
try:
    import trimesh
except (ImportError, OSError):
    trimesh = None


def _is_headless_environment():
    """Check if running in a headless environment (no display available)"""
    # Check DISPLAY environment variable (Linux)
    if os.name == 'posix':
        display = os.environ.get('DISPLAY')
        if not display or display.strip() == '':
            return True
    # Check for common headless indicators
    if os.environ.get('PYOPENGL_PLATFORM') == 'osmesa':
        return True
    # Try to detect if we can actually create a window (will be caught later)
    return False


def _create_video_from_frames(frame_paths, output_path, fps, cleanup=True):
    """
    Create a video from a list of frame image paths using ffmpeg.
    
    Args:
        frame_paths: List of paths to frame images (should be in order)
        output_path: Path to output video file
        fps: Frame rate for the video
        cleanup: Whether to clean up frame images after creating video
    
    Returns:
        True if successful, False otherwise
    """
    if len(frame_paths) == 0:
        if hasattr(logger, 'warning'):
            logger.warning("No frames provided, cannot create video.")
        return False
    
    # Get the directory of the first frame
    frame_dir = os.path.dirname(frame_paths[0])
    
    try:
        # Use ffmpeg to combine frames into video
        # Pattern: frame_%06d.png matches frame_000000.png, frame_000001.png, etc.
        # First, we need to ensure frames are numbered sequentially starting from 0
        # Get the base filename pattern
        first_frame_name = os.path.basename(frame_paths[0])
        # Extract the pattern (e.g., "frame_000000.png" -> "frame_%06d.png")
        if '_' in first_frame_name:
            base_name = first_frame_name.split('_')[0]
            ext = os.path.splitext(first_frame_name)[1]
            pattern = f"{base_name}_%06d{ext}"
        else:
            # Fallback: use frame number in filename
            pattern = "frame_%06d.png"
        
        # Try to find ffmpeg with libx264 support
        # Prefer system ffmpeg over conda ffmpeg (conda version may not have libx264)
        ffmpeg_paths = ['/usr/bin/ffmpeg', 'ffmpeg']
        ffmpeg_binary = None
        
        for path in ffmpeg_paths:
            try:
                # Check if this ffmpeg supports libx264
                check_cmd = [path, '-encoders']
                check_result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if 'libx264' in check_result.stdout:
                    ffmpeg_binary = path
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        if ffmpeg_binary is None:
            # Fallback: try to use any available ffmpeg with alternative codec
            ffmpeg_binary = 'ffmpeg'
            codec = 'libopenh264'  # Usually available in conda ffmpeg
            logger.warning(f"libx264 not found, trying {codec} instead")
        else:
            codec = 'libx264'
            if ffmpeg_binary != 'ffmpeg':
                logger.info(f"Using system ffmpeg at {ffmpeg_binary} (supports libx264)")
        
        ffmpeg_cmd = [
            ffmpeg_binary,
            '-y',  # Overwrite output file if it exists
            '-r', str(fps),  # Input frame rate
            '-i', os.path.join(frame_dir, pattern),  # Input pattern
            '-c:v', codec,  # Video codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-r', str(fps),  # Output frame rate
            output_path
        ]
        
        if hasattr(logger, 'info'):
            logger.info(f"Running ffmpeg to create video: {output_path}")
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if hasattr(logger, 'info'):
            logger.info(f"Successfully created video: {output_path}")
        
        # Clean up frame images if requested
        if cleanup:
            for frame_path in frame_paths:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception as e:
                    if hasattr(logger, 'warning'):
                        logger.warning(f"Failed to remove frame {frame_path}: {e}")
            
            # Try to remove the directory if it's empty
            try:
                if os.path.exists(frame_dir) and not os.listdir(frame_dir):
                    os.rmdir(frame_dir)
            except:
                pass
        
        return True
                
    except subprocess.CalledProcessError as e:
        if hasattr(logger, 'error'):
            logger.error(f"ffmpeg failed to create video: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
        # Clean up frame images even if video creation failed
        if cleanup:
            for frame_path in frame_paths:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception as cleanup_error:
                    if hasattr(logger, 'warning'):
                        logger.warning(f"Failed to remove frame {frame_path}: {cleanup_error}")
        return False
    except FileNotFoundError:
        if hasattr(logger, 'error'):
            logger.error("ffmpeg not found. Please install ffmpeg to use video synthesis.")
        raise RuntimeError("ffmpeg not found. Please install ffmpeg: apt-get install ffmpeg (or equivalent)")


def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
):
    # Deprecated function, use visualize_pc instead
    FPS = cfg.FPS
    width, height = cfg.WH
    intrinsic = cfg.intrinsics[vis_cam_idx]
    w2c = cfg.w2cs[vis_cam_idx]
    
    # Early exit if in headless environment and cannot save video
    if _is_headless_environment() and not save_video and visualize:
        if hasattr(logger, 'warning'):
            logger.warning("Running in headless environment and cannot visualize. Skipping visualization.")
        return

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    else:
        if object_colors.shape[1] < object_points.shape[1]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            object_colors.shape[0],
                            object_points.shape[1] - object_colors.shape[1],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=1,
            )

    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    # Check headless environment early to avoid segfault
    is_headless = _is_headless_environment()
    
    # In headless mode, only proceed if we're saving video (not visualizing)
    if is_headless and visualize and not save_video:
        if hasattr(logger, 'warning'):
            logger.warning("Headless environment detected. Cannot visualize interactively. Skipping.")
        return
    
    vis = o3d.visualization.Visualizer()
    
    # Try to create window, handle headless environment gracefully
    window_created = False
    try:
        # In headless mode, always create invisible window
        window_visible = visualize and not is_headless
        vis.create_window(visible=window_visible, width=width, height=height)
        window_created = True
    except Exception as e:
        # If window creation fails, try one more time with invisible window
        if save_video:
            try:
                vis.create_window(visible=False, width=width, height=height)
                window_created = True
                if hasattr(logger, 'warning'):
                    logger.warning(f"Running in headless mode, using offscreen rendering: {e}")
            except Exception as e2:
                # Complete failure - cannot render at all
                if hasattr(logger, 'warning'):
                    logger.warning(f"Cannot create visualization window in headless environment: {e2}. Skipping visualization.")
                # Clean up and return
                try:
                    vis.destroy_window()
                except:
                    pass
                return
        else:
            # Cannot visualize and not saving video - just return
            if hasattr(logger, 'warning'):
                logger.warning(f"Cannot create visualization window: {e}. Skipping visualization.")
            try:
                vis.destroy_window()
            except:
                pass
            return

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    # Initialize frame storage for ffmpeg-based video creation
    temp_frame_dir = None
    frame_paths = []
    if save_video:
        # Create temporary directory for storing frames
        temp_frame_dir = tempfile.mkdtemp(prefix="video_frames_")
        if hasattr(logger, 'info'):
            logger.info(f"Using ffmpeg for video synthesis. Temporary frames directory: {temp_frame_dir}")

    if controller_points is not None:
        controller_meshes = []
        prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_visibilities is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_visibilities[i])[0], :]
            )
        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            if controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
            # Adjust the viewpoint
            if window_created:
                view_control = vis.get_view_control()
                if view_control is None:
                    # Headless mode: skip view control setup, will use default view
                    if hasattr(logger, 'warning'):
                        logger.warning("View control is None (headless mode), using default camera view")
                else:
                    try:
                        camera_params = o3d.camera.PinholeCameraParameters()
                        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                            width, height, intrinsic
                        )
                        camera_params.intrinsic = intrinsic_parameter
                        camera_params.extrinsic = w2c
                        view_control.convert_from_pinhole_camera_parameters(
                            camera_params, allow_arbitrary=True
                        )
                    except Exception as e:
                        if hasattr(logger, 'warning'):
                            logger.warning(f"Could not set camera parameters: {e}. Using default view.")
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            if controller_points is not None:
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
        if window_created:
            vis.poll_events()
            vis.update_renderer()

            # Capture frame and save as image if save_video is True
        if save_video:
            try:
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                frame = (frame * 255).astype(np.uint8)
            except Exception as e:
                if hasattr(logger, 'warning'):
                    logger.warning(f"Could not capture frame {i}: {e}. Skipping this frame.")
                continue
            
            # Apply overlay if configured
            if cfg.overlay_path is not None:
                # Get the mask where the pixel is white
                mask = np.all(frame == [255, 255, 255], axis=-1)
                image_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
                if os.path.exists(image_path):
                    overlay = cv2.imread(image_path)
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    frame[mask] = overlay[mask]
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save frame as PNG with zero-padded frame number for ffmpeg
            frame_filename = f"frame_{i:06d}.png"
            frame_path = os.path.join(temp_frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame_bgr)
            frame_paths.append(frame_path)

        if visualize and window_created:
            time.sleep(1 / FPS)

    if window_created:
        try:
            vis.destroy_window()
        except:
            pass  # Ignore errors when destroying window in headless mode
    
    # Use ffmpeg to combine frames into video
    if save_video:
        if len(frame_paths) == 0:
            if hasattr(logger, 'warning'):
                logger.warning("No frames captured, cannot create video.")
            # Clean up temp directory
            if temp_frame_dir and os.path.exists(temp_frame_dir):
                try:
                    shutil.rmtree(temp_frame_dir)
                except:
                    pass
            return
        
        # Use helper function to create video from frames
        success = _create_video_from_frames(frame_paths, save_path, FPS, cleanup=True)
        
        # Clean up temporary frame directory
        if temp_frame_dir and os.path.exists(temp_frame_dir):
            try:
                shutil.rmtree(temp_frame_dir)
                if hasattr(logger, 'debug'):
                    logger.debug(f"Cleaned up temporary frames directory: {temp_frame_dir}")
            except Exception as e:
                if hasattr(logger, 'warning'):
                    logger.warning(f"Failed to clean up temporary directory {temp_frame_dir}: {e}")
        
        if not success:
            raise RuntimeError("Failed to create video from frames")
