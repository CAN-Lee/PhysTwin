import os
import glob
import argparse
import multiprocessing as mp
import subprocess
from queue import Empty
import time

def worker(gpu_id, task_queue, config_path, iters):
    gpu_id = str(gpu_id).strip()
    while True:
        try:
            scene_id = task_queue.get_nowait()
        except Empty:
            break
            
        selected_config = config_path if config_path else "configs/softbody.yaml"
        print(f"\n[GPU {gpu_id}] >>> [SOFTBODY MODE] Training Scene: {scene_id} using {selected_config}", flush=True)
        
        cmd = ["python", "train_mpm.py", "--case_name", scene_id, "--iters", str(iters), "--config", selected_config]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] <<< Finished {scene_id} Successfully", flush=True)
        except subprocess.CalledProcessError:
            print(f"[GPU {gpu_id}] !!! Error in {scene_id}", flush=True)
        finally:
            task_queue.task_done()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/softbody.yaml", help="Path to config file")
    args = parser.parse_args()
    
    from omegaconf import OmegaConf

    scenes_paths = sorted(glob.glob(os.path.join(args.base_path, "*")))
    # Filter zebra, sloth, dinosor
    valid_scenes = [os.path.basename(p) for p in scenes_paths if os.path.isdir(p) and 
                    any(kw in os.path.basename(p).lower() for kw in ['zebra', 'sloth', 'dinosor'])]

    # [NEW] Check for target_scenes AND iters/gpus in config
    if os.path.exists(args.config):
        try:
            cfg = OmegaConf.load(args.config)
            # 1. Override params if present in config
            if 'train' in cfg:
                if 'iters' in cfg.train:
                    print(f"Config Override: Using iters={cfg.train.iters} from {args.config}")
                    args.iters = cfg.train.iters
                if 'gpus' in cfg.train:
                    print(f"Config Override: Using gpus={cfg.train.gpus} from {args.config}")
                    args.gpus = str(cfg.train.gpus)
                if 'num_workers' in cfg.train:
                    print(f"Config Override: Using num_workers={cfg.train.num_workers} from {args.config}")
                    args.num_workers = int(cfg.train.num_workers)
            
            # 2. Filter scenes
            if 'target_scenes' in cfg and cfg.target_scenes:
                targets = list(cfg.target_scenes)
                print(f"Filter: Applying target_scenes from config: {targets}")
                valid_scenes = [s for s in valid_scenes if s in targets]
        except Exception as e:
            print(f"WARNING: Failed to parse target_scenes from {args.config}: {e}")

    print(f"Found {len(valid_scenes)} SOFTBODY scenes. Using GPU(s): {args.gpus}. Config: {args.config}", flush=True)
    task_queue = mp.JoinableQueue()
    for scene_id in valid_scenes:
        task_queue.put(scene_id)

    gpu_list = args.gpus.split(",")
    processes = []
    if os.name != 'nt': mp.set_start_method('fork', force=True)

    for gpu_id in gpu_list:
        for i in range(args.num_workers):
            p = mp.Process(target=worker, args=(gpu_id, task_queue, args.config, args.iters))
            p.start()
            processes.append(p)
            time.sleep(0.5)

    try:
        while not task_queue.empty() or any(p.is_alive() for p in processes):
            for p in processes: p.join(timeout=1.0)
            if task_queue.empty() and not any(p.is_alive() for p in processes): break
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        os._exit(1)
    print("\n[ALL DONE] All softbody scenes processed.", flush=True)

if __name__ == "__main__":
    main()
