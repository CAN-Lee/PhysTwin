#!/usr/bin/env python3
"""
Headless PhysTwin MPM + browser mouse drag demo.

  pip install -r demo_web_mpm/requirements.txt
  cd /path/to/PhysTwin
  python demo_web_mpm/interactive_mpm_server.py \\
      --case_name single_push_rope --config configs/rope.yaml \\
      --host 0.0.0.0 --port 8765

Open http://<server>:8765 in a browser. Click a particle to grab, drag, release.

Requires: mpm.use_warp=true, CUDA, trained best_checkpoint.pt for the scene.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import socket
import struct
import sys
import warnings

warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor", module="warp")

# Repo root on sys.path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import warp as wp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
import uvicorn

from phys_expert.engine.trainer_mpm import PhysExpertMPMTrainer
from phys_expert.model.diff_simulator.warp_solver.warp_utils import torch2warp_float, torch2warp_vec3


def _pack_positions(pos: np.ndarray) -> bytes:
    """pos: (N,3) float32"""
    pos = np.ascontiguousarray(pos.astype(np.float32))
    n = pos.shape[0]
    header = struct.pack("<II", n, 0)
    return header + pos.tobytes()


def _connected_particle_indices(sim) -> list[int]:
    """Unique particle ids under PD control (Warp)."""
    if sim.controller_indices_wp is None or sim.num_connections == 0:
        return []
    idx = wp.to_torch(sim.controller_indices_wp).int().cpu().numpy()
    mask = wp.to_torch(sim.controller_mask_wp).int().cpu().numpy()
    flat = idx[mask > 0]
    if flat.size == 0:
        return []
    return sorted({int(x) for x in np.asarray(flat).ravel()})


class InteractiveMPMSession:
    def __init__(
        self,
        trainer: PhysExpertMPMTrainer,
        steps_per_tick: int,
        max_send_particles: int,
        use_neural: bool,
        settle_iters: int,
    ):
        if not getattr(trainer.cfg.mpm, "use_warp", False):
            raise RuntimeError("demo_web_mpm requires mpm.use_warp: true in config")
        self.trainer = trainer
        self.device = trainer.device
        self.dt = float(trainer.cfg.mpm.dt)
        self.steps_per_tick = int(steps_per_tick)
        self.max_send_particles = int(max_send_particles)
        self.use_neural = use_neural and trainer.residual_net is not None

        self.init_pos = (trainer.data["init_pos"].to(self.device) + trainer.auto_offset).detach()
        self.n_total = self.init_pos.shape[0]

        pc = trainer.data.get("particle_counts", {})
        n_pkl = pc.get("surface", 0) + pc.get("other_surface", 0) + pc.get("interior", 0)
        self.n_export = min(n_pkl, self.n_total) if n_pkl > 0 else self.n_total
        if self.n_export <= 0:
            self.n_export = self.n_total

        self._stride = max(1, int(np.ceil(self.n_total / max(1, self.max_send_particles))))
        expert_order = ["nh", "co", "st", "fi"]
        active = getattr(trainer.cfg.mpm, "active_experts", expert_order)
        self.mask_active = [1 if e in active else 0 for e in expert_order]
        self.active_mask_wp = wp.array(self.mask_active, dtype=wp.int32, device=trainer.simulator.warp_device)
        self.moe_params_wp = self._build_moe_wp()

        H = getattr(
            trainer.cfg.residual if hasattr(trainer.cfg, "residual") else None,
            "n_history",
            2,
        )
        self._H = H
        self._reset_hist_from(self.init_pos)

        self.last_ctrl: torch.Tensor | None = None
        self.grabbing = False

        res_cfg = getattr(trainer.cfg, "residual", None)
        self._K_damping = getattr(res_cfg, "damping_interval", 20) if res_cfg else 20
        self._res_mode = getattr(res_cfg, "mode", "both") if res_cfg else "both"

        self.settle_iters = max(0, int(settle_iters))
        self._settled_pos: torch.Tensor | None = None

    def _build_moe_wp(self):
        w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, _, yield_patch, _, _, visc_patch = (
            self.trainer.get_current_phys_props()
        )

        def gather(patch_data):
            flat_idx = self.trainer.patch_idx.squeeze(0).view(-1)
            gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
            return torch.sum(self.trainer.interp_weights * gathered, dim=2).squeeze(0)

        p_weights = gather(w_patch)
        p_mu = gather(mu_patch.unsqueeze(-1)).squeeze()
        p_lam = gather(lam_patch.unsqueeze(-1)).squeeze()
        p_fk = gather(fk_patch.unsqueeze(-1)).squeeze()
        p_fdir = torch.nn.functional.normalize(gather(fdir_patch), dim=1, eps=1e-8)
        p_yield = gather(yield_patch.unsqueeze(-1)).squeeze()
        p_visc = gather(visc_patch.unsqueeze(-1)).squeeze()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                "weights": torch2warp_float(p_weights),
                "mu": torch2warp_float(p_mu),
                "lam": torch2warp_float(p_lam),
                "fk": torch2warp_float(p_fk),
                "fdir": torch2warp_vec3(p_fdir),
                "active_mask": self.active_mask_wp,
            }

    def _reset_hist_from(self, x0: torch.Tensor):
        self.x_history = [x0.clone() for _ in range(self._H)]
        self.v_history = [torch.zeros_like(x0) for _ in range(self._H)]

    def _ensure_settled(self):
        """First frame config + gravity until roughly on floor (cached)."""
        if self._settled_pos is not None:
            return
        print(f"[demo] Settling: {self.settle_iters} ticks (no controller, gravity) …")
        self.trainer.simulator.reset(self.init_pos.clone(), controller_pos=None)
        with torch.no_grad():
            for _ in range(self.settle_iters):
                self._step_frame(None, None)
        self._settled_pos = (self.trainer.simulator.x - self.trainer.simulator.shift).detach().clone()
        print("[demo] Settling done — rest pose on ground cached.")

    def reset_rest(self):
        """Rest pose on ground, no controller."""
        self._ensure_settled()
        assert self._settled_pos is not None
        self.trainer.simulator.reset(self._settled_pos.clone(), controller_pos=None)
        self._reset_hist_from(self._settled_pos)
        self.last_ctrl = None
        self.grabbing = False

    def grab(self, particle_index: int):
        self._ensure_settled()
        assert self._settled_pos is not None
        rest = self._settled_pos
        idx = int(particle_index) % self.n_total
        ctrl = rest[idx : idx + 1].clone()
        self.trainer.simulator.reset(rest.clone(), controller_pos=ctrl)
        self._reset_hist_from(rest)
        self.last_ctrl = ctrl.clone()
        self.grabbing = True
        if self.trainer.simulator.num_connections == 0:
            from pytorch3d.ops import knn_points

            q = rest[idx : idx + 1].unsqueeze(0)
            dist, ix, _ = knn_points(q, rest.unsqueeze(0), K=min(32, self.n_total))
            neigh = rest[ix.squeeze(0)].mean(dim=0, keepdim=True)
            self.trainer.simulator.reset(rest.clone(), controller_pos=neigh)
            self.last_ctrl = neigh.clone()

    def frame_meta(self, max_connected_vis: int = 512) -> dict:
        sim = self.trainer.simulator
        ctrl = []
        if self.grabbing and self.last_ctrl is not None:
            p = self.last_ctrl.detach().cpu().numpy().reshape(-1, 3)
            ctrl = p.astype(float).tolist()
        conn_ids: list[int] = []
        conn_pts: list[list[float]] = []
        if self.grabbing:
            conn_ids = _connected_particle_indices(sim)
            if conn_ids:
                x = (sim.x - sim.shift).detach().cpu().numpy()
                for i in conn_ids[:max_connected_vis]:
                    conn_pts.append([float(x[i, 0]), float(x[i, 1]), float(x[i, 2])])
        return {
            "type": "frame_meta",
            "grabbing": self.grabbing,
            "ctrl": ctrl,
            "connected": conn_ids,
            "connected_points": conn_pts,
        }

    def positions_numpy(self) -> np.ndarray:
        x = (self.trainer.simulator.x - self.trainer.simulator.shift).detach()
        if self._stride > 1:
            x = x[:: self._stride]
        else:
            x = x[: self.n_export]
        return x.cpu().numpy().astype(np.float32)

    def _step_frame(self, controller_pos: torch.Tensor | None, controller_vel: torch.Tensor | None):
        sim = self.trainer.simulator
        moe = self.moe_params_wp
        frame_dt = self.dt * self.steps_per_tick

        x_start = (sim.x - sim.shift).detach().unsqueeze(0)
        current_damping = None

        for s in range(self.steps_per_tick):
            if (
                self.use_neural
                and self.trainer.residual_net is not None
                and s % self._K_damping == 0
                and self._res_mode in ("damping", "both")
            ):
                x_his = torch.stack(self.x_history, dim=1).unsqueeze(0)
                v_his = torch.stack(self.v_history, dim=1).unsqueeze(0)
                curr_x = (sim.x - sim.shift).unsqueeze(0)
                curr_v = sim.v.unsqueeze(0)
                self.trainer.residual_net.eval()
                with torch.no_grad():
                    current_damping = self.trainer.residual_net(
                        curr_x, curr_v, x_start, x_his, v_his, mode="damping"
                    ).squeeze(0)

            with torch.no_grad():
                sim.step(
                    moe,
                    controller_pos=controller_pos,
                    controller_vel=controller_vel,
                    residual_v=None,
                    damping_override=current_damping,
                )

        if self.use_neural and self.trainer.residual_net is not None and self._res_mode in ("residual", "both"):
            self.trainer.residual_net.eval()
            x_his = torch.stack(self.x_history, dim=1).unsqueeze(0)
            v_his = torch.stack(self.v_history, dim=1).unsqueeze(0)
            curr_x = (sim.x - sim.shift).unsqueeze(0)
            curr_v = sim.v.unsqueeze(0)
            with torch.no_grad():
                delta_v = self.trainer.residual_net(
                    curr_x, curr_v, x_start, x_his, v_his, mode="residual"
                ).squeeze(0)
            sim.v = sim.v + delta_v
            sim.x = sim.x + delta_v * frame_dt

        sim.x = sim.x.detach()
        sim.v = sim.v.detach()
        sim.F = sim.F.detach()
        sim.C = sim.C.detach()

        if self.use_neural:
            self.x_history.pop(0)
            self.x_history.append((sim.x - sim.shift).detach())
            self.v_history.pop(0)
            self.v_history.append(sim.v.detach())

    def tick_drag(self, pos_xyz: list[float]):
        pos = torch.tensor([pos_xyz], device=self.device, dtype=torch.float32)
        if self.last_ctrl is None:
            v = torch.zeros_like(pos)
        else:
            v = (pos - self.last_ctrl) / max(self.dt * self.steps_per_tick, 1e-9)
        self._step_frame(pos, v)
        self.last_ctrl = pos.clone()

    def tick_free(self):
        self._step_frame(None, None)


def create_app(session: InteractiveMPMSession, static_dir: str) -> FastAPI:
    app = FastAPI()
    lock = asyncio.Lock()

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "hello",
                "n_total": session.n_total,
                "n_render": session.positions_numpy().shape[0],
                "stride": session._stride,
                "steps_per_tick": session.steps_per_tick,
                "use_neural": session.use_neural,
            }
        )
        try:
            while True:
                msg = await websocket.receive_json()
                cmd = msg.get("cmd")
                async with lock:
                    if cmd == "reset":
                        session.reset_rest()
                    elif cmd == "grab":
                        session.grab(int(msg.get("particle_index", 0)))
                    elif cmd == "drag":
                        if session.grabbing:
                            session.tick_drag([float(msg["x"]), float(msg["y"]), float(msg["z"])])
                    elif cmd == "release":
                        session.grabbing = False
                        session.last_ctrl = None
                        for _ in range(max(1, msg.get("free_ticks", 2))):
                            session.tick_free()
                    elif cmd == "step_free":
                        session.tick_free()
                    else:
                        await websocket.send_json({"error": f"unknown cmd {cmd}"})
                        continue

                await websocket.send_json(session.frame_meta())
                blob = _pack_positions(session.positions_numpy())
                await websocket.send_bytes(blob)
        except WebSocketDisconnect:
            pass

    return app


def _pick_listen_port(host: str, preferred: int, span: int = 50) -> int:
    """Use `preferred` if free, else try preferred+1, … (avoids 'address already in use')."""
    bind_host = "" if host in ("0.0.0.0", "::") else host
    for p in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((bind_host, p))
                return p
            except OSError:
                continue
    print(f"ERROR: no free port in [{preferred}, {preferred + span})", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--steps_per_tick", type=int, default=56)
    parser.add_argument("--max_send_particles", type=int, default=35000)
    parser.add_argument("--use_neural", action="store_true", help="Slower: run ResidualPGND each tick")
    parser.add_argument(
        "--settle_iters",
        type=int,
        default=220,
        help="Gravity-only ticks to reach ground pose from data frame-0 (first reset is slow)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = "cuda"

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = os.path.join(cfg.output_dir, args.case_name, "best_checkpoint.pt")
    if not os.path.isfile(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading {args.case_name} from {ckpt} ...")
    trainer = PhysExpertMPMTrainer(cfg, args.case_name)
    trainer.load_from_checkpoint(ckpt)

    session = InteractiveMPMSession(
        trainer,
        steps_per_tick=args.steps_per_tick,
        max_send_particles=args.max_send_particles,
        use_neural=args.use_neural,
        settle_iters=args.settle_iters,
    )

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = create_app(session, static_dir)

    port = _pick_listen_port(args.host, args.port)
    if port != args.port:
        print(f"[demo] Port {args.port} 已被占用 → 改用 {port}")
    print(f"Open http://127.0.0.1:{port}  (WebSocket ws://127.0.0.1:{port}/ws)")
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
