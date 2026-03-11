from isaacgym import gymapi
import argparse
import os
import sys

import cv2
import numpy as np
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403
from humanoid.utils import get_args, task_registry


def _infer_task_from_checkpoint(ckpt_path: str, explicit_task: str = None) -> str:
    if explicit_task is not None:
        return explicit_task
    p = ckpt_path.lower()
    if "cgrpo" in p:
        return "humanoid_cgrpo"
    if "grpo" in p:
        return "humanoid_grpo"
    raise ValueError("Unable to infer task from checkpoint path. Pass --task explicitly.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Overlay humanoid motion from a policy checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Absolute or relative path to model_*.pt")
    parser.add_argument("--task", type=str, default=None, help="Task name (e.g., humanoid_grpo, humanoid_cgrpo)")
    parser.add_argument("--num_steps", type=int, default=400, help="Total rollout steps")
    parser.add_argument("--sample_every", type=int, default=30, help="Capture one frame every N steps")
    parser.add_argument("--alpha", type=float, default=0.15, help="Overlay alpha for each sampled frame (alpha mode)")
    parser.add_argument(
        "--blend_mode",
        type=str,
        default="max",
        choices=["max", "alpha"],
        help="Frame compositing mode: 'max' is sharper, 'alpha' is smoother.",
    )
    parser.add_argument("--camera_distance", type=float, default=2.8, help="Side-view camera Y distance from robot")
    parser.add_argument("--camera_height", type=float, default=1.05, help="Side-view camera Z height")
    parser.add_argument("--camera_target_height", type=float, default=0.95, help="Look-at Z height")
    parser.add_argument(
        "--camera_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Choose left or right side-view camera placement.",
    )
    parser.add_argument("--output_image", type=str, default=None, help="Output PNG path")

    custom, remaining = parser.parse_known_args()

    # Keep Isaac Gym args parsing intact by passing only remaining CLI flags.
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining
        isaac_args = get_args()
    finally:
        sys.argv = old_argv

    return custom, isaac_args


def _default_output_path(checkpoint_path: str) -> str:
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "images", "checkpoint_overlays")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{ckpt_name}_overlay.png")


def main():
    custom, isaac_args = _parse_args()
    task = _infer_task_from_checkpoint(custom.checkpoint_path, custom.task)

    env_cfg, train_cfg = task_registry.get_cfgs(name=task)


    env_cfg.env.num_envs = 1
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=task, args=isaac_args, env_cfg=env_cfg)


    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=isaac_args, train_cfg=train_cfg)
    print(f"Loading checkpoint: {custom.checkpoint_path}")
    runner.load(custom.checkpoint_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()

    camera_props = gymapi.CameraProperties()
    camera_props.width = 1920
    camera_props.height = 1080
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)

    side_sign = -1.0 if custom.camera_side == "left" else 1.0
    cam_pos = gymapi.Vec3(0.0, side_sign * custom.camera_distance, custom.camera_height)
    cam_tgt = gymapi.Vec3(0.0, 0.0, custom.camera_target_height)
    env.gym.set_camera_location(camera_handle, env.envs[0], cam_pos, cam_tgt)

    composite = None
    sampled = 0

    for i in range(custom.num_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        if i % custom.sample_every != 0:
            continue

        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

        img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_f = img.astype(np.float32)
        if composite is None:
            composite = img_f
        else:
            if custom.blend_mode == "max":

                composite = np.maximum(composite, img_f)
            else:
                composite = (1.0 - custom.alpha) * composite + custom.alpha * img_f
        sampled += 1

    if sampled == 0 or composite is None:
        raise RuntimeError("No frames were sampled. Reduce --sample_every or increase --num_steps.")

    out_path = custom.output_image if custom.output_image else _default_output_path(custom.checkpoint_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_img = np.clip(composite, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out_img)
    print(f"Saved overlay image with {sampled} sampled frames to: {out_path}")


if __name__ == "__main__":
    main()
