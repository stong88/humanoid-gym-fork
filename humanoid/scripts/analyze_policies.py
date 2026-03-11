import csv
import json
import math
import os
import sys
import time
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from humanoid.algo.grpo.actor import Actor
from humanoid.algo.ppo.actor_critic import ActorCritic

GLOBAL_SEED = 12345


@dataclass
class RunSpec:
    policy: str
    event_file: Path
    run_dir: Path


def find_latest_model(run_dir: Path) -> Optional[Path]:
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return models[-1] if models else None


def load_scalars(event_file: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        scalars[tag] = [(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)]
    return scalars


def first_reach(series: List[Tuple[int, float, float]], threshold: float) -> Tuple[Optional[int], Optional[float]]:
    for step, wall, value in series:
        if value >= threshold:
            return step, wall
    return None, None


def reward_auc_over_time(series: List[Tuple[int, float, float]]) -> float:
    if len(series) < 2:
        return float("nan")
    t = np.array([x[1] for x in series])
    y = np.array([x[2] for x in series])
    dt = t[-1] - t[0]
    if dt <= 0:
        return float("nan")
    return float(np.trapz(y, t) / dt)


def parse_wandb_configs(root: Path) -> List[dict]:
    cfgs = []
    for path in sorted(root.glob("**/wandb/run-*/files/config.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfgs.append(yaml.safe_load(f))
        except Exception:
            continue
    return cfgs


def _g(cfg: dict, *keys):
    x = cfg
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x


def get_run_hparams(policy_exp_name: str, cfgs: List[dict]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    for cfg in cfgs:
        exp_name = _g(cfg, "runner", "value", "experiment_name")
        if exp_name != policy_exp_name:
            continue
        num_envs = _g(cfg, "env", "value", "num_envs")
        nsteps = _g(cfg, "runner", "value", "num_steps_per_env")
        alg = _g(cfg, "runner", "value", "algorithm_class_name")
        if num_envs is not None and nsteps is not None:
            return int(num_envs), int(nsteps), str(alg)
    return None, None, None


def infer_arch_from_state_dict(state_dict: dict) -> Tuple[str, List[int], Optional[List[int]], int, int, int]:
    actor_layers = []
    i = 0
    while f"actor.{i}.weight" in state_dict:
        w = state_dict[f"actor.{i}.weight"]
        actor_layers.append((int(w.shape[1]), int(w.shape[0])))
        i += 2
    if not actor_layers:
        raise RuntimeError("No actor layers found in checkpoint")

    num_actor_obs = actor_layers[0][0]
    actor_hidden = [pair[1] for pair in actor_layers[:-1]]
    num_actions = actor_layers[-1][1]

    if any(k.startswith("critic.") for k in state_dict.keys()):
        critic_layers = []
        j = 0
        while f"critic.{j}.weight" in state_dict:
            w = state_dict[f"critic.{j}.weight"]
            critic_layers.append((int(w.shape[1]), int(w.shape[0])))
            j += 2
        num_critic_obs = critic_layers[0][0]
        critic_hidden = [pair[1] for pair in critic_layers[:-1]]
        return "ActorCritic", actor_hidden, critic_hidden, num_actor_obs, num_critic_obs, num_actions

    return "Actor", actor_hidden, None, num_actor_obs, num_actor_obs, num_actions


def build_model_from_checkpoint(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model_state_dict"]
    kind, actor_hidden, critic_hidden, num_actor_obs, num_critic_obs, num_actions = infer_arch_from_state_dict(state)

    if kind == "ActorCritic":
        model = ActorCritic(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden,
            critic_hidden_dims=critic_hidden,
        )
    else:
        model = Actor(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden,
            critic_hidden_dims=[],
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, kind, num_actor_obs


def benchmark_actor_forward(model, obs_dim: int, batch_size: int, device: str, repeats: int = 200) -> Tuple[float, float]:
    gen = torch.Generator(device=device)
    gen.manual_seed(GLOBAL_SEED + batch_size)
    obs = torch.randn(batch_size, obs_dim, generator=gen, device=device)
    actor = model.actor

    # Warm-up
    with torch.inference_mode():
        for _ in range(30):
            _ = actor(obs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            _ = actor(obs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    per_forward = total / repeats
    per_sample = per_forward / batch_size
    return per_forward, per_sample


def clip_curves_to_common_domain(curves: Dict[str, Tuple[List[float], List[float]]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    mins = [float(np.min(x)) for x, _ in curves.values()]
    maxs = [float(np.max(x)) for x, _ in curves.values()]
    common_min = max(mins)
    common_max = min(maxs)

    clipped = {}
    for name, (x, y) in curves.items():
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        m = (xx >= common_min) & (xx <= common_max)
        clipped[name] = (xx[m], yy[m])
    return clipped


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def permutation_test_mean_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = GLOBAL_SEED) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        d = abs(np.mean(combined[:n]) - np.mean(combined[n:]))
        if d >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


def pairwise_significance(curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
    keys = sorted(curves.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_name, b_name = keys[i], keys[j]
            xa, ya = curves[a_name]
            xb, yb = curves[b_name]
            lo = max(float(np.min(xa)), float(np.min(xb)))
            hi = min(float(np.max(xa)), float(np.max(xb)))
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, 300)
            ya_i = np.interp(grid, xa, ya)
            yb_i = np.interp(grid, xb, yb)
            p = permutation_test_mean_diff(ya_i, yb_i)
            d = cohen_d(ya_i, yb_i)
            rows.append(
                {
                    "policy_a": a_name,
                    "policy_b": b_name,
                    "domain_min": round(lo, 6),
                    "domain_max": round(hi, 6),
                    "mean_reward_a": round(float(np.mean(ya_i)), 6),
                    "mean_reward_b": round(float(np.mean(yb_i)), 6),
                    "mean_diff_a_minus_b": round(float(np.mean(ya_i) - np.mean(yb_i)), 6),
                    "cohen_d": round(d, 6) if not np.isnan(d) else None,
                    "permutation_p": round(p, 8),
                }
            )
    return rows


def rolling_std(x: np.ndarray, window: int = 20) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.std(x[lo : i + 1])
    return out


def max_drawdown(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(x)
    dd = running_max - x
    return float(np.max(dd))


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    p = np.polyfit(x, y, 1)
    return float(p[0])


def summarize_reward_components(scalars: Dict[str, List[Tuple[int, float, float]]], tail_n: int = 40) -> Dict[str, float]:
    comp = {}
    for tag, series in scalars.items():
        if not tag.startswith("Episode/rew_"):
            continue
        vals = np.array([v for _, _, v in series], dtype=np.float64)
        if len(vals) == 0:
            continue
        key = tag.replace("Episode/", "")
        comp[f"{key}_tail_mean"] = float(np.mean(vals[-tail_n:]))
        comp[f"{key}_tail_std"] = float(np.std(vals[-tail_n:]))
    return comp


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Policy comparison analysis")
    parser.add_argument("--snapshot", type=str, default="frozen_v1", help="Snapshot name for frozen inputs")
    parser.add_argument("--refresh_snapshot", action="store_true", help="Refresh snapshot files from live logs")
    args = parser.parse_args()

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "analysis" / "policy_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_dir / "snapshots" / args.snapshot
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        RunSpec(
            policy="PPO",
            run_dir=repo_root / "logs" / "XBot_ppo" / "Mar10_14-22-40_v1",
            event_file=repo_root / "logs" / "XBot_ppo" / "Mar10_14-22-40_v1" / "events.out.tfevents.1773177763.iris-ws-10.stanford.edu.1153330.0",
        ),
        RunSpec(
            policy="GRPO",
            run_dir=repo_root / "logs" / "XBot_grpo" / "Mar10_14-53-29_",
            event_file=repo_root / "logs" / "XBot_grpo" / "Mar10_14-53-29_" / "events.out.tfevents.1773179611.iris7.stanford.edu.3308242.0",
        ),
        RunSpec(
            policy="CGRPO",
            run_dir=repo_root / "logs" / "XBot_cgrpo" / "Mar10_15-56-57_",
            event_file=repo_root / "logs" / "XBot_cgrpo" / "Mar10_15-56-57_" / "events.out.tfevents.1773183420.iris-ws-10.stanford.edu.1172532.0",
        ),
    ]

    cfgs = parse_wandb_configs(repo_root)

    # Freeze event/checkpoint inputs to make repeated analyses reproducible.
    frozen_runs = []
    for run in runs:
        dst_run = snapshot_dir / run.policy.lower()
        dst_run.mkdir(parents=True, exist_ok=True)
        frozen_event = dst_run / run.event_file.name
        if args.refresh_snapshot or (not frozen_event.exists()):
            shutil.copy2(run.event_file, frozen_event)

        src_latest = find_latest_model(run.run_dir)
        frozen_model = None
        if src_latest is not None:
            frozen_model = dst_run / src_latest.name
            if args.refresh_snapshot or (not frozen_model.exists()):
                shutil.copy2(src_latest, frozen_model)

        frozen_runs.append(
            {
                "policy": run.policy,
                "event": frozen_event,
                "model": frozen_model,
                "run_dir": dst_run,
            }
        )

    training_rows = []
    speed_rows = []
    behavior_rows = []
    threshold_rows = []
    component_rows = []

    reward_curves = {}
    wall_curves = {}
    samples_curves = {}
    reward_volatility_curves = {}

    for f_run in frozen_runs:
        policy = f_run["policy"]
        scalars = load_scalars(f_run["event"])
        reward = scalars["Train/mean_reward"]
        fps = scalars["Perf/total_fps"]
        ctime = scalars["Perf/collection time"]
        ltime = scalars["Perf/learning_time"]

        rew_vals = [x[2] for x in reward]
        steps = [x[0] for x in reward]
        walls = [x[1] for x in reward]
        wall_rel = [w - walls[0] for w in walls]

        exp_name = f"XBot_{policy.lower()}" if policy != "CGRPO" else "XBot_cgrpo"
        num_envs, nsteps, alg = get_run_hparams(exp_name, cfgs)
        if num_envs is None or nsteps is None:
            # fallback estimates from timing and fps
            m_fps = float(np.mean([x[2] for x in fps]))
            m_it = float(np.mean([x[2] for x in ctime]) + np.mean([x[2] for x in ltime]))
            est_batch = int(m_fps * m_it)
            num_envs, nsteps = est_batch, 1


        n = min(len(reward), len(fps), len(ctime), len(ltime))
        samples_per_iter = np.array([fps[i][2] * (ctime[i][2] + ltime[i][2]) for i in range(n)], dtype=np.float64)
        env_samples = np.cumsum(samples_per_iter)
        reward_trim = reward[:n]
        steps = [x[0] for x in reward_trim]
        rew_vals = [x[2] for x in reward_trim]
        walls = [x[1] for x in reward_trim]
        wall_rel = [w - walls[0] for w in walls]

        reward_curves[policy] = (steps, rew_vals)
        wall_curves[policy] = (wall_rel, rew_vals)
        samples_curves[policy] = (env_samples.tolist(), rew_vals)

        s5, _ = first_reach(reward, 5)
        s20, _ = first_reach(reward, 20)
        s50, _ = first_reach(reward, 50)
        s80, _ = first_reach(reward, 80)
        s100, t100 = first_reach(reward, 100)

        training_rows.append(
            {
            "policy": policy,
                "iterations_logged": len(reward),
                "iter_start": steps[0],
                "iter_end": steps[-1],
                "reward_start": round(rew_vals[0], 4),
                "reward_end": round(rew_vals[-1], 4),
                "reward_max": round(float(np.max(rew_vals)), 4),
                "reward_auc_over_walltime": round(reward_auc_over_time(reward), 4),
                "iter_to_5": s5,
                "iter_to_20": s20,
                "iter_to_50": s50,
                "iter_to_80": s80,
                "iter_to_100": s100,
                "walltime_to_100_s": round(t100 - walls[0], 3) if t100 is not None else None,
                "num_envs": num_envs,
                "num_steps_per_env": nsteps,
                "algorithm": alg,
            }
        )

        fps_vals = np.array([x[2] for x in fps], dtype=np.float64)
        c_vals = np.array([x[2] for x in ctime], dtype=np.float64)
        l_vals = np.array([x[2] for x in ltime], dtype=np.float64)
        iter_time = c_vals + l_vals
        est_samples_per_iter = fps_vals * iter_time

        speed_rows.append(
            {
                "policy": policy,
                "fps_mean": round(float(np.mean(fps_vals)), 3),
                "fps_median": round(float(np.median(fps_vals)), 3),
                "iter_time_mean_s": round(float(np.mean(iter_time)), 4),
                "collection_time_mean_s": round(float(np.mean(c_vals)), 4),
                "learning_time_mean_s": round(float(np.mean(l_vals)), 4),
                "learn_to_collect_ratio": round(float(np.mean(l_vals) / np.mean(c_vals)), 4),
                "est_samples_per_iter_mean": round(float(np.mean(est_samples_per_iter)), 3),
            }
        )

        rew_np = np.array(rew_vals, dtype=np.float64)
        step_np = np.array(steps, dtype=np.float64)
        wall_np = np.array(wall_rel, dtype=np.float64)
        samp_np = np.array(env_samples, dtype=np.float64)

        vol = rolling_std(rew_np, window=20)
        reward_volatility_curves[policy] = (step_np, vol)

        q1 = max(1, len(step_np) // 3)
        q2 = max(q1 + 1, 2 * len(step_np) // 3)
        behavior_rows.append(
            {
                "policy": policy,
                "reward_std_full": round(float(np.std(rew_np)), 6),
                "reward_std_tail40": round(float(np.std(rew_np[-40:])), 6),
                "reward_max_drawdown": round(max_drawdown(rew_np), 6),
                "rolling_std20_mean": round(float(np.mean(vol)), 6),
                "slope_early": round(slope(step_np[:q1], rew_np[:q1]), 6),
                "slope_mid": round(slope(step_np[q1:q2], rew_np[q1:q2]), 6),
                "slope_late": round(slope(step_np[q2:], rew_np[q2:]), 6),
                "frac_iter_reward_ge_100": round(float(np.mean(rew_np >= 100.0)), 6),
            }
        )

        for th in [20.0, 50.0, 80.0, 100.0]:
            sid, wid = first_reach(reward_trim, th)
            if sid is None:
                threshold_rows.append(
                    {
                        "policy": policy,
                        "threshold": th,
                        "iter_to_threshold": None,
                        "walltime_to_threshold_s": None,
                        "samples_to_threshold": None,
                    }
                )
                continue
            idx = next(i for i, (s, _, v) in enumerate(reward_trim) if s == sid and v >= th)
            threshold_rows.append(
                {
                    "policy": policy,
                    "threshold": th,
                    "iter_to_threshold": int(sid),
                    "walltime_to_threshold_s": round(float(wall_rel[idx]), 6),
                    "samples_to_threshold": round(float(env_samples[idx]), 6),
                }
            )

        comp = summarize_reward_components(scalars, tail_n=40)
        row = {"policy": policy}
        for k, v in sorted(comp.items()):
            row[k] = round(v, 6)
        component_rows.append(row)

    write_csv(out_dir / "training_summary.csv", training_rows)
    write_csv(out_dir / "speed_summary.csv", speed_rows)
    write_csv(out_dir / "behavior_summary.csv", behavior_rows)
    write_csv(out_dir / "threshold_efficiency.csv", threshold_rows)
    write_csv(out_dir / "reward_component_summary.csv", component_rows)

    reward_curves_common = clip_curves_to_common_domain(reward_curves)
    wall_curves_common = clip_curves_to_common_domain(wall_curves)
    samples_curves_common = clip_curves_to_common_domain(samples_curves)

    sig_rows = pairwise_significance(reward_curves_common)
    write_csv(out_dir / "significance_summary.csv", sig_rows)

    tr_by = {r["policy"]: r for r in training_rows}
    bh_by = {r["policy"]: r for r in behavior_rows}
    cgain_rows = []
    if "CGRPO" in tr_by:
        for baseline in ["GRPO", "PPO"]:
            if baseline not in tr_by:
                continue
            cg = tr_by["CGRPO"]
            bl = tr_by[baseline]
            cgain_rows.append(
                {
                    "comparison": f"CGRPO_vs_{baseline}",
                    "final_reward_ratio": round(cg["reward_end"] / max(1e-8, bl["reward_end"]), 6),
                    "max_reward_ratio": round(cg["reward_max"] / max(1e-8, bl["reward_max"]), 6),
                    "auc_over_time_ratio": round(cg["reward_auc_over_walltime"] / max(1e-8, bl["reward_auc_over_walltime"]), 6),
                    "iter_to_100_diff": None if (cg["iter_to_100"] is None or bl["iter_to_100"] is None) else int(cg["iter_to_100"] - bl["iter_to_100"]),
                    "tail_std_ratio": round(bh_by["CGRPO"]["reward_std_tail40"] / max(1e-8, bh_by[baseline]["reward_std_tail40"]), 6) if ("CGRPO" in bh_by and baseline in bh_by) else None,
                    "drawdown_ratio": round(bh_by["CGRPO"]["reward_max_drawdown"] / max(1e-8, bh_by[baseline]["reward_max_drawdown"]), 6) if ("CGRPO" in bh_by and baseline in bh_by) else None,
                }
            )
    write_csv(out_dir / "cgrpo_gain_summary.csv", cgain_rows)

    inf_rows = []
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    torch.set_num_threads(1)
    for f_run in frozen_runs:
        policy = f_run["policy"]
        ckpt = f_run["model"]
        if ckpt is None:
            continue
        model_size_mb = ckpt.stat().st_size / (1024 * 1024)

        for device in devices:
            model, arch, obs_dim = build_model_from_checkpoint(ckpt, device=device)
            actor_params = sum(p.numel() for p in model.actor.parameters()) + int(model.std.numel())
            total_params = sum(p.numel() for p in model.parameters())

            for bsz in [1, 256, 2048]:
                rep = 160 if bsz >= 2048 else 240

                fwd_runs = []
                sample_runs = []
                for _ in range(5):
                    per_fwd, per_sample = benchmark_actor_forward(model, obs_dim, bsz, device, repeats=rep)
                    fwd_runs.append(per_fwd)
                    sample_runs.append(per_sample)
                per_fwd = float(np.median(np.array(fwd_runs)))
                per_sample = float(np.median(np.array(sample_runs)))
                inf_rows.append(
                    {
                        "policy": policy,
                        "checkpoint": ckpt.name,
                        "device": device,
                        "arch": arch,
                        "batch_size": bsz,
                        "actor_params": actor_params,
                        "total_params": total_params,
                        "checkpoint_size_mb": round(model_size_mb, 4),
                        "forward_time_ms": round(per_fwd * 1000.0, 6),
                        "per_sample_us": round(per_sample * 1e6, 6),
                        "samples_per_sec": round((1.0 / per_sample), 3),
                        "benchmark_repeats": rep,
                        "benchmark_trials": 5,
                    }
                )

    write_csv(out_dir / "inference_benchmark.csv", inf_rows)

    fig1_rows = []
    for p, (x, y) in reward_curves_common.items():
        for xi, yi in zip(x, y):
            fig1_rows.append({"policy": p, "iteration": float(xi), "mean_reward": float(yi)})
    write_csv(out_dir / "fig1_reward_vs_iteration_data.csv", fig1_rows)

    fig2_rows = []
    for p, (x, y) in wall_curves_common.items():
        for xi, yi in zip(x, y):
            fig2_rows.append({"policy": p, "wall_time_s": float(xi), "mean_reward": float(yi)})
    write_csv(out_dir / "fig2_reward_vs_walltime_data.csv", fig2_rows)

    fig3_rows = []
    for p, (x, y) in samples_curves_common.items():
        for xi, yi in zip(x, y):
            fig3_rows.append({"policy": p, "env_samples": float(xi), "mean_reward": float(yi)})
    write_csv(out_dir / "fig3_reward_vs_samples_data.csv", fig3_rows)

    fig4_rows = []
    for r in speed_rows:
        fig4_rows.append(
            {
                "policy": r["policy"],
                "collection_time_mean_s": r["collection_time_mean_s"],
                "learning_time_mean_s": r["learning_time_mean_s"],
                "iter_time_mean_s": r["iter_time_mean_s"],
            }
        )
    write_csv(out_dir / "fig4_iteration_time_breakdown_data.csv", fig4_rows)

    # Figure 1: Reward vs iteration
    plt.figure(figsize=(8, 5))
    for p, (x, y) in reward_curves_common.items():
        plt.plot(x, y, label=p, linewidth=2)
    plt.xlabel("Training iteration")
    plt.ylabel("Train/mean_reward")
    plt.title("Reward Progression vs Iteration")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_reward_vs_iteration.png", dpi=180)
    plt.close()

    # Figure 2: Reward vs wall time
    plt.figure(figsize=(8, 5))
    for p, (x, y) in wall_curves_common.items():
        plt.plot(x, y, label=p, linewidth=2)
    plt.xlabel("Wall time since run start (s)")
    plt.ylabel("Train/mean_reward")
    plt.title("Reward Progression vs Wall Time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_reward_vs_walltime.png", dpi=180)
    plt.close()

    # Figure 3: Reward vs environment samples
    plt.figure(figsize=(8, 5))
    for p, (x, y) in samples_curves_common.items():
        plt.plot(x, y, label=p, linewidth=2)
    plt.xlabel("Environment samples (iteration * num_envs * num_steps_per_env)")
    plt.ylabel("Train/mean_reward")
    plt.title("Sample Efficiency Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_reward_vs_samples.png", dpi=180)
    plt.close()

    # Figure 4: Iteration time decomposition
    policies = [r["policy"] for r in speed_rows]
    cvals = [r["collection_time_mean_s"] for r in speed_rows]
    lvals = [r["learning_time_mean_s"] for r in speed_rows]

    plt.figure(figsize=(8, 5))
    plt.bar(policies, cvals, label="Collection")
    plt.bar(policies, lvals, bottom=cvals, label="Learning")
    plt.ylabel("Mean seconds per iteration")
    plt.title("Iteration Time Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_iteration_time_breakdown.png", dpi=180)
    plt.close()

    # Figure 5: Inference per-sample latency (CPU, batch=1)
    inf_cpu_b1 = [r for r in inf_rows if r["device"] == "cpu" and r["batch_size"] == 1]
    write_csv(out_dir / "fig5_inference_latency_cpu_b1_data.csv", inf_cpu_b1)
    if inf_cpu_b1:
        inf_cpu_b1 = sorted(inf_cpu_b1, key=lambda x: x["policy"])
        plt.figure(figsize=(8, 5))
        plt.bar([r["policy"] for r in inf_cpu_b1], [r["per_sample_us"] for r in inf_cpu_b1])
        plt.ylabel("Per-sample latency (us)")
        plt.title("Actor Inference Latency (CPU, batch=1)")
        plt.tight_layout()
        plt.savefig(out_dir / "fig5_inference_latency_cpu_b1.png", dpi=180)
        plt.close()

    # Figure 6: Reward volatility (rolling std, common iteration domain)
    vol_common = clip_curves_to_common_domain(
        {k: (v[0].tolist(), v[1].tolist()) for k, v in reward_volatility_curves.items()}
    )
    fig6_rows = []
    for p, (x, y) in vol_common.items():
        for xi, yi in zip(x, y):
            fig6_rows.append({"policy": p, "iteration": float(xi), "rolling_std20": float(yi)})
    write_csv(out_dir / "fig6_reward_volatility_data.csv", fig6_rows)
    plt.figure(figsize=(8, 5))
    for p, (x, y) in vol_common.items():
        plt.plot(x, y, label=p, linewidth=2)
    plt.xlabel("Training iteration")
    plt.ylabel("Rolling std (window=20)")
    plt.title("Reward Volatility Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_reward_volatility.png", dpi=180)
    plt.close()

    # Figure 7: CGRPO reward component profile (tail means)
    key_comps = [
        "rew_tracking_lin_vel_tail_mean",
        "rew_tracking_ang_vel_tail_mean",
        "rew_orientation_tail_mean",
        "rew_action_smoothness_tail_mean",
        "rew_torques_tail_mean",
    ]
    comp_by = {r["policy"]: r for r in component_rows}
    if "CGRPO" in comp_by:
        labels = []
        vals = []
        for c in key_comps:
            if c in comp_by["CGRPO"]:
                labels.append(c.replace("_tail_mean", ""))
                vals.append(comp_by["CGRPO"][c])
        if labels:
            fig7_rows = []
            for lbl, val in zip(labels, vals):
                fig7_rows.append({"policy": "CGRPO", "component": lbl, "tail_mean": float(val)})
            write_csv(out_dir / "fig7_cgrpo_reward_components_data.csv", fig7_rows)
            plt.figure(figsize=(9, 5))
            plt.bar(labels, vals)
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("Tail mean (last 40 points)")
            plt.title("CGRPO Reward Component Profile")
            plt.tight_layout()
            plt.savefig(out_dir / "fig7_cgrpo_reward_components.png", dpi=180)
            plt.close()

    # Export a compact JSON for paper scripting.
    with open(out_dir / "analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": [
                    {
                        "policy": f["policy"],
                        "run_dir": str(f["run_dir"]),
                        "event_file": str(f["event"]),
                        "model_file": str(f["model"]) if f["model"] is not None else None,
                    }
                    for f in frozen_runs
                ],
                "snapshot": {"name": args.snapshot, "refresh_snapshot": args.refresh_snapshot},
                "outputs": {
                    "tables": [
                        "training_summary.csv",
                        "speed_summary.csv",
                        "behavior_summary.csv",
                        "threshold_efficiency.csv",
                        "reward_component_summary.csv",
                        "cgrpo_gain_summary.csv",
                        "significance_summary.csv",
                        "inference_benchmark.csv",
                        "fig1_reward_vs_iteration_data.csv",
                        "fig2_reward_vs_walltime_data.csv",
                        "fig3_reward_vs_samples_data.csv",
                        "fig4_iteration_time_breakdown_data.csv",
                        "fig5_inference_latency_cpu_b1_data.csv",
                        "fig6_reward_volatility_data.csv",
                        "fig7_cgrpo_reward_components_data.csv",
                    ],
                    "figures": [
                        "fig1_reward_vs_iteration.png",
                        "fig2_reward_vs_walltime.png",
                        "fig3_reward_vs_samples.png",
                        "fig4_iteration_time_breakdown.png",
                        "fig5_inference_latency_cpu_b1.png",
                        "fig6_reward_volatility.png",
                        "fig7_cgrpo_reward_components.png",
                    ],
                },
            },
            f,
            indent=2,
        )

    print(f"Analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
