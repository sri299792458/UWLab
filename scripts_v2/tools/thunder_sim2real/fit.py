"""Fit or evaluate Thunder dynamics using timestamp-checked 500 Hz recordings."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from isaaclab.app import AppLauncher

from records import load_record, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--evaluate", type=Path, help="Replay an existing fit instead of optimizing")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--delay_max", type=int, default=5)
parser.add_argument("--armature_max", type=float, default=10)
parser.add_argument("--friction_max", type=float, default=20)
parser.add_argument("--viscous_max", type=float, default=20,
                    help="UWLab default upper bound; increase if the fitted solution reaches this bound")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
record, record_report = load_record(args.record)  # reject bad inputs before starting simulation
if args.output.exists():
    raise FileExistsError(args.output)
if args.num_envs < 2 or args.iterations < 1 or args.delay_max < 1:
    parser.error("Use num_envs >= 2, iterations >= 1, delay_max >= 1")
for limit in (args.armature_max, args.friction_max, args.viscous_max):
    if not np.isfinite(limit) or limit <= 0:
        parser.error("Dynamics upper bounds must be finite and positive")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

from replay import ThunderReplay, parameter_dict


def write_json(name, content):
    path = args.output / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plot_evaluation(real, simulated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    t = np.arange(len(real)) * record["dt"]
    for j, ax in enumerate(axes.flat):
        ax.plot(t, np.degrees(real[:, j]), label="Recorded")
        ax.plot(t, np.degrees(simulated[:, j]), label="Simulation", alpha=0.8)
        ax.set_title(record["joint_names"][j]); ax.set_ylabel("Degrees"); ax.grid(alpha=0.25)
    axes[0, 0].legend(); axes[-1, 0].set_xlabel("Seconds"); axes[-1, 1].set_xlabel("Seconds")
    fig.tight_layout(); fig.savefig(args.output / "joint_overlay.png", dpi=150); plt.close(fig)


def main():
    base = {"schema_version": 1, "robot": "thunder", "source_kind": record["source_kind"],
            "record_path": str(args.record.resolve()), "record_sha256": sha256(args.record),
            "record_validation": record_report, "comparison_phase": "pre_command", "dt": record["dt"],
            "controller": record["osc_params"], "seed": args.seed}
    if args.evaluate:
        fit = json.loads(args.evaluate.read_text())
        if fit["source_kind"] != record["source_kind"] or fit["comparison_phase"] != "pre_command":
            raise ValueError("Expected a Thunder fit matching the recording source kind and pre-command alignment")
        replay = ThunderReplay(record, num_envs=1, device=args.device,
                              delay_max=max(args.delay_max, fit["delay_steps"]), seed=args.seed)
        try:
            result = replay.run(np.asarray(fit["best_params"])[None], trajectory=True)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            replay.close()
        metrics = {**base, "fit_path": str(args.evaluate.resolve()), "fit_sha256": sha256(args.evaluate),
                   "held_out_record": fit["record_sha256"] != base["record_sha256"],
                   "per_joint_rmse_deg": np.degrees(result["per_joint_rmse_rad"][0]).tolist(),
                   "score_sum_joint_mse_rad2": float(result["scores"][0]),
                   "upstream_post_step_score_rad2": float(result["upstream_post_step_scores"][0])}
        write_json("evaluation.json", metrics)
        np.savez_compressed(args.output / "replay.npz", recorded=record["joint_positions"].numpy(),
                            simulated=result["joint_positions"][:, 0], dt=record["dt"])
        plot_evaluation(record["joint_positions"].numpy(), result["joint_positions"][:, 0])
        print(json.dumps(metrics, indent=2))
        return
    from cmaes import CMA
    high = np.array([args.armature_max]*6 + [args.friction_max]*6 + [1.0]*6 + [args.viscous_max]*6 + [args.delay_max])
    optimizer = CMA(mean=np.full(25, 0.5), sigma=0.3, population_size=args.num_envs,
                    bounds=np.column_stack([np.zeros(25), np.ones(25)]), seed=args.seed)
    replay = ThunderReplay(record, num_envs=args.num_envs, device=args.device,
                          delay_max=args.delay_max, seed=args.seed)
    best, best_score, history = None, float("inf"), []
    try:
        for iteration in range(args.iterations):
            started = time.monotonic()
            normalized = np.array([optimizer.ask() for _ in range(args.num_envs)])
            candidates = normalized * high
            result = replay.run(candidates)
            scores = result["scores"]
            if not np.isfinite(scores).all():
                raise RuntimeError("Non-finite replay score; inspect candidate dynamics")
            optimizer.tell(list(zip(normalized, scores.astype(float))))
            index = int(scores.argmin())
            if scores[index] < best_score:
                best_score = float(scores[index]); best = candidates[index].copy()
            history.append({"iteration": iteration+1, "best_score": best_score,
                            "population_mean": float(scores.mean()), "seconds": time.monotonic()-started})
            checkpoint = {**base, "iteration": iteration+1, "best_params": best.tolist(),
                          "score_sum_joint_mse_rad2": best_score, "sysid": parameter_dict(best),
                          "delay_steps": int(np.rint(best[24])), "bounds_upper": high.tolist(), "history": history}
            write_json("best_fit.json", checkpoint)
            print(f"Iteration {iteration+1}/{args.iterations}: best joint-averaged RMSE "
                  f"{np.degrees(np.sqrt(best_score/6)):.4f} deg; delay {checkpoint['delay_steps']} steps", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        replay.close()


try:
    main()
except BaseException as exc:
    traceback.print_exc()
    write_json("failure.json", {"status": "FAIL", "error": str(exc)})
    raise
finally:
    app.close()
