# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Check native wrist limits, successful reset export and invalid-bank rejection."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--lab-scene", action="store_true", help="Also check the measured Thunder mount/table scene.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import copy
import json
import numpy as np
import torch

import gymnasium as gym
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from pxr import UsdGeom

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset import mdp
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import easy_cube_cfg as easy
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import thunder_calibration_cfg as thunder

if args.lab_scene:
    from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import thunder_lab_cfg as profile
    class_prefix = "ThunderLab"
    task_suffix = "CubeEasyThunderLab"
else:
    profile = thunder
    class_prefix = "ThunderCalibration"
    task_suffix = "CubeEasyThunderCalibration"


def main():
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "Small native reset-generation probe; no training or production bank."}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    families = (
        "ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
        "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped", "Train",
    )
    for family in families:
        cfg = getattr(profile, f"{class_prefix}{family}Cfg")()
        parent = getattr(easy, f"CubeEasy{family}Cfg")()
        term = cfg.events.reset_from_reset_states if family == "Train" else cfg.terminations.success
        parent_term = parent.events.reset_from_reset_states if family == "Train" else parent.terminations.success
        assert term.params["joint_limit_joint_names"] == thunder.WRIST_JOINTS
        assert "joint_limit_joint_names" not in parent_term.params
    report["all_five_thunder_configs_opt_in_upstream_configs_unchanged"] = True

    cfg = getattr(profile, f"{class_prefix}ObjectAnywhereEEAnywhereCfg")()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 20260925
    cfg.recorders = mdp.StableStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(args.output)
    cfg.recorders.dataset_filename = "generated_resets.pt"
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler
    env = gym.make(
        f"OmniReset-UR5eRobotiq2f85-{task_suffix}-ObjectAnywhereEEAnywhere-v0", cfg=cfg
    ).unwrapped
    try:
        with torch.inference_mode():
            env.reset(seed=20260925)
            robot = env.scene["robot"]
            ids = [robot.joint_names.index(name) for name in thunder.WRIST_JOINTS]
            expected = torch.tensor([-torch.pi, torch.pi], device=env.device)
            assert torch.allclose(robot.data.joint_pos_limits[:, ids], expected.expand(env.num_envs, 3, 2))
            native = robot.root_physx_view.get_dof_limits()[:, ids].to(env.device)
            assert torch.allclose(native, expected.expand(env.num_envs, 3, 2))
            report["native_wrist_limits_degrees"] = {
                name: torch.rad2deg(native[0, i]).cpu().tolist() for i, name in enumerate(thunder.WRIST_JOINTS)
            }
            report["default_wrists_inside_limits"] = bool(mdp.utils.joint_positions_within_limits(
                robot.data.default_joint_pos[:, ids], robot.data.joint_pos_limits[:, ids]
            ).all())
            assert report["default_wrists_inside_limits"]
            if args.lab_scene:
                roots = robot.data.root_pos_w - env.scene.env_origins
                offsets = roots - torch.tensor(profile.ROBOT_POS, device=env.device)
                assert (offsets.abs() <= torch.tensor([0.01001, 0.02001, 0.01001], device=env.device)).all()
                expected_rot = torch.tensor(profile.ROBOT_ROT, device=env.device)
                assert torch.allclose(robot.data.root_quat_w.abs(), expected_rot.expand(env.num_envs, 4), atol=1e-6)
                table_prim = next(p for p in env.sim.stage.Traverse() if p.GetName() == "vention_mat")
                vertices = np.asarray(UsdGeom.Mesh(table_prim).GetPointsAttr().Get())
                transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(table_prim))
                world = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:, :3]
                world -= env.scene.env_origins[0].cpu().numpy()
                bounds = np.array([world.min(0), world.max(0)])
                assert abs(bounds[1, 2] - profile.TABLE_TOP_Z) < 1e-5, bounds.tolist()
                goals = env.scene["receptive_object"].data.root_pos_w - env.scene.env_origins
                expected_goal = torch.tensor([0.45, 0.15, profile.TABLE_TOP_Z + 0.02], device=env.device)
                assert torch.allclose(goals, expected_goal.expand(env.num_envs, 3), atol=1e-5), goals
                report["lab_scene"] = {
                    "tabletop_bounds_m": bounds.tolist(),
                    "robot_root_offset_min_m": offsets.min(0).values.cpu().tolist(),
                    "robot_root_offset_max_m": offsets.max(0).values.cpu().tolist(),
                    "robot_quaternion_wxyz": robot.data.root_quat_w[0].cpu().tolist(),
                    "fixed_goal_root_m": goals[0].cpu().tolist(),
                }

            # Inject invalid numerical joint angles only during acceptance checks.
            # Restore buffers before recording or advancing physics.
            term_cfg = env.termination_manager.get_term_cfg("success")
            term = term_cfg.func
            injected_checks = []

            def observe_success(current_env, **kwargs):
                result = term(current_env, **kwargs)
                if result.any() and not injected_checks:
                    row = int(result.nonzero()[0, 0])
                    original_q = robot.data.joint_pos.clone()
                    original_counter = term.stability_counter.clone()
                    original_ids = term.joint_limit_ids
                    try:
                        for name, joint_id in zip(thunder.WRIST_JOINTS, ids):
                            for label, angle in (("above", torch.pi + 0.01), ("below", -torch.pi - 0.01),
                                                 ("nonfinite", float("nan"))):
                                robot.data.joint_pos.copy_(original_q)
                                robot.data.joint_pos[row, joint_id] = angle
                                term.stability_counter.copy_(original_counter)
                                assert not term(current_env, **kwargs)[row], (name, label)
                                term.joint_limit_ids = None
                                term.stability_counter.copy_(original_counter)
                                assert term(current_env, **kwargs)[row], (name, label, "unrestricted control")
                                term.joint_limit_ids = original_ids
                                injected_checks.append({"joint": name, "case": label, "rejected": True})
                    finally:
                        robot.data.joint_pos.copy_(original_q)
                        term.stability_counter.copy_(original_counter)
                        term.joint_limit_ids = original_ids
                return result

            observe_success.reset = term.reset
            term_cfg.func = observe_success
            actions = torch.zeros(env.action_space.shape, device=env.device)
            actions[:, -1] = torch.randint(0, 2, (env.num_envs,), device=env.device) * 2 - 1
            for step in range(args.max_steps):
                env.step(actions)
                if env.recorder_manager.exported_successful_episode_count:
                    break
            term_cfg.func = term
            assert len(injected_checks) == 9, "No accepted state reached for negative acceptance checks"
            report["acceptance_negative_checks"] = injected_checks
            report["policy_steps"] = step + 1

            source = args.output / "generated_resets.pt"
            dataset = torch.load(source, map_location="cpu", weights_only=False)
            positions = torch.stack(dataset["initial_state"]["articulation"]["robot"]["joint_position"])
            limits = robot.data.joint_pos_limits[0, ids].cpu()
            assert mdp.utils.joint_positions_within_limits(positions[:, ids], limits).all()
            report["exported_reset_count"] = len(positions)
            report["exported_wrist_ranges_degrees"] = {
                name: torch.rad2deg(torch.stack((positions[:, i].min(), positions[:, i].max()))).tolist()
                for name, i in zip(thunder.WRIST_JOINTS, ids)
            }

            # Exercise the actual training-bank loader with valid and damaged copies.
            loader_root = args.output / "loader_cases"
            pair = mdp.utils.compute_pair_dir(
                env.scene["insertive_object"].cfg.spawn.usd_path,
                env.scene["receptive_object"].cfg.spawn.usd_path,
            )
            target = loader_root / "Resets" / pair / "resets_probe.pt"
            target.parent.mkdir(parents=True)
            loader_cfg = EventTermCfg(func=mdp.MultiResetManager, mode="reset", params={
                "dataset_dir": str(loader_root), "reset_types": ["probe"], "probs": [1.0],
                "joint_limit_joint_names": thunder.WRIST_JOINTS.copy(),
            })
            torch.save(dataset, target)
            loader = mdp.MultiResetManager(loader_cfg, env)
            loaded_q = loader.datasets[0]["initial_state"]["articulation"]["robot"]["joint_position"]
            assert torch.equal(loaded_q.cpu(), positions)
            loader_checks = []
            for name, joint_id in zip(thunder.WRIST_JOINTS, ids):
                for label, angle in (("above", torch.pi + 0.01), ("below", -torch.pi - 0.01),
                                     ("nonfinite", float("nan"))):
                    damaged = copy.deepcopy(dataset)
                    damaged["initial_state"]["articulation"]["robot"]["joint_position"][0][joint_id] = angle
                    torch.save(damaged, target)
                    try:
                        mdp.MultiResetManager(loader_cfg, env)
                    except ValueError as error:
                        assert "violate native limits" in str(error), str(error)
                    else:
                        raise AssertionError((name, label, "invalid bank loaded"))
                    loader_checks.append({"joint": name, "case": label, "rejected": True})
            # The default upstream loader retains its existing behavior when not opted in.
            unrestricted_cfg = loader_cfg.copy()
            unrestricted_cfg.params.pop("joint_limit_joint_names")
            mdp.MultiResetManager(unrestricted_cfg, env)
            torch.save(dataset, target)
            report["loader_negative_checks"] = loader_checks
            report["valid_bank_loaded_without_modification"] = True
            report["upstream_loader_opt_out_preserved"] = True
            report["status"] = "PASS"
            save()
            print("WRIST_LIMITS_PASS", json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status="failed", failure=repr(error))
        save()
        raise
    finally:
        env.close()


try:
    main()
finally:
    app.close()
