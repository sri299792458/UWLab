"""Validate the live UMI/AprilCube task's physics and original alignment success.

The no-dataset mode isolates contact and success checks. With --datasets, also
check that every reset family loads and survives brief policy rollouts.
"""

import argparse
import faulthandler
import hashlib
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--datasets", action="store_true")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--dataset-dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils
from pxr import Usd, UsdGeom, UsdPhysics

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg, UmiCubeEvalCfg
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import umi_reset_cfg as reset_cfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.rewards import ProgressContext, success_reward
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.commands import TaskCommand
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_reset_cfg import (
    UMI_ROBOT_USD as UMI_ROBOT_USD_PATH, INSERTIVE_USD as INSERTIVE_CUBE_USD,
    RECEPTIVE_USD as RECEPTIVE_CUBE_USD, TABLE_Z as LAB_TABLETOP_TOP_Z,
)

TASK = "OmniReset-UMI-Defaults-State-Train-v0"
active_env = None


def finite_nested(x):
    if isinstance(x, torch.Tensor):
        assert torch.isfinite(x).all(), "Non-finite tensor in rollout"
    elif isinstance(x, dict):
        for value in x.values():
            finite_nested(value)


def main():
    global active_env
    configuration_motion_modes = {}
    for config_type in (UmiCubeTrainCfg, UmiCubeEvalCfg, reset_cfg.UmiPartialAssembliesCfg,
                        reset_cfg.UmiObjectAnywhereEEAnywhereCfg, reset_cfg.UmiObjectRestingEEGraspedCfg,
                        reset_cfg.UmiObjectAnywhereEEGraspedCfg, reset_cfg.UmiObjectPartiallyAssembledEEGraspedCfg):
        candidate = config_type()
        modes = {name: getattr(candidate.scene, name).spawn.rigid_props.kinematic_enabled
                 for name in ("insertive_object", "receptive_object")}
        assert modes == {"insertive_object": False, "receptive_object": True}, (config_type.__name__, modes)
        configuration_motion_modes[config_type.__name__] = modes
    cfg = UmiCubeTrainCfg()
    if args.dataset_dir is not None:
        cfg.events.reset_from_reset_states.params["dataset_dir"] = args.dataset_dir
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 37
    if not args.datasets:
        cfg.events.reset_from_reset_states = None
    # Keep synthetic fixtures in place while testing even if the arm starts in
    # an awkward default pose. The training configuration itself is unchanged.
    cfg.terminations.abnormal_robot = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    active_env = env
    faulthandler.dump_traceback_later(30, repeat=True)
    print("VALIDATION: resetting", flush=True)
    env.reset()
    print("VALIDATION: reset done", flush=True)
    robot, top, bottom = (env.scene[k] for k in ("robot", "insertive_object", "receptive_object"))
    context = env.reward_manager.get_term_cfg("progress_context").func
    assert type(context) is ProgressContext
    assert cfg.commands.task_command.class_type is TaskCommand
    assert cfg.rewards.success_reward.func is success_reward
    assert cfg.rewards.success_reward.weight == 1.0
    finger_id = robot.joint_names.index("finger_joint")
    base_id = robot.body_names.index("robotiq_base_link")
    report = {"robot_usd": robot.cfg.spawn.usd_path, "task": TASK, "datasets_enabled": args.datasets,
              "cube_assets": [top.cfg.spawn.usd_path, bottom.cfg.spawn.usd_path],
              "dt": env.physics_dt, "policy_dt": env.step_dt,
              "success_definition": "may_pose_alignment", "checks": {}}
    report["configuration_kinematic_flags"] = configuration_motion_modes
    report["controller"] = {"law": "explicit_torque",
                            "kp": list(cfg.actions.arm.motion_stiffness),
                            "damping_ratio": list(cfg.actions.arm.motion_damping_ratio)}
    assert not hasattr(cfg.actions.arm, "implicit_damping"), "Removed implicit-controller API unexpectedly present"
    report["controller"]["kd"] = [2 * kp**.5 * ratio for kp, ratio in zip(cfg.actions.arm.motion_stiffness, cfg.actions.arm.motion_damping_ratio)]
    assert report["controller"]["kp"] == [500., 500., 500., 60., 60., 60.]
    assert report["controller"]["kd"] == [160., 160., 160., .1, .1, .1]
    if args.datasets:
        dataset_dir = Path(cfg.events.reset_from_reset_states.params["dataset_dir"])
        report["dataset_dir"] = str(dataset_dir)
        report["dataset_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(dataset_dir.rglob('resets_*.pt'))}
    assert robot.cfg.spawn.usd_path == UMI_ROBOT_USD_PATH
    assert [top.cfg.spawn.usd_path, bottom.cfg.spawn.usd_path] == [INSERTIVE_CUBE_USD, RECEPTIVE_CUBE_USD]
    for name, cube in (("top", top), ("bottom", bottom)):
        mass = cube.root_physx_view.get_masses().cpu().reshape(-1)
        assert ((mass >= 0.02) & (mass <= 0.20)).all(), mass
        expected_kinematic = name == "bottom"
        assert cube.cfg.spawn.rigid_props.kinematic_enabled is expected_kinematic
        prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{'InsertiveObject' if name == 'top' else 'ReceptiveObject'}")
        body = next(p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI))
        assert bool(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()) is expected_kinematic
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        size = bbox.ComputeLocalBound(prim).ComputeAlignedRange().GetSize()
        report[name] = {"mass_kg": mass.tolist(), "authored_size_m": list(size),
                        "dynamic": not expected_kinematic, "kinematic": expected_kinematic}
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, -1] = 1
    ids = torch.arange(env.num_envs, device=env.device)

    def put_cubes(top_xy_shift=0.0, height_shift=0.0):
        context.reset(ids)
        # Isolate cube contacts from random dataset hand poses. Grasp capability
        # is tested separately below with the unmodified sampled robot states.
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                      torch.zeros_like(robot.data.joint_vel), env_ids=ids)
        for cube, z, dx in ((bottom, 0.03, 0), (top, 0.09, top_xy_shift)):
            pose = torch.tensor([0.30 + dx, -0.20, LAB_TABLETOP_TOP_Z + z + height_shift, 1, 0, 0, 0], device=env.device).repeat(env.num_envs, 1)
            pose[:, :3] += env.scene.env_origins
            cube.write_root_pose_to_sim(pose)
            cube.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))

    def steps(n):
        for _ in range(n):
            obs, rew, terminated, truncated, extras = env.step(actions)
            finite_nested(obs)
            finite_nested(rew)
            finite_nested(env.scene.get_state(is_relative=True))

    put_cubes()
    print("VALIDATION: settling synthetic stack", flush=True)
    steps(25)
    print("VALIDATION: stack settled", flush=True)
    report["settled_stack"] = {"success": context.success.cpu().tolist(),
                               "aligned": (context.orientation_aligned & context.position_aligned).cpu().tolist(),
                               "top_pos": (top.data.root_pos_w-env.scene.env_origins).cpu().tolist(),
                               "bottom_pos": (bottom.data.root_pos_w-env.scene.env_origins).cpu().tolist(),
                               "driver_angle": robot.data.joint_pos[:, finger_id].cpu().tolist()}
    failed = ~context.success
    if failed.any():
        report["settled_stack"]["failed_conditions"] = {
            "env_ids": ids[failed].cpu().tolist(),
            "aligned": (context.orientation_aligned & context.position_aligned)[failed].cpu().tolist(),
            "top_speed": torch.linalg.vector_norm(top.data.root_lin_vel_w[failed],dim=1).cpu().tolist(),
            "top_angular_speed": torch.linalg.vector_norm(top.data.root_ang_vel_w[failed],dim=1).cpu().tolist(),
            "bottom_speed": torch.linalg.vector_norm(bottom.data.root_lin_vel_w[failed],dim=1).cpu().tolist(),
            "bottom_angular_speed": torch.linalg.vector_norm(bottom.data.root_ang_vel_w[failed],dim=1).cpu().tolist(),
            "hand_pos": (robot.data.body_pos_w[failed,base_id]-env.scene.env_origins[failed]).cpu().tolist()}
        print("FAILED_STACK_CONDITIONS",json.dumps(report["settled_stack"]["failed_conditions"]),flush=True)
    print("VALIDATION STACK RESULT", json.dumps(report["settled_stack"]), flush=True)
    assert context.success.all(), report["settled_stack"]
    report["checks"]["aligned_stack_accepted"] = True
    assert torch.equal(success_reward(env), torch.ones(env.num_envs, device=env.device))
    report["checks"]["alignment_reward_matches_may"] = True
    # Check the original success predicate directly, without advancing physics:
    # an aligned state earns its bonus regardless of finger closure.
    put_cubes()
    joint_pos = robot.data.joint_pos.clone()
    joint_pos[:, finger_id] = 0.8
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)
    context(env, **cfg.rewards.progress_context.params)
    assert context.success.all() and (success_reward(env) == 1).all()
    report["checks"]["closed_hand_does_not_gate_alignment"] = True
    # Motion and absolute table height do not enter May's alignment predicate.
    put_cubes(height_shift=0.15)
    for cube in (top, bottom):
        velocity = torch.zeros((env.num_envs, 6), device=env.device)
        velocity[:, 0] = 0.1
        cube.write_root_velocity_to_sim(velocity)
    context(env, **cfg.rewards.progress_context.params)
    assert context.success.all() and (success_reward(env) == 1).all()
    report["checks"]["moving_aligned_pair_accepted"] = True
    put_cubes(top_xy_shift=0.08)
    steps(20)
    assert not context.success.any()
    report["checks"]["separate_cubes_rejected"] = True
    # The upper cube falls when unsupported; the lower cube remains fixed even
    # above the table and under a horizontal force. Separate them for this check.
    put_cubes(top_xy_shift=0.15, height_shift=0.15)
    start_top_z = top.data.root_pos_w[:, 2].clone()
    start_bottom_pose = bottom.data.root_pose_w.clone()
    force = torch.zeros((env.num_envs, 1, 3), device=env.device)
    force[:, :, 0] = 10.0
    bottom.set_external_force_and_torque(force, torch.zeros_like(force))
    steps(2)
    bottom.set_external_force_and_torque(torch.zeros_like(force), torch.zeros_like(force))
    assert ((start_top_z - top.data.root_pos_w[:, 2]) > 0.05).all()
    assert torch.allclose(bottom.data.root_pose_w, start_bottom_pose, atol=1e-6, rtol=0)
    report["fixed_cube_probe"] = {
        "upper_drop_m": (start_top_z - top.data.root_pos_w[:, 2]).cpu().tolist(),
        "lower_max_pose_change": float((bottom.data.root_pose_w - start_bottom_pose).abs().max()),
        "lower_applied_force_n": 10.0,
        "duration_s": 2 * env.step_dt,
    }
    report["checks"]["upper_cube_falls_under_gravity"] = True
    report["checks"]["lower_cube_fixed_under_gravity_and_force"] = True
    if args.datasets:
        resetter = env.event_manager.get_term_cfg("reset_from_reset_states").func
        report["dataset_counts"] = resetter.num_states.cpu().tolist()
        for family in range(resetter.num_tasks):
            resetter.probs.zero_()
            resetter.probs[family] = 1
            env.reset()
            assert (resetter.task_id == family).all()
            loaded_bottom_pose = bottom.data.root_pose_w.clone()
            steps(10)
            assert torch.allclose(bottom.data.root_pose_w, loaded_bottom_pose, atol=1e-6, rtol=0)
        report["checks"]["all_reset_families_finite"] = True
        report["checks"]["lower_cube_fixed_after_all_reset_families"] = True
        # Exercise full-arm loaded grasps with the training controller. Its
        # translation action is in the rotated robot-root frame: convert an
        # upward world displacement before dividing by the action scale.
        families = cfg.events.reset_from_reset_states.params["reset_types"]
        grasp_family = families.index("ObjectAnywhereEEGrasped")
        resetter.probs.zero_()
        resetter.probs[grasp_family] = 1
        env.reset()
        actions.zero_()
        actions[:, -1] = -1
        steps(10)
        start_top = top.data.root_pos_w.clone()
        start_hand = robot.data.body_pos_w[:, base_id].clone()
        start_relative = start_top - start_hand
        world_delta = torch.zeros((env.num_envs, 3), device=env.device)
        world_delta[:, 2] = 0.01
        root_delta = math_utils.quat_apply_inverse(robot.data.root_quat_w, world_delta)
        scale = torch.tensor(cfg.actions.arm.scale_xyz_axisangle[:3], device=env.device)
        actions[:, :3] = root_delta / scale
        steps(25)
        actions[:, :3] = 0
        steps(5)
        rise = top.data.root_pos_w[:, 2] - start_top[:, 2]
        hand_rise = robot.data.body_pos_w[:, base_id, 2] - start_hand[:, 2]
        relative_drift = torch.linalg.vector_norm(
            top.data.root_pos_w - robot.data.body_pos_w[:, base_id] - start_relative, dim=1)
        lifted = (rise > 0.02) & (hand_rise > 0.02) & (relative_drift < 0.03)
        before_release = top.data.root_pos_w[:, 2].clone()
        actions[:, -1] = 1
        steps(10)
        drop = before_release - top.data.root_pos_w[:, 2]
        released = lifted & (drop > 0.02)
        report["loaded_grasp_probe"] = {
            "environments": env.num_envs, "lifted": int(lifted.sum()), "released": int(released.sum()),
            "cube_rise_m": rise.cpu().tolist(), "hand_rise_m": hand_rise.cpu().tolist(),
            "relative_drift_m": relative_drift.cpu().tolist(), "release_drop_m": drop.cpu().tolist(),
            "mass_kg": top.root_physx_view.get_masses().cpu().reshape(-1).tolist(),
        }
        print("LOADED_GRASP_PROBE", json.dumps(report["loaded_grasp_probe"]), flush=True)
        # This is a controller/physics capability check, not a trained-policy
        # success estimate. Require multiple complete lift/release examples.
        if int(released.sum()) < max(1, env.num_envs // 8):
            report["validation_status"] = "failed_loaded_lift_and_release"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert int(released.sum()) >= max(1, env.num_envs // 8), report["loaded_grasp_probe"]
        report["checks"]["loaded_lift_and_release"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("UMI TASK VALIDATION PASSED", json.dumps(report), flush=True)
    faulthandler.cancel_dump_traceback_later()
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    if active_env is not None:
        active_env.close()
    app.close()
