"""Verify direct Isaac state restoration at a table-supported, open-gripper skill boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IsaacOpenCheckpoint:
	joint_pos: object
	joint_vel: object
	joint_target: object
	rigids: dict


def capture(scene, target):
	robot = scene["robot"]
	return IsaacOpenCheckpoint(
		robot.data.joint_pos.torch.clone(),
		robot.data.joint_vel.torch.clone(),
		target.clone(),
		{name: obj.data.root_state_w.torch.clone() for name, obj in scene.rigid_objects.items()},
	)


def restore(scene, checkpoint):
	"""Restore observable articulation, targets and rigid roots. PhysX solver caches are internal."""
	scene.reset()
	robot = scene["robot"]
	robot.write_joint_position_to_sim_index(position=checkpoint.joint_pos)
	robot.write_joint_velocity_to_sim_index(velocity=checkpoint.joint_vel)
	robot.actuators.target_command.set_position_index(value=checkpoint.joint_target)
	for name, state in checkpoint.rigids.items():
		scene[name].write_root_state_to_sim(state)
	scene.write_data_to_sim()


def probe(sim, scene, target, output: Path):
	"""Run from the active Stage 2 cell. Compare native restore continuations without a grasp."""
	import torch

	dt = sim.get_physics_dt()

	def step(command, count):
		robot = scene["robot"]
		robot.actuators.target_command.set_position_index(value=command)
		for _ in range(count):
			scene.write_data_to_sim()
			sim.step()
			scene.update(dt)

	def maxabs(value):
		return float(torch.max(torch.abs(value)).item())

	start = capture(scene, target)
	command = target.clone()
	pan_id = scene["robot"].joint_names.index("shoulder_pan")
	command[:, pan_id] += 0.10
	step(command, 72)
	baseline = capture(scene, command)
	drift = maxabs(baseline.joint_pos - start.joint_pos)
	# Move farther on the failed branch, then restore the original checkpoint.
	step(target - (command - target), 144)
	restore(scene, start)
	step(command, 72)
	replayed = capture(scene, command)
	joint = maxabs(baseline.joint_pos - replayed.joint_pos)
	brick = maxabs(baseline.rigids["brick"][:, :3] - replayed.rigids["brick"][:, :3])
	angular = float(
		2
		* torch.acos(
			torch.abs(
				torch.nn.functional.normalize(baseline.rigids["brick"][:, 3:7], dim=1)
				.mul(torch.nn.functional.normalize(replayed.rigids["brick"][:, 3:7], dim=1))
				.sum(dim=1)
			).clamp(0, 1)
		)
		.max()
		.item()
	)
	report = {
		"engine": "isaac-physx",
		"checkpoint": "open_gripper_table_supported",
		"settle_steps": 72,
		"movement_rad": drift,
		"joint_error_rad": joint,
		"brick_position_error_m": brick,
		"brick_angle_error_rad": angular,
		"criteria": {"min_test_motion_rad": 0.01, "joint_rad": 0.02, "brick_m": 0.002, "brick_angle_rad": 0.05},
		"status": "PASS"
		if drift >= 0.01 and joint <= 0.02 and brick <= 0.002 and angular <= 0.05
		else "REPLAY_DIVERGED",
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(json.dumps(report, indent=2))
	return report
