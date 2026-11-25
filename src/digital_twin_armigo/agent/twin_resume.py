"""Restart an interrupted episode from its latest persisted skill-boundary checkpoint.

What is restored (all read from the parent episode directory, plus the LangGraph SQLite thread when present), so the
resumed episode continues exactly as the uninterrupted one would have (``TwinMujoco.resume_state``):

* MuJoCo integration state, command filter (latency queue, backlash) and the relocate-expert scalars, also for a
  checkpoint taken while the brick is held (lift pose, grasp roll, believed grip-from-brick transform);
* the agent's sensing state: brick belief, perceived bowls, top-camera calibration, contact filter, depth-noise RNG,
  park spot and expected put-down yaw, and the empty-bowl depth reference the placement check compares against
  (``empty-bowl-depth-<digest>.npy``, written once per episode);
* the per-frame history up to the checkpoint (``<id>.history.npz``: telemetry rows, policy targets and phases), so
  the first resumed observation sees the previous gripper command, effort and jaw reading, frame numbers continue,
  and the final health and smoothness evaluation covers the whole episode;
* the expert parameters, recovery variants, clipped-target and LLM counters, and the segment rows of the segment
  twins (``agent/student_rollout.py``);
* the live checkpoint list (rewind targets), reconstructed by replaying ``checkpoint``/``rewound`` events, so the
  resumed run can still rewind to an earlier boundary;
* the graph state ``views`` (from the checkpoint) and ``failures`` (from the saved LangGraph thread);
* ``branch``/``rewinds`` (a branch id equals the rewind count), ``retries`` (in-place recoveries, which share the
  recovery budget) and the wrist/bottom view counters.

What is not restored: the parent's trace rows and video (the resumed trace directory records from the checkpoint on),
failures raised after the last boundary, and any node that was mid-flight. Checkpoints written before the history and
depth files existed still load; their resumed episodes lack that history. Resumed episodes are never
``student_eligible``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .twin_mujoco import HISTORY_SUFFIX, Checkpoint


@dataclass
class ResumePlan:
	parent_dir: Path
	checkpoints: list[Checkpoint]
	failures: list[dict]
	parent_job_id: str | None = None
	history: dict[str, np.ndarray] | None = None  # the latest checkpoint's per-frame history
	empty_bowl_depth: np.ndarray | None = None

	@property
	def latest(self):
		return self.checkpoints[-1]

	def graph_state(self):
		state = {"views": self.latest.views, "failures": self.failures, "error": None, "label": "unknown"}
		agent = self.latest.agent or {}
		if "recalibration_attempted" in agent:
			state["recalibration_attempted"] = agent["recalibration_attempted"]
		return state


def live_checkpoint_ids(events_path):
	"""Checkpoint ids still valid at the end of the log (``rewound`` discards the ones after its target)."""
	live: list[str] = []
	with Path(events_path).open(encoding="utf-8") as events:
		for line in events:
			if not line.endswith("\n"):
				break  # torn final line from a hard kill
			event = json.loads(line)
			if event["kind"] == "checkpoint":
				if event["checkpoint"] in live:
					live = live[: live.index(event["checkpoint"])]
				live.append(event["checkpoint"])
			elif event["kind"] == "rewound" and event["checkpoint"] in live:
				live = live[: live.index(event["checkpoint"]) + 1]
	return live


def load_checkpoint(directory, checkpoint_id):
	raw = json.loads((Path(directory) / "checkpoints" / f"{checkpoint_id}.json").read_text(encoding="utf-8"))
	integration = np.load(Path(directory) / "checkpoints" / f"{checkpoint_id}.npy", allow_pickle=False)
	return Checkpoint(integration=integration, **raw)


def load_history(directory, checkpoint):
	"""``(history arrays, empty-bowl depth)`` persisted with ``checkpoint``; ``None`` for files that do not exist."""
	directory = Path(directory) / "checkpoints"
	history = depth = None
	if (directory / f"{checkpoint.id}{HISTORY_SUFFIX}").is_file():
		with np.load(directory / f"{checkpoint.id}{HISTORY_SUFFIX}", allow_pickle=False) as rows:
			history = {key: rows[key] for key in rows.files}
	name = (checkpoint.agent or {}).get("empty_bowl_depth")
	if name and (directory / name).is_file():
		depth = np.load(directory / name, allow_pickle=False)
	return history, depth


def saved_failures(saver, thread_id):
	if saver is None or not thread_id:
		return []
	tuple_ = saver.get_tuple({"configurable": {"thread_id": thread_id}})
	return list(tuple_.checkpoint["channel_values"].get("failures", [])) if tuple_ else []


def load_plan(parent_dir, *, saver=None, parent_job_id=None):
	"""A ``ResumePlan`` for the episode directory, or ``None`` when no checkpoint was persisted."""
	parent_dir = Path(parent_dir)
	events = parent_dir / "events.jsonl"
	if not events.is_file():
		return None
	ids = live_checkpoint_ids(events)
	if not ids:
		return None
	checkpoints = [load_checkpoint(parent_dir, name) for name in ids]
	history, depth = load_history(parent_dir, checkpoints[-1])
	return ResumePlan(parent_dir, checkpoints, saved_failures(saver, parent_job_id), parent_job_id, history, depth)


def restore(backend, plan):
	"""Put a freshly built ``TwinMujoco`` at the plan's latest checkpoint (mirrors ``TwinMujoco.restore_checkpoint``)."""
	ck = plan.latest
	backend.resume_state(ck, plan.checkpoints, plan.history, plan.empty_bowl_depth)
	backend.branch = backend.rewinds = ck.branch
	backend.retries = ck.retries
	backend.total_frames = ck.frame
	backend.record.restored(backend.cell)
	trace = backend.record
	# The resumed trace is a linked episode: its first branch starts at the restored boundary.
	trace.branch = ck.branch
	trace.branches[0].update(
		id=ck.branch, start_frame=ck.frame, parent_branch=ck.branch, checkpoint=ck.id, resumed_from=str(plan.parent_dir)
	)
	# Make the new directory self-contained: copy checkpoints, keyframes and view events of the parent.
	source = plan.parent_dir / "checkpoints"
	for item in plan.checkpoints:
		names = [f"{item.id}{suffix}" for suffix in (".npy", ".json", HISTORY_SUFFIX)]
		names.append((item.agent or {}).get("empty_bowl_depth"))
		for name in names:
			if name and (source / name).is_file() and not (trace.checkpoints / name).is_file():
				shutil.copy2(source / name, trace.checkpoints / name)
	for image in (plan.parent_dir / "keyframes").glob("*.png"):
		shutil.copy2(image, trace.keyframes / image.name)
	wanted = {view["id"] for view in ck.views}
	with (plan.parent_dir / "events.jsonl").open(encoding="utf-8") as events:
		for line in events:
			if not line.endswith("\n"):
				break
			event = json.loads(line)
			if event["kind"] == "view" and event["observation"]["id"] in wanted:
				payload = {
					k: v for k, v in event.items() if k not in ("seq", "kind", "node", "branch", "frame", "sim_time_s")
				}
				payload.pop("wall_time_s", None)
				trace.event("view", event["node"], event["branch"], event["frame"], inherited=True, **payload)
	for item in plan.checkpoints:
		backend.event("checkpoint", checkpoint=item.id, restored=True)
	backend.event("resumed", checkpoint=ck.id, parent=str(plan.parent_dir), parent_frame=ck.frame)
	trace_json = {
		"parent_dir": str(plan.parent_dir),
		"parent_job_id": plan.parent_job_id,
		"checkpoint": ck.id,
		"frame": ck.frame,
		"branch": ck.branch,
	}
	return trace_json
