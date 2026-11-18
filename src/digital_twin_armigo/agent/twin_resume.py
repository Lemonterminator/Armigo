"""Restart an interrupted episode from its latest persisted skill-boundary checkpoint.

What is restored (all read from the parent episode directory, plus the LangGraph SQLite thread when present):

* MuJoCo integration state, command filter (latency queue, backlash) and relocate-expert scalars, exactly as
  ``TwinMujoco.rewind`` restores them in memory: the brick is on the table, the gripper is open, the arm is at a
  ready pose;
* the live checkpoint list (rewind targets), reconstructed by replaying ``checkpoint``/``rewound`` events, so the
  resumed run can still rewind to an earlier boundary;
* the graph state ``views`` (from the checkpoint) and ``failures`` (from the saved LangGraph thread);
* ``branch``/``rewinds`` (a branch id equals the rewind count) and the wrist/bottom view counters.

What is not restored: telemetry rows and video of the parent run (the resumed trace starts at the checkpoint, its
physical-health verdict covers only the resumed segment, ``active_frames`` restarts at 0 and ``sim_frames`` restarts
from the checkpoint frame), failures raised after the last boundary, and any node that was mid-flight. Resumed
episodes are never ``student_eligible``.
"""

from __future__ import annotations

import json
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mujoco_cell._mj import mujoco
from ..mujoco_cell.table_relocation import RelocateExpert
from .twin_mujoco import INITIAL_Q, STATE_SPEC, Checkpoint


@dataclass
class ResumePlan:
	parent_dir: Path
	checkpoints: list[Checkpoint]
	failures: list[dict]
	parent_job_id: str | None = None

	@property
	def latest(self):
		return self.checkpoints[-1]

	def graph_state(self):
		return {"views": self.latest.views, "failures": self.failures, "error": None, "label": "unknown"}


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
	return ResumePlan(parent_dir, checkpoints, saved_failures(saver, parent_job_id), parent_job_id)


def restore(backend, plan):
	"""Put a freshly built ``TwinMujoco`` at the plan's latest checkpoint (mirrors ``TwinMujoco.rewind``)."""
	ck = plan.latest
	m, d = backend.cell.model, backend.cell.data
	mujoco.mj_setState(m, d, ck.integration, STATE_SPEC)
	mujoco.mj_forward(m, d)
	mujoco.mj_setState(m, d, ck.integration, STATE_SPEC)
	filt = backend.ep.filt
	filt._queue = deque((np.array(q) for q in ck.filter["queue"]), maxlen=filt._queue.maxlen)
	filt._prev_out = np.array(ck.filter["prev"])
	filt._backlash_pos = np.array(ck.filter["backlash"])
	backend.expert = ex = RelocateExpert(
		backend.cell, INITIAL_Q, backend.layout.brick_xy, backend.expert.p, backend.expert.rp
	)
	ex.last = np.array(ck.expert["last"])
	ex._active_open_g = ck.expert["active_open"]
	ex._frame_idx = ck.expert["frame_idx"]
	ex.gripper_monitor._prev_meas = ck.expert["monitor_position"]
	ex.gripper_monitor._run = ck.expert["monitor_count"]
	backend.held = None
	backend.actual_destination = None
	backend.execution_error = None
	backend.wrist_count = ck.wrist_count
	backend.bottom_count = ck.bottom_count
	backend.branch = backend.rewinds = ck.branch
	backend.total_frames = ck.frame
	backend.checkpoints = list(plan.checkpoints)
	backend.record.restored(backend.cell)
	trace = backend.record
	# The resumed trace is a linked episode: its first branch starts at the restored boundary.
	trace.branch = ck.branch
	trace.branches[0].update(
		id=ck.branch, start_frame=ck.frame, parent_branch=ck.branch, checkpoint=ck.id, resumed_from=str(plan.parent_dir)
	)
	# Make the new directory self-contained: copy checkpoints, keyframes and view events of the parent.
	for item in plan.checkpoints:
		for suffix in (".npy", ".json"):
			shutil.copy2(
				plan.parent_dir / "checkpoints" / f"{item.id}{suffix}", trace.checkpoints / f"{item.id}{suffix}"
			)
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
