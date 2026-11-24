"""Paired opposite-target carry demonstrations (docs/vla/BOWL_CHANNEL_PAIRED_PLAN_2026-11-23.md).

The bowl swap probe (docs/vla/BOWL_SWAP_PROBE_2026-11-23.md) showed the ACT student ignores the target-bowl input:
every training layout demonstrates one target, so images and pose identify the bowl and the instruction never has to
be read. Here the expert runs a train layout to its held ``present_bottom`` checkpoint; one branch places the brick by
its label (the normal demonstration), the other into the other bowl. Both branches share the physical state at the
fork, so the target-bowl instruction is the only input that tells their carries apart. Only carry and release frames
are supervised (the prefix duplicates the base dataset's episode of the same layout).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..agent.perturbations import HeldCheckpointExpertTracer

SWAP = {"green": "pink", "pink": "green"}
CARRY_KINDS: tuple[str, ...] = ("carry_to_bowl", "release_in_bowl")


class TargetSwapTracer(HeldCheckpointExpertTracer):
	"""Expert tracer that also checkpoints the held brick after ``present_bottom``. With ``target_swap`` the brick goes
	to the other bowl (or ``forced_target`` when set); the placement check and the final verdict are judged against
	that bowl and the result carries ``target_swap: True`` (so a swapped episode is never mistaken for a correct
	sort)."""

	def __init__(self, *args, target_swap: bool = True, forced_target: str | None = None, **kwargs):
		self.target_swap = bool(target_swap)
		self.forced_target = forced_target
		self._in_place = False
		super().__init__(*args, **kwargs)

	def _place_checkpoint(self) -> None:
		"""Held checkpoint at the start of the release motion: the fork point of a pair. A defective brick is often
		decided from the wrist views alone and grasped inside ``place`` (no ``present_bottom``), so the checkpoint is
		taken here for every label, with the views of the latest skill-boundary checkpoint."""
		views = self.checkpoints[-1].views if self.checkpoints else []
		self._save(list(views))

	def _grasp(self):
		error = super()._grasp()
		if error is None and self._in_place and self.held is not None:
			self._place_checkpoint()
		return error

	def place(self, label):
		if label != "unknown" and self.held is not None:
			self._place_checkpoint()
		self._in_place = True
		try:
			return super().place(label)
		finally:
			self._in_place = False

	def target_bowl(self, label: str) -> str:
		forced = getattr(self, "forced_target", None)
		if forced is not None:
			return forced
		bowl = "pink" if label == "defective" else "green"
		return SWAP[bowl] if self.target_swap else bowl


def restrict_supervision(
	arrays: Mapping[str, np.ndarray],
	segment_kinds: Sequence[str],
	keep: Sequence[str] = CARRY_KINDS,
	from_frame: int = 0,
) -> dict[str, np.ndarray]:
	"""Copy of ``arrays`` (an ``actions.npz`` row set) whose ``supervise_action`` is True only on frames of a ``keep``
	segment at or after ``from_frame``, and only where it was True before. ``segment_kinds[i]`` is the kind of segment
	id ``i``; frames outside any segment (id -1) are not supervised."""
	seg = np.asarray(arrays["segment"], dtype=np.int64)
	n = seg.shape[0]
	prior = np.asarray(arrays["supervise_action"], dtype=bool) if "supervise_action" in arrays else np.ones(n, bool)
	keep = set(keep)
	kind_ok = np.array([0 <= s < len(segment_kinds) and segment_kinds[s] in keep for s in seg], dtype=bool)
	frame_ok = np.arange(n) >= int(from_frame)
	out = {k: np.array(v, copy=True) for k, v in arrays.items()}
	out["supervise_action"] = prior & kind_ok & frame_ok
	return out


def _carry_only(ep_dir, from_frame: int) -> dict:
	"""Rewrite ``branch-0/actions.npz`` of an exportable episode so only carry/release frames from ``from_frame`` on
	are supervised; returns the supervised frame count and the carry segment's bowl."""
	from ..agent.twin_record import segment_labels

	segments = segment_labels(ep_dir)["segments"]
	kinds = [s["kind"] for s in segments]
	path = ep_dir / "branch-0" / "actions.npz"
	with np.load(path, allow_pickle=False) as z:
		arrays = {k: z[k] for k in z.files}
	out = restrict_supervision(arrays, kinds, from_frame=from_frame)
	np.savez_compressed(path, **out)
	carries = [s for s in segments if s["kind"] == "carry_to_bowl" and s["start"] >= from_frame]
	return {
		"supervised_frames": int(out["supervise_action"].sum()),
		"carry_bowls": [s["params"].get("bowl") for s in carries],
	}


def run_pair(case, scorer, out_dir, *, crop_size: int, sensing: str = "realistic") -> dict:
	"""Run one paired-target demonstration of ``case`` into ``out_dir`` (see the module docstring).

	``out_dir/raw/parent``: the expert episode (by label, held checkpoint after ``present_bottom``);
	``out_dir/raw/swap``: the swapped branch resumed from that checkpoint in a fresh twin;
	``out_dir/parent`` and ``out_dir/swap``: the exportable single-branch episodes (the parent moved out of ``raw``, the
	swap stitched onto the parent prefix), both supervised on carry/release frames after the fork only; ``raw`` is
	removed once they exist. Returns the pair record (``pair.json``)."""
	import json
	import shutil
	from pathlib import Path

	from langgraph.checkpoint.memory import InMemorySaver

	from ..agent.twin_graph import Policy, build_graph
	from ..agent.twin_mujoco import sample_case
	from ..agent.twin_record import EpisodeTrace, stitch_perturbation_episode, student_frames
	from ..agent.twin_resume import load_history, load_plan, restore

	out_dir = Path(out_dir)
	raw_parent, raw_swap = out_dir / "raw" / "parent", out_dir / "raw" / "swap"
	rec: dict = {"case": list(case), "episodes": {"parent": None, "swap": None}}
	provenance = {"engine": "mujoco", "sensing": sensing, "case": list(case), "paired_carry": True}

	twin = TargetSwapTracer(
		sample_case(*case),
		scorer,
		EpisodeTrace(raw_parent, video=True),
		crop_size=crop_size,
		sensing=sensing,
		target_swap=False,
	)
	try:
		parent_result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
		twin.record.finish(parent_result, {**provenance, "target_swap": False})
	finally:
		twin.close()
	rec["parent_verdict"] = parent_result["verdict"]
	if parent_result["verdict"] != "SUCCESS" or not parent_result.get("student_eligible"):
		rec["skip"] = "parent_not_eligible"
		return rec

	plan = load_plan(raw_parent)
	held = [i for i, c in enumerate(plan.checkpoints) if c.held is not None] if plan else []
	if not held:
		rec["skip"] = "no_held_checkpoint"
		return rec
	plan.checkpoints = plan.checkpoints[: held[-1] + 1]
	ck = plan.latest
	plan.history, plan.empty_bowl_depth = load_history(raw_parent, ck)
	rec["fork_frame"] = int(ck.frame)

	twin = TargetSwapTracer(
		sample_case(*case),
		scorer,
		EpisodeTrace(raw_swap, video=True),
		crop_size=crop_size,
		sensing=sensing,
		target_swap=True,
	)
	try:
		link = restore(twin, plan)
		graph = build_graph(twin, Policy(), InMemorySaver())
		config = {"recursion_limit": 64, "configurable": {"thread_id": "swap"}}
		graph.update_state(config, plan.graph_state(), as_node="prepare")
		swap_result = graph.invoke(None, config=config)["result"]
		twin.record.finish(swap_result, {**provenance, "target_swap": True, "resumed_from": link})
	finally:
		twin.close()
	rec["swap_verdict"] = swap_result["verdict"]
	rec["labels"] = {"parent": parent_result.get("label"), "swap": swap_result.get("label")}
	if swap_result["verdict"] != "SUCCESS":
		rec["skip"] = "swap_failed"
		rec["swap_failures"] = [f.get("code") for f in swap_result.get("failures", [])]
		return rec

	# The fork: the swapped branch's first observation must be the parent's observation at the checkpoint frame.
	with (
		np.load(raw_parent / "branch-0" / "actions.npz", allow_pickle=False) as pz,
		np.load(raw_swap / f"branch-{ck.branch}" / "actions.npz", allow_pickle=False) as sz,
	):
		diff = float(np.max(np.abs(pz["state_f64"][ck.frame] - sz["state_f64"][0])))
	rec["fork_state_max_diff_rad"] = diff
	rec["fork_state_match"] = diff <= 1e-6

	ep_parent, ep_swap = out_dir / "parent", out_dir / "swap"
	for ep in (ep_parent, ep_swap):
		if ep.exists():
			shutil.rmtree(ep)
	stitch_perturbation_episode(raw_parent, ck.branch, ep_swap, branch_episode=raw_swap)
	# Disk: the parent episode is moved (not copied) out of raw/, the raw swap branch is dropped once stitched, and
	# the lossy review video is not needed by the exporter.
	shutil.move(str(raw_parent), str(ep_parent))
	shutil.rmtree(out_dir / "raw")
	for ep in (ep_parent, ep_swap):
		for mp4 in ep.rglob("replay.mp4"):
			mp4.unlink()
	rec["parent"] = _carry_only(ep_parent, ck.frame)
	rec["swap"] = _carry_only(ep_swap, ck.frame)
	swap_manifest = json.loads((ep_swap / "episode.json").read_text(encoding="utf-8"))
	rec["swap_target_swap_flag"] = bool(swap_manifest["result"].get("target_swap"))
	bowls = (rec["parent"]["carry_bowls"][:1], rec["swap"]["carry_bowls"][:1])
	rec["bowls"] = {"parent": (bowls[0] or [None])[0], "swap": (bowls[1] or [None])[0]}
	ok = (
		rec["fork_state_match"]
		and rec["swap_target_swap_flag"]
		and rec["labels"]["parent"] == rec["labels"]["swap"]
		and SWAP.get(rec["bowls"]["parent"]) == rec["bowls"]["swap"]
	)
	for name, ep in (("parent", ep_parent), ("swap", ep_swap)):
		frames = student_frames(ep)
		try:
			next(frames)
		except Exception as exc:  # noqa: BLE001 - record why an episode is not exportable
			rec.setdefault("export_errors", {})[name] = f"{type(exc).__name__}: {exc}"
			ok = False
		finally:
			frames.close()
	if ok:
		rec["episodes"] = {"parent": str(ep_parent), "swap": str(ep_swap)}
	else:
		rec["skip"] = "pair_check_failed"
	return rec


def _resolve_detector(detector, crop_size: int | None):
	import json
	from pathlib import Path

	from ..inspection_scorer import default_sup_scorer

	if isinstance(detector, (str, Path)):
		path = Path(detector)
		crop = (
			int(json.loads(path.read_text(encoding="utf-8")).get("crop_size", 256))
			if crop_size is None
			else int(crop_size)
		)
		return default_sup_scorer(path), crop, True
	crop = int(getattr(detector, "crop_size", 256) if crop_size is None else crop_size)
	return detector, crop, False


def _available_slots(layout) -> tuple[str, ...]:
	extra = getattr(layout, "extra_bowls", ()) or ()
	return ("green", "pink", *(f"extra:{i}" for i in range(len(extra))))


def _select_branch_targets(layout, true_slot: str, seed: int) -> list[str]:
	wrong = [s for s in _available_slots(layout) if s != true_slot]
	if len(wrong) > 2:
		rng = np.random.default_rng([int(seed), 0xB8A4C4])
		idxs = sorted(int(i) for i in rng.choice(len(wrong), size=2, replace=False))
		wrong = [wrong[i] for i in idxs]
	return [true_slot, *wrong]


def _fork_state_diff(raw_parent, raw_branch, ck) -> float:
	with (
		np.load(raw_parent / "branch-0" / "actions.npz", allow_pickle=False) as pz,
		np.load(raw_branch / f"branch-{ck.branch}" / "actions.npz", allow_pickle=False) as sz,
	):
		return float(np.max(np.abs(pz["state_f64"][ck.frame] - sz["state_f64"][0])))


def _verify_exportable(ep_dir) -> bool:
	from ..agent.twin_record import student_frames

	frames = student_frames(ep_dir)
	try:
		next(frames)
		return True
	except Exception:  # noqa: BLE001 - report branch as non-exportable
		return False
	finally:
		if hasattr(frames, "close"):
			frames.close()


def _strip_replay_mp4(ep_dir) -> None:
	for mp4 in ep_dir.rglob("replay.mp4"):
		mp4.unlink()


def run_branches(
	case,
	out,
	detector,
	*,
	crop_size: int | None = None,
	sensing: str = "realistic",
	action_noise=None,
) -> dict:
	"""Run a multi-branch demonstration from the held checkpoint for ``case`` into ``out``.

	Branches to the true target bowl plus up to 2 distinct wrong bowls (all 3 bowls on 3-bowl v3 scenes), recording
	carry-only supervision per branch. Returns ``{"fork_frame", "branches", "fork_state_match", ...}``."""
	import shutil
	from pathlib import Path

	from langgraph.checkpoint.memory import InMemorySaver

	from ..agent.twin_graph import Policy, build_graph
	from ..agent.twin_mujoco import sample_case
	from ..agent.twin_record import EpisodeTrace, stitch_perturbation_episode
	from ..agent.twin_resume import load_history, load_plan, restore

	out_dir = Path(out)
	raw_dir = out_dir / "raw"
	if raw_dir.exists():
		shutil.rmtree(raw_dir)
	raw_parent = raw_dir / "parent"
	scorer, crop, owns_scorer = _resolve_detector(detector, crop_size)
	tracer_kwargs: dict = {"crop_size": crop, "sensing": sensing}
	if action_noise is not None:
		tracer_kwargs["action_noise"] = action_noise
	provenance = {"engine": "mujoco", "sensing": sensing, "case": list(case), "paired_carry": True}

	rec: dict = {
		"case": list(case),
		"parent_verdict": None,
		"fork_frame": None,
		"branches": [],
		"fork_state_match": False,
	}
	try:
		layout = sample_case(*case)
		twin = TargetSwapTracer(
			layout,
			scorer,
			EpisodeTrace(raw_parent, video=True),
			target_swap=False,
			**tracer_kwargs,
		)
		try:
			parent_result = build_graph(twin, Policy()).invoke({}, config={"recursion_limit": 64})["result"]
			twin.record.finish(parent_result, {**provenance, "target_swap": False})
			expected_label = "defective" if layout.defective else "normal"
			true_slot = twin.target_bowl(expected_label)
			slot_colours = {s: twin.bowl_colour(s) for s in _available_slots(layout)}
		finally:
			twin.close()

		rec["parent_verdict"] = parent_result["verdict"]
		if parent_result["verdict"] != "SUCCESS" or not parent_result.get("student_eligible"):
			rec["skip"] = "parent_not_eligible"
			return rec

		plan = load_plan(raw_parent)
		held = [i for i, c in enumerate(plan.checkpoints) if c.held is not None] if plan else []
		if not held:
			rec["skip"] = "no_held_checkpoint"
			return rec
		plan.checkpoints = plan.checkpoints[: held[-1] + 1]
		ck = plan.latest
		plan.history, plan.empty_bowl_depth = load_history(raw_parent, ck)
		rec["fork_frame"] = int(ck.frame)

		targets = _select_branch_targets(layout, true_slot, int(case[0]))
		wrong_runs = []
		for idx, slot in enumerate(targets[1:], start=1):
			raw_branch = raw_dir / f"branch-{idx}"
			twin = TargetSwapTracer(
				sample_case(*case),
				scorer,
				EpisodeTrace(raw_branch, video=True),
				target_swap=True,
				forced_target=slot,
				**tracer_kwargs,
			)
			try:
				link = restore(twin, plan)
				graph = build_graph(twin, Policy(), InMemorySaver())
				config = {"recursion_limit": 64, "configurable": {"thread_id": f"branch-{idx}"}}
				graph.update_state(config, plan.graph_state(), as_node="prepare")
				branch_result = graph.invoke(None, config=config)["result"]
				twin.record.finish(branch_result, {**provenance, "target_swap": True, "resumed_from": link})
			finally:
				twin.close()
			branch_npz = raw_branch / f"branch-{ck.branch}" / "actions.npz"
			diff = _fork_state_diff(raw_parent, raw_branch, ck) if branch_npz.is_file() else None
			match = diff is not None and diff <= 1e-6
			wrong_runs.append((slot, raw_branch, branch_result, match, diff))

		branch_entries = []
		for slot, raw_branch, branch_result, match, diff in wrong_runs:
			colour = slot_colours[slot]
			ep_branch = out_dir / slot.replace(":", "_")
			if ep_branch.exists():
				shutil.rmtree(ep_branch)
			if branch_result["verdict"] == "SUCCESS" and match:
				stitch_perturbation_episode(raw_parent, ck.branch, ep_branch, branch_episode=raw_branch)
				_strip_replay_mp4(ep_branch)
				info = _carry_only(ep_branch, ck.frame)
				ok_ep = _verify_exportable(ep_branch)
				if not ok_ep:
					shutil.rmtree(ep_branch, ignore_errors=True)
				branch_entries.append(
					{
						"target": slot,
						"colour": colour,
						"verdict": branch_result["verdict"],
						"supervised_frames": info["supervised_frames"] if ok_ep else 0,
						"dir": str(ep_branch) if ok_ep else "",
						"fork_state_match": match,
						"fork_state_max_diff_rad": diff,
					}
				)
			else:
				branch_entries.append(
					{
						"target": slot,
						"colour": colour,
						"verdict": branch_result["verdict"],
						"supervised_frames": 0,
						"dir": "",
						"fork_state_match": match,
						"fork_state_max_diff_rad": diff,
					}
				)

		ep_parent = out_dir / true_slot.replace(":", "_")
		if ep_parent.exists():
			shutil.rmtree(ep_parent)
		shutil.move(str(raw_parent), str(ep_parent))
		_strip_replay_mp4(ep_parent)
		parent_info = _carry_only(ep_parent, ck.frame)
		ok_parent = _verify_exportable(ep_parent)
		parent_entry = {
			"target": true_slot,
			"colour": slot_colours[true_slot],
			"verdict": parent_result["verdict"],
			"supervised_frames": parent_info["supervised_frames"] if ok_parent else 0,
			"dir": str(ep_parent) if ok_parent else "",
			"fork_state_match": True,
			"fork_state_max_diff_rad": 0.0,
		}
		shutil.rmtree(raw_dir, ignore_errors=True)
		rec["branches"] = [parent_entry, *branch_entries]
		rec["fork_state_match"] = bool(wrong_runs) and all(m for _, _, _, m, _ in wrong_runs)
		return rec
	finally:
		if owns_scorer and hasattr(scorer, "close"):
			scorer.close()
