"""Failure ledger clustering, factor skew analysis, and Markdown rendering (Task F2).

Impact definition (from plan section 1):
    Each failure episode contributes a severity loss weight:
    - explicit ``loss`` in factors/review if provided;
    - physical outcome ``"table"`` -> ``0.5``, ``"lost"`` / bowl outcome -> ``1.0``;
    - otherwise by effective root cause: ``"evidence_insufficient"`` -> ``0.3``,
      ``"premature_stop"`` or ``code`` in ``("UNKNOWN", "ABSTAIN")`` -> ``0.5``,
      all other failures -> ``1.0``.
    Cluster ``impact`` is the sum of episode losses in that cluster; ``freq_share``
    (and backwards-compatible alias ``share``) is ``n / failure_episodes`` while
    ``impact_share`` is ``impact / total_impact`` (used by stopping criteria).
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from .contract import case_id as make_case_id
from .scoring import _json
from .triage import TriageRow, triage_episode

VIEWER_TRAJ_URL = "http://127.0.0.1:8767/#"
MAX_EXAMPLES = 10

_SKIP_DIRS = frozenset(
	{"checkpoints", "keyframes", "llm_images", "replay", "branch-0", "poses", "videos", "images", "isaac"}
)
_ATTEMPT_RE = re.compile(r"-a(\d+)$")
_OUTCOME_LOSS = {"table": 0.5, "lost": 1.0, "green_bowl": 1.0, "pink_bowl": 1.0}


def _strip_attempt(name: str) -> str:
	return _ATTEMPT_RE.sub("", name)


def _attempt_from_name(name: str) -> int:
	m = _ATTEMPT_RE.search(name)
	return int(m.group(1)) if m else 1


def _size_class(brick_size) -> str | None:
	if isinstance(brick_size, str):
		return brick_size
	if isinstance(brick_size, (list, tuple)) and len(brick_size) >= 2:
		lx, ly = float(brick_size[0]), float(brick_size[1])
		lo, hi = min(lx, ly), max(lx, ly)
		if abs(lo - 0.0318) < 2e-3:
			if abs(hi - 0.0318) < 2e-3:
				return "2x2"
			if abs(hi - 0.0477) < 2e-3:
				return "2x3"
			if abs(hi - 0.0636) < 2e-3:
				return "2x4"
		return f"{round(hi * 1000.0, 1)}x{round(lo * 1000.0, 1)}mm"
	return None


def _coerce_row(item: TriageRow | dict) -> TriageRow:
	if isinstance(item, TriageRow):
		return item
	return TriageRow(
		traj_id=str(item["traj_id"]),
		stage=str(item.get("stage") or "unknown"),
		code=item.get("code"),
		root_cause=str(item.get("root_cause") or "unclassified"),
		evidence=dict(item.get("evidence") or {}),
		rule=str(item.get("rule") or "unclassified"),
		status=item.get("status"),
		mechanism=item.get("mechanism"),
		physical_success=item.get("physical_success"),
	)


def episode_loss(
	row: TriageRow,
	review: dict | None = None,
	ep_factors: dict | None = None,
	*,
	effective_cause: str | None = None,
) -> float:
	"""Compute severity weight for one episode."""
	for src in (ep_factors, review):
		if isinstance(src, dict) and isinstance(src.get("loss"), (int, float)):
			return float(src["loss"])
	for src in (review, ep_factors):
		if isinstance(src, dict):
			po = src.get("physical_outcome")
			if po in _OUTCOME_LOSS:
				return _OUTCOME_LOSS[po]
	cause = effective_cause or row.root_cause
	if cause == "none":
		return 0.0
	if cause == "evidence_insufficient":
		return 0.3
	if cause == "premature_stop" or row.code in ("UNKNOWN", "ABSTAIN"):
		return 0.5
	return 1.0


def _resolve_review(row: TriageRow, review: dict | None) -> tuple[str, bool | None]:
	"""Return ``(effective_root_cause, agreed_bool_or_none)``."""
	if not isinstance(review, dict):
		return row.root_cause, None
	human_cause = review.get("root_cause") or review.get("decision")
	state = review.get("state")
	if human_cause not in (None, ""):
		cause_str = str(human_cause)
		return cause_str, (cause_str == row.root_cause)
	if state == "confirmed":
		return row.root_cause, True
	if state == "disputed":
		return row.root_cause, False
	return row.root_cause, None


def build_ledger(
	rows: list[TriageRow],
	reviews: dict[str, dict] | None = None,
	factors: dict[str, dict] | None = None,
) -> dict:
	"""Cluster triaged failure episodes by ``(stage, code, effective_root_cause)`` and sort by impact."""
	reviews = reviews or {}
	factors = factors or {}
	coerced = [_coerce_row(r) for r in rows]

	groups: dict[tuple[str, str | None, str], dict] = {}
	ok_episodes = 0
	recovered_rows = []
	unscored_rows = []
	total_reviewed = 0
	total_agreed = 0

	for row in coerced:
		rev = reviews.get(row.traj_id)
		ep_fac = factors.get(row.traj_id) or {}
		effective_cause, agreed = _resolve_review(row, rev)

		if agreed is not None:
			total_reviewed += 1
			if agreed:
				total_agreed += 1

		if row.status == "recovered":
			recovered_rows.append(
				{"traj_id": row.traj_id, "stage": row.stage, "code": row.code, "mechanism": row.mechanism}
			)
			continue
		if row.status == "unscored":
			unscored_rows.append(row.traj_id)
			continue
		if row.status == "ok" or (row.status is None and effective_cause == "none"):
			ok_episodes += 1
			continue
		if effective_cause == "none":
			effective_cause = row.root_cause

		key = (row.stage, row.code, effective_cause)
		grp = groups.get(key)
		if grp is None:
			grp = {
				"n": 0,
				"impact": 0.0,
				"factors": defaultdict(Counter),
				"examples": [],
				"reviewed": 0,
				"agreed": 0,
				"mechanisms": Counter(),
			}
			groups[key] = grp

		grp["n"] += 1
		if row.mechanism:
			grp["mechanisms"][row.mechanism] += 1
		grp["impact"] += episode_loss(row, rev, ep_fac, effective_cause=effective_cause)
		if len(grp["examples"]) < MAX_EXAMPLES:
			grp["examples"].append(row.traj_id)
		if agreed is not None:
			grp["reviewed"] += 1
			if agreed:
				grp["agreed"] += 1

		if isinstance(ep_fac, dict):
			for fname, fval in ep_fac.items():
				if fname == "loss" or fval is None:
					continue
				grp["factors"][str(fname)][str(fval)] += 1

	total_failures = sum(g["n"] for g in groups.values())
	total_impact = sum(g["impact"] for g in groups.values())

	clusters = []
	for (stage, code, cause), grp in groups.items():
		n = grp["n"]
		freq_share = round(n / total_failures, 6) if total_failures else 0.0
		impact_share = round(grp["impact"] / total_impact, 6) if total_impact > 0 else 0.0
		factor_dist = {
			fname: {val: cnt for val, cnt in sorted(cnts.items(), key=lambda kv: (-kv[1], kv[0]))}
			for fname, cnts in sorted(grp["factors"].items())
		}
		clusters.append(
			{
				"key": [stage, code, cause],
				"n": n,
				"impact": round(grp["impact"], 6),
				"freq_share": freq_share,
				"impact_share": impact_share,
				"share": freq_share,  # backwards-compatible alias of freq_share; stopping criteria use impact_share
				"factors": factor_dist,
				"examples": grp["examples"],
				"mechanisms": dict(sorted(grp["mechanisms"].items())),
				"review_agreement": round(grp["agreed"] / grp["reviewed"], 6) if grp["reviewed"] else None,
			}
		)

	clusters.sort(key=lambda c: (-c["impact"], -c["n"], [str(x or "") for x in c["key"]]))

	return {
		"total_episodes": len(coerced),
		"ok_episodes": ok_episodes,
		"recovered_episodes": len(recovered_rows),
		"recovered": recovered_rows,
		"unscored_episodes": len(unscored_rows),
		"unscored": unscored_rows,
		"failure_episodes": total_failures,
		"total_impact": round(total_impact, 6),
		"reviewed_episodes": total_reviewed,
		"review_agreement": round(total_agreed / total_reviewed, 6) if total_reviewed else None,
		"clusters": clusters,
	}


def check_stop_criterion(
	ledger: dict,
	*,
	max_top_impact_share: float = 0.15,
	max_top_impact: float | None = None,
) -> dict:
	"""Evaluate loop stopping criterion using severity-weighted ``impact_share`` (not frequency count share)."""
	clusters = ledger.get("clusters") or []
	if not clusters:
		return {
			"stop": True,
			"reason": "no_failures",
			"top_cluster": None,
			"top_impact_share": 0.0,
			"top_freq_share": 0.0,
			"top_impact": 0.0,
		}
	top = clusters[0]
	total_impact = float(ledger.get("total_impact") or 0.0)
	top_impact = float(top.get("impact") or 0.0)
	impact_share = (
		float(top["impact_share"])
		if top.get("impact_share") is not None
		else (top_impact / total_impact if total_impact > 0 else 0.0)
	)
	freq_share = float(top.get("freq_share") if top.get("freq_share") is not None else (top.get("share") or 0.0))
	stop_by_share = impact_share <= max_top_impact_share
	stop_by_impact = max_top_impact is not None and top_impact <= max_top_impact
	stop = stop_by_share or stop_by_impact
	reason = (
		"top_impact_share_below_threshold"
		if stop_by_share
		else ("top_impact_below_threshold" if stop_by_impact else "continue")
	)
	return {
		"stop": stop,
		"reason": reason,
		"top_cluster": top.get("key"),
		"top_impact_share": round(impact_share, 6),
		"top_freq_share": round(freq_share, 6),
		"top_impact": round(top_impact, 6),
	}


def extract_episode_factors(
	ep_dir: Path,
	plan: dict | None = None,
	*,
	episode: dict | None = None,
	task: dict | None = None,
	scene: dict | None = None,
) -> dict:
	"""Read scene factors (camera condition, bowl colour, brick size, clutter, etc.) without rendering."""
	ep_dir = Path(ep_dir)
	if plan is None:
		for candidate in (ep_dir / "plan.json", ep_dir.parent / "plan.json"):
			loaded = _json(candidate)
			if isinstance(loaded, dict):
				plan = loaded
				break

	episode = episode if episode is not None else (_json(ep_dir / "episode.json") or {})
	result = episode.get("result") or {}
	exp = ((episode.get("provenance") or {}).get("experiment")) or {}
	task = task if task is not None else (_json(ep_dir / "task.json") or {})
	scene = scene if scene is not None else (_json(ep_dir / "replay" / "scene.json") or {})
	layout = scene.get("layout") if isinstance(scene.get("layout"), dict) else {}

	case = task.get("case") or []
	case_opts = case[3] if len(case) > 3 and isinstance(case[3], dict) else {}
	plan_opts = plan.get("options") if isinstance(plan, dict) and isinstance(plan.get("options"), dict) else {}

	camera_condition = (
		exp.get("camera_condition")
		or (plan or {}).get("camera_condition")
		or case_opts.get("cameras")
		or plan_opts.get("cameras")
		or result.get("camera_condition")
	)

	defective: bool | None = None
	if len(case) > 1 and isinstance(case[1], bool):
		defective = case[1]
	elif isinstance(layout.get("defective"), bool):
		defective = layout["defective"]

	target_slot = "pink" if defective is True else ("green" if defective is False else None)

	bowl_colour = case_opts.get("bowl_colour") or result.get("bowl_colour")
	if bowl_colour is None and target_slot is not None:
		bowl_names = layout.get("bowl_names")
		if isinstance(bowl_names, dict) and bowl_names.get(target_slot):
			bowl_colour = str(bowl_names[target_slot])
		elif case_opts.get("scene") == "v3" and len(case) >= 1:
			try:
				from ..mujoco_cell.scene_v3 import sample_scene_v3

				sc_v3 = sample_scene_v3(int(case[0]), str(case_opts.get("split", "train")))
				bowl_colour = sc_v3.defect if defective else sc_v3.normal
			except (ImportError, OSError, ValueError, TypeError):
				bowl_colour = target_slot
		else:
			bowl_colour = target_slot

	raw_size = (
		case_opts.get("size_class")
		or case_opts.get("brick_size")
		or (plan or {}).get("brick_size")
		or layout.get("brick_size")
		or result.get("size_class")
		or result.get("brick_size")
	)
	raw_clutter = (
		case_opts.get("clutter")
		if "clutter" in case_opts
		else (plan or {}).get("clutter", layout.get("clutter", result.get("clutter")))
	)

	if (raw_size is None or raw_clutter is None) and len(case) >= 3:
		try:
			from ..agent.twin_mujoco import sample_case

			lay = sample_case(*case)
			if raw_size is None:
				raw_size = lay.brick_size
			if raw_clutter is None:
				raw_clutter = lay.clutter
		except (ImportError, OSError, ValueError, TypeError):
			pass

	brick_size = _size_class(raw_size)
	clutter: int | None = None
	if isinstance(raw_clutter, (list, tuple)):
		clutter = len(raw_clutter)
	elif raw_clutter is not None:
		clutter = int(raw_clutter)

	out: dict = {}
	for k, v in (
		("camera_condition", camera_condition),
		("bowl_colour", bowl_colour),
		("brick_size", brick_size),
		("clutter", clutter),
		("strategy", exp.get("strategy") or (plan or {}).get("strategy")),
		("split", case_opts.get("split") or plan_opts.get("split") or (plan or {}).get("split")),
	):
		if v is not None:
			out[k] = v
	return out


def _scan_batch_records(root: Path, batch: str) -> list[tuple[Path, dict | None, dict, dict]]:
	"""Discover latest-attempt episodes under ``<root>/runs/<batch>`` with cached plan/episode/task."""
	batch_dir = Path(root) / "runs" / batch
	if not batch_dir.is_dir():
		raise FileNotFoundError(batch_dir)

	plan_cache: dict[Path, dict | None] = {}

	def _plan_for(parent: Path) -> dict | None:
		if parent not in plan_cache:
			loaded = _json(parent / "plan.json")
			plan_cache[parent] = loaded if isinstance(loaded, dict) else None
		return plan_cache[parent]

	latest: dict[tuple[str, str], tuple[tuple[int, str], Path, dict | None, dict, dict]] = {}
	for cur, dirs, files in os.walk(batch_dir):
		if "episode.json" in files or "run_status.json" in files:
			ep_dir = Path(cur)
			dirs[:] = []  # Never descend into checkpoints/, keyframes/, replay/, etc.
			ep = _json(ep_dir / "episode.json") or {}
			exp = ((ep.get("provenance") or {}).get("experiment")) or {}
			task = _json(ep_dir / "task.json") or {}
			case = task.get("case") or []
			cid = exp.get("case_id")
			if not cid:
				if len(case) >= 3:
					cid = make_case_id(case[0], bool(case[1]), case[2])
				else:
					cid = _strip_attempt(ep_dir.name)
			attempt = int(exp.get("attempt") or _attempt_from_name(ep_dir.name))
			group_rel = ep_dir.parent.relative_to(batch_dir).as_posix() if ep_dir.parent != batch_dir else ""
			key = (group_rel, str(cid))
			rank = (attempt, ep_dir.name)
			if key not in latest or rank > latest[key][0]:
				latest[key] = (rank, ep_dir, _plan_for(ep_dir.parent), ep, task)
		else:
			dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))

	selected = sorted(latest.values(), key=lambda item: item[1].as_posix())
	return [(ep_dir, plan, ep, task) for _, ep_dir, plan, ep, task in selected]


def scan_batch_episodes(root: Path, batch: str) -> list[tuple[Path, dict | None]]:
	"""Discover episode directories under ``<root>/runs/<batch>`` keeping the latest attempt per case."""
	return [(ep_dir, plan) for ep_dir, plan, _, _ in _scan_batch_records(root, batch)]


def triage_batch(root: Path, batch: str) -> tuple[list[TriageRow], dict[str, dict]]:
	"""Scan and triage all latest-attempt episodes in ``<root>/runs/<batch>``."""
	runs_dir = Path(root) / "runs"
	records = _scan_batch_records(root, batch)
	rows: list[TriageRow] = []
	factors: dict[str, dict] = {}
	for ep_dir, plan, ep, task in records:
		exp = ((ep.get("provenance") or {}).get("experiment")) or {}
		has_exp_id = all(exp.get(k) for k in ("batch", "engine", "group"))
		fallback_id = ep_dir.relative_to(runs_dir).as_posix() if not has_exp_id else None
		row = triage_episode(ep_dir, traj_id=fallback_id)
		rows.append(row)
		factors[row.traj_id] = extract_episode_factors(ep_dir, plan=plan, episode=ep, task=task)
	return rows, factors


def _fmt_pct(val: float | None) -> str:
	if val is None:
		return "-"
	return f"{100.0 * val:.1f}%"


def render_ledger_md(
	ledger: dict,
	*,
	batch: str | None = None,
	round_name: str | None = None,
) -> str:
	"""Render a human-readable Markdown failure ledger with viewer links."""
	header = "# Failure Ledger"
	if batch or round_name:
		header += " — " + " / ".join(p for p in (batch, round_name) if p)
	lines = [
		header,
		"",
		f"- Total episodes: {ledger['total_episodes']}",
		f"- Clean (ok/none): {ledger['ok_episodes']}",
		f"- Recovered (physically successful): {ledger.get('recovered_episodes', 0)}",
		f"- Unscored (outcome unavailable): {ledger.get('unscored_episodes', 0)}",
		f"- Failure episodes: {ledger['failure_episodes']}",
		f"- Total impact: {ledger['total_impact']:.2f}",
		f"- Review agreement: {_fmt_pct(ledger.get('review_agreement'))} ({ledger.get('reviewed_episodes', 0)} reviewed)",
		"",
		"## Clusters",
		"",
		"| # | Stage | Code | Root Cause | N | Freq Share | Impact | Impact Share | Review Agreement |",
		"|---:|---|---|---|---:|---:|---:|---:|---:|",
	]
	for idx, c in enumerate(ledger["clusters"], 1):
		stage, code, cause = c["key"]
		freq_share = c.get("freq_share", c.get("share"))
		impact_share = c.get("impact_share")
		lines.append(
			f"| {idx} | {stage} | {code or '-'} | {cause} | {c['n']} | "
			f"{_fmt_pct(freq_share)} | {c['impact']:.2f} | {_fmt_pct(impact_share)} | "
			f"{_fmt_pct(c['review_agreement'])} |"
		)

	for idx, c in enumerate(ledger["clusters"], 1):
		stage, code, cause = c["key"]
		freq_share = c.get("freq_share", c.get("share"))
		impact_share = c.get("impact_share")
		lines += [
			"",
			f"### {idx}. {stage} / {code or '-'} / {cause}",
			"",
			(
				f"- Count: {c['n']} ({_fmt_pct(freq_share)} of failures) · "
				f"Impact: {c['impact']:.2f} ({_fmt_pct(impact_share)} of impact) · "
				f"Review agreement: {_fmt_pct(c['review_agreement'])}"
			),
		]
		if c["factors"]:
			lines.append("- Factors:")
			for fname, dist in c["factors"].items():
				dist_str = ", ".join(f"{val}={cnt}" for val, cnt in dist.items())
				lines.append(f"  - `{fname}`: {dist_str}")
		if c.get("mechanisms"):
			counts = ", ".join(f"`{name}`={n}" for name, n in c["mechanisms"].items())
			lines.append(f"- Mechanisms: {counts}")
		if c["examples"]:
			lines.append("- Examples:")
			for tid in c["examples"]:
				url = VIEWER_TRAJ_URL + quote(tid, safe="/")
				lines.append(f"  - [{tid}]({url}) (`{url}`)")

	if ledger.get("recovered"):
		lines += ["", "## Recovered episodes", "", "| Trajectory | Stage | Code | Mechanism |", "|---|---|---|---|"]
		for row in ledger["recovered"]:
			tid = row["traj_id"]
			url = VIEWER_TRAJ_URL + quote(tid, safe="/")
			lines.append(f"| [{tid}]({url}) | {row['stage']} | {row['code'] or '-'} | {row['mechanism'] or '-'} |")

	return "\n".join(lines) + "\n"
