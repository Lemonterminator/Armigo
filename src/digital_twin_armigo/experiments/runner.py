"""Batch runner of the v2 experiments (plan 2026-11-24): one experiment group per call, resumable, disk-bounded.

	python -m digital_twin_armigo.experiments.runner run --batch p1-bench16 --engine mujoco --sensing truth \\
		--suite bench16 --detector v7bot --workers 4
	python -m digital_twin_armigo.experiments.runner run --batch p2-dev --sensing realistic --camera recal \\
		--strategy C --suite p2_dev12 --detector v7bot --review-llm-url http://127.0.0.1:8038/v1
	python -m digital_twin_armigo.experiments.runner status --batch p1-bench16

Every episode is written to ``<root>/runs/<batch>/<engine>/<group>/<case_id>[-a<n>]/`` with ``run_status.json``
and ``provenance.experiment`` (see :mod:`.contract`) and a 3D pose replay (``ARMIGO_RERUN=record``). Running the same
command again continues the group: finished cases are skipped, unfinished ones (interrupted, crashed, infra
failure) get a new attempt directory whose ``parent_run`` points at the previous attempt. Sealed suites run only
with ``--freeze <freeze.json>`` whose code commit, detector and options match the current ones exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .contract import (
	CAMERA_CONDITIONS,
	DETECTORS,
	ENGINES,
	EVIDENCE,
	MOTIONS,
	ORACLE_PARTS,
	SEGMENTERS,
	SENSING,
	STRATEGIES,
	case_id,
	freeze_digest,
	layout_id,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path("D:/Armigo_data_v2")
SUITES = REPO / "config/agent_bench/v2"
CAMERA_MODE = {"fixed": "fixed", "recal": "jitter_extrinsic", "drift": "drift"}
GIB, MIB = 1024**3, 1024**2
MAX_ATTEMPTS = 3


def _now() -> str:
	return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
	with path.open("rb") as f:
		return hashlib.file_digest(f, "sha256").hexdigest()


def _write_json(path: Path, data) -> None:
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	os.replace(tmp, path)


def code_commit() -> str:
	try:
		head = subprocess.run(
			["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
		).stdout.strip()
		dirty = subprocess.run(
			["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "config", "scripts"],
			capture_output=True,
			text=True,
			check=True,
		).stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return "unknown"
	return head + ("-dirty" if dirty else "")


def _parse_oracle(raw) -> tuple[str, ...]:
	if not raw:
		return ()
	items = raw.split(",") if isinstance(raw, str) else list(raw)
	parts = sorted({str(p).strip() for p in items if str(p).strip()})
	bad = [p for p in parts if p not in ORACLE_PARTS]
	if bad:
		raise SystemExit(f"unknown oracle part(s) {bad}; expected a subset of {ORACLE_PARTS}")
	return tuple(parts)


def group_name(
	sensing: str,
	camera: str,
	strategy: str,
	detector: str,
	oracle=(),
	*,
	motion: str = "open",
	evidence: str = "hold",
	calib_offset_mm: float = 0,
	segmenter: str = "lab",
	localise: str = "top",
	drift_alarm: bool = False,
) -> str:
	if sensing == "truth":
		base = f"truth-{detector}" if strategy == "A" else f"truth-{strategy}-{detector}"
	else:
		base = f"realistic-{camera}-{strategy}-{detector}"
	parts = sorted({str(p).strip() for p in (oracle or ()) if str(p).strip()})
	if parts:
		base = f"{base}-oracle-{'+'.join(parts)}"
	off = float(calib_offset_mm or 0)
	if motion != "open" or evidence != "hold" or off != 0:
		base = f"{base}-m{motion}"
		if evidence != "hold":
			base = f"{base}-e{evidence}"
		if off != 0:
			mm = int(off) if off.is_integer() else off
			base = f"{base}-c{mm}"
	if segmenter != "lab":
		base = f"{base}-s{segmenter}"
	if localise != "top":
		base = f"{base}-loc-{localise}"
	if drift_alarm:
		base = f"{base}-alarm"
	return base


def _attempt_dir(group_dir: Path, cid: str, n: int) -> Path:
	return group_dir / (cid if n == 1 else f"{cid}-a{n}")


def _attempts(group_dir: Path, cid: str) -> list[tuple[int, Path, dict]]:
	out = []
	for n in range(1, MAX_ATTEMPTS + 10):
		d = _attempt_dir(group_dir, cid, n)
		if not d.exists():
			break
		status_file = d / "run_status.json"
		status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {"state": "planned"}
		out.append((n, d, status))
	return out


def _resolve_suite_src(suites_dir: Path, name: str, meta: dict) -> Path:
	rel = meta.get("path")
	if not rel:
		return suites_dir / f"{name}.json"
	p = Path(rel)
	if p.is_absolute():
		return p
	candidate = (suites_dir / p).resolve()
	if candidate.is_file():
		return candidate
	repo_candidate = (REPO / p).resolve()
	return repo_candidate if repo_candidate.is_file() else candidate


def _suite_source(name: str, suites_dir: Path) -> tuple[Path, dict, dict]:
	manifest = json.loads((suites_dir / "manifest.json").read_text(encoding="utf-8"))
	meta = manifest["suites"].get(name)
	if meta is None:
		raise SystemExit(f"unknown suite {name!r}; known: {sorted(manifest['suites'])}")
	src = _resolve_suite_src(suites_dir, name, meta)
	if _sha256(src) != meta["sha256"]:
		raise SystemExit(f"{src} does not match its manifest sha256")
	return src, meta, manifest


def _freeze_suite(root: Path, name: str, *, suites_dir: Path | None = None) -> tuple[Path, str, bool]:
	"""Copy the suite into ``<root>/suites`` (once) and check it against the repository manifest."""
	src, meta, manifest = _suite_source(name, SUITES if suites_dir is None else Path(suites_dir))
	dst_dir = root / "suites"
	dst_dir.mkdir(parents=True, exist_ok=True)
	dst = dst_dir / f"{name}.json"
	if not dst.exists():
		shutil.copyfile(src, dst)
		_write_json(dst_dir / "manifest.json", manifest)
	if _sha256(dst) != meta["sha256"]:
		raise SystemExit(f"{dst} differs from the frozen suite")
	return dst, meta["sha256"], bool(meta["sealed"])


def detector_path(name: str) -> Path:
	manifest = json.loads((SUITES / "manifest.json").read_text(encoding="utf-8"))
	return REPO / manifest["detectors"][name]


def _selected_detector(a) -> Path:
	spec = getattr(a, "detector_spec", None)
	return Path(spec).resolve() if spec else detector_path(a.detector)


def _checkpoint_hash(det: Path, *, required: bool = False) -> str | None:
	checkpoint = json.loads(det.read_text(encoding="utf-8")).get("checkpoint")
	if not checkpoint:
		return None
	path = det.parent / checkpoint
	if path.is_file():
		return _sha256(path)
	if required:
		raise SystemExit(f"missing detector checkpoint: {path.resolve()}")
	return None


# ---- worker side ----

_SERVICES: dict[tuple, object] = {}


def _service(config: dict):
	key = (
		config["sensing"],
		config["detector"],
		json.dumps(config["agent_options"], sort_keys=True),
		config.get("review_llm_url"),
		config.get("review_model"),
	)
	svc = _SERVICES.get(key)
	if svc is None:
		from ..agent.twin_mcp import TwinService

		review = config.get("review_llm_url")
		review_spec = (
			{
				"base_url": review,
				"model": config["review_model"],
				"max_tokens": 96,
				"timeout_s": 60,
				"retries": 0,
				"json_mode": True,
				"max_model_len": 16384,
			}
			if review
			else None
		)
		svc = TwinService(
			Path(config["scratch"]),
			Path(config["detector"]),
			sensing=config["sensing"],
			agent_options=config["agent_options"],
			review_llm_spec=review_spec,
		)
		_SERVICES[key] = svc
	return svc


def run_job(config: dict, job: dict) -> dict:
	"""Run one episode in this process; never raises for episode failures (they are recorded)."""
	from ..agent.twin_graph import Policy

	os.environ["ARMIGO_RERUN"] = "record"
	path = Path(job["path"])
	path.mkdir(parents=True, exist_ok=False)
	status = {
		"state": "running",
		"started": _now(),
		"finished": None,
		"error": None,
		"host": socket.gethostname(),
		"pid": os.getpid(),
		"experiment": job["experiment"],
	}
	_write_json(path / "run_status.json", status)
	started = time.perf_counter()
	try:
		cell_factory = None
		if config["engine"] == "isaac":
			from ..isaac_bridge.mirror import mirror_cell_factory

			cell_factory = mirror_cell_factory(log_dir=path / "isaac")
		result = _service(config).run(
			tuple(job["case"]),
			Policy(**config["policy"]),
			path,
			video=False,
			use_llm=False,
			hook=None,
			provenance={"experiment": job["experiment"]},
			cell_factory=cell_factory,
		)
		status.update(state="done", verdict=result.get("verdict"), label=result.get("label"))
	except KeyboardInterrupt:
		status.update(state="interrupted", error="KeyboardInterrupt")
		raise
	except BaseException as exc:  # noqa: BLE001 - the record says what broke; the batch goes on
		status.update(state="infra_failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
	finally:
		status["finished"] = _now()
		status["wall_s"] = round(time.perf_counter() - started, 2)
		_write_json(path / "run_status.json", status)
	return {"case_id": job["experiment"]["case_id"], "state": status["state"], "verdict": status.get("verdict")}


def _init_worker():
	os.environ["ARMIGO_RERUN"] = "record"


# ---- driver ----


def _dir_mib(path: Path) -> float:
	return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / MIB


def _group_setup(a) -> tuple[str, dict]:
	oracle = _parse_oracle(getattr(a, "oracle", ""))
	motion = str(getattr(a, "motion", "open") or "open")
	evidence = str(getattr(a, "evidence", "hold") or "hold")
	calib_offset_mm = float(getattr(a, "calib_offset_mm", 0) or 0)
	segmenter = str(getattr(a, "segmenter", "lab") or "lab")
	default_alarm = a.sensing == "realistic" and a.camera == "drift"
	requested_alarm = getattr(a, "drift_alarm", None)
	drift_alarm = default_alarm if requested_alarm is None else requested_alarm
	localise = getattr(a, "localise", None) or ("vote" if default_alarm or drift_alarm else "top")
	if motion not in MOTIONS:
		raise SystemExit(f"unknown motion {motion!r}; expected one of {MOTIONS}")
	if evidence not in EVIDENCE:
		raise SystemExit(f"unknown evidence {evidence!r}; expected one of {EVIDENCE}")
	if segmenter not in SEGMENTERS:
		raise SystemExit(f"unknown segmenter {segmenter!r}; expected one of {SEGMENTERS}")
	if not math.isfinite(calib_offset_mm) or calib_offset_mm < 0:
		raise SystemExit("--calib-offset-mm must be finite and >= 0")
	if a.sensing == "truth" and a.camera != "fixed":
		raise SystemExit("omniscient (truth) runs use the fixed cameras")
	if a.sensing == "truth" and (motion != "open" or evidence != "hold" or calib_offset_mm != 0 or segmenter != "lab"):
		raise SystemExit("motion, evidence, segmenter and calib_offset_mm variants are only for realistic sensing")
	if calib_offset_mm and a.camera == "drift":
		raise SystemExit("--calib-offset-mm cannot be combined with --camera drift")
	if oracle and a.sensing == "truth":
		raise SystemExit("oracle substitution is only for realistic sensing")
	if localise != "top" and a.sensing != "realistic":
		raise SystemExit("--localise vote is only for realistic sensing")
	if drift_alarm and (localise != "vote" or {"brick", "camera"}.intersection(oracle)):
		raise SystemExit("--drift-alarm requires --localise vote and measured brick and camera poses")

	max_views = a.max_scored_views if a.max_scored_views is not None else (6 if a.sensing == "realistic" else None)
	agent_options: dict = {
		"strategy": a.strategy,
		"max_scored_views": max_views,
		"localise": localise,
		"drift_alarm": drift_alarm,
	}
	if a.sensing == "realistic":
		agent_options["boundary"] = a.boundary
	if motion != "open":
		agent_options["motion"] = motion
	if evidence != "hold":
		agent_options["evidence"] = evidence
	if segmenter != "lab":
		agent_options["segmenter"] = segmenter
	if oracle:
		agent_options["oracle"] = list(oracle)

	from ..agent.twin_graph import Policy

	policy = json.loads(Path(a.policy).read_text(encoding="utf-8")) if a.policy else {}
	review_url = getattr(a, "review_llm_url", None)
	if a.strategy == "C" and not review_url:
		raise SystemExit("strategy C needs --review-llm-url")
	group = group_name(
		a.sensing,
		a.camera,
		a.strategy,
		a.detector,
		oracle=oracle,
		motion=motion,
		evidence=evidence,
		calib_offset_mm=calib_offset_mm,
		segmenter=segmenter,
		localise=localise,
		drift_alarm=drift_alarm,
	)
	return group, {
		"sensing": a.sensing,
		"camera_condition": a.camera,
		"agent_options": agent_options,
		"policy": asdict(Policy(**policy)),
		"calib_offset_mm": calib_offset_mm,
		"review_model": a.review_model if review_url else None,
		"review_endpoint_sha256": hashlib.sha256(review_url.encode("utf-8")).hexdigest() if review_url else None,
	}


def _validate_freeze(a, freeze: dict, identity: dict, group: str, setup: dict) -> None:
	if "groups" in freeze:
		if not isinstance(freeze["groups"], dict):
			raise SystemExit("invalid paired freeze: groups must be an object")
		expected = {**identity, "engine": a.engine, "suite": a.suite}
		diffs = [k for k, v in expected.items() if freeze.get(k) != v]
		if freeze["groups"].get(group) != setup:
			diffs.append(f"groups.{group}")
	else:
		# Accept older single-group freezes for the single-group entry point.
		if getattr(a, "real_cameras", None) is not None:
			raise SystemExit("sealed run-pair needs a freeze written by freeze-pair")
		expected = {
			"code_commit": identity["code_commit"],
			"detector_sha256": identity["detector_sha256"],
			"agent_options": setup["agent_options"],
			"policy": setup["policy"],
		}
		if "group_setup" in freeze:
			expected.update(
				group=group,
				group_setup=setup,
				detector=identity["detector"],
				detector_checkpoint_sha256=identity["detector_checkpoint_sha256"],
			)
		diffs = [k for k, v in expected.items() if freeze.get(k) != v]
	if diffs or identity["code_commit"].endswith("-dirty") or identity["code_commit"] == "unknown":
		raise SystemExit(f"freeze file does not match the current setup: {diffs or ['uncommitted or unknown code']}")


def plan_group(a) -> tuple[dict, list[dict], Path]:
	root = Path(a.root)
	group, setup = _group_setup(a)
	agent_options = setup["agent_options"]
	policy = setup["policy"]
	max_views = agent_options["max_scored_views"]
	oracle = agent_options.get("oracle", ())
	motion = agent_options.get("motion", "open")
	evidence = agent_options.get("evidence", "hold")
	segmenter = agent_options.get("segmenter", "lab")
	calib_offset_mm = setup["calib_offset_mm"]
	suite_path, suite_sha, sealed = _freeze_suite(root, a.suite)
	if sealed and not a.freeze:
		raise SystemExit(f"{a.suite} is sealed: pass --freeze <freeze.json> written after development")
	det = _selected_detector(a)
	identity = {
		"code_commit": code_commit(),
		"detector": a.detector,
		"detector_sha256": _sha256(det),
		"detector_checkpoint_sha256": _checkpoint_hash(det, required=sealed),
		"suite_sha256": suite_sha,
	}
	freeze_sha = None
	if sealed:
		freeze = json.loads(Path(a.freeze).read_text(encoding="utf-8"))
		_validate_freeze(a, freeze, identity, group, setup)
		freeze_sha = freeze_digest(freeze)
	group_dir = root / "runs" / a.batch / a.engine / group
	from ..agent.twin_mujoco import load_suite

	cases = load_suite(suite_path)
	if getattr(a, "case_limit", None) is not None:
		cases = cases[: a.case_limit]
	extra_opts: dict = {}
	if a.camera != "fixed":
		extra_opts["cameras"] = CAMERA_MODE[a.camera]
	if calib_offset_mm:
		extra_opts["calib_offset_mm"] = calib_offset_mm
	if oracle:
		extra_opts["oracle"] = list(oracle)
	if extra_opts:
		cases = [(*c[:3], {**(c[3] if len(c) > 3 else {}), **extra_opts}) for c in cases]
	if agent_options["drift_alarm"] and any(
		len(c) > 3 and {"brick", "camera"}.intersection(_parse_oracle(c[3].get("oracle") or "")) for c in cases
	):
		raise SystemExit("--drift-alarm requires measured brick and camera poses in every suite case")
	group_dir.mkdir(parents=True, exist_ok=True)
	ids = [case_id(*c[:3]) for c in cases]
	plan = {
		"batch": a.batch,
		"engine": a.engine,
		"sensing": a.sensing,
		"group": group,
		"camera_condition": a.camera,
		"strategy": a.strategy,
		**identity,
		**({"freeze_sha256": freeze_sha} if freeze_sha else {}),
		**({"oracle": list(oracle)} if oracle else {}),
		"motion": motion,
		"evidence": evidence,
		"calib_offset_mm": calib_offset_mm,
		"segmenter": segmenter,
		"suite": a.suite,
		"sealed": sealed,
		"agent_options": agent_options,
		"policy": policy,
		"review_model": setup["review_model"],
		"review_endpoint_sha256": setup["review_endpoint_sha256"],
		"cases": ids,
	}
	plan_file = group_dir / "plan.json"
	if plan_file.exists():
		old = json.loads(plan_file.read_text(encoding="utf-8"))
		if "code_commit" not in old:
			# Legacy plans did not pin code. Resume only when every existing attempt identifies this commit.
			commits = {
				row.get("experiment", {}).get("code_commit") for cid in ids for _, _, row in _attempts(group_dir, cid)
			}
			if not commits or commits == {identity["code_commit"]}:
				old["code_commit"] = identity["code_commit"]
				old["detector_checkpoint_sha256"] = identity["detector_checkpoint_sha256"]
		changed = [k for k, value in plan.items() if old.get(k) != value]
		if changed:
			raise SystemExit(f"{plan_file} was planned differently ({changed}); use another batch or group")
		plan["created"] = old.get("created")
	else:
		plan["created"] = _now()
		_write_json(plan_file, plan)
	config = {
		"engine": a.engine,
		"sensing": a.sensing,
		"detector": str(det),
		"agent_options": agent_options,
		"policy": policy,
		"scratch": str(root / "scratch" / f"{os.getpid()}"),
		"review_llm_url": a.review_llm_url,
		"review_model": a.review_model,
	}
	jobs = []
	budget = {"max_scored_views": max_views, "max_review_views": 2 if a.strategy != "A" else 0, **policy}
	for case, cid in zip(cases, ids, strict=True):
		attempts = _attempts(group_dir, cid)
		if attempts and attempts[-1][2].get("state") == "done":
			continue
		if len(attempts) >= MAX_ATTEMPTS:
			print(f"  {cid}: {len(attempts)} attempts without success, skipped", flush=True)
			continue
		parent = attempts[-1][2].get("experiment", {}).get("run_id") if attempts else None
		if attempts and attempts[-1][2].get("state") == "running":  # left by a dead runner
			status = attempts[-1][2]
			status.update(state="interrupted", error="runner stopped", finished=_now())
			_write_json(attempts[-1][1] / "run_status.json", status)
		n = len(attempts) + 1
		experiment = {
			"batch": a.batch,
			"engine": a.engine,
			"sensing": a.sensing,
			"group": group,
			"camera_condition": a.camera,
			"strategy": a.strategy,
			**identity,
			**({"freeze_sha256": freeze_sha} if freeze_sha else {}),
			"sealed": sealed,
			"oracle": list(oracle) if oracle else None,
			"motion": motion,
			"evidence": evidence,
			"calib_offset_mm": calib_offset_mm,
			"segmenter": segmenter,
			"localise": agent_options.get("localise", "top"),
			"drift_alarm": agent_options.get("drift_alarm", False),
			"suite": a.suite,
			"layout_id": layout_id(case[0]),
			"case_id": cid,
			"run_id": uuid.uuid4().hex,
			"attempt": n,
			"parent_run": parent,
			"budget": budget,
		}
		jobs.append({"case": list(case), "path": str(_attempt_dir(group_dir, cid, n)), "experiment": experiment})
	return config, jobs, group_dir


def pair_jobs_order(jobs: list[dict]) -> list[dict]:
	"""Order ``jobs`` so truth and realistic runs for the same case stay adjacent (truth first)."""
	order_index: dict[tuple, int] = {}

	def _case_key(job: dict) -> tuple:
		c = job.get("case") or []
		return tuple(c[:3])

	def _group_str(job: dict) -> str:
		return str(job.get("group") or (job.get("experiment") or {}).get("group") or "")

	for j in jobs:
		ck = _case_key(j)
		if ck not in order_index:
			order_index[ck] = len(order_index)
	return sorted(
		jobs,
		key=lambda j: (order_index[_case_key(j)], 0 if _group_str(j).startswith("truth") else 1),
	)


def _execute_jobs(
	items: list[tuple[dict, dict]],
	root: Path,
	engine: str,
	workers: int,
	min_free_gib: float,
	est_mib: float,
) -> int:
	if not items:
		return 0
	workers = 1 if engine == "isaac" else max(1, workers)

	def room(inflight: int) -> bool:
		free = shutil.disk_usage(root).free
		return free > min_free_gib * GIB + (inflight + 1) * est_mib * MIB

	done = 0
	stopped = False
	infra_failed = 0
	if workers == 1:
		for config, job in items:
			if not room(0):
				stopped = True
				break
			row = run_job(config, job)
			infra_failed += int(row["state"] != "done")
			done += 1
			print(f"  [{done}/{len(items)}] {row['case_id']}: {row['state']} {row.get('verdict') or ''}", flush=True)
	else:
		import multiprocessing

		ctx = multiprocessing.get_context("spawn")
		with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as pool:
			pending = {}
			queue = list(items)
			while queue or pending:
				while queue and len(pending) < workers:
					if not room(len(pending)):
						stopped = True
						queue.clear()
						break
					config, job = queue.pop(0)
					pending[pool.submit(run_job, config, job)] = job
				if not pending:
					break
				future = next(as_completed(pending))
				pending.pop(future)
				done += 1
				try:
					row = future.result()
				except Exception as exc:  # noqa: BLE001 - worker died; the attempt dir keeps its running status
					row = {"case_id": "?", "state": f"worker error {exc!r}"}
				infra_failed += int(row["state"] != "done")
				print(
					f"  [{done}/{len(items)}] {row['case_id']}: {row['state']} {row.get('verdict') or ''}", flush=True
				)
	if stopped:
		print(f"stopped at the disk boundary ({min_free_gib} GiB free kept); rerun to continue", flush=True)
		return 3
	sizes = [_dir_mib(Path(j["path"])) for _, j in items if Path(j["path"]).exists()]
	if sizes:
		print(f"episode size: mean {sum(sizes) / len(sizes):.1f} MiB, max {max(sizes):.1f} MiB", flush=True)
	if infra_failed:
		print(f"{infra_failed} infrastructure failure(s); see the attempt status files", flush=True)
		return 2
	return 0


def cmd_run(a) -> int:
	config, jobs, group_dir = plan_group(a)
	if a.limit:
		jobs = jobs[: a.limit]
	print(f"{group_dir}: {len(jobs)} episode(s) to run", flush=True)
	return _execute_jobs([(config, j) for j in jobs], Path(a.root), a.engine, a.workers, a.min_free_gib, a.est_mib)


def _pair_groups(a) -> list[argparse.Namespace]:
	cams = [c.strip() for c in str(a.real_cameras).split(",") if c.strip()]
	if not cams or any(c not in CAMERA_CONDITIONS for c in cams) or len(cams) != len(set(cams)):
		raise SystemExit(f"--real-cameras must be a comma-separated list of distinct values from {CAMERA_CONDITIONS}")
	truth = argparse.Namespace(
		**{
			**vars(a),
			"sensing": "truth",
			"camera": "fixed",
			"oracle": "",
			"motion": "open",
			"evidence": "hold",
			"calib_offset_mm": 0.0,
			"segmenter": "lab",
			"localise": "top",
			"drift_alarm": False,
		}
	)
	return [truth, *(argparse.Namespace(**{**vars(a), "sensing": "realistic", "camera": cam}) for cam in cams)]


def cmd_run_pair(a) -> int:
	pairs_planned: list[dict] = []
	for args in _pair_groups(a):
		config, jobs, group_dir = plan_group(args)
		print(f"{group_dir}: {len(jobs)} {args.sensing} episode(s) planned", flush=True)
		for job in jobs:
			pairs_planned.append({**job, "group": job["experiment"]["group"], "_cfg": config})
	ordered = pair_jobs_order(pairs_planned)
	if a.limit:
		ordered = ordered[: a.limit]
	items = [(j.pop("_cfg"), {k: v for k, v in j.items() if k != "group"}) for j in ordered]
	return _execute_jobs(items, Path(a.root), a.engine, a.workers, a.min_free_gib, a.est_mib)


def cmd_status(a) -> int:
	root = Path(a.root) / "runs" / a.batch
	for plan_file in sorted(root.glob("*/*/plan.json")):
		plan = json.loads(plan_file.read_text(encoding="utf-8"))
		counts: dict[str, int] = {}
		for cid in plan["cases"]:
			attempts = _attempts(plan_file.parent, cid)
			state = attempts[-1][2].get("state", "planned") if attempts else "planned"
			counts[state] = counts.get(state, 0) + 1
		rel = plan_file.parent.relative_to(root)
		print(f"{rel}: {len(plan['cases'])} planned  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
	return 0


def cmd_freeze(a) -> int:
	"""Freeze either a single configuration or every configuration of a paired run."""
	commit = code_commit()
	if commit.endswith("-dirty") or commit == "unknown":
		raise SystemExit("freeze needs a committed development state")
	det = _selected_detector(a)
	freeze = {
		"created": _now(),
		"code_commit": commit,
		"detector": a.detector,
		"detector_sha256": _sha256(det),
		"detector_checkpoint_sha256": _checkpoint_hash(det, required=True),
		"note": a.note,
	}
	if a.command == "freeze-pair":
		_, suite, _ = _suite_source(a.suite, SUITES)
		freeze.update(
			engine=a.engine,
			suite=a.suite,
			suite_sha256=suite["sha256"],
			groups=dict(_group_setup(args) for args in _pair_groups(a)),
		)
	else:
		group, setup = _group_setup(a)
		freeze.update(group=group, group_setup=setup, agent_options=setup["agent_options"], policy=setup["policy"])
	out = Path(a.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	_write_json(out, freeze)
	print(json.dumps(freeze, indent=2))
	return 0


def _common(p, *, include_sensing: bool = True):
	p.add_argument("--root", default=str(DEFAULT_ROOT))
	if include_sensing:
		p.add_argument("--sensing", choices=SENSING, default="truth")
	p.add_argument("--camera", choices=CAMERA_CONDITIONS, default="fixed")
	p.add_argument("--strategy", choices=STRATEGIES, default="A")
	p.add_argument("--detector", choices=DETECTORS, default="v7bot")
	p.add_argument("--detector-spec", type=Path, help="detector JSON with the local checkpoint path")
	p.add_argument("--policy", help="Policy JSON (twin_graph.Policy fields); default Policy()")
	p.add_argument("--max-scored-views", type=int, help="default: 6 for realistic, none for truth")
	p.add_argument("--boundary", choices=("belief_cell", "legacy"), default="belief_cell")
	p.add_argument("--oracle", default="", help="comma list of camera,brick,bowls (evaluation only)")
	p.add_argument("--motion", choices=MOTIONS, default="open")
	p.add_argument("--evidence", choices=EVIDENCE, default="hold")
	p.add_argument("--calib-offset-mm", type=float, default=0.0)
	p.add_argument("--segmenter", choices=SEGMENTERS, default="lab")
	p.add_argument("--localise", choices=("top", "vote"), help="default: vote for drift, top otherwise")
	p.add_argument(
		"--drift-alarm",
		action=argparse.BooleanOptionalAction,
		help="enable one top-camera recalibration (default for realistic drift); --no-drift-alarm disables it",
	)
	p.add_argument("--review-llm-url")
	p.add_argument("--review-model", default="qwen3.8-vllm")


def _suite_args(p):
	p.add_argument("--engine", choices=ENGINES, default="mujoco")
	p.add_argument("--suite", required=True, help="suite name in config/agent_bench/v2/manifest.json")


def _positive_int(raw: str) -> int:
	value = int(raw)
	if value < 1:
		raise argparse.ArgumentTypeError("expected a positive integer")
	return value


def _add_run_args(p):
	_suite_args(p)
	p.add_argument("--batch", required=True)
	p.add_argument("--workers", type=int, default=4)
	p.add_argument("--limit", type=_positive_int, help="maximum jobs attempted by this invocation")
	p.add_argument("--case-limit", type=_positive_int, help="plan only the first N suite cases in each group")
	p.add_argument("--freeze", help="freeze.json required for sealed suites")
	p.add_argument("--min-free-gib", type=float, default=8.0)
	p.add_argument("--est-mib", type=float, default=25.0, help="expected episode size for the disk check")


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	sub = p.add_subparsers(dest="command", required=True)
	run = sub.add_parser("run")
	_common(run, include_sensing=True)
	_add_run_args(run)
	run_pair = sub.add_parser("run-pair")
	_common(run_pair, include_sensing=False)
	_add_run_args(run_pair)
	run_pair.add_argument("--real-cameras", default="fixed,recal,drift")
	status = sub.add_parser("status")
	status.add_argument("--root", default=str(DEFAULT_ROOT))
	status.add_argument("--batch", required=True)
	freeze = sub.add_parser("freeze")
	_common(freeze)
	freeze_pair = sub.add_parser("freeze-pair")
	_common(freeze_pair, include_sensing=False)
	_suite_args(freeze_pair)
	freeze_pair.add_argument("--real-cameras", default="fixed,recal,drift")
	for parser in (freeze, freeze_pair):
		parser.add_argument("--out", required=True)
		parser.add_argument("--note", default="")
	a = p.parse_args(argv)
	return {
		"run": cmd_run,
		"run-pair": cmd_run_pair,
		"status": cmd_status,
		"freeze": cmd_freeze,
		"freeze-pair": cmd_freeze,
	}[a.command](a)


if __name__ == "__main__":
	sys.exit(main())
