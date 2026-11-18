"""Explicit, frozen export requests with one local worker and append-only task events."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

from digital_twin_armigo.runtime.canonical import sha256_file
from digital_twin_armigo.runtime.errors import LedgerError
from digital_twin_armigo.runtime.ledger import JobState, OperationLedger, job_resource
from digital_twin_armigo.runtime.snapshot import hash_tree

from .store import Catalog, dumps, now

FORMATS = {"archive", "existing-video", "mujoco-video", "isaac-rtx", "lerobot", "rerun"}


@contextlib.contextmanager
def worker_lock(home: Path):
	"""An OS-released process lock, including Windows; serializes GPU work across CLI/UI."""
	f = (home / "worker.lock").open("a+b")
	f.seek(0)
	f.write(b"0")
	f.flush()
	f.seek(0)
	try:
		if os.name == "nt":
			import msvcrt

			msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
		else:
			import fcntl

			fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
		yield
	finally:
		f.close()


class Jobs:
	def __init__(self, catalog: Catalog):
		self.catalog = catalog
		self.ledger = OperationLedger(catalog.home / "jobs.sqlite")
		self.stop = threading.Event()
		self.thread = None

	def submit(
		self, ids, format="archive", profile="preview", request_id=None, allow_incomplete=False, actor="local-user"
	):
		if (
			not isinstance(ids, (list, tuple))
			or not all(isinstance(i, str) for i in ids)
			or not isinstance(allow_incomplete, bool)
		):
			raise ValueError("episode_ids must be a list of ids and allow_incomplete must be boolean")
		if not ids or format not in FORMATS or profile not in ("preview", "quality"):
			raise ValueError("explicit selection, valid format and profile required")
		if request_id:
			existing = self.ledger.job_by_key(request_id)
			if existing:
				old = existing.params
				if (sorted(set(ids)), format, profile, bool(allow_incomplete)) != (
					sorted(e["id"] for e in old["episodes"]),
					old["format"],
					old["profile"],
					old["allow_incomplete"],
				):
					raise ValueError("request id already belongs to another selection")
				return self.get(existing.job_id)
		episodes = [self.catalog.episode(i) for i in sorted(set(ids))]
		if format == "rerun" and len(episodes) != 1:
			raise ValueError("rerun selects exactly one parent experiment")
		if format == "rerun" and episodes[0]["engine"] != "mujoco":
			raise ValueError("Isaac new captures use the stage2 --catalog-capture option")
		for e in episodes:
			if format == "isaac-rtx" and not e["capabilities"]["rtx"]:
				raise ValueError(f"{e['alias']}: complete motion is missing; request a separate rerun first")
			if format == "mujoco-video" and not e["capabilities"]["mujoco"]:
				raise ValueError("MuJoCo full scene motion is required")
			if format == "lerobot" and not e["capabilities"]["observations"]:
				raise ValueError("camera observations are missing")
			if (
				format == "lerobot"
				and (e["quality_status"] != "PASS" or e["verdict"] != "SUCCESS")
				and not allow_incomplete
			):
				raise ValueError("quality is not PASS; explicitly allow a research exception or select other episodes")
			if format == "existing-video" and not any(a["role"].endswith("video") for a in e["artifacts"]):
				raise ValueError("selected experiment has no existing video")
		params = {
			"schema": "catalog-export/v1",
			"episodes": episodes,
			"format": format,
			"profile": profile,
			"allow_incomplete": bool(allow_incomplete),
			"actor": actor,
			"memberships": {e["id"]: self.catalog.memberships(e["id"]) for e in episodes},
		}
		job, _ = self.ledger.register(
			idempotency_key=request_id or uuid.uuid4().hex,
			op_type="catalog-export",
			params=params,
			run_id="catalog",
			branch_id="exports",
		)
		return self.get(job.job_id)

	def get(self, ident):
		job = self.ledger.job(ident)
		event = self.ledger.latest_event(ident)
		progress = event.payload.get("progress", str(event.state))
		if event.state == JobState.RUNNING:
			for logfile in (self.catalog.home / "exports" / ident).glob("*/process.log"):
				try:
					with logfile.open("rb") as f:
						f.seek(max(0, logfile.stat().st_size - 8192))
						lines = f.read().decode("utf-8", errors="replace").splitlines()
					for line in reversed(lines):
						if line.startswith('{"frame":'):
							v = json.loads(line)
							progress = f"{v['frame']} / {v['total']} frames"
							break
				except (OSError, ValueError, KeyError):
					pass
		return {
			"id": ident,
			"state": str(event.state),
			"format": job.params["format"],
			"error": event.payload.get("error"),
			"progress": progress,
			"result": event.payload,
			"download_url": f"/api/jobs/{ident}/download" if event.state == JobState.SUCCEEDED else None,
		}

	def list(self):
		return {"items": [self.get(j.job_id) for j in reversed(self.ledger.jobs())]}

	def run_process(self, args, folder, ident):
		with (folder / "process.log").open("w", encoding="utf-8") as log:
			proc = subprocess.Popen(
				list(map(str, args)),
				cwd=self.catalog.root,
				stdout=log,
				stderr=subprocess.STDOUT,
				creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
			)
			try:
				while proc.poll() is None:
					if self.ledger.state(ident) == JobState.CANCEL_REQUESTED:
						proc.terminate()
						proc.wait(timeout=20)
						raise InterruptedError("cancelled")
					time.sleep(0.25)
				if proc.returncode:
					tail = (folder / "process.log").read_text(encoding="utf-8", errors="replace")[-2000:]
					raise RuntimeError(f"export process exited {proc.returncode}: {tail}")
			finally:
				if proc.poll() is None:
					proc.kill()
					proc.wait()

	def execute(self, ident):
		with worker_lock(self.catalog.home):
			job = self.ledger.job(ident)
			state = self.ledger.state(ident)
			if state not in (JobState.QUEUED, JobState.NEEDS_RECOVERY):
				return self.get(ident)
			lease = self.ledger.acquire_lease(job_resource(ident), f"export-{os.getpid()}")
			folder = self.catalog.home / "exports" / ident
			self.ledger.append_event(ident, JobState.RUNNING, lease=lease)
			try:
				# A retry gets a new job id. A crashed attempt cannot silently overwrite files.
				folder.mkdir(parents=True, exist_ok=False)
				request = job.params
				episodes = request["episodes"]
				for e in episodes:
					for a in e["artifacts"]:
						p = self.catalog.project_path(a["path"])
						if not p.is_file() or p.stat().st_size != a["size"] or sha256_file(p) != a["sha256"]:
							raise ValueError(f"source file missing or changed: {p}")
				manifest = {
					**request,
					"exported_at": now(),
					"job_id": ident,
					"memberships": request.get("memberships", {}),
					"files": [],
					"exporter_code_sha256": hash_tree(
						Path(__file__).resolve().parents[1],
						["catalog/*.py", "lerobot_export.py", "runtime/canonical.py"],
					),
				}
				fmt = request["format"]
				for e in episodes:
					if fmt == "lerobot" and e is not episodes[0]:
						continue
					if self.ledger.state(ident) == JobState.CANCEL_REQUESTED:
						raise InterruptedError("cancelled")
					out = folder / e["id"][:16]
					out.mkdir()
					if fmt in ("archive", "existing-video"):
						portable = []
						for a in e["artifacts"]:
							if fmt == "existing-video" and not a["role"].endswith("video"):
								continue
							name = f"{a['role']}_{a['sha256'][:10]}_{Path(a['path']).name}"
							if a["role"].startswith("resource_"):
								(out / "resources").mkdir(exist_ok=True)
								name = "resources/" + Path(a["path"]).name
							shutil.copyfile(a["path"], out / name)
							portable.append({**a, "path": name})
							if sha256_file(out / name) != a["sha256"]:
								raise ValueError("source changed while copying")
						if fmt == "archive" and {"scene", "motion"} <= {a["role"] for a in portable}:
							(out / "replay.json").write_text(
								dumps({"episode": {**e, "artifacts": portable}, "profile": request["profile"]}),
								encoding="utf-8",
							)
					else:
						requestfile = out / "request.json"
						requestfile.write_text(
							dumps(
								{
									"episode": e,
									"episodes": episodes,
									"memberships": request.get("memberships", {}),
									"profile": request["profile"],
									"allow_incomplete": request["allow_incomplete"],
								}
							),
							encoding="utf-8",
						)
						if fmt == "rerun":
							args = [
								self.catalog.root / ".venv/Scripts/python.exe",
								self.catalog.root / "scripts/catalog_record_mujoco.py",
								"--request",
								requestfile,
								"--out",
								out / "recording",
							]
						elif fmt == "isaac-rtx":
							args = [
								os.environ.get("ARMIGO_ISAAC_PYTHON", r"E:\il\v\Scripts\python.exe"),
								self.catalog.root / "scripts/isaac/catalog_render.py",
								"--request",
								requestfile,
								"--out",
								out / "render",
							]
						elif fmt == "mujoco-video":
							args = [
								self.catalog.root / ".venv/Scripts/python.exe",
								"-m",
								"digital_twin_armigo.catalog.playback",
								"--request",
								requestfile,
								"--out",
								out / "render",
							]
						else:
							args = [
								os.environ.get("ARMIGO_LEROBOT_PYTHON", r"E:\il\lerobot\Scripts\python.exe"),
								self.catalog.root / "src/digital_twin_armigo/catalog/lerobot.py",
								"--request",
								requestfile,
								"--out",
								out / "lerobot",
							]
						self.run_process(args, out, ident)
						if fmt == "rerun":
							produced = json.loads((out / "recording/episode.json").read_text(encoding="utf-8"))
							if produced.get("parent_episode_id") != e["id"]:
								raise ValueError("rerun ancestry differs from the frozen parent")
							manifest.setdefault("new_episodes", []).extend(
								self.catalog.import_file(out / "recording/episode.json")
							)
				for p in sorted(folder.rglob("*")):
					if self.ledger.state(ident) == JobState.CANCEL_REQUESTED:
						raise InterruptedError("cancelled during hashing")
					if p.is_file():
						manifest["files"].append(
							{
								"path": p.relative_to(folder).as_posix(),
								"sha256": sha256_file(p),
								"size": p.stat().st_size,
							}
						)
				for e in episodes:
					for a in e["artifacts"]:
						if sha256_file(self.catalog.project_path(a["path"])) != a["sha256"]:
							raise ValueError("source changed before publication")
				(folder / "manifest.json").write_text(dumps(manifest), encoding="utf-8")
				(folder / "README.txt").write_text(
					"Armigo experiment export. See manifest.json for original evidence, missing provenance, selection, rules and file hashes. Historical replay checks establish summary agreement only.\n",
					encoding="utf-8",
				)
				with zipfile.ZipFile(folder / "bundle.zip", "w", compression=zipfile.ZIP_STORED) as bundle:
					for p in sorted(folder.rglob("*")):
						if self.ledger.state(ident) == JobState.CANCEL_REQUESTED:
							raise InterruptedError("cancelled during packaging")
						if p.is_file() and p.name != "bundle.zip":
							bundle.write(p, p.relative_to(folder).as_posix())
				self.ledger.assert_current(lease)
				if self.ledger.state(ident) == JobState.CANCEL_REQUESTED:
					raise InterruptedError("cancelled before publication")
				self.ledger.append_event(
					ident,
					JobState.SUCCEEDED,
					lease=lease,
					payload={
						"bundle": str(folder / "bundle.zip"),
						"sha256": sha256_file(folder / "bundle.zip"),
						"progress": "complete",
						"new_episodes": manifest.get("new_episodes", []),
					},
				)
			except Exception as exc:  # noqa: BLE001 -- task boundary must record failures.
				self.ledger.append_event(
					ident,
					JobState.CANCELLED if isinstance(exc, InterruptedError) else JobState.FAILED,
					lease=lease,
					payload={"error": str(exc)},
				)
		if self.ledger.state(ident) == JobState.SUCCEEDED and job.params["format"] in ("isaac-rtx", "mujoco-video"):
			for e in job.params["episodes"]:
				self.catalog.add_artifact(
					e["id"],
					folder / e["id"][:16] / "render/animation.mp4",
					"rtx_video" if job.params["format"] == "isaac-rtx" else "video",
				)
		return self.get(ident)

	def start(self):
		if self.thread and self.thread.is_alive():
			return

		def loop():
			while not self.stop.is_set():
				for job in self.ledger.incomplete_jobs():
					try:
						if self.ledger.state(job.job_id) in (JobState.RUNNING, JobState.CANCEL_REQUESTED):
							with worker_lock(self.catalog.home):
								if self.ledger.state(job.job_id) not in (JobState.RUNNING, JobState.CANCEL_REQUESTED):
									continue
								self.ledger.append_event(
									job.job_id,
									JobState.NEEDS_RECOVERY,
									payload={"reason": "worker exited before publication"},
								)
								self.ledger.append_event(
									job.job_id,
									JobState.FAILED,
									payload={"error": "interrupted export; retry creates a separate attempt"},
								)
							continue
						self.execute(job.job_id)
					except OSError:
						pass  # another process owns the single worker
					except LedgerError:
						# Reconcile a superseded lease on the next pass after releasing the OS lock.
						continue
				self.stop.wait(0.5)

		self.thread = threading.Thread(target=loop, daemon=True)
		self.thread.start()
