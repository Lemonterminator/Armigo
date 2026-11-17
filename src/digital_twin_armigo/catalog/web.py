"""Loopback-only browser interface. All writes require a per-process token."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from .jobs import Jobs
from .store import Catalog


def create_app(catalog: Catalog):
	from fastapi import FastAPI, HTTPException, Request
	from fastapi.responses import FileResponse, JSONResponse

	jobs = Jobs(catalog)
	token = secrets.token_urlsafe(32)

	@asynccontextmanager
	async def lifespan(app):
		jobs.start()
		yield
		jobs.stop.set()

	app = FastAPI(title="Armigo local catalog", lifespan=lifespan)

	@app.middleware("http")
	async def local_only(request: Request, call_next):
		host = request.headers.get("host", "").split(":")[0]
		if host not in ("127.0.0.1", "localhost", "testserver"):
			return JSONResponse({"detail": "local requests only"}, status_code=403)
		if request.method not in ("GET", "HEAD") and not secrets.compare_digest(
			request.headers.get("X-Catalog-Token", ""), token
		):
			return JSONResponse({"detail": "catalog token required"}, status_code=403)
		try:
			return await call_next(request)
		except (ValueError, KeyError, FileNotFoundError) as exc:
			return JSONResponse({"detail": str(exc)}, status_code=400)

	@app.get("/")
	def index():
		return FileResponse(Path(__file__).with_name("index.html"))

	@app.get("/api/token")
	def get_token():
		return {"token": token}

	@app.get("/api/summary")
	def summary():
		return catalog.summary()

	@app.get("/api/episodes")
	def episodes(
		search: str = "", verdict: str = "", engine: str = "", quality: str = "", limit: int = 50, offset: int = 0
	):
		return catalog.list(search, verdict, engine, quality, limit, offset)

	@app.get("/api/episodes/{ident}")
	def detail(ident: str):
		return catalog.episode(ident)

	@app.get("/api/episodes/{ident}/telemetry")
	def telemetry(ident: str):
		import numpy as np

		e = catalog.episode(ident)
		a = next((a for a in e["artifacts"] if a["role"] in ("telemetry", "legacy_telemetry")), None)
		if a is None:
			return {"series": {}}
		series = {}
		with np.load(catalog.project_path(a["path"]), allow_pickle=False) as z:
			for key in ("brick_vel", "jaw_forces", "pinch_normal"):
				if key not in z:
					continue
				v = z[key]
				if key == "brick_vel":
					series["积木速度 (m/s)"] = np.linalg.norm(v, axis=1).tolist()
				else:
					series["夹持力 (N)"] = v.max(axis=1).tolist()
		return {"series": series}

	@app.get("/api/artifacts/{ident}")
	def artifact(ident: str):
		a = catalog.artifact(ident)
		return FileResponse(catalog.project_path(a["path"]))

	@app.post("/api/import")
	def import_(body: dict):
		return catalog.import_path(body["path"])

	@app.post("/api/exports")
	def export(body: dict):
		return jobs.submit(
			body["episode_ids"],
			body.get("format", "archive"),
			body.get("profile", "preview"),
			body.get("request_id"),
			body.get("allow_incomplete", False),
			"browser",
		)

	@app.post("/api/splits")
	def split(body: dict):
		return catalog.freeze_split(body["episode_ids"], body["name"], int(body.get("seed", 0)))

	@app.post("/api/quality")
	def evaluate(body: dict):
		return catalog.evaluate(body["episode_ids"], body.get("rules", {}))

	@app.post("/api/validate")
	def validate(body: dict):
		return catalog.validate_motion(body["episode_ids"])

	@app.get("/api/jobs")
	def list_jobs():
		return jobs.list()

	@app.get("/api/jobs/{ident}")
	def job(ident: str):
		return jobs.get(ident)

	@app.post("/api/jobs/{ident}/cancel")
	def cancel(ident: str):
		jobs.ledger.request_cancel(ident, reason="explicit browser cancellation")
		return jobs.get(ident)

	@app.post("/api/jobs/{ident}/retry")
	def retry(ident: str):
		p = jobs.ledger.job(ident).params
		j, _ = jobs.ledger.register(
			idempotency_key=secrets.token_hex(16),
			op_type="catalog-export",
			params=p,
			run_id="catalog",
			branch_id="exports",
		)
		return jobs.get(j.job_id)

	@app.get("/api/jobs/{ident}/download")
	def download(ident: str):
		j = jobs.get(ident)
		if j["state"] != "SUCCEEDED":
			raise HTTPException(409, "export is not published")
		return FileResponse(j["result"]["bundle"], filename=f"armigo-{ident[:12]}.zip")

	app.state.jobs = jobs
	return app
