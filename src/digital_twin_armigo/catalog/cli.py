"""Command-line access to the same catalog used by the browser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .store import Catalog, dumps


def main(argv=None):
	p = argparse.ArgumentParser(description="Local experiment catalog; nothing exports automatically.")
	p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
	p.add_argument("--home", type=Path)
	s = p.add_subparsers(dest="command", required=True)
	i = s.add_parser("import")
	i.add_argument("path")
	i = s.add_parser("list")
	for flag in ("search", "verdict", "engine", "quality"):
		i.add_argument("--" + flag, default="")
	i.add_argument("--limit", type=int, default=50)
	i.add_argument("--offset", type=int, default=0)
	i = s.add_parser("show")
	i.add_argument("id")
	s.add_parser("summary")
	i = s.add_parser("backup")
	i.add_argument("--out", type=Path, required=True)
	i = s.add_parser("import-snapshot")
	i.add_argument("--store", type=Path, required=True)
	i.add_argument("--ledger", type=Path, required=True)
	i.add_argument("--snapshot", required=True)
	i = s.add_parser("serve")
	i.add_argument("--port", type=int, default=8765)
	for name in ("export", "rerun"):
		i = s.add_parser(name)
		i.add_argument("--ids", nargs="+", required=True)
		i.add_argument(
			"--format", default="archive", choices=["archive", "existing-video", "mujoco-video", "isaac-rtx", "lerobot"]
		)
		i.add_argument("--profile", default="preview", choices=["preview", "quality"])
		i.add_argument("--allow-incomplete", action="store_true")
		i.add_argument("--request-id")
		i.add_argument("--enqueue", action="store_true")
	i = s.add_parser("split")
	i.add_argument("--ids", nargs="+", required=True)
	i.add_argument("--name", required=True)
	i.add_argument("--seed", type=int, default=0)
	i = s.add_parser("verify")
	i.add_argument("--ids", nargs="+", required=True)
	i = s.add_parser("validate")
	i.add_argument("--ids", nargs="+", required=True)
	i = s.add_parser("evaluate")
	i.add_argument("--ids", nargs="+", required=True)
	i.add_argument("--rules", default="{}", help="JSON threshold overrides")
	i = s.add_parser("job")
	i.add_argument("id")
	i.add_argument("--cancel", action="store_true")
	a = p.parse_args(argv)
	try:
		c = Catalog(a.root, a.home)
		if a.command == "import":
			result = c.import_path(a.path)
		elif a.command == "list":
			result = c.list(a.search, a.verdict, a.engine, a.quality, a.limit, a.offset)
		elif a.command == "show":
			result = c.episode(a.id)
		elif a.command == "summary":
			result = c.summary()
		elif a.command == "backup":
			result = c.backup(a.out)
		elif a.command == "import-snapshot":
			result = c.import_snapshot(a.store, a.ledger, a.snapshot)
		elif a.command == "split":
			result = c.freeze_split(a.ids, a.name, a.seed)
		elif a.command == "verify":
			result = c.verify(a.ids)
		elif a.command == "validate":
			result = c.validate_motion(a.ids)
		elif a.command == "evaluate":
			result = c.evaluate(a.ids, json.loads(a.rules))
		elif a.command == "serve":
			import uvicorn

			from .web import create_app

			uvicorn.run(create_app(c), host="127.0.0.1", port=a.port)
			return 0
		else:
			from .jobs import Jobs

			jobs = Jobs(c)
			if a.command == "job":
				if a.cancel:
					jobs.ledger.request_cancel(a.id, reason="explicit CLI cancellation")
				result = jobs.get(a.id)
			else:
				result = jobs.submit(
					a.ids,
					"rerun" if a.command == "rerun" else a.format,
					a.profile,
					a.request_id,
					a.allow_incomplete,
					"cli-or-agent",
				)
				if not a.enqueue:
					result = jobs.execute(result["id"])
		print(dumps(result))
		return 1 if result.get("state") == "FAILED" or result.get("ok") is False else 0
	except Exception as exc:  # noqa: BLE001 -- CLI boundary emits a structured failure.
		print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
