"""``armigo-track``: dataset manifests and the local MLflow UI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .lineage import build_manifest, verify_manifest
from .runs import store_dir, tracking_uri


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description="Dataset manifests and local MLflow store.")
	s = p.add_subparsers(dest="command", required=True)
	i = s.add_parser("dataset-manifest", help="hash a dataset directory and write dataset_manifest.json there")
	i.add_argument("path", type=Path)
	i.add_argument("--name")
	i.add_argument("--kind", help="e.g. anomaly-render, lerobot, recordings")
	i.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="generation parameter to record")
	i = s.add_parser("verify", help="re-hash a dataset directory against its manifest")
	i.add_argument("path", type=Path)
	i = s.add_parser("ui", help="serve the local MLflow store (blocks)")
	i.add_argument("--port", type=int, default=5000)
	a = p.parse_args(argv)
	if a.command == "dataset-manifest":
		params = dict(kv.split("=", 1) for kv in a.param) if a.param else None
		m = build_manifest(a.path, name=a.name, kind=a.kind, params=params)
		print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=1, ensure_ascii=False))
		return 0
	if a.command == "verify":
		result = verify_manifest(a.path)
		print(json.dumps(result, indent=1))
		return 0 if result["ok"] else 1
	cmd = [sys.executable, "-m", "mlflow", "server", "--backend-store-uri", tracking_uri(), "--port", str(a.port)]
	if "ARMIGO_MLFLOW_URI" not in os.environ:
		cmd += ["--default-artifact-root", (store_dir() / "artifacts").as_uri()]
	return subprocess.call(cmd)


if __name__ == "__main__":
	raise SystemExit(main())
