"""Isolated batch LeRobot adapter with typed labels and stable source ids."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from importlib.metadata import version as distribution_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Direct execution must not shadow the installed `lerobot` package with this filename.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT / "src"))


def convert(request: dict, out: Path):
	import numpy as np
	import pyarrow as pa
	import pyarrow.parquet as pq

	from digital_twin_armigo.catalog.quality import label_definitions, normalize_labels
	from digital_twin_armigo.catalog.store import dumps
	from digital_twin_armigo.lerobot_export import export
	from digital_twin_armigo.runtime.canonical import sha256_file

	results = request.get("episodes") or [request["episode"]]
	base = results[0]["provenance"].get("recording_index")
	if not base:
		raise ValueError("recording camera feature contract missing")
	rows = {
		k: []
		for k in (
			"episode_index",
			"frame_index",
			"index",
			"phase",
			"defect_visible_pixels",
			"brick_visible_pixels",
			"defect_visibility_proxy_score",
			"brick_color_mask_pixels",
		)
	}
	mapping = []
	entries = []
	cursor = 0
	with tempfile.TemporaryDirectory(prefix="armigo-lerobot-") as temporary:
		dir_ = Path(temporary)
		for i, e in enumerate(results):
			index = e["provenance"].get("recording_index")
			if not index or any(index.get(k) != base.get(k) for k in ("fps", "state_names", "cameras")):
				raise ValueError("selected recordings have incompatible camera/state/fps contracts")
			obs = next(r for r in e["artifacts"] if r["role"] == "observations")
			with np.load(obs["path"], allow_pickle=False) as z:
				length = len(z["state"])
				if len(z["action"]) != length:
					raise ValueError("state/action length mismatch")
				for camera, shape in base["cameras"].items():
					if camera not in z:
						raise ValueError(f"missing camera {camera}")
					with zipfile.ZipFile(obs["path"]) as archive, archive.open(camera + ".npy") as stream:
						header_version = np.lib.format.read_magic(stream)
						reader = (
							np.lib.format.read_array_header_1_0
							if header_version == (1, 0)
							else np.lib.format.read_array_header_2_0
						)
						actual, _, _ = reader(stream)
						if actual != (length, *shape):
							raise ValueError(f"camera shape/frame mismatch: {camera}")
				labels = normalize_labels(e["engine"], {k: z[k] for k in ("phase", "defect_px", "brick_px") if k in z})
				for key, column in rows.items():
					if key == "episode_index":
						values = [i] * length
					elif key == "frame_index":
						values = list(range(length))
					elif key == "index":
						values = list(range(cursor, cursor + length))
					else:
						values = labels[key].tolist() if key in labels else [None] * length
					if len(values) != length:
						raise ValueError("label/frame length mismatch")
					column.extend(values)
			filename = f"episode_{i:04d}.npz"
			shutil.copyfile(obs["path"], dir_ / filename)
			entry = next((r for r in index["episodes"] if Path(r["file"]).name == Path(obs["path"]).name), None)
			if entry is None:
				raise ValueError("observation has no matching index entry")
			entries.append({**entry, "file": filename, "length": length})
			mapping.append(
				{
					"episode_index": i,
					"first_frame_index": cursor,
					"length": length,
					"catalog_episode_id": e["id"],
					"provenance": e["provenance"],
					"quality": e["quality"],
					"quality_policy": e.get("quality_policy"),
					"role": e["role"],
					"label_definitions": label_definitions(e["engine"]),
				}
			)
			cursor += length
		(dir_ / "index.json").write_text(dumps({**base, "episodes": entries}), encoding="utf-8")
		export(dir_, out, "local/armigo-catalog", include_failures=True, video=True, robot_type="SO101_Follower")
	pq.write_table(
		pa.table({k: pa.array(v, type=pa.int64()) for k, v in rows.items()}), out / "meta/privileged.parquet"
	)
	(out / "meta/source.json").write_text(
		dumps(
			{
				"schema": "catalog-dataset-source/v1",
				"exporter_code_sha256": sha256_file(Path(__file__)),
				"lerobot_version": distribution_version("lerobot"),
				"episode_map": mapping,
				"research_exception": request.get("allow_incomplete", False),
				"memberships": request.get("memberships", {}),
			}
		),
		encoding="utf-8",
	)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--request", type=Path, required=True)
	p.add_argument("--out", type=Path, required=True)
	a = p.parse_args()
	convert(json.loads(a.request.read_text(encoding="utf-8")), a.out)


if __name__ == "__main__":
	main()
