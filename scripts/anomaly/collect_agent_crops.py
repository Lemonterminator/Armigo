"""Collect the detector crops of six-face agent runs into a scoring dataset (``infer_sup.py --holdout``).

Every ``view`` event of ``--camera`` with a score becomes one row; the crop is the exact image the agent scored
(``keyframes/<branch>-<frame>-<camera>-crop.png``). Labels come from the fixture: a defective brick's view is
``abnormal`` (the agent suites put the defect on the face the camera reads), a normal brick's is ``normal``; every
row is split ``test``. Run with the main venv::

    python scripts/anomaly/collect_agent_crops.py --out results/anomaly/dataset-agentbot \
        data/catalog/agent/bottom48 data/catalog/agent/bottom2x4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
	from digital_twin_armigo.agent.twin_mujoco import sample_case

	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("runs", type=Path, nargs="+")
	ap.add_argument("--camera", default="top")
	ap.add_argument("--out", type=Path, required=True)
	a = ap.parse_args()
	(a.out / "png").mkdir(parents=True, exist_ok=True)
	rows = []
	for layout, task in enumerate(sorted(t for run in a.runs for t in run.rglob("task.json"))):
		ep = task.parent
		seed, defective, face = json.loads(task.read_text(encoding="utf-8"))["case"]
		size = sample_case(seed, defective, face).brick_size
		with (ep / "events.jsonl").open(encoding="utf-8") as f:
			views = [e["observation"] for e in map(json.loads, f) if e["kind"] == "view"]
		for o in views:
			if o["camera"] != a.camera or o["score"] is None:
				continue
			branch, frame, camera = o["id"].split(":")
			src = ep / "keyframes" / f"{branch}-{frame}-{camera}-crop.png"
			name = f"{layout:04d}_{branch}_{frame}_{camera}.png"
			shutil.copyfile(src, a.out / "png" / name)
			rows.append(
				{
					"path": f"png/{name}",
					"layout_index": layout,
					"episode": str(ep),
					"fixture_seed": seed,
					"split": "test",
					"camera": camera,
					"stage": "agent",
					"defective": defective,
					"defect_face": face,
					"brick_2x4": bool(size[0] > 0.05),
					"agent_score": o["score"],
					"faces_px": o["face_pixels"],
					"label": "abnormal" if defective else "normal",
				}
			)
	(a.out / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
	print(f"{len(rows)} crops from {len({r['episode'] for r in rows})} episodes -> {a.out}")


if __name__ == "__main__":
	main()
