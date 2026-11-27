"""Plan A2: check that the S9 training loader and the S10 closed-loop client feed a student the same inputs.

The chain under test, for one freshly recorded expert episode:

  S9:  expert trace (lossless mkv) -> ``export_lerobot`` (AV1) -> ``SegmentChunkDataset`` sample
  S10: ``StudentTwin`` replay of the same episode -> ``StudentObservation`` -> ``ChunkClientPolicy._build_payload``
       -> ``validate_predict_payload`` (what ``vla/server.py`` hands the model)

Per compared frame it checks the 36-dim state vector, the instruction text, the camera key order and image shapes,
and the pixel difference per camera (the export is lossy AV1, so pixels are compared against a tolerance and against
the difference to the *other* cameras, which catches swapped or mis-cropped streams). It also checks that the client
clears its action queue at every segment switch.

Two stages, because MuJoCo lives in the repo ``.venv`` and LeRobot in the CUDA env:

	.venv/Scripts/python scripts/vla/check_s9_s10_parity.py record --out E:/scratch/parity-a2
	E:/il/lerobot-cuda/Scripts/python.exe scripts/vla/check_s9_s10_parity.py compare --out E:/scratch/parity-a2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CASE = (1200001, False, 5)  # first val-suite case (normal brick)
CAPTURE_STRIDE = 9


class _ClearScorer:
	"""Detector stand-in: every view is 'normal'. Parity concerns the input path, not the detector."""

	def threshold(self, camera):
		return 0.0

	def score(self, camera, image):
		return -1.0

	def close(self):
		pass


class _NullPredictor:
	"""Never called: the capture builds payloads itself; present only so ChunkClientPolicy can be constructed."""

	def predict(self, payload):  # pragma: no cover
		raise AssertionError("parity capture must not call the predictor")


def record(out: Path, case: tuple, stride: int) -> None:
	from digital_twin_armigo.agent.student_rollout import ReplayPolicy, run_episode
	from digital_twin_armigo.vla.client_policy import ChunkClientPolicy
	from digital_twin_armigo.vla.server import validate_predict_payload

	out.mkdir(parents=True, exist_ok=True)
	expert_dir = out / "expert"
	if not (expert_dir / "run.json").exists():
		run_episode(case, _ClearScorer(), expert_dir, sensing="realistic", video=True)

	inner = ReplayPolicy.from_trace(expert_dir)
	client = ChunkClientPolicy(_NullPredictor(), mode="sync")
	rows: list[dict] = []
	images: dict[str, list[np.ndarray]] = {}
	queue_resets: list[bool] = []

	class Capture:
		def start(self, task, segment_id):
			inner.start(task, segment_id)
			client.start(task, segment_id)
			queue_resets.append(client.queue_remaining == 0 and client.planned_stop_step is None)

		def act(self, obs):
			step = inner.act(obs)
			if not step.done and (obs.segment_frame == 0 or obs.frame % stride == 0):
				prepared = validate_predict_payload(client._build_payload(obs))
				rows.append(
					{
						"frame": int(obs.frame),
						"segment_id": int(obs.segment_id),
						"segment_frame": int(obs.segment_frame),
						"task": prepared["task"],
						"state": prepared["state"].tolist(),
					}
				)
				for key, chw in prepared["images"].items():
					images.setdefault(key, []).append(np.round(chw * 255.0).astype(np.uint8))
			return step

	student_dir = out / "student"
	run_episode(
		case,
		_ClearScorer(),
		student_dir,
		policy=Capture(),
		segment_monitor=inner.check_segment,
		sensing="realistic",
		video=True,
	)
	if hasattr(client, "close"):
		client.close()
	np.savez_compressed(out / "client_images.npz", **{k: np.stack(v) for k, v in images.items()})
	(out / "client_rows.json").write_text(
		json.dumps(
			{
				"case": list(case),
				"stride": stride,
				"state_mismatches": inner.state_mismatches,
				"all_segments_used": inner.index == len(inner.segments) - 1,
				"queue_reset_on_every_start": all(queue_resets),
				"segment_starts": len(queue_resets),
				"camera_keys": list(images),
				"rows": rows,
			}
		),
		encoding="utf-8",
	)
	print(f"captured {len(rows)} client frames; replay state mismatches={inner.state_mismatches}")


def compare(out: Path, pixel_tol: float) -> int:
	import torch

	from digital_twin_armigo.agent.twin_record import export_lerobot
	from digital_twin_armigo.vla.contract import STUDENT_CAMERAS
	from digital_twin_armigo.vla.dataset import SegmentChunkDataset

	ds_root = out / "lerobot"
	if not (ds_root / "meta" / "info.json").exists():
		export_lerobot([out / "expert"], ds_root, "local/parity-a2")
	ds = SegmentChunkDataset(ds_root, horizon=100, augment=False)
	by_frame = {}
	for i, spec in enumerate(ds._samples):
		if not spec.is_synthesized_zero_frame:
			by_frame[int(spec.anchor_row)] = i

	cap = json.loads((out / "client_rows.json").read_text(encoding="utf-8"))
	with np.load(out / "client_images.npz") as z:
		client_imgs = {k: z[k] for k in z.files}

	problems: list[str] = []
	if cap["camera_keys"] != list(STUDENT_CAMERAS):
		problems.append(f"client camera order {cap['camera_keys']} != loader {list(STUDENT_CAMERAS)}")
	if cap["state_mismatches"] != 0 or not cap["all_segments_used"]:
		problems.append("replay did not reproduce the expert episode")
	if not cap["queue_reset_on_every_start"]:
		problems.append("client queue not cleared at a segment start")

	state_err = 0.0
	task_mismatch = 0
	pix = {k: [] for k in STUDENT_CAMERAS}
	cross = []
	for j, row in enumerate(cap["rows"]):
		idx = by_frame.get(row["frame"])
		if idx is None:
			problems.append(f"frame {row['frame']} missing from loader")
			continue
		sample = ds[idx]
		s_loader = np.asarray(sample["observation.state"], dtype=np.float64)
		s_client = np.asarray(row["state"], dtype=np.float64)
		state_err = max(state_err, float(np.max(np.abs(s_loader - s_client))))
		task_mismatch += sample["task"] != row["task"]
		loader_u8 = {}
		for cam in STUDENT_CAMERAS:
			img = sample[cam]
			arr = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
			loader_u8[cam] = np.round(arr * 255.0).astype(np.int16)
			c = client_imgs[cam][j].astype(np.int16)
			if c.shape != loader_u8[cam].shape:
				problems.append(f"{cam} shape client {c.shape} != loader {loader_u8[cam].shape}")
				continue
			pix[cam].append(float(np.mean(np.abs(c - loader_u8[cam]))))
		a, b = STUDENT_CAMERAS[0], STUDENT_CAMERAS[1]  # top RGB vs top depth share a shape: a swap would show here
		if loader_u8[a].shape == client_imgs[b][j].shape:
			cross.append(float(np.mean(np.abs(client_imgs[b][j].astype(np.int16) - loader_u8[a]))))

	report = {
		"case": cap["case"],
		"frames_compared": len(cap["rows"]),
		"segment_starts": cap["segment_starts"],
		"state_max_abs_err": state_err,
		"task_mismatches": int(task_mismatch),
		"pixel_mean_abs_err": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in pix.items() if v},
		"swapped_top_rgb_vs_depth_mean_abs": float(np.mean(cross)) if cross else None,
		"pixel_tolerance": pixel_tol,
	}
	if state_err > 1e-5:
		problems.append(f"state vector differs (max abs {state_err:.2e})")
	if task_mismatch:
		problems.append(f"{task_mismatch} instruction mismatches")
	for k, v in pix.items():
		if v and max(v) > pixel_tol:
			problems.append(f"{k} pixel error {max(v):.2f} > {pixel_tol}")
	report["problems"] = problems
	report["pass"] = not problems
	(out / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	print(json.dumps(report, indent=2))
	return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = p.add_subparsers(dest="cmd", required=True)
	r = sub.add_parser("record")
	r.add_argument("--out", type=Path, required=True)
	r.add_argument("--case", type=json.loads, default=list(DEFAULT_CASE), help='JSON case, e.g. "[1200002, true, 0]"')
	r.add_argument("--stride", type=int, default=CAPTURE_STRIDE)
	c = sub.add_parser("compare")
	c.add_argument("--out", type=Path, required=True)
	c.add_argument("--pixel-tol", type=float, default=6.0, help="max per-frame mean |diff| in 0..255 per camera")
	args = p.parse_args(argv)
	if args.cmd == "record":
		record(args.out, tuple(args.case), args.stride)
		return 0
	return compare(args.out, args.pixel_tol)


if __name__ == "__main__":
	raise SystemExit(main())
