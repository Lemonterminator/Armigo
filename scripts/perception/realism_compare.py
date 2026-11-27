#!/usr/bin/env python
"""Side-by-side real vs MuJoCo visual comparison (no real-robot metrics or thresholds, read-only on hackjunction).

Plan: docs/superpowers/plans/2026-11-24-failure-observability-harness-learning.md Task U5.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.agent.brick_pose_nn import BrickSegNet, peaks
from digital_twin_armigo.constants import SO101_MJCF_DIR, TOP_CROP
from digital_twin_armigo.joints import normalized_to_rad, rad_to_normalized
from digital_twin_armigo.mujoco_cell.cell import Cell
from digital_twin_armigo.mujoco_cell.layout import sample_layout

DEFAULT_REAL_REF = Path("C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction/artifacts/isaac_test/real_ref")
DEFAULT_STUDIO = Path(
	"C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction/data/studio/af9b173a-ee59-4e24-97c8-be71dba701ce"
)
DEFAULT_OUT = Path("C:/scratch/brickseg-real/compare")
CLASS_COLOURS_RGB = {
	1: (0, 255, 0),
	2: (0, 0, 255),
	3: (255, 165, 0),
}


def real_state_to_mjcf_rad(state: np.ndarray | list[float]) -> np.ndarray:
	"""Convert real Studio/LeRobot 6-DOF joint state to MuJoCo joint radians."""
	return normalized_to_rad(np.asarray(state, dtype=np.float64))


def mjcf_rad_to_real_state(q_rad: np.ndarray | list[float]) -> np.ndarray:
	"""Inverse of :func:`real_state_to_mjcf_rad`."""
	return rad_to_normalized(np.asarray(q_rad, dtype=np.float64))


def sample_frame_indices(total_frames: int, count: int, seed: int = 0) -> list[int]:
	"""Deterministically choose ``count`` unique sorted frame indices evenly spread with seeded jitter."""
	if total_frames <= 0 or count <= 0:
		return []
	if count >= total_frames:
		return list(range(total_frames))
	rng = np.random.default_rng(int(seed))
	edges = np.linspace(0, total_frames, count + 1, dtype=int)
	picked = []
	for i in range(count):
		lo, hi = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
		picked.append(int(rng.integers(lo, hi)))
	return sorted(set(picked))


def render_overlay(rgb: np.ndarray, net: BrickSegNet | None) -> np.ndarray:
	"""Render predicted segmentation mask + corner peaks on top of ``rgb`` (copy if net is None)."""
	out = rgb.copy()
	if net is None:
		return out
	pred = net.predict(rgb)
	cls_full = cv2.resize(pred.cls.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
	tint = out.copy()
	for cval, col in CLASS_COLOURS_RGB.items():
		tint[cls_full == cval] = col
	out = cv2.addWeighted(out, 0.65, tint, 0.35, 0.0)
	for ch, col in ((0, (255, 255, 0)), (1, (255, 0, 255))):
		for u, v, _score in peaks(pred.heat[ch], thr=0.3, radius=2, scale=pred.scale):
			cv2.circle(out, (round(float(u)), round(float(v))), 4, col, -1)
	return out


def make_triptych(
	real_rgb: np.ndarray,
	sim_rgb: np.ndarray,
	overlay_rgb: np.ndarray | None = None,
	*,
	unpaired: bool = False,
) -> np.ndarray:
	"""Build fixed-height ``[Real | Sim | Overlay]`` side-by-side comparison strip (RGB uint8)."""
	h, w = real_rgb.shape[:2]

	def _fit(img: np.ndarray) -> np.ndarray:
		return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img.copy()

	panels = [(_fit(real_rgb), "REAL" + (" [UNPAIRED]" if unpaired else "")), (_fit(sim_rgb), "SIM")]
	ov = _fit(real_rgb if overlay_rgb is None else overlay_rgb)
	panels.append((ov, "OVERLAY"))
	labelled = []
	for panel, title in panels:
		bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
		cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		labelled.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
	return np.concatenate(labelled, axis=1)


def _sim_cell() -> Cell:
	raw = json.loads((ROOT / "config/sim/top_camera_joint.json").read_text(encoding="utf-8"))
	from digital_twin_armigo.camera import CameraPose

	layout = sample_layout(np.random.default_rng(0), CameraPose(**raw["camera"]), defective=False)
	return Cell(layout, SO101_MJCF_DIR, dynamic_clutter=True, assisted_grasp=False)


def _sim_render_for_camera(cell: Cell, cam: str, q_rad: np.ndarray) -> np.ndarray:
	cell.set_joints(q_rad)
	if cam == "top_rgb_v2":
		x, y, w, h = TOP_CROP
		return cell.rgb("top")[y : y + h, x : x + w].copy()
	return cell.rgb("wrist_cam" if cam == "wrist" else cam).copy()


def _write_notes_and_html(out_dir: Path, entries: list[dict]) -> None:
	notes = [
		"# Real-to-Sim Visual Observation Notes (No Metrics / No Thresholds)",
		"",
		"Fill by visual inspection only (do NOT tune parameters against real frames):",
		"",
		"- Colour tone & brightness:",
		"- Specular reflections & table glare:",
		"- Depth of field & motion blur:",
		"- Robot link & gripper geometry alignment:",
		"- Obvious network false positives / misses on real wrist frames:",
		"",
	]
	(out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8", newline="\n")

	cards = []
	for e in entries:
		cards.append(
			f"<div style='margin:16px 0'><h3>{html.escape(e['label'])}</h3>"
			f"<img src='{html.escape(e['file'])}' style='max-width:100%'/></div>"
		)
	doc = (
		"<!doctype html><html><meta charset='utf-8'><body>"
		"<h1>BrickSeg Real vs Sim Visual Comparison</h1>" + "".join(cards) + "</body></html>\n"
	)
	(out_dir / "index.html").write_text(doc, encoding="utf-8", newline="\n")


def run_compare(
	*,
	out_dir: Path = DEFAULT_OUT,
	real_ref: Path = DEFAULT_REAL_REF,
	studio_dir: Path | None = None,
	onnx_path: Path | None = None,
	seed: int = 0,
	n_wrist: int = 16,
	n_top: int = 8,
) -> list[Path]:
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	net = BrickSegNet(onnx_path) if onnx_path is not None and Path(onnx_path).is_file() else None
	cell = _sim_cell()
	written: list[Path] = []
	entries: list[dict] = []
	try:
		meta = json.loads((Path(real_ref) / "frame_meta.json").read_text(encoding="utf-8"))
		q_rad = real_state_to_mjcf_rad(meta["observation.state"])
		idx = int(meta.get("index", 0))
		for cam, fname, use_net in (
			("wrist", f"wrist_{idx}.png", True),
			("top_rgb_v2", f"top_rgb_v2_{idx}.png", False),
		):
			p = Path(real_ref) / fname
			if not p.is_file():
				continue
			real_rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
			sim_rgb = _sim_render_for_camera(cell, cam, q_rad)
			ov = render_overlay(real_rgb, net if use_net else None)
			trip = make_triptych(real_rgb, sim_rgb, ov)
			out_name = f"real_ref_{cam}_{idx}.png"
			out_path = out_dir / out_name
			cv2.imwrite(str(out_path), cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
			written.append(out_path)
			entries.append({"label": f"real_ref {cam} #{idx}", "file": out_name})

		if studio_dir is not None and Path(studio_dir).is_dir():
			import av
			import pyarrow.parquet as pq

			pq_table = pq.read_table(min((Path(studio_dir) / "data").rglob("*.parquet")))
			ep_table = pq.read_table(min((Path(studio_dir) / "meta" / "episodes").rglob("*.parquet")))
			ep_map = {int(ep_table["episode_index"][r].as_py()): r for r in range(len(ep_table))}
			total = len(pq_table)
			for cam, count, use_net in (("wrist", n_wrist, True), ("top_rgb_v2", n_top, False)):
				vkey = f"observation.images.{cam}"
				for k_i, g_idx in enumerate(sample_frame_indices(total, count, seed=seed + (0 if use_net else 100))):
					ep_idx = int(pq_table["episode_index"][g_idx].as_py())
					fr_idx = int(pq_table["frame_index"][g_idx].as_py())
					state = pq_table["observation.state"][g_idx].as_py()
					erow = ep_map[ep_idx]
					c_idx = int(ep_table[f"videos/{vkey}/chunk_index"][erow].as_py())
					f_idx = int(ep_table[f"videos/{vkey}/file_index"][erow].as_py())
					t0 = float(ep_table[f"videos/{vkey}/from_timestamp"][erow].as_py())
					vid = Path(studio_dir) / "videos" / vkey / f"chunk-{c_idx:03d}" / f"file-{f_idx:03d}.mp4"
					if not vid.is_file():
						continue
					target_pts_s = t0 + fr_idx / 30.0
					frame_rgb = None
					with av.open(str(vid)) as container:
						stream = container.streams.video[0]
						container.seek(max(0, int((target_pts_s - 0.5) / float(stream.time_base))), stream=stream)
						for fr in container.decode(video=0):
							if fr.time is not None and fr.time >= target_pts_s - 1e-4:
								frame_rgb = fr.to_ndarray(format="rgb24")
								break
					if frame_rgb is None:
						continue
					sim_rgb = _sim_render_for_camera(cell, cam, real_state_to_mjcf_rad(state))
					ov = render_overlay(frame_rgb, net if use_net else None)
					trip = make_triptych(frame_rgb, sim_rgb, ov)
					out_name = f"studio_{cam}_{k_i:02d}_idx{g_idx}.png"
					out_path = out_dir / out_name
					cv2.imwrite(str(out_path), cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
					written.append(out_path)
					entries.append({"label": f"studio {cam} ep={ep_idx} idx={g_idx}", "file": out_name})
	finally:
		cell.close()

	_write_notes_and_html(out_dir, entries)
	return written


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
	p.add_argument("--real-ref", type=Path, default=DEFAULT_REAL_REF)
	p.add_argument("--studio-dir", type=Path, default=None, help="Optional LeRobot v3 studio dir for 24 video frames")
	p.add_argument("--onnx", type=Path, default=None)
	p.add_argument("--seed", type=int, default=0)
	args = p.parse_args(argv)
	files = run_compare(
		out_dir=args.out_dir,
		real_ref=args.real_ref,
		studio_dir=args.studio_dir,
		onnx_path=args.onnx,
		seed=args.seed,
	)
	print(f"wrote {len(files)} panels + index.html + notes.md to {args.out_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
