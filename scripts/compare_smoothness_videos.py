"""Side-by-side before/after videos and jerk plots for two recorded runs of the same suite.

Usage: python scripts/compare_smoothness_videos.py BEFORE_RUN AFTER_RUN OUT_DIR

Both runs come from ``twin_mcp run`` without ``--headless``. Needs ffmpeg on PATH and matplotlib (not a project
dependency; any Python with numpy and matplotlib works with ``PYTHONPATH=src``). Writes per case a full-episode
video, a quarter-speed close-up of the return, a jerk plot, and ``cases.json``. The font path is Windows-only."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from digital_twin_armigo.constants import FPS

BEFORE, AFTER, OUT = (Path(p) for p in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
FONT = "C\\:/Windows/Fonts/arial.ttf"
CROP = "crop=608:456:16:44"  # the ROBOT / FRONT panel of replay.mp4


def jerk(state):
	q = state[:, :5]
	return np.abs(np.diff(q, 3, axis=0)).max(axis=1) * FPS**3


def label(text):
	return f"drawtext=fontfile='{FONT}':text='{text}':x=12:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"


def frame_label():
	return f"drawtext=fontfile='{FONT}':text='frame %{{n}}':x=12:y=h-40:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"


def ffmpeg(*args):
	subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


rows = []
for case_dir in sorted(BEFORE.glob("case-*")):
	name = case_dir.name
	ep = json.loads((case_dir / "episode.json").read_text(encoding="utf-8"))
	fixture = json.loads((case_dir / "task.json").read_text(encoding="utf-8"))["case"]
	ep_after = json.loads((AFTER / name / "episode.json").read_text(encoding="utf-8"))
	branches = sorted(p.name for p in case_dir.glob("branch-*"))
	last = branches[-1]
	b = np.load(case_dir / last / "actions.npz")
	a = np.load(AFTER / name / last / "actions.npz")
	ret = "return"
	start = int(np.flatnonzero(b["phase"] == ret)[0])
	jb, ja = jerk(b["state"]), jerk(a["state"])

	fig, ax = plt.subplots(figsize=(10, 3.2))
	ax.plot(np.arange(3, len(b["state"])), jb, lw=0.9, color="#c0392b", label="before")
	ax.plot(np.arange(3, len(a["state"])), ja, lw=0.9, color="#2471a3", label="after")
	ax.axvline(start, color="grey", ls="--", lw=0.8)
	ax.text(start + 5, ax.get_ylim()[1] * 0.9, "return", color="grey")
	ax.set_xlabel(f"frame ({FPS} fps)")
	ax.set_ylabel("max |jerk| over joints 0-4, rad/s³")
	ax.set_title(f"{name}  fixture {fixture}  ({last})")
	ax.legend(loc="upper left")
	fig.tight_layout()
	fig.savefig(OUT / f"{name}_jerk.png", dpi=110)
	plt.close(fig)

	vb, va = case_dir / last / "replay.mp4", AFTER / name / last / "replay.mp4"
	# Full episode, before left, after right.
	ffmpeg(
		"-i",
		str(vb),
		"-i",
		str(va),
		"-filter_complex",
		f"[0:v]{CROP},{label('before')},{frame_label()}[l];[1:v]{CROP},{label('after')},{frame_label()}[r];"
		"[l][r]hstack=inputs=2:shortest=0[v]",
		"-map",
		"[v]",
		"-c:v",
		"libx264",
		"-crf",
		"23",
		"-preset",
		"veryfast",
		"-pix_fmt",
		"yuv420p",
		str(OUT / f"{name}_full.mp4"),
	)
	# Return close-up at quarter speed, from 1 s before the return to the end.
	t0 = max(0.0, (start - FPS) / FPS)
	ffmpeg(
		"-ss",
		f"{t0:.3f}",
		"-i",
		str(vb),
		"-ss",
		f"{t0:.3f}",
		"-i",
		str(va),
		"-filter_complex",
		f"[0:v]{CROP},{label('before')},setpts=4*PTS[l];[1:v]{CROP},{label('after')},setpts=4*PTS[r];"
		"[l][r]hstack=inputs=2:shortest=0[v]",
		"-map",
		"[v]",
		"-r",
		"30",
		"-c:v",
		"libx264",
		"-crf",
		"20",
		"-preset",
		"veryfast",
		"-pix_fmt",
		"yuv420p",
		str(OUT / f"{name}_return_quarter_speed.mp4"),
	)
	rows.append(
		{
			"case": name,
			"fixture": fixture,
			"branch": last,
			"frames_before": len(b["state"]),
			"frames_after": len(a["state"]),
			"return_start": start,
			"return_peak_before": round(float(jb[start - 3 :].max()), 1),
			"return_peak_after": round(float(ja[start - 3 :].max()), 1),
			"peak_before": round(float(jb.max()), 1),
			"peak_after": round(float(ja.max()), 1),
			"result_before": {k: ep.get("result", ep).get(k) for k in ("verdict", "destination", "label", "rewinds")},
			"result_after": {
				k: ep_after.get("result", ep_after).get(k) for k in ("verdict", "destination", "label", "rewinds")
			},
			"smoothness_before": ep["result"].get("smoothness"),
			"smoothness_after": ep_after["result"].get("smoothness"),
		}
	)
	print(name, "done", flush=True)

(OUT / "cases.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
print(json.dumps([{k: r[k] for k in list(r)[:10]} for r in rows], indent=1, default=str))
