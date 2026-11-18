"""Candidate bowl collision builders and monkeypatch install/uninstall helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco

from digital_twin_armigo.constants import BOWL_H, BOWL_R, SIM_ASSETS_DIR
from digital_twin_armigo.mujoco_cell import physics_expert, scene

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "bowl-experiments"
HULLS_DIR = ROOT / "assets" / "meshes" / "bowl_hulls"
PROFILE_POLYS_PATH = RESULTS_DIR / "profile_polys.json"

N_STRIPS = 24
STRIP_W = 2.0 * BOWL_R / N_STRIPS
BOWL_XY = (0.26, -0.18)

ALIASES = {
	"R0": "R0_current",
	"R0_current": "R0_current",
	"R1": "R1_coacd_full",
	"R1_coacd_full": "R1_coacd_full",
	"R2": "R2_wedge_full",
	"R2_wedge_full": "R2_wedge_full",
	"R3": "R3_strips_floor_wedge_walls",
	"R3_strips_floor_wedge_walls": "R3_strips_floor_wedge_walls",
	"R4": "R4_strips_floor_box_walls",
	"R4_strips_floor_box_walls": "R4_strips_floor_box_walls",
	"R5": "R5_sdf",
	"R5_sdf": "R5_sdf",
	"R6": "R6_box_floor_wedge_walls",
	"R6_box_floor_wedge_walls": "R6_box_floor_wedge_walls",
	"R9": "R9_slab_box_200",
	"R9_slab_box_200": "R9_slab_box_200",
}


def resolve_id(cid: str) -> str:
	if cid not in ALIASES:
		raise KeyError(f"Unknown candidate id: {cid!r}")
	return ALIASES[cid]


def strip_params(n: int = N_STRIPS) -> list[tuple[float, float]]:
	"""Return ``[(x_center, half_length), ...]`` for ``n`` parallel box strips covering the bowl disc."""
	w = 2.0 * BOWL_R / n
	out: list[tuple[float, float]] = []
	for j in range(n):
		x = -BOWL_R + (j + 0.5) * w
		hl = math.sqrt(max(BOWL_R**2 - x**2, 0.0))
		out.append((x, hl))
	return out


def floor_strips(xy: tuple[float, float], rgba: str, group: str, condim: str, n: int = N_STRIPS) -> str:
	"""``n`` parallel box strips covering the disc; analytic box-box contacts only."""
	w = 2.0 * BOWL_R / n
	half_w = w / 2.0 + 1e-4  # tiny overlap prevents ray/contact seams between adjacent strips
	out = []
	for x, hl in strip_params(n):
		if hl <= 1e-6:
			continue
		out.append(
			f'<geom type="box" size="{half_w:.5f} {hl:.5f} 0.0015" '
			f'pos="{xy[0] + x:.5f} {xy[1]:.5f} 0.0015" rgba="{rgba}"{condim}{group}/>'
		)
	return "".join(out)


def _current_wall_boxes(xy: tuple[float, float], rgba: str, group: str) -> str:
	segs = []
	n = 64
	for k in range(n):
		a = 2.0 * math.pi * k / n
		cx, cy = xy[0] + BOWL_R * math.cos(a), xy[1] + BOWL_R * math.sin(a)
		half = BOWL_R * math.pi / n * 1.05
		segs.append(
			f'<geom type="box" size="0.003 {half:.4f} {BOWL_H / 2:.4f}" pos="{cx:.4f} {cy:.4f} {BOWL_H / 2:.4f}" '
			f'euler="0 -0.35 {a:.4f}" rgba="{rgba}"{group}/>'
		)
	return "".join(segs)


def wedge_mesh_xml(
	profile_polys: list[list[tuple[float, float]]],
	n_wedges: int = 48,
	prefix: str = "bowl_wedge",
) -> tuple[str, list[str]]:
	"""Build inline ``<mesh>`` assets and mesh names for revolved convex 2D profile polygons."""
	assets: list[str] = []
	names: list[str] = []
	eps_a = 1e-4  # tiny angular overlap prevents zero-width seams between adjacent wedges
	for p_idx, poly in enumerate(profile_polys):
		if len(poly) < 3:
			continue
		for k in range(n_wedges):
			a0 = 2.0 * math.pi * k / n_wedges - eps_a
			a1 = 2.0 * math.pi * (k + 1) / n_wedges + eps_a
			c0, s0 = math.cos(a0), math.sin(a0)
			c1, s1 = math.cos(a1), math.sin(a1)
			verts: list[str] = []
			for r, z in poly:
				verts.append(f"{r * c0:.6f} {r * s0:.6f} {z:.6f}")
				if r > 1e-7:
					verts.append(f"{r * c1:.6f} {r * s1:.6f} {z:.6f}")
			mname = f"{prefix}_{p_idx}_{k}"
			names.append(mname)
			assets.append(f'<mesh name="{mname}" vertex="{" ".join(verts)}"/>')
	return "".join(assets), names


def _load_profile_polys(wall_only: bool = False) -> list[list[tuple[float, float]]]:
	if not PROFILE_POLYS_PATH.is_file():
		raise FileNotFoundError(f"Missing {PROFILE_POLYS_PATH}; run decompose_plate.py first.")
	data = json.loads(PROFILE_POLYS_PATH.read_text(encoding="utf-8"))
	key = "wall_polys" if wall_only and "wall_polys" in data else "polys"
	return [[(float(r), float(z)) for r, z in poly] for poly in data[key]]


def _r1_hull_files() -> list[Path]:
	files = sorted(HULLS_DIR.glob("r1_*.obj"))
	if not files:
		raise FileNotFoundError(f"Missing R1 hulls in {HULLS_DIR}; run decompose_plate.py first.")
	return files


@dataclass(frozen=True)
class Candidate:
	"""MJCF bowl collision candidate."""

	id: str
	assets_xml: str
	body_geoms_fn: Callable[[tuple[float, float], str, str, str], str]

	def bowl_body_xml(
		self,
		name: str,
		xy: tuple[float, float],
		rgba: str,
		use_visual_mesh: bool = True,
		torsional: bool = False,
	) -> str:
		group = ' group="3"' if use_visual_mesh else ""
		condim = ' condim="4"' if torsional else ""
		visual = (
			f'<geom type="mesh" mesh="plate_200mm" pos="{xy[0]:.4f} {xy[1]:.4f} 0" '
			f'contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
			if use_visual_mesh
			else ""
		)
		col = self.body_geoms_fn(xy, rgba, group, condim)
		return f'<body name="{name}">{visual}{col}</body>'


def build_candidate(cid: str) -> Candidate:
	rcid = resolve_id(cid)
	if rcid == "R0_current":
		return Candidate(
			id=rcid,
			assets_xml="",
			body_geoms_fn=lambda xy, rgba, group, condim: (
				f'<geom type="cylinder" size="{BOWL_R:.3f} 0.0015" pos="{xy[0]:.4f} {xy[1]:.4f} 0.0015" '
				f'rgba="{rgba}"{condim}{group}/>' + _current_wall_boxes(xy, rgba, group)
			),
		)
	if rcid == "R9_slab_box_200":
		return Candidate(
			id=rcid,
			assets_xml="",
			body_geoms_fn=lambda xy, rgba, group, condim: (
				f'<geom type="box" size="{BOWL_R:.3f} {BOWL_R:.3f} 0.0015" pos="{xy[0]:.4f} {xy[1]:.4f} 0.0015" '
				f'rgba="{rgba}"{condim}{group}/>' + _current_wall_boxes(xy, rgba, group)
			),
		)
	if rcid == "R4_strips_floor_box_walls":
		return Candidate(
			id=rcid,
			assets_xml="",
			body_geoms_fn=lambda xy, rgba, group, condim: (
				floor_strips(xy, rgba, group, condim) + _current_wall_boxes(xy, rgba, group)
			),
		)
	if rcid == "R5_sdf":
		plate_file = (SIM_ASSETS_DIR / "plate_200mm.obj").as_posix()
		return Candidate(
			id=rcid,
			assets_xml=f'<mesh name="plate_200mm_sdf" file="{plate_file}"/>',
			body_geoms_fn=lambda xy, rgba, group, condim: (
				f'<geom type="sdf" mesh="plate_200mm_sdf" pos="{xy[0]:.4f} {xy[1]:.4f} 0" '
				f'rgba="{rgba}"{condim}{group}/>'
			),
		)
	if rcid == "R6_box_floor_wedge_walls":
		polys = _load_profile_polys(wall_only=True)
		assets, names = wedge_mesh_xml(polys, n_wedges=48, prefix="r6_wedge")
		return Candidate(
			id=rcid,
			assets_xml=assets,
			body_geoms_fn=lambda xy, rgba, group, condim: (
				f'<geom type="box" size="0.070 0.070 0.00125" pos="{xy[0]:.4f} {xy[1]:.4f} 0.00175" '
				f'rgba="{rgba}"{condim}{group}/>'
				+ "".join(
					f'<geom type="mesh" mesh="{m}" pos="{xy[0]:.4f} {xy[1]:.4f} 0" rgba="{rgba}"{group}/>'
					for m in names
				)
			),
		)
	if rcid == "R1_coacd_full":
		files = _r1_hull_files()
		assets = "".join(f'<mesh name="{p.stem}" file="{p.as_posix()}"/>' for p in files)
		stems = [p.stem for p in files]
		return Candidate(
			id=rcid,
			assets_xml=assets,
			body_geoms_fn=lambda xy, rgba, group, condim: "".join(
				f'<geom type="mesh" mesh="{m}" pos="{xy[0]:.4f} {xy[1]:.4f} 0" rgba="{rgba}"{condim}{group}/>'
				for m in stems
			),
		)
	if rcid == "R2_wedge_full":
		polys = _load_profile_polys(wall_only=False)
		assets, names = wedge_mesh_xml(polys, n_wedges=48, prefix="r2_wedge")
		return Candidate(
			id=rcid,
			assets_xml=assets,
			body_geoms_fn=lambda xy, rgba, group, condim: "".join(
				f'<geom type="mesh" mesh="{m}" pos="{xy[0]:.4f} {xy[1]:.4f} 0" rgba="{rgba}"{condim}{group}/>'
				for m in names
			),
		)
	if rcid == "R3_strips_floor_wedge_walls":
		polys = _load_profile_polys(wall_only=True)
		assets, names = wedge_mesh_xml(polys, n_wedges=48, prefix="r3_wedge")
		return Candidate(
			id=rcid,
			assets_xml=assets,
			body_geoms_fn=lambda xy, rgba, group, condim: (
				floor_strips(xy, rgba, group, condim)
				+ "".join(
					f'<geom type="mesh" mesh="{m}" pos="{xy[0]:.4f} {xy[1]:.4f} 0" rgba="{rgba}"{condim}{group}/>'
					for m in names
				)
			),
		)
	raise AssertionError(f"Unhandled candidate {rcid}")


def model_builder(
	cid: str,
	bowl_xy: tuple[float, float] = BOWL_XY,
	opt_xml: str = 'timestep="0.0041667" cone="elliptic" impratio="10"',
	torsional: bool = False,
	friction: str | None = None,
) -> Callable[[str], tuple[mujoco.MjModel, mujoco.MjData]]:
	"""Return ``builder(extra_body_xml) -> (model, data)`` for a standalone bowl model."""
	cand = build_candidate(cid)

	def _build(extra_body_xml: str = "") -> tuple[mujoco.MjModel, mujoco.MjData]:
		bowl_xml = cand.bowl_body_xml(
			"green_bowl", bowl_xy, "0.25 0.65 0.3 1", use_visual_mesh=False, torsional=torsional
		)
		# Put table in group 0 and bowl collision geoms in group 3 so mj_ray can isolate bowl geoms cleanly.
		bowl_xml = bowl_xml.replace("<geom ", '<geom group="3" ')
		default_fr = f'<default><geom friction="{friction}"/></default>' if friction else ""
		xml = (
			f"<mujoco><option {opt_xml}/>{default_fr}"
			f"<asset>{cand.assets_xml}</asset>"
			f'<worldbody><geom name="table" type="plane" size="3 3 0.05"/>'
			f"{bowl_xml}{extra_body_xml}</worldbody></mujoco>"
		)
		m = mujoco.MjModel.from_xml_string(xml)
		d = mujoco.MjData(m)
		mujoco.mj_forward(m, d)
		return m, d

	return _build


_ORIG = {
	"bowl": scene._bowl,
	"visual": scene._visual_mesh_assets,
	"floor_z": physics_expert.bowl_floor_z,
}


def install(cid: str) -> Candidate:
	"""Monkeypatch ``scene._bowl``, ``scene._visual_mesh_assets``, and ``physics_expert.bowl_floor_z``."""
	cand = build_candidate(cid)
	orig_visual = _ORIG["visual"]

	def _bowl_patch(name, xy, rgba, use_visual_mesh=True, torsional=False):
		return cand.bowl_body_xml(name, xy, rgba, use_visual_mesh=use_visual_mesh, torsional=torsional)

	def _visual_patch(assets_dir: Path | None = None) -> str:
		return orig_visual(assets_dir) + cand.assets_xml

	scene._bowl = _bowl_patch
	scene._visual_mesh_assets = _visual_patch
	physics_expert.bowl_floor_z = lambda *a, **k: 0.003
	return cand


def uninstall() -> None:
	"""Restore monkeypatched ``scene`` and ``physics_expert`` functions."""
	scene._bowl = _ORIG["bowl"]
	scene._visual_mesh_assets = _ORIG["visual"]
	physics_expert.bowl_floor_z = _ORIG["floor_z"]


def visual_plate_profile_fallback() -> dict:
	"""Compute reference profile directly from ``plate_200mm.obj`` vertices if JSON is not generated yet."""
	verts = []
	for line in (SIM_ASSETS_DIR / "plate_200mm.obj").read_text(encoding="utf-8").splitlines():
		if line.startswith("v "):
			verts.append([float(x) for x in line.split()[1:4]])
	# not needed once decompose_plate.py writes plate_profile.json
	raise FileNotFoundError(f"Missing plate_profile.json; verts={len(verts)}")
