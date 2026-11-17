"""Offline plate cross-section extraction, Hertel-Mehlhorn 2D convex decomposition, and CoACD 3D decomposition.

Run in the isolated Python 3.12 environment (``C:\\tmp\\bowl_decomp_env\\Scripts\\python.exe``).
The pure-numpy 2D Hertel-Mehlhorn helpers can also be imported in ``.venv`` for unit tests.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLATE_OBJ = ROOT / "assets" / "meshes" / "plate_200mm.obj"
RESULTS_DIR = ROOT / "results" / "bowl-experiments"
HULLS_DIR = ROOT / "assets" / "meshes" / "bowl_hulls"


def polygon_area(pts: Sequence[tuple[float, float]]) -> float:
	"""Signed area of a 2D polygon (positive for CCW)."""
	n = len(pts)
	return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
	return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex_ccw(pts: Sequence[tuple[float, float]], eps: float = -1e-12) -> bool:
	"""Return True if ``pts`` is a convex CCW polygon."""
	n = len(pts)
	if n < 3:
		return False
	return all(_cross(pts[i], pts[(i + 1) % n], pts[(i + 2) % n]) >= eps for i in range(n))


def _point_in_triangle(
	p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> bool:
	c1 = _cross(a, b, p)
	c2 = _cross(b, c, p)
	c3 = _cross(c, a, p)
	return c1 >= -1e-12 and c2 >= -1e-12 and c3 >= -1e-12


def ear_clip_triangulate(poly: Sequence[tuple[float, float]]) -> list[list[int]]:
	"""Ear-clipping triangulation of a simple CCW polygon; returns index triples into ``poly``."""
	pts = list(poly)
	if polygon_area(pts) < 0:
		raise ValueError("polygon must be CCW")
	rem = list(range(len(pts)))
	tris: list[list[int]] = []
	guard = 0
	while len(rem) > 3:
		n = len(rem)
		ear_found = False
		for k in range(n):
			i0, i1, i2 = rem[(k - 1) % n], rem[k], rem[(k + 1) % n]
			a, b, c = pts[i0], pts[i1], pts[i2]
			if _cross(a, b, c) <= 1e-12:
				continue
			if any(_point_in_triangle(pts[j], a, b, c) for j in rem if j not in (i0, i1, i2)):
				continue
			tris.append([i0, i1, i2])
			rem.pop(k)
			ear_found = True
			break
		if not ear_found:
			guard += 1
			if guard > 5:
				raise RuntimeError("ear clipping failed on degenerate polygon")
			break
	if len(rem) == 3:
		tris.append(rem)
	return tris


def hertel_mehlhorn(poly: Sequence[tuple[float, float]]) -> list[list[tuple[float, float]]]:
	"""2D Hertel-Mehlhorn convex decomposition: triangulate and greedily merge adjacent convex faces."""
	pts = [(float(x), float(y)) for x, y in poly]
	if polygon_area(pts) < 0:
		pts = list(reversed(pts))
	faces: list[list[int]] = ear_clip_triangulate(pts)

	changed = True
	while changed:
		changed = False
		for i in range(len(faces)):
			for j in range(i + 1, len(faces)):
				fi, fj = faces[i], faces[j]
				# Find shared directed edge (u -> v in fi, v -> u in fj)
				shared = None
				for ei in range(len(fi)):
					u, v = fi[ei], fi[(ei + 1) % len(fi)]
					for ej in range(len(fj)):
						if fj[ej] == v and fj[(ej + 1) % len(fj)] == u:
							shared = (ei, ej)
							break
					if shared is not None:
						break
				if shared is None:
					continue
				ei, ej = shared
				merged = fi[: ei + 1] + [fj[(ej + 2 + k) % len(fj)] for k in range(len(fj) - 2)] + fi[ei + 1 :]
				if is_convex_ccw([pts[idx] for idx in merged]):
					faces[i] = merged
					faces.pop(j)
					changed = True
					break
			if changed:
				break

	return [[pts[idx] for idx in f] for f in faces]


def _chain_segments_to_loop(segs: np.ndarray, tol: float = 1e-6) -> list[tuple[float, float]]:
	"""Chain unordered 2D line segments ``(N, 2, 2)`` into a closed CCW polygon."""
	# Keep segments in r >= -tol (the x >= 0 half-section)
	kept = []
	for p0, p1 in segs:
		if p0[0] >= -tol and p1[0] >= -tol:
			kept.append(((float(max(0.0, p0[0])), float(p0[1])), (float(max(0.0, p1[0])), float(p1[1]))))

	def _key(p: tuple[float, float]) -> tuple[int, int]:
		return (round(p[0] * 1e5), round(p[1] * 1e5))

	adj: dict[tuple[int, int], list[tuple[float, float]]] = {}
	pts_map: dict[tuple[int, int], tuple[float, float]] = {}
	for a, b in kept:
		ka, kb = _key(a), _key(b)
		if ka == kb:
			continue
		pts_map[ka], pts_map[kb] = a, b
		adj.setdefault(ka, []).append(b)
		adj.setdefault(kb, []).append(a)

	# Open endpoints at r == 0 get closed along r == 0 axis
	open_ends = [k for k, nbrs in adj.items() if len(nbrs) == 1]
	if len(open_ends) == 2:
		k0, k1 = open_ends
		adj[k0].append(pts_map[k1])
		adj[k1].append(pts_map[k0])

	start = min(adj.keys())
	loop = [pts_map[start]]
	prev_k = None
	curr_k = start
	for _ in range(len(adj) + 2):
		nbrs = [n for n in adj[curr_k] if _key(n) != prev_k]
		if not nbrs:
			break
		nxt = nbrs[0]
		nxt_k = _key(nxt)
		if nxt_k == start:
			break
		loop.append(nxt)
		prev_k, curr_k = curr_k, nxt_k

	if polygon_area(loop) < 0:
		loop = list(reversed(loop))
	return loop


def _clip_wall_polygon(loop: Sequence[tuple[float, float]], r_min: float = 0.062) -> list[tuple[float, float]]:
	"""Extract the outer wall portion ``r >= r_min`` of the closed cross-section polygon."""
	pts = [p for p in loop if p[0] >= r_min - 1e-6]
	if polygon_area(pts) < 0:
		pts = list(reversed(pts))
	return pts


def _wall_inner_r_mm(loop: Sequence[tuple[float, float]]) -> dict[str, float]:
	"""Interpolate inner wall radius (mm) at z = 6, 12, 18 mm along the inner profile from floor to rim."""
	pts = list(loop)
	n = len(pts)
	idx_top = min(range(n), key=lambda i: (pts[i][0], -pts[i][1]))
	idx_rim = max(range(n), key=lambda i: pts[i][1])
	# Walk both directions from center top (0, 3mm) to rim top (97.2mm, 22mm); pick the path staying at z >= 2.9mm
	fwd, bwd = [], []
	i = idx_top
	while True:
		fwd.append(pts[i])
		if i == idx_rim:
			break
		i = (i + 1) % n
	i = idx_top
	while True:
		bwd.append(pts[i])
		if i == idx_rim:
			break
		i = (i - 1) % n
	inner_path = fwd if min(z for _, z in fwd) >= min(z for _, z in bwd) else bwd
	inner = [(r * 1e3, z * 1e3) for r, z in inner_path]
	out: dict[str, float] = {}
	for z_target in (6, 12, 18):
		for (r0, z0), (r1, z1) in itertools.pairwise(inner):
			if z0 <= z_target <= z1 and z1 > z0:
				t = (z_target - z0) / (z1 - z0)
				out[str(z_target)] = round(r0 + t * (r1 - r0), 4)
				break
	return out


def main(argv: list[str] | None = None) -> int:
	import argparse

	import coacd
	import trimesh

	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--skip-coacd", action="store_true")
	args = ap.parse_args(argv)

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	HULLS_DIR.mkdir(parents=True, exist_ok=True)

	mesh = trimesh.load(str(PLATE_OBJ), force="mesh")
	lines_3d = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0])
	segs_rz = lines_3d[:, :, [0, 2]]
	loop = _chain_segments_to_loop(segs_rz)

	profile_data = {
		"floor_top_mm": round(max(z * 1e3 for r, z in loop if r <= 0.0625), 3),
		"flat_floor_r_mm": 62.0,
		"rim_top_mm": round(max(z * 1e3 for _, z in loop), 3),
		"profile": [[round(r, 6), round(z, 6)] for r, z in loop],
		"wall_inner_r_mm": _wall_inner_r_mm(loop),
	}
	(RESULTS_DIR / "plate_profile.json").write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
	print("wrote plate_profile.json:", {k: v for k, v in profile_data.items() if k != "profile"})

	# 2D Hertel-Mehlhorn for R2 (full profile) and R3 (wall r >= 62 mm)
	polys = hertel_mehlhorn(loop)
	wall_loop = _clip_wall_polygon(loop, r_min=0.062)
	wall_polys = hertel_mehlhorn(wall_loop)
	poly_data = {
		"full_area_m2": polygon_area(loop),
		"polys": [[[round(r, 6), round(z, 6)] for r, z in p] for p in polys],
		"wall_area_m2": polygon_area(wall_loop),
		"wall_polys": [[[round(r, 6), round(z, 6)] for r, z in p] for p in wall_polys],
	}
	(RESULTS_DIR / "profile_polys.json").write_text(json.dumps(poly_data, indent=2), encoding="utf-8")
	print(f"wrote profile_polys.json: {len(polys)} full polys, {len(wall_polys)} wall polys")

	if args.skip_coacd:
		return 0

	# R1 CoACD 3D decomposition
	for old in HULLS_DIR.glob("r1_*.obj"):
		old.unlink()
	coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
	parts = coacd.run_coacd(coacd_mesh, threshold=0.03, max_convex_hull=32, seed=0)
	hull_stats = []
	for idx, (verts, faces) in enumerate(parts):
		tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
		out_path = HULLS_DIR / f"r1_{idx:02d}.obj"
		tm.export(str(out_path))
		r_vals = np.hypot(verts[:, 0], verts[:, 1])
		hull_stats.append(
			{
				"file": out_path.name,
				"n_verts": len(verts),
				"r_min_mm": round(float(r_vals.min()) * 1e3, 2),
				"r_max_mm": round(float(r_vals.max()) * 1e3, 2),
				"z_min_mm": round(float(verts[:, 2].min()) * 1e3, 2),
				"z_max_mm": round(float(verts[:, 2].max()) * 1e3, 2),
			}
		)
	(RESULTS_DIR / "r1_coacd_stats.json").write_text(json.dumps(hull_stats, indent=2), encoding="utf-8")
	print(f"wrote {len(parts)} R1 CoACD hulls to {HULLS_DIR}")
	for s in hull_stats:
		print(" ", s)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
