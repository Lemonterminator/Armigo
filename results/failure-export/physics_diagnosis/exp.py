"""Re-run episodes with an in-memory collider variant of the bowl floor (nothing is written to the repo).

usage: exp.py VARIANT cond seed [seed ...]     VARIANT in: none, thick, condim3, both
"""
import sys

sys.path.insert(0, r"E:\digital_twin_armigo\src")
sys.path.insert(0, r"E:\digital_twin_armigo\scripts")
import math

import mujoco
import numpy as np

import bench_physics_expert as bp
import digital_twin_armigo.mujoco_cell as mc
from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
from digital_twin_armigo.physics import load_physics_spec

variant, cname = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]]
orig_cell = mc.Cell


def patched(*a, **k):
    cell = orig_cell(*a, **k)
    m = cell.model
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if body.endswith("_bowl") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            if variant in ("thick", "both"):
                r, h_old = m.geom_size[g][0], m.geom_size[g][1]
                top = m.geom_pos[g][2] + h_old  # keep the top face where it was (3 mm)
                h = 0.02
                m.geom_size[g][1] = h
                m.geom_pos[g][2] = top - h
                m.geom_rbound[g] = math.hypot(r, h)
                m.geom_aabb[g] = [0, 0, 0, r, r, h]
            if variant in ("condim3", "both"):
                m.geom_condim[g] = 3
    if variant == "libccd":
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    if variant == "boxfloor":
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name.endswith("_bowl") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = m.geom_size[g][0], m.geom_size[g][1]
                m.geom_type[g] = mujoco.mjtGeom.mjGEOM_BOX
                m.geom_size[g] = [r, r, h]
                m.geom_rbound[g] = math.sqrt(2 * r * r + h * h)
                m.geom_aabb[g] = [0, 0, 0, r, r, h]
    mujoco.mj_forward(m, cell.data)
    return cell


from digital_twin_armigo.mujoco_cell import physics_expert as pe
if variant == "boxfloor":
    pe.bowl_floor_z = lambda *a, **k: 0.003  # the floor's top face is unchanged (3 mm)
mc.Cell = patched
cond = {c.name: c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)}[cname]
spec = load_physics_spec()
top = bp.load_top_camera()
ok = 0
for s in seeds:
    row = bp.run_episode(cond, spec, top, s)
    i = row["info"]
    ok += row["verdict"] == "SUCCESS"
    print(
        f"{variant:8s} {cname} seed {s}: {row['verdict']} {row['reason_codes']} "
        f"release_peak={i.get('release_peak_speed_mps')} n={i.get('n_frames')} aborted={i.get('aborted')}",
        flush=True,
    )
print(f"{variant} {cname}: {ok}/{len(seeds)} SUCCESS")
