"""Which jaw touches the environment during idx32's grasp attempts, and when."""
import sys

sys.path.insert(0, r"E:\digital_twin_armigo\src")
sys.path.insert(0, r"E:\digital_twin_armigo\scripts")
import mujoco
import numpy as np

import bench_physics_expert as bp
from digital_twin_armigo.mujoco_cell import physics_expert as pe
from digital_twin_armigo.physics import load_physics_spec

state = {"phase": "rest", "cell": None}
log = []
orig_step, orig_act = pe.PhysicsEpisode.step, pe.ClosedLoopExpert.act


def act(self, obs):
    t, p = orig_act(self, obs)
    state["phase"] = p
    state["retries"] = self.grasp_retries
    return t, p


def step(self, target):
    orig_step(self, target)
    cell = self.cell
    m, d = cell.model, cell.data
    bg = {int(g) for g in cell.brick_geoms}
    cf = np.zeros(6)
    hits = {}
    for cid in range(d.ncon):
        c = d.contact[cid]
        g1, g2 = int(c.geom1), int(c.geom2)
        for jaw, gs in (("fixed", cell.fixed_jaw_geoms), ("moving", cell.moving_jaw_geoms)):
            if g1 in gs or g2 in gs:
                o = g2 if g1 in gs else g1
                if o in gs or o in bg or o in cell.robot_all_geoms:
                    continue
                mujoco.mj_contactForce(m, d, cid, cf)
                key = (jaw, m.body(m.geom_bodyid[o]).name + f"/g{o}")
                hits[key] = max(hits.get(key, 0.0), float(np.linalg.norm(cf[:3])))
    log.append(
        (self.n_frames - 1, state["phase"], state.get("retries", 0), float(target[5]), float(cell.joints()[5]),
         [float(x) for x in self.last("pinch_normal")[-1:]], hits, cell.brick_pos().tolist())
    )


pe.PhysicsEpisode.step, pe.ClosedLoopExpert.act = step, act
from digital_twin_armigo.mujoco_cell import physics_expert as _  # noqa
spec = load_physics_spec()
top = bp.load_top_camera()
layout = bp.sample_layouts(60, 1000, top)[32]
row = bp.run_layout_episode(layout, spec, 32)
print("verdict", row["verdict"], row["reason_codes"], row["info"]["aborted"], "retries", row["info"]["grasp_retries"])
print("grasp_geometry", row["info"]["grasp_geometry"])
prev = None
for f, ph, r, gcmd, g, fn, hits, bp_ in log:
    if not hits and ph != "grasp":
        continue
    sig = (ph, r, tuple(sorted(hits)))
    if hits or sig != prev:
        print(f"f{f:3d} {ph:8s} try{r} g_cmd={gcmd:.3f} g_meas={g:.3f} brick_normal={fn[0]:.1f}N hits={ {k: round(v, 1) for k, v in hits.items()} } brick=({bp_[0]:.3f},{bp_[1]:.3f},{bp_[2]*1000:.1f}mm)")
    prev = sig
print("total frames with jaw-env hits:", sum(1 for x in log if x[6]))
