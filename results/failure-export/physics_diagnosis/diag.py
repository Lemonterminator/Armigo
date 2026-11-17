"""Replay one episode and dump solver / contact state of the brick at 240 Hz substep resolution in a frame window."""
import sys

sys.path.insert(0, r"E:\digital_twin_armigo\src")
sys.path.insert(0, r"E:\digital_twin_armigo\scripts")
import mujoco
import numpy as np

import bench_physics_expert as bp
from digital_twin_armigo.benchmarks import HELD_OUT_PERTURBATIONS, SENSITIVITY_MATRIX
from digital_twin_armigo.mujoco_cell import physics_expert as pe
from digital_twin_armigo.physics import load_physics_spec

name, seed, f0, f1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cond = {c.name: c for c in (*SENSITIVITY_MATRIX, *HELD_OUT_PERTURBATIONS)}[name]
state = {"frame": -1, "cell": None, "sub": 0, "warn_first": {}, "max_niter": 0, "printed_opt": False}
orig_step = pe.PhysicsEpisode.step
orig_mj_step = mujoco.mj_step


def step(self, target):
    state["frame"] = self.n_frames
    state["cell"] = self.cell
    state["sub"] = 0
    orig_step(self, target)


def gname(m, g):
    b = m.body(m.geom_bodyid[g]).name
    return f"{b}/g{g}({m.geom(g).name or '-'})"


def mj_step(m, d, *a, **k):
    orig_mj_step(m, d, *a, **k)
    cell = state["cell"]
    if cell is None or d is not cell.data:
        return
    state["sub"] += 1
    for i in range(len(d.warning)):
        if d.warning[i].number and i not in state["warn_first"]:
            state["warn_first"][i] = (state["frame"], state["sub"], int(d.warning[i].number))
    state["max_niter"] = max(state["max_niter"], int(d.solver_niter[0]))
    f = state["frame"]
    if not (f0 <= f <= f1):
        return
    bg = set(int(g) for g in cell.brick_geoms)
    bq = cell.brick_qadr
    bd = cell.brick_dofadr
    v = d.qvel[bd : bd + 3]
    w = d.qvel[bd + 3 : bd + 6]
    rows = []
    cf = np.zeros(6)
    if not state.get("info_done"):
        state["info_done"] = True
        mm = m
        print("OPT timestep", mm.opt.timestep, "integrator", mm.opt.integrator, "cone", mm.opt.cone, "impratio", mm.opt.impratio, "solver", mm.opt.solver, "iters", mm.opt.iterations, "tol", mm.opt.tolerance, "noslip", mm.opt.noslip_iterations, "ccd_iter", getattr(mm.opt, "ccd_iterations", None), "ccd_tol", getattr(mm.opt, "ccd_tolerance", None), "disable", mm.opt.disableflags, "enable", mm.opt.enableflags)
        for g in sorted(bg):
            print("BRICK geom", g, mujoco.mjtGeom(mm.geom_type[g]).name, mm.geom_size[g], "pos", mm.geom_pos[g], "solref", mm.geom_solref[g], "solimp", mm.geom_solimp[g], "margin", mm.geom_margin[g], "fric", mm.geom_friction[g], "mass", mm.body_mass[cell.brick_body])
        bb = mm.geom_bodyid[118]
        for g in range(mm.ngeom):
            if mm.geom_bodyid[g] == bb and g < 125:
                print("BOWL geom", g, mujoco.mjtGeom(mm.geom_type[g]).name, mm.geom_size[g], "pos", mm.geom_pos[g], "contype", mm.geom_contype[g], "solref", mm.geom_solref[g], "solimp", mm.geom_solimp[g], "fric", mm.geom_friction[g], "margin", mm.geom_margin[g])
                if g > 120: break
        print("TABLE", mujoco.mjtGeom(mm.geom_type[0]).name, mm.geom_size[0], mm.geom_friction[0])
    for cid in range(d.ncon):
        c = d.contact[cid]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in bg or g2 in bg:
            mujoco.mj_contactForce(m, d, cid, cf)
            o = g2 if g1 in bg else g1
            rows.append(f"cf={np.round(cf,4).tolist()} dimm={int(c.dim)} {gname(m, o)}<-b{g1 if g1 in bg else g2} dist={c.dist * 1000:+.2f}mm F={np.linalg.norm(cf[:3]):.1f}N n=({c.frame[0]:+.2f},{c.frame[1]:+.2f},{c.frame[2]:+.2f}) pos=({c.pos[0]:.3f},{c.pos[1]:.3f},{c.pos[2]*1000:.1f}mm)")
    print(f"   qfrc_constraint(brick)={np.array2string(d.qfrc_constraint[bd:bd+6], precision=4)} qacc(brick)={np.array2string(d.qacc[bd:bd+6], precision=1)} nefc={d.nefc}")
    if f == 552 and state['sub'] in (1, 2, 3):
        np.save(str(__import__("pathlib").Path(__file__).parent / ("pose_f552_s%d.npy" % state["sub"])), np.concatenate([d.qpos[bq:bq+7], [m.body_mass[cell.brick_body]]]))
    q = d.qpos[bq + 3 : bq + 7]
    print(f"   quat(wxyz)={np.round(q,3)}")
    print(
        f"f{f} s{state['sub']} z={d.qpos[bq + 2] * 1000:6.1f}mm xy=({d.qpos[bq]:.3f},{d.qpos[bq + 1]:.3f}) "
        f"|v|={np.linalg.norm(v):6.2f} |w|={np.linalg.norm(w):7.1f} niter={int(d.solver_niter[0])} "
        f"ncon={d.ncon} qacc_max={np.abs(d.qacc).max():.2e} g={d.qpos[cell.qadr[5]]:.3f} :: "
        + " | ".join(rows)
    )


pe.PhysicsEpisode.step = step
mujoco.mj_step = mj_step
spec = load_physics_spec()
top = bp.load_top_camera()
row = bp.run_episode(cond, spec, top, seed)
print("verdict", row["verdict"], row["reason_codes"], "n_frames", row["info"]["n_frames"])
print("warnings (frame, substep, count):", {mujoco.mjtWarning(i).name: v for i, v in state["warn_first"].items()})
print("max solver_niter", state["max_niter"])
