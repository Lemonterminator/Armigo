"""Isolated reproduction: one box on a static bowl-floor collider, no robot, no gravity-driven dynamics (mj_forward only)."""
import numpy as np
import mujoco

pose = np.load(str(__import__("pathlib").Path(__file__).parent / "pose_f552_s2.npy"))
pos, quat = pose[:3], pose[3:7]


def run(floor_xml, label, opt="", flag=""):
    xml = f"""
<mujoco><option timestep="0.0041667" cone="elliptic" impratio="10" {opt}>{flag}</option>
<worldbody>
 <geom name="table" type="plane" size="3 3 0.05"/>
 {floor_xml}
 <body name="brick" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
  <freejoint/><geom name="b" type="box" size="0.0159 0.0159 0.0096" mass="0.0112"/>
 </body>
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"--- {label}: ncon={d.ncon}")
    for i in range(d.ncon):
        c = d.contact[i]
        print(f"   {m.geom(c.geom1).name or c.geom1}/{m.geom(c.geom2).name or c.geom2} dist={c.dist*1000:+.2f}mm n=({c.frame[0]:+.2f},{c.frame[1]:+.2f},{c.frame[2]:+.2f}) pos_z={c.pos[2]*1000:.1f}mm")


R = 0.1
run(f'<geom name="floor" type="cylinder" size="{R} 0.0015" pos="0.26 -0.18 0.0015"/>', "cylinder R=100mm thickness 3mm (as in scene)")
run(f'<geom name="floor" type="cylinder" size="{R} 0.02" pos="0.26 -0.18 -0.017"/>', "cylinder, same top face, 40mm thick")
run(f'<geom name="floor" type="box" size="{R} {R} 0.0015" pos="0.26 -0.18 0.0015"/>', "box 200x200x3mm")
run(f'<geom name="floor" type="box" size="{R} {R} 0.02" pos="0.26 -0.18 -0.017"/>', "box 200x200x40mm, same top face")
run(f'<geom name="floor" type="cylinder" size="0.03 0.0015" pos="0.26 -0.18 0.0015"/>', "cylinder R=30mm thickness 3mm")

cyl = f'<geom name="floor" type="cylinder" size="{R} 0.0015" pos="0.26 -0.18 0.0015"/>'
run(cyl, "cylinder, ccd_iterations=500 ccd_tolerance=1e-10", opt='ccd_iterations="500" ccd_tolerance="1e-10"')
try:
    run(cyl, "cylinder, nativeccd disabled", flag='<flag nativeccd="disable"/>')
except Exception as e:
    print("nativeccd flag not available:", str(e)[:120])
print(mujoco.__version__)
