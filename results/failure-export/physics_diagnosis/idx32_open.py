"""idx32 with a narrower pre-grasp jaw opening (ExpertParams.open_g); nothing is written to the repo."""
import dataclasses
import sys

sys.path.insert(0, r"E:\digital_twin_armigo\src")
sys.path.insert(0, r"E:\digital_twin_armigo\scripts")
import bench_physics_expert as bp
from digital_twin_armigo.mujoco_cell.physics_expert import ExpertParams
from digital_twin_armigo.physics import load_physics_spec

spec = load_physics_spec()
top = bp.load_top_camera()
layout_seed, idx = int(sys.argv[1]), int(sys.argv[2])
for open_g in [float(x) for x in sys.argv[3:]]:
    layout = bp.sample_layouts(60, layout_seed, top)[idx]
    p = dataclasses.replace(ExpertParams(), open_g=open_g)
    row = bp.run_layout_episode(layout, spec, idx, p)
    i = row["info"]
    print(
        f"open_g={open_g:.2f}: {row['verdict']} {row['reason_codes']} aborted={i['aborted']} retries={i['grasp_retries']} "
        f"n={i['n_frames']} jaw_env_max={i['jaw_env_contact_max_N']:.1f}N clearance={i['grasp_geometry'].get('clearance_m')}",
        flush=True,
    )
