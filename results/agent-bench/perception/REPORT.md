# bench16 with the brick located through a calibrated camera

The six-face agent (orbit scan, `sup-v7bot` detector, rule choices, 4 workers) normally plans with the simulator's
brick pose. With `ARMIGO_BRICK_PERCEPTION` set to one of the configs in `config/agent_bench/perception/`:

1. Before the episode, the top camera is calibrated: card on the gripper, 40 random poses (spread ×2), Park-Martin
   start and LM refinement (`mujoco_cell/handeye_rig.py`, see `results/handeye-sim/`). The joint readings carry the
   configured noise and bias.
2. Whenever `scan_wrist` or a grasp needs the brick on the table, the brick is located in the top depth image
   (`agent/perception.py`):
   - D415-like noise, σ 1.5 mm at 0.5 m, growing with the square of the depth;
   - back-projection through the estimated camera;
   - table plane fit;
   - blob 6 to 50 mm above the plane, outside the bowls;
   - top-face rectangle, with occlusion correction.
3. The scan planners see the perceived pose: the simulated brick is moved there while they plan, then put back. The
   grasp gets it in place of the simulator's brick pose.

Still from the simulator:

- the pose of the brick in the hand;
- the bowl positions;
- the face-visibility counts and detector crops;
- the per-jaw contact forces used to notice a lost brick;
- the placement check.

| Config | Calibration joints | Camera error, median (max) | Localisation xy, median (max) | Yaw, median (max) | Sorted | Labels | Rewinds |
|---|---|---|---|---|---|---|---|
| truth (no perception) | | | | | 16/16 | 16/16 | 0 |
| `ideal` | exact, offsets solved | 0.41 (0.55) mm, 0.02° | 1.28 (2.05) mm | 0.66° (6.0°) | 16/16 | 16/16 | 0 |
| `realistic` | noise 0.1°, bias σ 0.5°, offsets solved | 2.26 (4.52) mm, 0.20° | 1.63 (3.24) mm | 0.67° (5.9°) | 16/16 | 16/16 | 0 |
| `stress` | noise 0.3°, bias σ 1.0°, offsets not solved | 7.96 (13.0) mm, 0.69° | 4.79 (6.69) mm | 0.59° (5.6°) | 16/16 | 16/16 | 0 |

32 localisations per run (one per wrist scan and one per grasp). All found the brick with the right size class, and
8 needed the occlusion correction.

- **Nothing was lost by dropping the true brick pose.** Every episode had the same outcome, and the wrist views went
  from 26 to 28.
- **Occlusion.** At the ready pose the arm hides part of the top face in 600004 and 600008. The fitted rectangle
  comes out 8 to 9 mm short. The locator extends the shortfall towards whichever side the depth image shows
  covered by something nearer. This brought both cases from 3 to 4 mm down to about 1 mm (`tests/test_perception.py`).
- **The table plane fit is needed.** The first `stress` run measured heights from z = 0 of the estimated frame. A
  0.7° to 1.5° camera rotation error then lifts the far table above the 6 mm threshold, and the table was taken
  for the brick:
  - localisation error 128 mm median, 14 of 40 size classes wrong;
  - 10/16 sorted. The other 6 episodes abstained (`unknown`) instead of misplacing the brick.

  With heights taken from a plane fitted to the depth points, `stress` sorts 16/16 and the localisation error
  follows the calibration error. The `ideal` and `realistic` rows did not change with the fix.
- The 2x2 brick's yaw is only known modulo 90°. The perceived face numbering can therefore differ from the
  simulator's. Only repeat wrist scans that target a subset of faces would notice this, and none failed here.

## Sensor-only skills (`--sensing realistic`)

`TwinMujoco(sensing="realistic")` goes further and removes every simulator signal the skills read:

| Signal | Before | Sensor-only replacement |
|---|---|---|
| brick on the table | perceived (above) | the same |
| brick in the hand | simulator | believed rigid with the gripper from the grasp: forward kinematics × the pose perceived before the grasp |
| bowls | layout | colour blob, then a circle fitted to the 3D outer rim (`locate_bowls`) |
| visible faces, detector crop | simulator brick | computed on the believed brick, cut from the real image |
| clamp and lost-brick checks | per-jaw contact forces | gripper position stalled above its command under high effort (`agent/sensing.py`; agrees with the force test on 98-99 % of frames) |
| brick speed (clamp and put-down settling) | simulator | none (reads zero) |
| placement check | brick position | top depth image against the empty-bowl image (`bowl_fill`) |
| recovery | physics rewind to a checkpoint | from the current state: park or let go, ready pose, locate the brick again; earlier views count only if it has not moved |
| motion health | checked after placing | final evaluation only |

Still from the simulator: the bowl collision geometry in the grasp clearance checks (true positions, about 1.5 mm
from the perceived ones). The final evaluation is ground truth, as before.

bench16, `realistic` calibration:

| Run | Sorted | Labels | Recoveries | Brick belief error, table (median / max) | in hand | Bowls |
|---|---|---|---|---|---|---|
| sensor-only | 16/16 | 16/16 | 0 | 3.0 / 3.4 mm | 4.9 / 7.9 mm | 1.5 / 2.6 mm |

Belief errors are 3D distances between the believed and true brick centres at each scored view.

- The 2x2 footprint fixes the yaw only up to 90°, so the agent's face numbering is a rotation of the simulator's.
  That is harmless as long as it stays consistent. After the agent puts the brick down itself, the new
  localisation takes the symmetric yaw nearest the one it was released with.
- **The first run sorted 15/16.** In 600006 (defect on the bottom), the bottom-view planner maximised the believed
  bottom-face pixels and chose a pose with the brick at the top edge of the image. The real brick, 5-8 mm away from
  the belief, was partly cut off and the cross was missed: the brick was labelled normal. Bottom views now keep the
  believed brick 40 px inside the image (`plan_bottom_view(border_px=...)`), and the rerun sorts 16/16.
- Placement check: the target bowl shows 1366-2732 filled pixels and the other bowl 83-282 (the arm's edges and
  noise). The threshold is 500.

Commands (run folders stay local under `runs/`):

	ARMIGO_BRICK_PERCEPTION=config/agent_bench/perception/realistic.json \
	  uv run armigo-mcp run --detector results/agent-bench/sup-v7bot-local.json --workers 4 \
	  --out results/agent-bench/perception/runs/bench16-realistic --suite config/agent_bench/bench16.json --headless
	python results/agent-bench/perception/summarize.py truth=runs/bench16-truth ideal=runs/bench16-ideal ...
	uv run armigo-mcp run --sensing realistic --detector results/agent-bench/sup-v7bot-local.json --workers 4 	  --out results/agent-bench/perception/runs/bench16-sensor2 --suite config/agent_bench/bench16.json --headless
