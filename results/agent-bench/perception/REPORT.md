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

Commands (run folders stay local under `runs/`):

	ARMIGO_BRICK_PERCEPTION=config/agent_bench/perception/realistic.json \
	  uv run armigo-mcp run --detector results/agent-bench/sup-v7bot-local.json --workers 4 \
	  --out results/agent-bench/perception/runs/bench16-realistic --suite config/agent_bench/bench16.json --headless
	python results/agent-bench/perception/summarize.py truth=runs/bench16-truth ideal=runs/bench16-ideal ...
