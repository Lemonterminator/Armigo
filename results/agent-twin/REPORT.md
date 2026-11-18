# Armigo agent acceptance, 2026-11-18

## MuJoCo closed loop

- An independent stdio MCP client discovered the tools, submitted two matched bottom-only fixtures, and checked request
  idempotency. Each episode recorded two wrist views and a bottom view. The normal brick went to green (1,726 frames),
  and the otherwise matched defective brick went to pink (1,700 frames). Both passed the audited motion evaluator.
  The bottom decision was at frame 1,063 in both episodes.
- The selected wrist-first policy then passed a separate 16-episode headless suite, with eight normal and eight
  defective fixtures and defects across all six faces. All 16 labels and audited sorts were correct. It used 40 views
  and 22,785 control frames. The median was 1,632 frames. The run spent about 152 seconds inside episodes, averaging
  9.5 wall-clock seconds per episode on this host. This puts 299 defective trials alone at roughly 47 minutes serial
  under the same workload. At 8/8, the one-sided exact 95% upper bound on the miss rate is 31.2%.
- The Qwen outer graph ran two candidates on the same four development fixtures. The recorded wrist-first baseline
  passed 2/4 physically and used 7,457 frames. The new Qwen proposal put bottom first and passed 1/4 with 9,670
  frames. Both labeled 4/4 correctly. The graph kept wrist-first. In this cohort, failures included torque stalls
  and placement misses. These data do not establish a general improvement over the existing heuristic.
- Qwen selected skills and a true rewind when a placement failure was injected. Branch 0 recorded its bottom evidence
  at frame 1,063 and stopped at 1,064. Qwen selected the earlier checkpoint at frame 191. It reacquired wrist and
  bottom evidence on branch 1 and sorted successfully from a new bottom image at frame 1,083. The teacher retained
  the failure history. The exporter rejected that rewound episode.
- With no usable detector scores, a physical MuJoCo trial returned UNKNOWN and put the brick back on the inspection
  table. This trial used a controlled detector substitution.
- The replay dashboard sought to the exact decision time, showed the source and crop, and preserved the separate
  branch timelines. Six archive camera frames matched their decision PNGs byte-for-byte. The LeRobot 0.6 reader
  opened a 2-episode, 3,426-frame export with one shared task and the full external RGB channel. PyAV was used for
  readback because TorchCodec could not load its Windows DLLs on this host.

## Bottom detector gate

| Cohort | v6 detects | v6 false alarms | Qwen detects | Qwen false alarms |
|---|---:|---:|---:|---:|
| MuJoCo separate 8-normal calibration and 8 paired test layouts | 8/8 | 2/8 | 8/8 | 0/8 |
| Earlier in-place presentation proof, the 6 visible pairs | 6/6 | 0/6 | 3/6 | 0/6 |

The earlier presentation script attempted eight pairs. One pair had no plan and one lost the brick, leaving six
readable pairs. The controlled camera-profile trial collected an image from every attempted fixture. Its threshold
stayed at v6's zero after the normal-only calibration. Qwen saw blinded RGB crops only. Its uneven sensitivity leaves
v6 in control of sorting.

## Isaac gates

- The native Isaac Lab 3.0 / Isaac Sim 6.2 checkpoint probe passed at an open-gripper, table-supported boundary.
  After direct writes and a repeat command, the maximum joint discrepancy was 7.15e-7 rad and the measured brick
  pose discrepancy was zero. This probe does not restore solver state during friction grasp.
- Isaac RTX rerendered the recorded MuJoCo bottom poses. The original v6 threshold detected 6/6 and produced
  5/6 false alarms. A separate calibration of six Isaac-rendered normal samples selected 0.52257. On six
  disjoint pairs it detected 4/6 and produced 1/6 false alarms. This is a pilot of RTX appearance transfer and
  is separate from native Isaac physics. The Isaac six-face closed loop and the 299-defect statistical
  acceptance each remain open.

## Verification

The Python suite passed 428 tests with one skipped. The MCP crop response was checked against the original PNG,
and an invalid observation path was rejected. Ruff passed for the changed Python files.

## Local evidence

- `data/catalog/agent/acceptance-final/episodes/32374db859f14466ab093797e70095c1` (normal)
- `data/catalog/agent/acceptance-final/episodes/820c6911801540c3b04a9d570df7ddf8` (defective)
- `data/catalog/agent/acceptance-final/case-0098` (Qwen recovery)
- `data/catalog/agent/vla-six-face` (accepted LeRobot demonstration dataset)
- `data/catalog/agent/bottom-pilot` and `data/catalog/agent/bottom-rtx` (detector crops and scores)
- `data/catalog/agent/qwen-search-proposals/search/222cbd4314b64a34ba950f44a1a5d7e9` (two-policy comparison)

The media files and model weights stay in local artifact directories. The fixtures and compact acceptance evidence
are versioned here. See [the operator guide](../../docs/ARMIGO_AGENT.md) for command and deployment details.
