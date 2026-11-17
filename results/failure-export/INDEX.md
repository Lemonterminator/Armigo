<!-- generated from the acceptance JSON files; see scripts/export_failure_animations.py -->
# Closed-loop physics expert: failure export

Everything here is a bit-exact MuJoCo 4.0.1 replay of episodes recorded in `results/physics-expert-v2/`. Simulation only; physics parameters are nominal assumed values.

## Failures (verdict != SUCCESS)

| id | verdict | reason | release peak speed (m/s) | jaw-env max (N) | frames | aborted | retries | source JSON | replay matches record |
|---|---|---|---|---|---|---|---|---|---|
| `high_mass_seed29` | FAILURE | bounce_out | 2.66 | 0.0 | 662 | None | 0 | matrix_seeds0-29_with_held_out.json | True |
| `layouts_seed1000_idx32` | FAILURE | pick_miss | - | 23.1 | 540 | no_clamp | 2 | layouts_seed1000.json | True |
| `low_friction_seed111` | FAILURE | bounce_out | 5.00 | 0.0 | 665 | None | 0 | matrix_seeds100-129_with_held_out.json | True |

## Flagged (verdict SUCCESS, but over a threshold)

| id | flag | release peak speed (m/s) | jaw-env max (N) | frames | source JSON | replay matches record |
|---|---|---|---|---|---|---|
| `layouts_seed1000_idx15` | JAW_ENV_OVER_5N | 0.22 | 9.7 | 819 | layouts_seed1000.json | True |
| `layouts_seed1000_idx48` | JAW_ENV_OVER_5N | 0.05 | 5.3 | 809 | layouts_seed1000.json | True |
| `layouts_seed1000_idx54` | JAW_ENV_OVER_5N | 0.07 | 12.1 | 787 | layouts_seed1000.json | True |
| `layouts_seed1001_idx11` | JAW_ENV_OVER_5N | 0.24 | 9.2 | 736 | layouts_seed1001.json | True |
| `layouts_seed1001_idx25` | JAW_ENV_OVER_5N | 0.05 | 17.2 | 820 | layouts_seed1001.json | True |
| `layouts_seed1001_idx37` | NEAR_FRAME_CAP | 0.02 | 0.0 | 899 | layouts_seed1001.json | True |
| `layouts_seed1_idx22` | JAW_ENV_OVER_5N | 0.11 | 7.1 | 724 | physics_expert_layouts_seed1_n60.json | True |
| `layouts_seed1_idx55` | NEAR_FRAME_CAP | 0.29 | 0.0 | 893 | physics_expert_layouts_seed1_n60.json | True |
| `layouts_seed2_idx39` | JAW_ENV_OVER_5N | 0.29 | 7.3 | 808 | physics_expert_layouts_seed2_n60.json | True |
| `layouts_seed2_idx47` | JAW_ENV_OVER_5N | 0.25 | 12.4 | 761 | physics_expert_layouts_seed2_n60.json | True |
What each failure is (from the recorded episode):

- `high_mass_seed29`, `low_friction_seed111`: the grasp, lift and carry succeed. The brick leaves the bowl after the jaws open (release peak speed 2.66 and 5.00 m/s against the 1.2 m/s gate) and the evaluator calls it `bounce_out`. These are the two over-gate releases in the 540-episode matrix. Per the earlier analysis the brick is held about 4 mm off the jaw centre and is squeezed out at opening.
- `layouts_seed1000_idx32`: `pick_miss`. The chosen grasp had -4.2 mm clearance to clutter or a bowl, the jaws pressed the environment at up to 23.1 N, two retries ended in `aborted=no_clamp` and the expert stopped without lifting (540 frames). There is no reachability pre-check, so this layout is never rejected up front.

## Layout

```
failure/   3 failing episodes:  <id>.mp4  <id>.json  <id>.npz
flagged/   10 episodes that pass the evaluator but exceed jaw-env 5 N or sit within 10 frames of the 900-frame cap
agent_logs/
  00_main_session_orchestrator.{jsonl,md}            this session's transcript (orchestration, acceptance runs, report)
  01_implementer_opus_closed_loop_expert.{jsonl,md}  the Opus 5.5 subagent that wrote physics_expert.py (both rounds)
  02_reviewer_sonnet_closed_loop_expert.{jsonl,md}   the Sonnet 5.5 review subagent
  *.meta.json                                        subagent type, model, description
export_run.log                                       console output of the replay run
```

- `<id>.mp4`: H.264, 30 fps, 2x2 grid (dataset top camera, overview, chase camera on the brick, wrist camera). The overlay shows frame, time, phase, brick speed, brick height, both jaw normal forces and the jaw-environment force. A red frame border marks brick speed above 1.2 m/s.
- `<id>.json`: the recorded episode (info and full audited-evaluator health) plus the replay check.
- `<id>.npz`: per-frame telemetry (qpos, q_cmd, brick pose and velocity, pinch forces, torques, phases, phase_names).
- `*.md` agent logs are readable renderings with each tool result cut to 4000 characters; the `.jsonl` files are the unmodified raw transcripts.

## Reproduce

```
.venv/Scripts/python scripts/export_failure_animations.py --category all     # or failure / flagged, --only <id>
.venv/Scripts/python scripts/export_agent_logs.py --out <dir> NAME=<transcript.jsonl> ...
```

Each replay reproduced the recorded verdict, reason codes, frame count and release speed exactly (all 13).
Not exported: the two stage-two subagents (snapshot store, operation ledger) in the same session, and the older
`results/verification-2026-11-17/*.log` files, which are render and simulation comparison checks that passed.
