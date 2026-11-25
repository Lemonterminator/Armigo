# Prompt for GPT: top-camera drift alarm and recalibration in the localize node

The text below the line is the prompt. Claude reviews the result against the acceptance list in section 6.

---

## 0. Your task in one paragraph

You are working in a MuJoCo digital twin of an SO-101 arm that sorts LEGO Duplo bricks. The brick is located by a calibrated top camera, then refined by a ring of wrist-camera views (`localize` node).

When the top camera has been bumped since its calibration (`cameras="drift"`), the ring has to correct the top-view position by 15–50 mm. After an in-place recalibration (`cameras="jitter_extrinsic"`, called "recal"), the correction should only be a few millimetres. A real system should notice the large correction, raise an alarm and recalibrate (about one minute; the calibration itself is free).

Implement that:
- a drift alarm based on the localize correction (plus a table-plane sanity check);
- a recalibration recovery that re-runs the hand-eye calibration on the camera as it is mounted now, then localizes again.

Pick the threshold from data, test it, and report the numbers.

## 1. Workspace and hard rules

**Workspace**
- Repository: `E:/digital_twin_armigo`. Do not touch the main checkout `E:/digital_twin_armigo` or `E:/digital_twin_armigo-v3`. `D:/Armigo_data_v2` is read-only.
- Create your own worktree from branch `claude/reverse-act-demos-dbc45c` (commit `a5db1cc` or later):
  `git -C E:/digital_twin_armigo/.claude/worktrees/fervent-keller-bb7998 worktree add E:/digital_twin_armigo-gpt-drift -b gpt/drift-alarm claude/reverse-act-demos-dbc45c`
- Work and commit only there. Do not push, merge or rebase other branches.

**Environment**
- Interpreter `E:/digital_twin_armigo/.venv/Scripts/python`, with `PYTHONPATH=src` and `CUDA_VISIBLE_DEVICES=""`. CPU only; do not start GPU jobs or the vLLM server. Do not install packages.

**Code style**
- Tab indentation. `ruff check` and `ruff format` must pass on every file you touch.
- Several source files are CRLF (`twin_mujoco.py`, `twin_graph.py`, `perception.py`, `step_mcp.py`, `student_rollout.py`). Keep their line endings: Git Bash `sed -i` turns them into LF, so edit with a tool that keeps CRLF, or read bytes, replace on `\n`-normalised text, and write CRLF back.

**Scope**
- Default behaviour must not change. Everything new is opt-in (a new `TwinMujoco` keyword, off by default).
- Do not change spec or contract files: `src/digital_twin_armigo/experiments/contract.py`, `ORACLE_PARTS`, the VLA contract `src/digital_twin_armigo/vla/contract.py`, or the experiment runner options. If you think one needs to change, write it in the report instead.
- Simulator truth may appear in events only as diagnostics registered in `TRUTH_DIAGNOSTIC_KEYS` (`src/digital_twin_armigo/agent/step_mcp.py`). The agent's decisions must never read truth: the alarm uses only what the agent itself measured.

## 2. What exists (read these first)

- `src/digital_twin_armigo/agent/twin_mujoco.py`:
  - `TwinMujoco.__init__`: keywords `localise="top"|"vote"`, `ring_localise`, `perception_llm`; the camera calibration block calls `calibrate_layout`. Under drift it is solved on `layout.calib_top` (the nominal rig) while the real camera sits elsewhere; see the `camera_calibrated` event fields `actual_cam_pos_err_mm` and `actual_cam_axis_err_deg`.
  - `_sense_bowls`, `_perceive(est=None, source=None)`, `_top_depth`.
  - `localise_brick()` (the graph's `localize` node): vote, wrist confirmation, `_ring_localise()`, then pins `_brick_hint = {"top_xy", "delta"}`. It emits `brick_localised` with `correction_mm` (the ring's correction of the top-view x, y, in mm).
  - `_ring_localise()` emits `ring_localised` with `innovation_xy_mm` and `accepted`.
  - `recovery_choices`, `failure`, `rewind`: recovery handling. See how failure codes such as `BRICK_NOT_FOUND` are offered recovery choices.
- `src/digital_twin_armigo/agent/twin_graph.py`: the LangGraph. `prepare → localize → choose_view`; `localize → recover` on error; `recover → choose_view | evaluate`.
- `src/digital_twin_armigo/agent/perception.py`: `locate_brick` and `table_plane`. The fitted brick height (`BrickEstimate.pos[2]`) comes out implausible under drift; for example seed 3001205 gives z = -0.013 m, when a brick resting on the table has its centre about 0.0095 m above it.
- `src/digital_twin_armigo/agent/act_grasp.py`: `grasp_trial(case, out, ..., localise="vote")` runs `prepare`, the localize node, then one grasp. It returns `localised` (including `correction_mm`) and `ring` (including `innovation_xy_mm`).
- `scripts/vla/eval_act_grasp.py`: 60-case grasp eval.
  - `--cameras drift|jitter_extrinsic`, `--localise vote`, `--descend-close-ratio 1.3`.
  - Without `--vlm-url`, the vote uses the shape rule only. That is fine for you (CPU only).
  - Writes `<out>/<arm>/results.jsonl`, and per case `<out>/<arm>/case-<seed>/events.jsonl`.
- Tests to keep green: `tests/test_brick_vote.py`, `tests/test_wrist_ring.py`, `tests/test_twin_agent.py`, `tests/test_checkpoint_resume.py`.
  - `tests/test_checkpoint_resume.py::test_real_detector_resume_from_disk_exactly` already fails before this work. Leave it, but say so.

**Existing data you can use**
- Drift, 60 cases, seeds 3001201–3001260:
  - `C:/scratch/reverse-act/eval/scripted-close1.3-vote/` (vote + ring, with VLM);
  - `C:/scratch/reverse-act/eval/scripted-close1.3-ring/`.
- Recal, same seeds: `C:/scratch/reverse-act/eval-recal/` (being produced now; use it when its `summary.json` has the `scripted-close1.3-vote` arm).
- Treat both as read-only. Write your own outputs under `C:/scratch/reverse-act/drift-alarm/`.

## 3. Step A: measure before you build

1. From the events, collect per case under drift and under recal:
   - `brick_localised.correction_mm`;
   - `ring_localised.innovation_xy_mm` and `accepted`;
   - the perceived brick height above the fitted table: `brick_perceived.position_m[2]` against the believed size, or recompute it from the perception.
2. If something you need is missing, rerun `eval_act_grasp.py` (shape-only vote, CPU) into your own output directory, for example with 8 workers.
3. Report both distributions: n, median, 90th percentile, max.
4. Find whether one threshold on `correction_mm` separates them, and how many cases fall on the wrong side.
5. Do the same for the height check.

This is the gate. If no threshold gives at most 1/60 false alarms under recal while flagging most drift cases with a correction ≥ 10 mm, stop and report the data instead of building.

## 4. Step B: build (only if step A passes)

1. **The alarm**: a new opt-in keyword `drift_alarm: bool = False` on `TwinMujoco` (only valid with `localise="vote"`).
   - In `localise_brick`, after the ring, raise the alarm when the ring was accepted and `correction_mm` exceeds the threshold from step A, or when the height check fails.
   - Emit an event `recalibration_needed`. Its fields are agent-side only: correction, height, threshold. Truth may be added only as registered diagnostics.
   - Return a new failure code `RECALIBRATE`.
2. **The recovery**: `RECALIBRATE` must lead to a recalibration, not a physics rewind.
   - Recalibration means re-running the existing hand-eye calibration (`calibrate_layout`) on the camera as it is mounted now. Under drift that is `layout.top` instead of `layout.calib_top`; the drift case then becomes recal.
   - Then re-sense the bowls (`_sense_bowls`), drop `_brick_hint`, and run the localize node again.
   - Allow at most one recalibration per episode. A second alarm is reported, not looped on.
   - Account its time: record `recalibration_s = 60` in the event. Decide whether it should advance sim time or only be logged, and say why.
   - Graph: prefer the smallest change that keeps the graph readable, for example `localize → recalibrate → localize`. Update `NODES` and `EDGES` in `twin_graph.py`.
   - Check the parts that list node names: `twin_mujoco.py` around the `kind == "node" and node in {...}` history code, `student_rollout.py` `_motion` (segment mapping), and `twin_record.py` (uses `NODES`/`EDGES`).
3. **Checkpoints**: the recalibrated camera estimate and the "already recalibrated" flag must survive `_save` / restore (see how `brick_hint` and the `camera` entry are stored in the agent checkpoint dict).
4. **Wiring**: `grasp_trial(..., drift_alarm=...)` and `eval_act_grasp.py --drift-alarm`. The arm name gets the suffix `-alarm`.

## 5. Step C: evaluate and test

**Evaluation**: shape-only vote, `--descend-close-ratio 1.3`, 60 cases each.
- `--cameras drift --localise vote --drift-alarm`
- `--cameras jitter_extrinsic --localise vote --drift-alarm`

Report:
- alarms raised per condition, and false alarms under recal;
- lifted counts with and without the alarm under drift and under recal;
- the localisation error after recalibration (truth diagnostic);
- the sim time per case.

**Tests** (fast where possible; at most one MuJoCo end-to-end test):
- threshold and height check logic as unit tests on plain numbers;
- the graph routes `localize → recalibrate → localize` exactly once on `RECALIBRATE`, and stops after a second alarm (fake backend, like `tests/test_brick_vote.py::test_graph_runs_localize_between_prepare_and_choose_view`);
- one end-to-end drift case where the alarm fires and the brick is then localised within 5 mm. Seed 3001205 is a known hard case: top view 51 mm off. Pick another if you find a reason.
- `drift_alarm=False` leaves events and results unchanged on one case (compare against a run without the keyword).

## 6. What Claude will check (acceptance)

1. A step A table with both distributions and the chosen threshold, and the data files under `C:/scratch/reverse-act/drift-alarm/`.
2. Recal: at most 1/60 false alarms. Drift: the alarm fires on most cases whose correction is ≥ 10 mm.
3. Drift with the alarm lifts at least as many as recal without it, minus 3. Recal with the alarm lifts the same as recal without it, ± 1.
4. No default behaviour change: `localise="top"` and `drift_alarm=False` give identical results; existing tests pass (except the known pre-existing failure).
5. No truth read by the agent; truth only in registered diagnostic keys.
6. CRLF files are still CRLF (`git diff --stat` shows no whole-file churn). `ruff check` and `ruff format` are clean on touched files.
7. Small commits on `gpt/drift-alarm`, each with a clear message. A short report (Chinese is fine) at `C:/scratch/reverse-act/drift-alarm/REPORT.md` with the numbers, the design choices (sim time, graph shape) and anything you think is a spec problem.
