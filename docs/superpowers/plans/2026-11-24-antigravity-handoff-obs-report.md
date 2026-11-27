# Failure Observability / CV Motion / UNet Brickseg / Paired Governance Report

Branch: `feat/failure-observability` in `E:/digital_twin_armigo-obs` (branched from `feat/smolvla-static-v3` at `16d17e7`).

---

## Part H (`H1–H8`)

### Task Summary

| Task | Commit | New & Modified Files | New Tests | Task / Full Pytest Result |
| :--- | :--- | :--- | :--- | :--- |
| Setup | `2d5b0f8` | `docs/superpowers/plans/{2026-11-24-antigravity-handoff-obs,2026-11-24-failure-observability-harness-learning,2026-11-24-paired-trajectory-governance}.md` | 0 | docs-only commit (`obs docs: plans and handoff`) |
| Test fix | `6c9068c` | `tests/test_palette_scene_v3.py` | 0 | Fixed worktree root normalization in `legacy_7cc24e2` XML hash check (`19 passed`) |
| H1 | `433de0f` | `src/digital_twin_armigo/agent/step_mcp.py`, `tests/test_truth_diagnostics.py` | 4 | Full `pytest -q`: `967 passed, 14 skipped, 2 xfailed, 16 warnings in 1389.79s (0:23:09)` |
| H2 | `b4f9879` | `src/digital_twin_armigo/experiments/{precursors,attribution,scoring,report}.py`, `tests/test_precursors.py` | 7 | Full `pytest -q`: `974 passed, 14 skipped, 2 xfailed, 16 warnings in 1356.32s (0:22:36)` |
| H3 | `520be0d` | `scripts/experiments/calibrate_precursors.py`, `config/experiments/precursor_thresholds.json`, `tests/test_precursors.py` | 4 | Full `pytest -q`: `978 passed, 14 skipped, 2 xfailed, 16 warnings in 1360.19s (0:22:40)` |
| H4 | `96ffa70` | `src/digital_twin_armigo/inventory/{scan,store,query,audit}.py`, `tests/test_inventory.py` | 4 | Full `pytest -q`: `982 passed, 14 skipped, 2 xfailed, 16 warnings in 1347.20s (0:22:27)` |
| H5 | `14ce496` | `src/digital_twin_armigo/agent/step_mcp.py`, `tests/test_step_mcp.py` | 1 | `tests/test_step_mcp.py + tests/test_truth_diagnostics.py`: `10 passed`; full `pytest -q`: `983 passed, 14 skipped, 2 xfailed, 16 warnings in 1372.75s (0:22:52)` |
| H6 | `b0de6bc` | `src/digital_twin_armigo/agent/{twin_mujoco,twin_record}.py`, `src/digital_twin_armigo/experiments/{contract,counterfactual,runner}.py`, `scripts/experiments/counterfactual.py`, `tests/test_counterfactual.py` | 8 | `tests/test_vla_paired_carry.py + tests/test_counterfactual.py`: `20 passed`; full `pytest -q`: `991 passed, 14 skipped, 2 xfailed, 16 warnings in 1376.86s (0:22:56)` |
| H7 | `0fb12d8` | `src/digital_twin_armigo/experiments/diagnosis.py`, `scripts/experiments/{make_diagnosis_bench,round_metrics}.py`, `config/agent_bench/diagnosis_bench/v1.json`, `tests/test_diagnosis.py` | 5 | `tests/test_diagnosis.py`: `5 passed`; full `pytest -q`: `996 passed, 14 skipped, 2 xfailed, 16 warnings in 1391.20s (0:23:11)` |
| H8 | `f7e5c3b` | `src/digital_twin_armigo/experiments/diag_mcp.py`, `tests/test_diag_mcp.py` | 8 | `tests/test_diag_mcp.py + tests/test_step_mcp.py`: `14 passed`; full `pytest -q`: `1004 passed, 14 skipped, 2 xfailed, 16 warnings in 1396.78s (0:23:16)` |

### Latest Full Pytest Suite Tail (`obs H8`)

```text
1004 passed, 14 skipped, 2 xfailed, 16 warnings in 1396.78s (0:23:16)
```

Skipped tests (`14 skipped`, unchanged across `H1–H8`): environment-gated skips (e.g., `test_vla_augment_batch.py` needing `torch`/`torchvision`, `test_vla_bowl_token.py` needing `torch`/`lerobot`, plus function-level `torch`/`lerobot`/`rerun`/local vLLM server/external fixture checks).

### Read-Only Checks & Calibration Outputs (`D:/Armigo_data_v2`)

- **H2 read-only rescore on `D:/Armigo_data_v2/runs/p2-dev-r2` (216 episodes)**:
  - `root_cause` counts match Plan §0: `none: 166, recovery_failure: 23, evidence_insufficient: 15, grasp_or_path_failure: 5, target_face_not_in_view: 3, placement_failure: 2, false_positive_visible: 2`.
  - Drift `recovery_failure` mechanism breakdown: `('brick_pose_error', 'calibration_drift'): 20`, `('brick_pose_error', None): 3`.
- **H3 calibration via `scripts/experiments/calibrate_precursors.py` on `D:/Armigo_data_v2/reports/p2-dev-r2/scores.jsonl`** -> `config/experiments/precursor_thresholds.json`:
  - `brick_xy_mm = 14.213` (AUC `0.892`)
  - `brick_yaw_deg = 3.409` (AUC `0.869`)
  - `bowl_xy_mm = 27.031` (AUC `1.000`)
  - `view_belief_mm = 20.690` (AUC `0.727 < 0.75`, flagged as weak precursor)
  - Calibrated §0 `root_cause` counts preserved.
- **H7 read-only benchmark build (`PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/experiments/make_diagnosis_bench.py --audit D:/Armigo_data_v2/audit/reviews.sqlite --inventory D:/Armigo_data_v2/inventory --out config/agent_bench/diagnosis_bench/v1.json`)**:
  - Output: `WARNING: only 1 confirmed entries (< 40); collect more human audits before drawing conclusions.`
  - `config\agent_bench\diagnosis_bench\v1.json: 1 entries (calib=1, heldout=0)` (`agent:v2:m0-smoke/mujoco/truth-v7bot/L600008-D1`).

### Deviations / Clarifications from Plan & Edge Fixes

1. **`test_palette_scene_v3.py` worktree path assumption (`6c9068c`)**: Existing test hardcoded main repo path instead of worktree root when hashing `legacy_7cc24e2` XML; normalized worktree root so baseline full suite passes cleanly in `E:/digital_twin_armigo-obs`.
2. **`precursor_thresholds.json` schema (`H3`)**: `precursors.load_thresholds()` reads the top-level `"metrics"` key (not `"thresholds"`); `calibrate_precursors.py` and `config/experiments/precursor_thresholds.json` align on `"metrics"`, and `tests/test_precursors.py` isolates `THRESHOLDS_FILE` via `monkeypatch`.
3. **`SimpleNamespace` stub compatibility in `TwinMujoco.evaluate()` (`H6`)**: `tests/test_vla_paired_carry.py::test_place_and_evaluate_route_extra_bowl_and_emit_v3_metadata` calls `TwinMujoco.evaluate()` with a `SimpleNamespace` stub lacking `.oracle`. Changed `not self.oracle` to `not getattr(self, "oracle", None)`.
4. **`tests/test_diagnosis.py` script loading & H4 schema alignment (`H7`)**: Loaded `scripts/experiments/{make_diagnosis_bench,round_metrics}.py` via `importlib.util.spec_from_file_location` (matching existing repo test conventions without altering `pythonpath`) and avoided duplicating `auto_root_cause`/`auto_mechanism`/`auto_upstream` already added to `store.TRAJECTORY_COLUMNS` in H4 (`96ffa70`).
5. **Transient full-suite render jitter during `H7` (`tests/test_d1_inputs.py`)**: On the first full `pytest -q` run for H7, untouched `tests/test_d1_inputs.py::test_realistic_top_depth_v2_uses_independent_d415_rng_and_perception_is_unchanged` failed once on `assert np.array_equal(imgs["top_rgb"], clean_rgb)`. Standalone rerun passed (`8 passed in 167.03s`) and full `pytest -q` rerun passed (`996 passed, 14 skipped, 2 xfailed`). Recorded as transient WGL/rendering jitter under load.

### Skipped / Unrun Steps

- **H6 Step 8 (GPU / live evaluation runs)**: Skipped per handoff scope (no GPU tasks or large agent runs).
- **H7**: Only `1` confirmed human audit entry exists in `D:/Armigo_data_v2/audit/reviews.sqlite` (`< 40`); needs additional confirmed human reviews before `v1.json` has statistical power across `calib` and `heldout`.


---

## Part C (`C1–C7`)

### Task Summary

| Task | Commit | New & Modified Files | New Tests | Task / Full Pytest Result |
| :--- | :--- | :--- | :--- | :--- |
| C1 | `4782356` | `src/digital_twin_armigo/agent/wrist_localize.py`, `tests/test_wrist_localize.py` | 5 | `tests/test_wrist_localize.py`: `5 passed`; full `pytest -q`: `1009 passed, 14 skipped, 2 xfailed, 16 warnings in 1391.55s (0:23:11)` |
| C2 | `97e240e` | `src/digital_twin_armigo/agent/grasp_loop.py`, `src/digital_twin_armigo/mujoco_cell/physics_expert.py`, `tests/test_grasp_loop.py` | 4 | `tests/test_grasp_loop.py + tests/test_physics_expert.py`: `41 passed`; full `pytest -q`: `1013 passed, 14 skipped, 2 xfailed, 16 warnings in 1496.38s (0:24:56)` |
| C3 | `665c8c3` | `src/digital_twin_armigo/agent/{step_mcp,twin_mujoco}.py`, `tests/test_grasp_loop.py` | 3 | `tests/test_grasp_loop.py`: `7 passed`; full `pytest -q`: `1016 passed, 14 skipped, 2 xfailed, 16 warnings in 1429.66s (0:23:49)` |
| C4 | `330993d` | `src/digital_twin_armigo/agent/twin_mujoco.py`, `src/digital_twin_armigo/mujoco_cell/{cell,inspection_views}.py`, `tests/test_belief_cameras.py` | 4 | `tests/test_belief_cameras.py`: `4 passed`; full `pytest -q`: `1020 passed, 14 skipped, 2 xfailed, 16 warnings in 1448.30s (0:24:08)` |
| C5 | `f3a573c` | `src/digital_twin_armigo/agent/twin_mujoco.py`, `tests/test_belief_cameras.py` | 2 | `tests/test_belief_cameras.py + tests/test_grasp_loop.py + tests/test_truth_diagnostics.py`: `17 passed`; full `pytest -q`: `1022 passed, 14 skipped, 2 xfailed, 16 warnings in 1481.19s (0:24:41)` |
| C6 | `d61d208` | `src/digital_twin_armigo/experiments/{contract,runner}.py`, `src/digital_twin_armigo/agent/twin_mujoco.py`, `tests/test_cv_motion_runner.py` | 4 | `tests/test_cv_motion_runner.py + tests/test_counterfactual.py`: `12 passed`; full `pytest -q`: `1026 passed, 14 skipped, 2 xfailed, 16 warnings in 1443.45s (0:24:03)` |
| C7a | `46c50be` | `docs/experiments/cv_motion_prereg.md` | 0 | Preregistration committed prior to `cv_motion_matrix.sh` |
| C7 | `092bca7` | `scripts/experiments/cv_motion_matrix.sh` | 0 | Matrix script committed (GPU/live Steps 3–5 skipped) |

### Latest Full Pytest Suite Tail (`obs C6`)

```text
1026 passed, 14 skipped, 2 xfailed, 16 warnings in 1443.45s (0:24:03)
```

### C1 Step 5 MuJoCo Integration Check (`cameras="drift"`, 4 cases, hover move)

Command run (CPU, `PYTHONPATH=src`): executed `twin.prepare()`, `twin._perceive()`, `twin._execute(move_hover())`, and `localize_from_wrist` with designed mount (`believed_wrist_from_cam`) across `sample_case(seed, positive=False, face=0, options={"cameras": "drift"})` for `seed in (700001, 700002, 700003, 700004)`:

```text
seed=700001: FAILED reason=BAD_FOOTPRINT tilt_deg=33.30 prior_xy_mm=16.76 prior_yaw_deg=1.65
seed=700002: FAILED reason=BAD_FOOTPRINT tilt_deg=33.21 prior_xy_mm=6.97 prior_yaw_deg=1.94
seed=700003: FAILED reason=BAD_FOOTPRINT tilt_deg=33.23 prior_xy_mm=15.79 prior_yaw_deg=1.64
seed=700004: FAILED reason=BAD_FOOTPRINT tilt_deg=33.28 prior_xy_mm=12.26 prior_yaw_deg=1.63
reasons: {'BAD_FOOTPRINT': 4}
```

- Failure reason counts: `{'BAD_FOOTPRINT': 4}` (0 localized from LAB blob alone; 0 `TILTED`/`NO_BLOB`).
- Measured wrist camera tilt (`tilt_deg`): `33.21°–33.30°`.
- Confirms Plan C1 Step 5 expectation: at ~33° camera tilt, brick side faces merge into the LAB colour blob and inflate the fitted footprint past the 30% gate (`BAD_FOOTPRINT`). No side-face heuristics were added in C1; left to Part U (`U1–U3`) top-face segmentation.

### Deviations / Clarifications from Plan & Edge Fixes

1. **`SimpleNamespace` stub compatibility in `TwinMujoco.evaluate()` (`C3`)**: Used `getattr(self, "motion", "open")` alongside `getattr(self, "oracle", None)` so existing `tests/test_vla_paired_carry.py` `SimpleNamespace` stub passes without adding runtime attrs to stub fixtures.
2. **Thread affinity in `StepService` test (`C3`)**: In `tests/test_grasp_loop.py::test_step_mcp_last_wrist_fix_strips_truth_keys`, calling `svc.start` via `svc.run_sync(...)` keeps MuJoCo GLFW WGL contexts confined to `StepService`'s simulation thread (calling `svc.start` directly on the main thread corrupted WGL contexts during `svc.close()` for subsequent MuJoCo tests).
3. **Un-forwarded `Cell` transforms & `detector.elapsed_ms` timing variance (`C4`, `C5`)**:
   - In `tests/test_belief_cameras.py`, fresh `Cell(layout, ..., dynamic_clutter=True, assisted_grasp=False)` leaves MuJoCo kinematics un-forwarded until `mujoco.mj_forward(cell.model, cell.data)`, so `test_camera_override_moves_the_model_camera_and_restores_it` and `test_center_px_crop_is_centred_on_the_given_point` call `mj_forward` first.
   - `diag_mcp.compare_runs` ignores top-level `{"wall_time_s", "latency_s", "seq"}` (`diag_mcp.py:28`), while `view` events nest wall-clock `detector.elapsed_ms` (`twin_mujoco.py`). When verifying `evidence="hold"` discrete invariance across `_wrist` calls (`test_hold_evidence_wrist_omits_motion_fields_and_matches_via_compare_runs`), the test freezes `twin_mujoco.time.perf_counter` via `monkeypatch`; `diag_mcp.py` was kept unchanged per H8 spec (note for acceptance `compare_runs` on real runs: sub-ms `detector.elapsed_ms` jitter can flag `"detector"` diffs even on identical discrete trajectories).

### Skipped / Unrun Steps

- **C7 Steps 3–5 (Smoke/full matrix runs on GPU with defect detector)**: Skipped per handoff scope (GPU occupied; no large runs).


---

## Part U (`U1–U5`)

### Task Summary

| Task | Commit | New & Modified Files | New Tests | Task / Full Pytest Result |
| :--- | :--- | :--- | :--- | :--- |
| U1 | `230ccde` | `src/digital_twin_armigo/mujoco_cell/cell.py`, `src/digital_twin_armigo/perception_nn/{__init__,labels}.py`, `scripts/perception/make_brickseg_dataset.py`, `tests/test_brickseg_labels.py` | 3 | `tests/test_brickseg_labels.py + tests/test_inspection_views.py`: `23 passed`; full `pytest -q`: `1029 passed, 14 skipped, 2 xfailed, 16 warnings in 1444.46s (0:24:04)` |
| U2 | `e3b9f9c` | `src/digital_twin_armigo/perception_nn/model.py`, `scripts/perception/{make_tiny_onnx_fixture,train_brickseg}.py`, `tests/data/brickseg_tiny.{onnx,json}`, `tests/test_brick_pose_nn.py` | 2 | `test_brickseg_unet_shape_and_params` ran in `lerobot-cuda` (`CUDA_VISIBLE_DEVICES=""`); `tests/test_brick_pose_nn.py` (`.venv`): `1 passed, 1 skipped`; full `pytest -q`: `1030 passed, 15 skipped, 2 xfailed, 16 warnings in 1503.67s (0:25:03)` |
| U3 | `27335b8` | `src/digital_twin_armigo/agent/{brick_pose_nn,wrist_localize}.py`, `tests/test_brick_pose_nn.py` | 6 | `tests/test_brick_pose_nn.py + tests/test_wrist_localize.py + tests/test_grasp_loop.py`: `19 passed, 1 skipped`; full `pytest -q`: `1036 passed, 15 skipped, 2 xfailed, 16 warnings in 1478.65s (0:24:38)` |
| U4a | `3404d9d` | `docs/experiments/brickseg_prereg.md` | 0 | Preregistration committed prior to U4 evaluation/code |
| U4b | `0a1660f` | `scripts/perception/eval_brick_pose.py`, `scripts/experiments/cv_motion_matrix.sh`, `src/digital_twin_armigo/experiments/{contract,runner}.py`, `src/digital_twin_armigo/agent/twin_mujoco.py`, `tests/{test_cv_motion_runner,test_grasp_loop}.py` | 2 | `tests/test_cv_motion_runner.py + test_grasp_loop.py + test_truth_diagnostics.py + test_vla_paired_carry.py`: `29 passed`; full `pytest -q`: `1038 passed, 15 skipped, 2 xfailed, 16 warnings in 1470.35s (0:24:30)` |
| U5 | `c2a620b` | `scripts/perception/realism_compare.py`, `tests/test_u5_realism_compare.py` | 1 | `tests/test_u5_realism_compare.py`: `1 passed`; full `pytest -q`: `1039 passed, 15 skipped, 2 xfailed, 16 warnings in 1460.00s (0:24:20)` |

### Latest Full Pytest Suite Tail (`obs U5`)

```text
1039 passed, 15 skipped, 2 xfailed, 16 warnings in 1460.00s (0:24:20)
```

The additional skip (+1 vs `14 skipped`) is `tests/test_brick_pose_nn.py::test_brickseg_unet_shape_and_params`, which requires `torch` (executed directly in `E:/il/lerobot-cuda/Scripts/python.exe` with `CUDA_VISIBLE_DEVICES=""`).

### Formal Artifacts & Smoke Checks

- **U1 50-case smoke (`C:/scratch/brickseg-dev/smoke`, `--preview-dir C:/scratch/brickseg-dev/preview`)**:
  - `total_frames = 276`, `total_dropped = 24`, `wall_s = 15.68`; 10 preview overlays in `C:/scratch/brickseg-dev/preview` (`preview_00_idx4.png`..`preview_09_idx227.png`).
- **U1 Formal CPU Dataset (`E:/Armigo_data_v3/perception/brickseg-v1`)**:
  - Command: `PYTHONPATH=src E:/digital_twin_armigo/.venv/Scripts/python scripts/perception/make_brickseg_dataset.py --train-cases 2000 --val-cases 250 --ood-cases 200 --workers 3 --out-dir E:/Armigo_data_v3/perception/brickseg-v1 --preview-dir E:/Armigo_data_v3/perception/brickseg-v1/preview --clean`
  - Recorded `E:/Armigo_data_v3/perception/brickseg-v1/manifest.json` (`git_commit = 230ccde902ede4a7ea1cd1b3c68e0d3a51382f3b`, `wall_s = 333.97`):
    - `total_frames`: `12936`
    - `total_dropped`: `1764`
    - `frames_by_split`: `{"train": 10598, "val": 1304, "ood-appearance": 1034}`
    - `dropped_by_split`: `{"train": 1402, "val": 196, "ood-appearance": 166}`
    - Preview PNGs (`10` files): `E:/Armigo_data_v3/perception/brickseg-v1/preview` (`preview_00_idx4.png`..`preview_09_idx227.png`).
- **U2 `CUDA_VISIBLE_DEVICES=""` CPU checks (`lerobot-cuda`)**:
  - `make_tiny_onnx_fixture.py --out-dir tests/data`: `tests/data/brickseg_tiny.onnx` = `133,113` bytes (`130.0 KB <= 200 KB` via `pure_python_onnx`) + `tests/data/brickseg_tiny.json` (`427` bytes).
  - `BrickSegUNet(base=24, classes=4, heat=2)`: `params = 1096734`, output shape `(2, 6, 240, 320)`.
  - 1-epoch CPU smoke (`train_brickseg.py --data-dir C:/scratch/brickseg-dev/smoke --out-dir C:/scratch/brickseg-dev/smoke_model --epochs 1 --limit 64 --batch-size 4 --base 8 --device cpu`): `train_samples = 52`, `val_samples = 12`, `wall_s = 1.94`, `val_loss = 2963.7862`, `top_iou = 0.0`, `pck_2px = 0.260274`.
- **U4b `eval_brick_pose.py` smoke (`C:/scratch/brickseg-dev/eval_smoke.md`)**:
  - Read `E:/Armigo_data_v3/perception/brickseg-v1` + `tests/data/brickseg_tiny.onnx` (`--limit 10`): `3` summary rows (`val` upright `lab`/`unet-pnp`/`unet-rect`, `n=6`), `cpu_infer_ms` median `6.371 ms`, p90 `7.049 ms`.
- **U5 `realism_compare.py` smoke (`C:/scratch/brickseg-real/compare`)**:
  - Ran against `real_ref` + `tests/data/brickseg_tiny.onnx`: wrote `2` panels (`real_ref_top_rgb_v2_3217.png`, `real_ref_wrist_3217.png`), `index.html`, `notes.md`. No real-robot metrics or thresholds computed; `C:/Users/Jiang/.gemini/antigravity/scratch/hackjunction` remained untouched (`file count before=107456 after=107456`).

### Deviations / Clarifications from Plan & Edge Fixes

1. **`BrickSegUNet` parameter count & pure-Python ONNX fallback (`U2`)**:
   - Plan prose estimated `1.5M–2.5M` params; `BrickSegUNet(base=24, classes=4, heat=2)` has `1,096,734` parameters (`> 1M`).
   - `E:/il/lerobot-cuda` lacks `onnx` / `onnxscript` / `onnxruntime`, so `export_onnx` (`dynamo=False`) automatically falls back to `_serialize_unet_onnx` (`pure_python_onnx` opset-17 protobuf serialization, verified loadable in `.venv` via `cv2.dnn.readNetFromONNX`). Caught `BLE001` by narrowing to `(ImportError, ModuleNotFoundError, RuntimeError)` in `model.py`.
2. **`tests/test_brick_pose_nn.py` preservation (`U3`)**: Kept both U2 (`test_brickseg_unet_shape_and_params`, `test_tiny_onnx_loads_in_cv2_dnn`) and U3 (6 pose/dispatch tests) in `tests/test_brick_pose_nn.py`.
3. **`SimpleNamespace` stub compatibility in `TwinMujoco.evaluate()` (`U4b`)**: Kept `getattr(self, "motion", "open")` and `getattr(self, "oracle", None)` in `evaluate()` so `tests/test_vla_paired_carry.py` passes alongside `--segmenter`.

### Skipped / Unrun Steps

- **U2 Step 5 (GPU training on `brickseg-v1`)**: Skipped per handoff scope (hardware GPU occupied).
- **U4 Step 4 (offline eval on trained ONNX & `cv_motion_matrix.sh stage3b` GPU matrix runs)**: Skipped per handoff scope.
- **U5 Step 2 (optional Studio video sampling + manual visual review in `notes.md`)**: Skipped per handoff scope (video sampling unrun; script exercised on `real_ref`).


---

## Part P (`P1–P6`)

### Task Summary

| Task | Commit | New & Modified Files | New Tests | Task / Full Pytest Result |
| :--- | :--- | :--- | :--- | :--- |
| P1 | `1146c89` | `src/digital_twin_armigo/governance/{__init__,pairs}.py`, `tests/test_gov_pairs.py` | 8 | `tests/test_gov_pairs.py`: `8 passed`; full `pytest -q`: `1047 passed, 15 skipped, 2 xfailed, 16 warnings in 1479.05s (0:24:39)` |
| P2 | `eb2c070` | `src/digital_twin_armigo/governance/{lake,bronze}.py`, `tests/test_gov_bronze.py` | 4 | `tests/test_gov_pairs.py + test_gov_bronze.py`: `12 passed`; full `pytest -q`: `1051 passed, 15 skipped, 2 xfailed, 16 warnings in 1471.21s (0:24:31)` |
| P3 | `e85aa3a` | `src/digital_twin_armigo/governance/{segments,silver}.py`, `tests/test_gov_silver.py` | 3 | `tests/test_gov_*.py`: `15 passed`; full `pytest -q`: `1054 passed, 15 skipped, 2 xfailed, 16 warnings in 1471.09s (0:24:31)` |
| P4 | `5ad91e6` | `src/digital_twin_armigo/governance/silver.py`, `tests/test_gov_silver.py` | 1 | `tests/test_gov_*.py`: `16 passed`; full `pytest -q`: `1055 passed, 15 skipped, 2 xfailed, 16 warnings in 1456.82s (0:24:16)` |
| P5 | `34c910b` | `src/digital_twin_armigo/governance/gold.py`, `tests/test_gov_gold.py` | 4 | `tests/test_gov_*.py`: `20 passed`; full `pytest -q` rerun: `1059 passed, 15 skipped, 2 xfailed, 16 warnings in 1450.25s (0:24:10)` |
| P6 | `3faacae` | `src/digital_twin_armigo/experiments/runner.py`, `src/digital_twin_armigo/governance/gold.py`, `scripts/vla/collect_paired_carry.py`, `scripts/governance/{lake.py,capacity_bench.sh}`, `docs/GOVERNANCE.md`, `tests/{test_run_pair,test_gov_gold}.py` | 4 | `tests/test_run_pair.py + test_gov_*.py`: `24 passed`; full `pytest -q -rs`: `1063 passed, 15 skipped, 2 xfailed, 16 warnings in 1456.51s (0:24:16)` |

### Latest Full Pytest Suite Tail & Skips (`gov P6`, `3faacae`)

```text
SKIPPED [1] tests\test_vla_augment_batch.py:6: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_vla_bowl_token.py:23: needs torch and lerobot (run under E:/il/lerobot-cuda)
SKIPPED [1] tests\test_agent_policies.py:175: local vLLM server not reachable
SKIPPED [1] tests\test_brick_pose_nn.py:16: could not import 'torch': No module named 'torch'
SKIPPED [2] tests\test_experiments_scoring.py:350: real episode missing
SKIPPED [1] tests\test_export_cameras.py:155: could not import 'lerobot': No module named 'lerobot'
SKIPPED [1] tests\test_instruction_variants.py:46: could not import 'transformers': No module named 'transformers'
SKIPPED [1] tests\test_tracking.py:90: could not import 'mlflow': No module named 'mlflow'
SKIPPED [1] tests\test_vla_s5_perturb.py:235: no stitched episode for an unrecovered branch
SKIPPED [1] tests\test_vla_s9_losses.py:545: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_vla_s9_losses.py:666: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_vla_s9_losses.py:736: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_vla_s9_losses.py:759: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_vla_s9_losses.py:856: could not import 'torch': No module named 'torch'
1063 passed, 15 skipped, 2 xfailed, 16 warnings in 1456.51s (0:24:16)
```

### P6 Step 5 v2 Read-Only Dry Run (`D:/Armigo_data_v2` -> `C:/scratch/gov-dry/`)

Commands executed cleanly under `PYTHONPATH=src` (after deleting misplaced `E:/Armigo_data_v3/governance/lake.sqlite` and empty `E:/Armigo_data_v3/governance/` directory):

```powershell
E:/digital_twin_armigo/.venv/Scripts/python.exe scripts/governance/lake.py --db C:/scratch/gov-dry/lake.sqlite register --root D:/Armigo_data_v2/runs --source v2
E:/digital_twin_armigo/.venv/Scripts/python.exe scripts/governance/lake.py --db C:/scratch/gov-dry/lake.sqlite silver --audit-path D:/Armigo_data_v2/audit/reviews.sqlite
E:/digital_twin_armigo/.venv/Scripts/python.exe scripts/governance/lake.py --db C:/scratch/gov-dry/lake.sqlite gold failure_bank --out-root C:/scratch/gov-dry
E:/digital_twin_armigo/.venv/Scripts/python.exe scripts/governance/lake.py --db C:/scratch/gov-dry/lake.sqlite gold harness_review --out-root C:/scratch/gov-dry
E:/digital_twin_armigo/.venv/Scripts/python.exe scripts/governance/lake.py --db C:/scratch/gov-dry/lake.sqlite report
```

Verbatim outputs:
- `register`: `{"new": 513, "unchanged": 0, "changed": 0}`
- `silver`: `{"build_id": "6f0ee23e9cc7", "episodes": 513, "pairs": 332}`
- `gold failure_bank`: `C:\scratch\gov-dry\gold\failure_bank\42ff984a4a23`
  - `manifest.jsonl`: `4883` bytes, `12` lines (`split=legacy: 12`, `camera_condition=drift: 12`, `pair_outcome=truth_only: 12`)
  - `card.md`: `383` bytes, `15` lines
- `gold harness_review`: `C:\scratch\gov-dry\gold\harness_review\a8582f9b0630`
  - `manifest.jsonl`: `990` bytes, `3` lines (`split=legacy: 3`, `pair_outcome=unpaired: 3`)
  - `card.md`: `457` bytes, `15` lines
- `lake.sqlite`: `C:\scratch\gov-dry\lake.sqlite` (`2,617,344` bytes); `E:/Armigo_data_v3/governance/lake.sqlite` confirmed absent (`False`).

Verbatim `report` output (`# Silver Pair Report (6f0ee23e9cc7)`):

```markdown
## Batch x Camera Outcomes

| batch | camera_condition | both_pass | truth_only | realistic_only | both_fail | void | unpaired |
|---|---|---:|---:|---:|---:|---:|---:|
| m2-coresid | recal | 0 | 0 | 0 | 0 | 0 | 2 |
| m2-smoke | fixed | 0 | 0 | 0 | 0 | 0 | 2 |
| m4-smoke | drift | 0 | 0 | 0 | 0 | 0 | 2 |
| m4-smoke | fixed | 0 | 0 | 0 | 0 | 0 | 2 |
| m4-smoke | recal | 0 | 0 | 0 | 0 | 0 | 2 |
| m4-smoke-c | recal | 0 | 0 | 0 | 0 | 0 | 2 |
| p1-bench16 | fixed | 0 | 0 | 0 | 0 | 0 | 32 |
| p1-dev | fixed | 0 | 0 | 0 | 0 | 0 | 72 |
| p2-dev | drift | 12 | 12 | 0 | 0 | 0 | 24 |
| p2-dev | fixed | 24 | 0 | 0 | 0 | 0 | 24 |
| p2-dev | recal | 24 | 0 | 0 | 0 | 0 | 24 |
| p2-dev-r2 | drift | 0 | 0 | 0 | 0 | 0 | 24 |
| p2-dev-r2 | fixed | 0 | 0 | 0 | 0 | 0 | 24 |
| p2-dev-r2 | recal | 0 | 0 | 0 | 0 | 0 | 24 |

## realistic_only Episodes

(none)

## truth_only (root_cause x mechanism)

| root_cause | (none) | bowl_pose_error | brick_pose_error | brick_yaw_error |
|---|---:|---:|---:|---:|
| false_positive_visible | 1 | 0 | 0 | 0 |
| grasp_or_path_failure | 0 | 0 | 2 | 1 |
| placement_failure | 0 | 1 | 0 | 0 |
| recovery_failure | 1 | 0 | 6 | 0 |

## Quarantine Reasons

| reason | count |
|---|---:|
| complete | 3 |
```

### Deviations / Bugs Caught by Dry Run & Edge Fixes (`Part P`)

1. **Clean package init without staging `extend_path` shim (`P1`)**: `src/digital_twin_armigo/governance/__init__.py` created without `pkgutil.extend_path`.
2. **Transient full-suite render jitter during `P5` (`tests/test_d1_inputs.py`)**: On the initial full `pytest -q` run for `P5` (`34c910b`), untouched `tests/test_d1_inputs.py::test_realistic_top_depth_v2_uses_independent_d415_rng_and_perception_is_unchanged` failed once under WGL load; standalone rerun (`8 passed`) and full `pytest -q` rerun (`1059 passed, 15 skipped, 2 xfailed`) both passed without code changes.
3. **`gold.py:71` `harness_review` `list` `pair_key` bug caught and fixed during `P6` Step 5**: On real v2 silver episodes (`silver.py:272`), `r["pair_key"]` is deserialized as a Python `list`; `ctx["void_streak"].get(r.get("pair_key_str"), ctx["void_streak"].get(r.get("pair_key"), 0))` eagerly evaluated the second `.get(...)` argument and raised `TypeError: cannot use 'list' as a dict key (unhashable type: 'list')`. Fixed by extracting `_void_streak_of(r, ctx)` in `src/digital_twin_armigo/governance/gold.py` and adding regression test `test_harness_review_handles_list_pair_key_and_void_streak` in `tests/test_gov_gold.py` (`3faacae`).
4. **`scripts/governance/lake.py` `--db` subparser default override caught and fixed during `P6` Step 5 verification**: Passing `parents=[common]` (with `default=DEFAULT_DB`) to both the parent parser and every subparser caused `lake.py --db C:/scratch/gov-dry/lake.sqlite <subcmd>` to overwrite `args.db` with `DEFAULT_DB` (`E:/Armigo_data_v3/governance/lake.sqlite`). Fixed in `scripts/governance/lake.py` by separating `root_common` (`default=DEFAULT_DB`) from `sub_common` (`default=argparse.SUPPRESS`), added regression test `test_lake_cli_db_flag_before_and_after_subcommand` in `tests/test_run_pair.py` (`3faacae`), removed `E:/Armigo_data_v3/governance/lake.sqlite` and empty `E:/Armigo_data_v3/governance/`, and re-ran Step 5 cleanly into `C:/scratch/gov-dry/lake.sqlite`.

### Skipped Steps (`Part P`)

- **P6 Step 6 (`scripts/governance/capacity_bench.sh`)**: Multi-seed GPU capacity throughput benchmark skipped per handoff scope.


---

## Overall Summary & §7 Acceptance Verification

### Consolidated Skipped Steps (GPU / Live / Optional)

1. **H6 Step 8**: Live counterfactual GPU/LLM evaluation runs skipped.
2. **C7 Steps 3–5**: Smoke and full CV-motion matrix GPU runs with defect detector (`scripts/experiments/cv_motion_matrix.sh`) skipped.
3. **U2 Step 5**: GPU training of `BrickSegUNet` on `E:/Armigo_data_v3/perception/brickseg-v1` skipped.
4. **U4 Step 4**: Offline `eval_brick_pose.py` on trained formal ONNX and `cv_motion_matrix.sh stage3b` GPU runs skipped (smoke-verified on `brickseg_tiny.onnx`).
5. **U5 Step 2**: Optional Studio video sampling and manual visual review notes skipped (`realism_compare.py` smoke-verified on `real_ref`).
6. **P6 Step 6**: Multi-seed throughput benchmark (`scripts/governance/capacity_bench.sh`) skipped.

### Items Requiring User / Claude Decision

1. **H7 `config/agent_bench/diagnosis_bench/v1.json` size (`1` entry, `< 40`)**: `D:/Armigo_data_v2/audit/reviews.sqlite` currently has only `1` confirmed human audit (`calib=1, heldout=0`). Collect more confirmed audits before re-running `make_diagnosis_bench.py` and benchmarking `diag_mcp`.
2. **H3 `view_belief_mm` weak precursor AUC (`0.727 < 0.75`)**: Recorded in `config/experiments/precursor_thresholds.json`; decide whether to retain, refine, or drop `view_belief_mm` in precursor attribution.
3. **C1 Step 5 `cameras="drift"` `BAD_FOOTPRINT` (`4/4` at `tilt_deg = 33.21°–33.30°`)**: LAB wrist localization fails the 30% footprint gate at ~33° tilt as expected; depends on scheduling `U2 Step 5` GPU training + `unet-pnp`/`unet-rect` eval (`U4 Step 4`).
4. **`diag_mcp.compare_runs` nested `detector.elapsed_ms` sensitivity**: `compare_runs` (`diag_mcp.py:28`) strips top-level `{"wall_time_s", "latency_s", "seq"}` while `view` events nest `detector.elapsed_ms` (`twin_mujoco.py`). Decide if `compare_runs` should leave nested timings verbatim or ignore them during live trajectory diffing.
5. **`docs/GOVERNANCE.md` §6 Open VLA Policy Decisions (from Paired Governance Plan §2.4)**:
   - Sampling / loss weight for `recovered=True` episodes during VLA student training.
   - Whether and in what ratio to mix `privileged_demos_v3` (`truth_only` trajectories) when retraining VLA policies.
6. **GPU execution schedule (once training hardware frees)**: Recommended sequence `U2 Step 5` -> `U4 Step 4` -> `C7 Steps 3–5` -> `H6 Step 8` -> `P6 Step 6`.

### §7 Acceptance Checklist Checks (Observed Commands & Outputs)

| Checklist Item | Verification Command | Observed Output |
| :--- | :--- | :--- |
| Worktree isolation & unpushed `feat/failure-observability` | `git -C E:/digital_twin_armigo log -n 1 --oneline`; `git -C E:/digital_twin_armigo-v3 status --short` + `log -n 1 --oneline`; `git -C E:/digital_twin_armigo-obs branch -vv` | `E:/digital_twin_armigo` HEAD `0b28091` (unmodified); `E:/digital_twin_armigo-v3` clean at `16d17e7`; `E:/digital_twin_armigo-obs` on local `feat/failure-observability` (no remote tracking/push) |
| Task commits & full test suite pass | `E:/digital_twin_armigo/.venv/Scripts/python -m pytest -q -rs` (post-`3faacae`) | `1063 passed, 15 skipped, 2 xfailed, 16 warnings in 1456.51s (0:24:16)` |
| `ROOT_CAUSES` unchanged & `p2-dev-r2` rescore | `git -C E:/digital_twin_armigo-obs diff 16d17e7..HEAD -- src/digital_twin_armigo/experiments/triage.py`; H2/H3 read-only rescore on `p2-dev-r2` (`precursors.py`/`triage.py` untouched since `520be0d`) | Empty diff (`triage_diff_len=0`); H2/H3 `p2-dev-r2` counts match §0 (`none: 166, recovery_failure: 23, evidence_insufficient: 15, grasp_or_path_failure: 5, target_face_not_in_view: 3, placement_failure: 2, false_positive_visible: 2`) |
| Truth diagnostics guard & oracle eligibility | Post-edit full `pytest -q -rs` (covers `tests/test_truth_diagnostics.py`, `tests/test_counterfactual.py`) | `assert_sensor_only` enforces `TRUTH_DIAGNOSTIC_KEYS`; oracle runs mark `student_eligible=False` (included in `1063 passed`) |
| Default options bitwise invariance | `inspect.signature(TwinMujoco.__init__)`; `Select-String runner.py`; unit invariance tests across `C2–C6`, `U3`, `U4b` | `TwinMujoco.__init__` defaults: `motion='open'`, `evidence='hold'`, `belief_cameras='actual'`, `segmenter='lab'`, `oracle=()`; `runner.py` defaults match; Claude to run cross-commit `p2_dev12` `compare_runs` check at acceptance |
| `wrist_localize` uses designed mount (`G10`) | Post-edit full `pytest -q -rs` (covers `tests/test_wrist_localize.py`) | Verified `believed_wrist_from_cam` equals designed mount under nonzero wrist camera jitter (included in `1063 passed`) |
| Tiny ONNX in `cv2.dnn`, no tracked real photos/weights, read-only `hackjunction` | `git ls-files tests/data/ \| Select-String brickseg`; `test_tiny_onnx_loads_in_cv2_dnn` in full `pytest -q -rs`; `Get-ChildItem -Recurse -File hackjunction` | `tests/data/brickseg_tiny.onnx` (`133,113` B) + `brickseg_tiny.json` (`427` B); `cv2.dnn` passed in `1063 passed`; `hackjunction` untouched (`107456` visible files before & after) |
| Trainable gold isolation invariants & deterministic manifest | `pytest tests/test_run_pair.py tests/test_gov_*.py -v` + full `pytest -q -rs` | `24 passed` in governance suite + `1063 passed` overall: trainable products reject `test` split, quarantined, `oracle`, and `realistic_only`; rebuilds produce byte-identical `manifest.jsonl`; `failure_bank` rejects `test` split |
| Preregistration docs committed before matrix/eval code | `git -C E:/digital_twin_armigo-obs log --oneline -- docs/experiments/` | `46c50be` (`obs C7a`) precedes `092bca7` (`obs C7`); `3404d9d` (`obs U4a`) precedes `0a1660f` (`obs U4b`) |
| No staging `extend_path` shims & clean formal output scope | `Select-String src/digital_twin_armigo/**/*.py -Pattern extend_path`; `Test-Path E:/Armigo_data_v3/governance` | `extend_path_matches=0`; `misplaced_gov_exists=False`; formal artifact solely `E:/Armigo_data_v3/perception/brickseg-v1` |


## Part WP（两轮独立审核整合修复 WP1–WP9）

审核来源：`feat/smolvla-static-v3@16d17e7` 的独立审核、`feat/failure-observability@1b6addb` 的独立审核和整合改进计划（`C:/scratch/obs-dev/independent-audit-1b6addb/`）。WP1–WP6 由 Antigravity 实施并提交；它的额度用尽后，WP7–WP9 由 Claude 接手完成。WP9 的代码由一个 DeepSeek 子代理编写，Claude 复核后提交。

### 每单提交

| 工作单 | 提交 | 改动文件 | 新增测试 | 红灯核对 |
|---|---|---|---:|---|
| WP1 保护来源视频与人工裁定历史 | `d99bb2a` | 7 | 5 | Antigravity 记录 |
| WP2 统一划分、冻结与封存守卫 | `da27de0` | 7 | 4 | Antigravity 记录 |
| WP3 配对覆盖与逐批分母 | `67ac74c` | 5 | 4 | Antigravity 记录 |
| WP4 预检、学生裁剪可见性、MCP 真值泄漏 | `1192a77` | 9 | 10 | Antigravity 记录 |
| WP5 学生输入契约、K 碗探针、导出/视频 | `f674a9e` | 11 | 6 | Antigravity 记录 |
| WP6 复现判定、反事实主键、F1–F4 | `c08c1dc` | 10 | 19 | Antigravity 记录 |
| WP7 闭环重试与密集取证 segment | `2a236b7` | 3 | 9 | 在 `c08c1dc` 上 7/9 失败 |
| WP8 U4 顶视/翻转/手中评测与辅助元数据 | `5530d18` | 4 | 6 | 在 `c08c1dc` 上 6/6 失败 |
| WP9 C7 矩阵单一来源与 `--plan-only` | `2300f3d` | 8 | `tests/test_cv_motion_plan.py` | 修复前 `--suite` 路径与 `-ehold` 组名均不可运行 |
| 全量后测试修正（WP4/WP5 遗留） | `acd9a7a` | 2 | 0 | 两条测试在 `1b6addb` 通过、在 `c08c1dc` 失败 |

“新增测试”一栏统计的是新增的 `def test_` 和 `parametrize` 行。

### WP7 核对（Claude 接手）

- 生产入口 `TwinService → ExpertTracer` 下，`dense-10` 和 `dense-20` 都覆盖两条路径：负分走到扫视末端再静止补拍，以及缺陷早停。两条路径都以 SUCCESS 结束，`actions.npz` 的 `segment` 列没有 `-1`，`segment_labels()` 通过。
- `w2s` 经过真实的 `_retry` 之后，`expert.recheck` 仍然非空，二次抓取产生了 `wrist_fix` 事件。`restore_state` 恢复 open 检查点时，`recheck` 被清空。
- 默认 `open + hold` 逐位不变：用固定 CPU 评分器，`c08c1dc` 与 WP7 各跑 8 个回合（策略 A/B × 案例 700001-N、700001-D、700002-D、600001-D），`actions.npz` 所有数组和 `comparable_events` 全部相同，0 处不一致。产物在 `C:/scratch/obs-dev/wp7-baseline/{head,new}`。
- 相关回归：14 个文件，120 passed、3 skipped、1 failed。失败的是 `test_checkpoint_resume.py::test_real_detector_resume_from_disk_exactly`，它在 `c08c1dc` 上同样失败（`assert 'FAILED' == 'SUCCESS'`）。原因是 `CUDA_VISIBLE_DEVICES=""` 下真实检测器无法给出原结果（审核已记录评分器硬编码 CUDA autocast），与 WP7 无关。

### WP8 核对

- `reconstruct_brickseg_aux.py` 以 brickseg-v1 manifest 记录的提交 `230ccde` 加载 `scene_v3`/`visibility`，对 val 和 ood-appearance 重建辅助元数据。2338/2338 帧对齐，max_pos/rot/cam 差 5e-7，没有修改任何分片，产物在 `C:/scratch/brickseg-dev/aux-v1/aux_meta.json`。
- 顶视帧数：val 493 帧（含翻转/侧放 53 帧），OOD 399 帧（含 51 帧），与审核数字一致。
- 用微型 ONNX 在正式 val/OOD 上跑冒烟：评测 2338 帧，各分表分母都非零，报表在 `C:/scratch/brickseg-dev/eval-tiny-v1/`。微型模型没有训练，预注册门槛判 FAIL 是预期结果，正式指标要等 U2 GPU 训练。
- 需要注意的数字：`lab` 基线在 val 腕相机上只成功 12.7%（103/811），失败原因以 BAD_FOOTPRINT 为主（478）。这和本报告 C1 Step 5 的已知结论一致：约 33° 倾角下侧面并入色块。这不是评测 bug，但意味着 U4 的“不劣于 lab”对照门槛偏低。
- 同色杂物分层很不均衡：val 1020/1304、OOD 811/1034 帧判为 `same_color=True`，主要来自干扰盒的色相判定（饱和度 ≥0.15 且色相差 ≤35°）。这个分层只作信息用途，定义是否合适需要人工确认。

### WP9 核对

- `PLAN_ONLY=1 bash scripts/experiments/cv_motion_matrix.sh all` 的输出是 `plan validation passed: stage1 stage2 stage3 stage4`。`--plan-only --stage stage3b` 退出码为 0。用 `WINNER_MOTION=w9x` 注入错误胜者时，退出码为 1，并报 `winner motion 'w9x' is not part of the stage2 matrix`。
- Stage 2 为 7 种 motion × {fixed, recal, drift}，共 21 次运行。对比组是 `realistic-drift-B-v7bot` 对 `realistic-drift-B-v7bot-m<motion>`，不再出现 `-ehold`。
- `loop_01_96` 已登记（sha256 `972680c0…`，96 例，48 布局，split 均为 loop-01，与 p2_dev12 不重叠）。`runner._freeze_suite` 对 `p2_dev12` 和 `loop_01_96` 均冻结成功。
- **额外发现（既有问题）**：在 Windows 检出（`core.autocrlf=true`）中，`config/agent_bench/v2/*.json` 是 CRLF，而 manifest 的 sha256 是按 LF 字节算的。因此在 v3/obs worktree 里，`runner._freeze_suite` 会拒绝所有 v2 套件，`make_suites.py --check` 也会失败。现已在 `.gitattributes` 里把 `config/agent_bench/**/*.json` 和 `scripts/experiments/*.sh` 固定为 LF，并重新检出了这些未修改的文件；索引内容没有变化。
- 预注册文档在任何 cv-motion 运行之前修订：Stage 2 加入 recal，Stage 4 套件写成 `loop_01_96`。第 3 节判据原文没有改动。

### WP3 演练库对账（Antigravity 记录，在 scratch 副本上执行）

- 513 行 v2 演练库：`eligible_latest=510`，`represented=510`，`unrepresented=0`。原先缺失的 154 行全部有了显式归类。
- 跨批次 144 条因严格的 `suite_sha256` 不一致标为 `suite_sha256_mismatch`，按历史子集口径是 `p2-dev` 72 对加 `p2-dev-r2` 72 对。`realistic_only=0`。

### F3 原始复现声明更正

第一轮审核在干净提交上把 `L700007-D0` 重跑了两次。第 372 帧腕部裁剪有 13 个像素相差 1 个灰阶，检测分数落在零阈值两侧，两次分别以 1099 帧 `UNKNOWN` 和 1461 帧 `FAILED` 结束。v3 报告中“`L700007-D0` 可复现”的说法不成立。F3 剔除该例是正确的；`L700006-N` 两次运行一致，保留。

### 全量测试（WP9 提交后，CPU only）

命令：`PYTHONPATH=src CUDA_VISIBLE_DEVICES="" E:/digital_twin_armigo/.venv/Scripts/python -m pytest -q -rs --durations=30`（在 `2300f3d` 上运行，日志在 `C:/scratch/obs-dev/wp-full-pytest.log`）

```
3 failed, 1137 passed, 15 skipped, 2 xfailed, 16 warnings in 1534.51s (0:25:34)
```

三项失败的归因与处理：

| 测试 | `1b6addb` | `c08c1dc` | 原因 | 处理 |
|---|---|---|---|---|
| `test_checkpoint_resume.py::test_real_detector_resume_from_disk_exactly` | — | 失败 | 加载真实检测器；`CUDA_VISIBLE_DEVICES=""` 下得不到原结果 | 既有问题，未改（待决策 3） |
| `test_twin_agent.py::test_recorded_camera_observation_precedes_its_action_and_rewound_trace_is_rejected` | 通过 | 失败 | WP5 让旧五相机导出默认不带 `mask.noisy`（为了合并兼容），但这条测试的期望没有同步更新（`test_action_noise.py` 已改） | 按 WP5 契约改正期望集合 |
| `test_vla_paired_carry.py::test_v3_three_bowl_run_branches_smoke` | 通过 | 失败 | WP4 把 `light_k` 接入光源颜色（色偏约 ±3%，且光照值精度从 2 位小数变为 3 位），v3 渲染随之改变；seed 3000001 的父回合在放置时两次 `no_clamp`。布局本身（种子 3000001–3000003、3100001、3500001）与 WP3 完全相同 | 见下 |

教师成功率对照（v3 train，种子 3000001–3000012，正负交替，策略 B，固定评分器，realistic）：WP3（`67ac74c`）为 9/12，当前为 11/12。失败落在哪些种子会随光照的微小变化而改变：WP3 失败的是 3000005 `no_clamp:place`、3000008、3000012 `BOWL_NOT_FOUND`，当前失败的是 3000001 `no_clamp:place`。色温没有让成功率下降，但教师对 1% 量级的光照变化敏感，与 `L700007-D0` 的 1-LSB 现象同类。这条测试检查的是分叉逻辑，所以改为依次尝试前 3 个三碗候选、取第一个父回合合格的布局，断言没有放宽。

测试修复提交为 `acd9a7a`，这两条测试单独复跑结果是 `2 passed`。复跑时还发现，WP4 收紧学生裁剪可见性之后，v3 train 种子 3000001–3000200 中有 7 个（3000042、3000085、3000104、3000130、3000142、3000169、3000186）在 20 次相机重采样内都无法通过，`sample_case` 会抛出 RuntimeError。`make_scene_suite_v3.py` 会捕获这种情况并丢弃这些种子，属于设计行为；但直接调用 `sample_case` 的代码需要自行处理。修复后没有再跑一次全量：本轮只改了上面两条测试，其余 1137 项的结果仍以 `2300f3d` 上的那次全量为准。

### 隔离与默认值

- 所有提交都在 `feat/failure-observability`，没有推送或合并。`E:/digital_twin_armigo`（`main@0b28091`）和 `E:/digital_twin_armigo-v3`（`feat/smolvla-static-v3@16d17e7`）的工作区与分支没有变化。
- `ROOT_CAUSES` 没有改动。WP6 只新增了只读的 `ROOT_CAUSE_TO_MECHANISM_COMPAT` 映射，并有测试保证它覆盖全部 `ROOT_CAUSES`。
- `motion`、`evidence`、`belief_cameras`、`segmenter` 的默认值都没有改。
- 没有启动 GPU 任务，没有安装包，没有重新渲染 U1 正式数据，`D:/Armigo_data_v2` 和 hackjunction 都只读。

### 阶段结论

- **本轮独立复现通过**：WP7 和 WP8 的红灯/绿灯、默认行为逐位对照、brickseg-v1 辅助元数据对齐、C7 `--plan-only` 全阶段校验。
- **待人工决策**：
  1. U4 中 `lab` 基线只有约 13% 的成功率，“不劣于 lab”是否仍作为 OOD 门槛；
  2. 同色杂物的判定口径；
  3. `test_real_detector_resume_from_disk_exactly` 在只用 CPU 时必然失败，是否要在没有 CUDA 时改为 skip（本轮没有改动）。
- **待 GPU 排期**：C7 Stage 1–4（含 3b）、U2 正式训练、U4 Step 4 正式评测、P6 Step 6 容量测试、H6 Step 8。

## Part S（腕部重定位：silhouette、检查点分层集、去除场景标记，2026-11-25）

### 改动（`feat/failure-observability`，未推送）

| 提交 | 内容 |
|---|---|
| `d3c81ed`、`9370605`、`62b7c23` | `segmenter="silhouette"`：已知长方体的三自由度（x、y、yaw）轮廓拟合，带阴影容忍前景和桌面高度假设；门限只约束拟合中心位置，不裁切计数区域 |
| `1a711bf` | 检查点分层集生成器（悬停，以及 0.050/0.045/0.040/0.035/0.030 m 五个高度）和同色干扰物分档 |
| `9370605`、`6461bdd` | 评测器：`closed_loop` 与 `oracle_camera` 两种输入、按高度和干扰物分层、门槛 |
| `540fd93`、`26c93dc` | 探索性 CPU 闭环（固定评分器，不计入判定） |
| `ce10a75`、`aadc1f3` | 用户决定：相机外参校正之后移除场景里的打印标记（10/11/16/17），校正结果逐位不变；step-MCP 旧值重新固定（亚毫米级、几个像素的变化） |
| `a55f6c4` | 标定卡纹理只在内容变化时才写入（并行时曾读到空 PNG） |

### 离线：`wrist-checkpoint-v2`（无标记，200 个案例，942 帧）

`silhouette`（修正后）；`lab` 全部输入下的成功率都低于 9%。

| 组 | 成功率 | 完整可见帧上的成功率 | xy 中位数 / p95（mm） | yaw 中位数（°） | 超过 3 mm |
|---|---:|---:|---|---:|---:|
| val · oracle_camera | 86.5% | 94.2% | 0.39 / 1.01 | 0.67 | 0 |
| ood · oracle_camera | 85.5% | 93.2% | 0.43 / 1.06 | 0.63 | 0 |
| val · closed_loop | 85.8% | 93.5% | 3.82 / 7.02 | 1.09 | 276 |
| ood · closed_loop | 80.3% | 87.3% | 3.51 / 6.38 | 1.03 | 236 |

- `closed_loop` 与 `oracle_camera` 的差距来自信念误差，不是分割误差。`drift` 下，顶相机估计的高度偏差为 5–12 mm，腕部安装偏差为 0.8–2.3°（不可观测），后者单独就带来约 4 mm 的误差。
- 第 3 条门槛“成功率 ≥ 98%”目前未通过。约 10% 的帧里积木有一部分在画面外，按设计应当拒绝，所以全帧成功率的上限约为 90%。口径需要用户决定（见 prereg 第 9 条）。
- 修正前（门限裁切了计数区域）：val 成功率 66.5%，完整可见的帧中有 107 帧被误判为 `CLIPPED`。

### 压力集（`wrist-checkpoint-stress-v2`，只作报告）

- 场景：在距积木 3–6 cm 处放一块同色 2×2。
- 目标是 2×4 时，`oracle_camera` 下没有误锁。
- 误锁（接受的定位偏差超过 10 mm）几乎都出现在目标也是 2×2 的情况，即与杂物完全相同。`oracle_camera` 下，杂物在门限圈内时 115 次接受中有 67 次锁错。
- 先验偏差约 20 mm，而杂物在 3–6 cm 处，这种歧义单帧无法分辨，换成 UNet 同样无法分辨。
- 可选对策：两个候选拟合都好而且相距较远时，返回 `AMBIGUOUS` 并放弃定位，而不是选离先验最近的一个。尚未实现，待决定。

### 探索性闭环（`p2_dev12`，24 回合每组，固定评分器，CPU）

| 相机 | open | w2s-lab | w2s-silhouette | silhouette 接受定位的误差中位数 |
|---|---:|---:|---:|---:|
| drift | 13/24 | 13/24 | **21/24** | 4.05 mm |
| fixed | 23/24（1 次基础设施失败，已修） | 24/24 | 22/24 | 0.53 mm |

- `lab` 在腕部视角下一次也没有被接受（`BAD_FOOTPRINT` 或 `CLIPPED`），所以 w2s-lab 的结果与 open 逐回合相同。
- drift 下 silhouette 的 3 个失败：
  - 700008 的两个回合，先验偏差约 21 mm，积木确实出了画面；
  - 700001-N，接受的定位偏差为 6.8 mm，属于信念误差。
- fixed 下 silhouette 的 2 个失败都在种子 700001 和 700006 上，但两次运行中失败的回合不同。单独复现抓取（真值位姿、真值加 ±1.7° yaw、真实 silhouette 输出）全部成功。原因待查。
- 抓取余量不是单调的：700001 在真值 x 方向偏 −1 mm 时会失败，在先验偏差 2.4 mm 时反而成功。所以“误差更小”并不等于“抓取更多”。

### 全量测试

- 去除标记后，全量测试有 2 项失败：
  - step-MCP 的固定旧值；
  - `test_twin_leakage` 修改了调用方传入的 layout，而 twin 现在持有一份副本。
- 两项都已在 `aadc1f3` 修正，并在开启标记时验证过原测试通过。

### 待决策

1. 门槛的输入口径：`oracle_camera` 还是 `closed_loop`（prereg 第 7 条）。
2. 成功率口径：完整可见帧，或者降低阈值（prereg 第 9 条）。
3. 同色同尺寸杂物：是否加入 `AMBIGUOUS` 拒绝。
4. 信念误差（顶相机高度、腕部安装）是下一步闭环成功率的主要瓶颈，需要另立工作项。
