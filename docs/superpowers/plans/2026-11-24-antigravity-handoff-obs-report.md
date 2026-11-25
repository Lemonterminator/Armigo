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
