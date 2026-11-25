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
