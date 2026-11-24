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
