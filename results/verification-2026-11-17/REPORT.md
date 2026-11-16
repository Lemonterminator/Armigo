# Refactor and performance verification — 2026-11-17

## Scope and provenance

Continued the interrupted refactor without resetting existing edits. Baseline is commit `6f85feb4f4a33d8f7cf37a00e74c224904140bc3`; the saved baseline simulation script was SHA-256 checked against that commit (line endings normalized). This report covers the current working tree, not a new commit.

The 1,600-line MuJoCo script is now a compatibility launcher over scene, layout, cell, IK, expert, views, calibration, pipeline and CLI modules. Camera geometry, joint conversion, constants and depth encoding are shared with Isaac. The public launcher imports remain available. New optional dependency groups and `armigo-sim` support source/editable installs; standalone wheels do not include repository assets.

## Verification

- Hardware-free tests: **32 passed** in the repository Python 3.14.4 environment.
- Ruff `F,E9,I001`: passed for `src`, `tests`, `scripts/sim` and the three changed Isaac stages. An unrelated existing I001 in `scripts/isaac/prepare_assets.py` remains outside this change.
- `git diff --check`: passed.
- Runtime interpreter: `E:/il/v/Scripts/python.exe`, Python 3.12.10, MuJoCo 3.12.0, Isaac Lab 3.0.0, PyTorch 2.11.0+cu130, RTX 5090. This environment has no pytest; runtime verification uses the actual CLIs below. The repository test environment has no MuJoCo/Isaac.
- MuJoCo: three seed-0 episodes, 567/612/647 frames, all successful. Every NPZ array and JSON metadata value matches baseline exactly. Running with two Windows spawn workers gives identical simulation outputs.
- Render: all three episodes, one view, seed 1, stride 10 (184 output frames). State, actions, phase, depth, visibility counts, metadata, file sets, shapes and dtypes match exactly.

### RGB control experiment

RGB is **not bitwise identical**. On the same GLFW backend, baseline versus baseline changes 3,370 / 456,720,384 color-channel values (0.000738%); baseline versus refactor changes 3,514 (0.000769%). Both have maximum absolute error **1/255**. One worker versus two changes 3,740 values, also at most 1. This control supports treating the observed differences as render variability; it does not establish bitwise RGB determinism.

The gate explicitly permits `--rgb-atol 1` only for uint8 `front`, `wrist` and `top_rgb_v2`. Everything else remains exact. Strict mode was also run and returned the expected exit code 1 for these RGB differences (`compare_render_strict.json`). `scripts/sim/compare_outputs.py` returns exit code 1 on mismatches, unlike the prior temporary comparison script which printed DIFFERENT but returned 0. Regression tests cover changed state, RGB tolerance, missing files, empty inputs, dtype changes and metadata changes.

The previous full stride-2 outputs (914 frames) were independently rechecked: exact non-RGB values and metadata, with 20,220 / 2,268,708,864 RGB values differing by at most one level. See `prior-render.json`.

## Timing

One fresh sequential measurement per configuration; wall time includes process startup, rendering and file compression. All runs used the same interpreter and `MUJOCO_GL=glfw`, without concurrent Isaac workloads. These are local measurements, not repeated statistical benchmarks.

| Workload | Baseline | Refactor, 1 worker | Refactor, 2 workers |
|---|---:|---:|---:|
| Simulate 3 episodes | 25.527 s | 3.809 s | 3.206 s |
| Render 184 frames | 22.738 s | 7.742 s | 5.744 s |

Single-worker improvements: **6.70x simulation**, **2.94x rendering**. Baseline render repeat: 22.685 s.

`timings.json` records exact commands, times and return codes; `run_benchmark.py` records the executed harness and baseline location. Raw recordings are retained locally in ignored `data/`; comparison JSON files are versionable.

## Hot paths and C++ decision

Recovered baseline profiles are preserved beside this report. The earlier four-episode simulation cProfile attributed 29.567 / 32.164 seconds cumulatively to visibility, including 27.647 seconds in that method. Its NumPy comparison with a pybind11 enum caused per-pixel Python overhead. Converting the enum once to int removes that path. The refactor also caches joint IDs, limits, segmentation lookup tables and top-camera rays; batches MuJoCo substeps where equivalent; reuses live data for IK with state restoration; and removes redundant RGB copies.

The old microbenchmark measured the enum comparison at 114,199 microseconds versus 97.8 microseconds for int comparison (full image, different from end-to-end timing). Physics and rendering already execute in native MuJoCo. A fresh three-episode profile took 3.916 seconds: visibility used 0.235 seconds of Python self-time / 1.467 seconds cumulative, renderer calls 1.219 seconds cumulative, and native `mj_step` 0.712 seconds. See `current-simulation-profile.txt`. The end-to-end gains do not justify a new C++ extension or its build/maintenance cost at this point.

## Isaac

Stage 1 ran headless and exited 0. Camera focal error: 0.00006104 px; maximum grasp reprojection error versus MuJoCo: 0.00008548 px; median error versus recorded real grasp points: 12.508 px. Six joints and a 480x640 wrist RGB frame were verified. Real reference PNGs are absent, so silhouette IoU is unavailable. Nonfatal Kit/extension warnings remain in the local log. Fresh metrics are copied to `isaac-stage1-metrics.json`; camera images and logs remain under ignored `isaac/stage1/` and `isaac/`.

Stage 2 completed the default friction-contact pick/inspect/place episode with **success=true**, **grasped=true**, and **595 frames**. Both jaw contact checks passed; maximum brick lift was 0.259 m and final distance to the pink plate center was 0.0489 m. Its metrics explicitly identify `pure_friction_contact_sensor_no_fixed_joint`. This is one successful synthetic episode, not a real-world grasp guarantee. See `isaac-stage2-metrics.json`.

Stage 3 completed **16 environments / 20 benchmark steps** after three warm-up steps. Top RGB, depth and wrist RGB outputs have the expected 16-environment shapes; observations are `[16, 6]`; inter-environment pixel std is 4.254. The script measured 672.6 environment-steps/s over 0.476 seconds (excludes initialization and export). This short benchmark is not a sustained dataset-output throughput measurement. `torch_peak_vram_mb` measures Torch allocations only, not total RTX/Kit VRAM. See `isaac-stage3-metrics.json`.

Original Stage 1 output files were restored to their clean initial git state after copying fresh results; existing Stage 2/3 output directories were preserved. New large outputs/logs are under ignored verification subdirectories.

## Knowledge graph and limits

Incremental Graphify code extraction produced **478 nodes, 1,067 edges, 19 communities**, references to all 11 `mujoco_cell` source modules and no dangling edges. `graphify-out/graph.html`, `graph.json` and `GRAPH_REPORT.md` were regenerated. No external LLM was used for this refresh. Earlier semantic relationships are retained; the subsequent README usage additions and this verification report are not semantically re-extracted. Historical token cost in `cost.json` is inherited and was not independently audited; no new model-token measurement is available.

MuJoCo uses assisted grasping; Isaac Stage 2 exercised friction-contact grasping. These checks do not validate real robot performance, broad grasp robustness, Linux runtime behavior, or standalone-wheel asset packaging.
