# Bowl Experiments Cost Estimate (Task 0)

- Single-episode wall clock (`baseline_2x2`, seed 1): `0.736 s` (`SUCCESS`)
- `bench_physics_expert.py` is single-process (`for s in seeds`), while `Cell.__init__` writes `_sorting_cell_{os.getpid()}.xml` per process. Parallelizing across 12 workers with `ProcessPoolExecutor` (pre-generating marker textures first) is PID-safe.
- Estimated wall-clock times (per candidate that reaches each gate):
  - G0 (Geometry raycast + profile): `< 0.5 s`
  - G1 (Collider micro repro, 400 + 1 poses, `mj_forward` only): `< 1 s`
  - G2 (Static soak, 4 conditions × 200 drops × 4 s @ 240 Hz): `~5–10 s`
  - G3 (Two failing conditions, 120 episodes): `~88 s` serial, `~10–15 s` with 12 workers
  - G4 (Full regression: 540 matrix/held-out + 120 layouts = 660 episodes): `~490 s` serial, `~50–70 s` with 12 workers
  - G5 (Determinism 3×2 episodes + timing): `~5 s`

## Baseline Summary (`results/physics-expert-v2/independent_acceptance`)

- Matrix + held-out: `538 / 540` (`high_mass`: `59/60`, `low_friction`: `59/60`, all others `60/60`)
- Layouts (`seed1000`, `seed1001`): `119 / 120` (`layouts_seed1000_idx32` failed)
- `jaw_env_over_5N`: `6` (in the 660-episode `independent_acceptance` suite; `9` across all 840 episodes in `results/physics-expert-v2/**/*.json`)
- `frames_ge_890`: `1` (`layouts_seed1001_idx37`; `2` across all 840 episodes)
