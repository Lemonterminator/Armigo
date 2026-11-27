# Running the tests

The suite has about 1,200 tests. Almost all of the time goes to a few dozen MuJoCo end-to-end tests (render, perceive,
record video, replay); the rest finish in a few minutes together.

| Command | What it runs | Wall time (measured 2026-11-27, 32 threads, another sim batch running) |
|---|---|---|
| `pytest` | everything, one process | 48 min |
| `pytest -n 6` | everything, 6 worker processes | 10 min |
| `pytest -n 6 -m "not slow"` | everything except the 67 `slow` tests | 3 min |

`pytest-xdist` is in the `dev` dependency group (`uv sync --group dev`). Each worker opens its own MuJoCo GL context and
uses one core, so choose `-n` by how busy the machine is (a running simulation batch needs its cores too); `-n auto`
starts one worker per logical CPU, which is more than this suite can use. The test that bounds a parallel run is
`test_vla_paired_carry.py::test_v3_three_bowl_run_branches_smoke` (5 to 7 minutes on its own); under xdist
`tests/conftest.py` starts the longest tests first so the run does not end waiting on one of them.

`slow` is the marker for tests that take 10 s or more. Besides the explicit `@pytest.mark.slow` decorators,
`tests/conftest.py` marks every node id listed in `tests/slow_tests.txt`. Refresh that list from a serial run:

```
pytest --durations=0 --durations-min=10 -q
```

The tests are process-safe: each MuJoCo cell writes its scene XML as `_sorting_cell_<pid>.xml` and marker textures are
written atomically, and tests write only under `tmp_path`.

## Where the time goes

Profiled on 2026-11-27 (`cProfile`, one process). In the long end-to-end tests the time is in native code: lossless
video encoding and re-encoding (PyAV, about 40% of the slowest test), MuJoCo rendering and pixel read-back, and
NumPy image arithmetic for the D415 depth-noise model. Two pure recomputations were removed: the per-pixel camera ray
grid in `depth.py` was rebuilt for every frame (now cached per camera and crop, bit-identical), and the failure ledger
parsed each episode's 18 MB `replay/scene.json` twice (now once).
