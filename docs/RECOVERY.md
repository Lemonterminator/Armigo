# Snapshots, execution service and recovery (stage two)

Scope: MuJoCo sessions on one machine. Everything here is verified by unit tests and MuJoCo runs in simulation only;
nothing has been validated on hardware, and Isaac is not covered (see "Not covered").

## Pieces

| Module | Role |
|---|---|
| `runtime/snapshot.py` | Content-addressed blobs (SHA-256, verified on every read) and immutable manifests. A manifest records run / episode / branch, parent snapshot, experiment identity (code, asset, config hashes; engine, Python, NumPy, platform), model fingerprint, task contract, scene specification and the producing job. |
| `runtime/ledger.py` | Append-only SQLite ledger: idempotency keys, job state events, leases with fencing tokens, stop latch, budget accounting. `UPDATE` and `DELETE` are blocked by database triggers. |
| `runtime/simulator.py` | Engine-neutral session protocol, `ControlProgram`, `SessionState`, state save / load / diff. |
| `mujoco_cell/session.py` | `MujocoSession`: steps a control program at 30 Hz with 240 Hz substep telemetry; captures and restores the full state. |
| `runtime/service.py` | `ExecutionService` (create / fork snapshot, submit / get / cancel rollout, stop, recover) and `RolloutWorker`. |
| `runtime/toy_sim.py` | Hardware-free reference simulator used only to fault-inject the service in milliseconds. No physical meaning. |

## What a snapshot holds

| State | Where | Why it is needed |
|---|---|---|
| MuJoCo integration state (`mjSTATE_INTEGRATION`: time, qpos, qvel, act, ctrl, applied forces, mocap, equality flags, userdata, plugin state, solver warm start), all bodies | blob `arrays/physics` | Dropping the warm start or the time changes the continuation (tests fail when either is removed). |
| Command filter: delay line, last output, backlash position | blob `arrays/controller` | In-flight commands and deadband position are not derivable from joint angles. |
| Assisted-grasp attachment and release counter (compatibility mode only) | `state.meta.cell` | Python-side state MuJoCo does not know. |
| Observation RNG: full bit-generator state | `state.meta.rng` | A seed replays draws that were already consumed. |
| Control program, cursor, program hash | blob `arrays/program`, `state.meta` | Trajectory cursor and targets. |
| Telemetry recorded so far | blob `arrays/telemetry` | The evaluator (`metrics.evaluate_episode_health`, `v2-audited`) is a pure function of telemetry, so this *is* the evaluator state: slip reference, release event, settle window and contact history are recomputed from it. |
| Model fingerprint (SHA-256 of the compiled model, including values changed in memory after loading) | manifest and `state.meta` | A restore into a different model is refused. |

Not stored, by design: renderer / GL contexts, connections, locks. Sessions here run without rendering and
observations are produced synchronously from the restored state, so there are no cached frames or in-flight
observations to restore.

Restore validates everything first (schema, engine version, fingerprint, shapes, finiteness, cursor, program hash) and
only then writes; a rejected restore leaves the session untouched.

## Commit protocol

1. The worker takes a lease (new fencing token) and appends `RUNNING` with the snapshot id it will write.
2. It runs on its own session, at chunk boundaries re-checking its lease and the cancel flag.
3. It writes blobs (staging file, fsync, re-read and hash check, rename into place).
4. It re-checks its lease, then commits the manifest under the announced id with an exclusive hard link.
5. It appends `SUCCEEDED`. Only this event publishes the snapshot. A manifest no `SUCCEEDED` event names is ignored by
   every service API, so a workflow checkpoint (LangGraph, stage three) must reference job ids / published snapshot
   ids only, and write its own checkpoint after step 5.

## Recovery rules

Call `ExecutionService.recover()` when the coordinator starts, or `recover([job_id, ...])` for workers known dead.
It treats every listed `RUNNING` / `CANCEL_REQUESTED` job as abandoned.

| Found after a crash | Action |
|---|---|
| Staging files, or blobs without a manifest | Ignored. |
| `RUNNING`, announced manifest exists, verifies, and was produced by this job under its last token | Old worker fenced; job adopted as `SUCCEEDED` (`result.recovered = true`). Not re-executed. |
| `RUNNING`, no such manifest | Old worker fenced; `NEEDS_RECOVERY`. Re-run with `RolloutWorker.run(job_id)` under a new token. |
| Manifest that no `SUCCEEDED` event names | Reported in `orphan_snapshots`; never usable, never deleted automatically. |
| `QUEUED` | Left queued. |
| Stop latch set | Stays set. Only `clear_stop(run_id, reason=..., actor=...)` releases it. |

Other rules the tests enforce:

- Same idempotency key and same request returns the existing job; same key with a different request is rejected.
  A finished job is never executed again.
- A superseded worker cannot write a manifest or a ledger event; it stops at its next chunk boundary.
- Identity mismatch (code, assets, config, engine / NumPy / Python version, platform, model fingerprint), a missing
  snapshot, or any failed hash is refused at submit and recorded as `FAILED` with a reason code at execution.
  There is no override: a changed source tree means a new root snapshot.
- Rollback is `fork_snapshot` onto a new branch; the parent is never modified. Budgets (`RunBudget`: rollouts and
  simulated seconds, plus the 60 s per-rollout limit) are computed per run from the append-only ledger, count
  cancelled and failed work and queued reservations, and are not reset by forking or restarting.
- Cancelling a queued job is immediate; a running job stops at a chunk boundary without an output snapshot. A cancel
  that arrives after the work finished records the real outcome (`SUCCEEDED`).

## Not covered

- **Isaac**: no checkpoint support here. The plan's commitment for Isaac is rebuild-and-replay, not state restore.
- **Hardware / latency model**: nothing in this layer is a real-time or real-robot claim.
- **Workflow state, LLM decisions, experience ranking**: stage three (LangGraph + MCP) on top of this service.
- **Parallel workers in one process**: `Cell` compiles its scene through a file named by process id, so concurrent
  workers must be separate processes. The ledger and store are safe across processes; the service has only been
  exercised with one coordinator.
- **Assisted-grasp mode** is restorable but is a compatibility mode: results carry `physics_mode = assisted_compat`
  and must be kept out of pure-physics rankings.
- **Scene specifications contain simulation truth** (defect flag and face). They belong in snapshots; anything shown
  to a planner must go through `inspection.sanitize_scene_for_planner`.
- **Portability**: asset paths are part of the model fingerprint, so a snapshot is tied to this checkout location.
- **Garbage collection** of orphan manifests and unreferenced blobs is not implemented.

## Test ownership

| Tests | Own |
|---|---|
| `tests/test_runtime_snapshot.py` | Blob / manifest integrity, atomic and exclusive commit, fork, identity check, tree hashing. |
| `tests/test_runtime_ledger.py` | Idempotency, full state-transition table, fencing, append-only triggers, stop latch, budgets, concurrency. |
| `tests/test_runtime_service.py` | Service on the toy simulator: resume across restart, crash before / after commit, stale workers, cancel races, refusal of mismatched or damaged snapshots, budgets across forks, stop latch. |
| `tests/test_mujoco_session_resume.py` | Paired straight-run vs save-restore on MuJoCo, bit-exact: undrained delay line, backlash deadband after a reversal, active velocity limiter, consumed observation noise, friction-only pinch, free fall after release, assisted attachment and release window, branch isolation, refusal of changed models and damaged state; service end to end on MuJoCo. |

Run: `uv sync --extra mujoco`, then `uv run --no-sync pytest`. Without the `mujoco` extra the MuJoCo file is skipped.
