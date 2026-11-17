# Handoff: how the failure data was found and exported (for data governance)

Repo: `E:\digital_twin_armigo` (MuJoCo 4.0.1, simulation only, nominal assumed physics). Everything below is untracked
in git (`results/failure-export/`, the two scripts). Python: `.venv/Scripts/python` (3.14); `uv` is not on PATH.
Do not use `python - <<EOF` heredocs with Windows paths (backslashes get mangled); write a script file instead.

## 1. Where the failures are recorded

Acceptance records (JSON, one per run) in `results/physics-expert-v2/`:

| file | content |
|---|---|
| `independent_acceptance/matrix_seeds0-29_with_held_out.json` | 9 conditions x seeds 0-29 |
| `independent_acceptance/matrix_seeds100-129_with_held_out.json` | 9 conditions x seeds 100-129 |
| `independent_acceptance/layouts_seed1000.json`, `layouts_seed1001.json` | 60 random layouts each (`sample_layout`) |
| `physics_expert_seeds0-29.json`, `physics_expert_layouts_seed{1,2,7}_n60.json` | the implementer's own runs |

Schema: matrix files have `conditions[<name>].episodes_detail[]`; layout files have `layouts.episodes_detail[]`.
Each episode has `verdict`, `reason_codes`, `info` (n_frames, release_peak_speed_mps, jaw_env_contact_max_N, aborted,
grasp_retries, grasp_geometry...) and `health` (full audited-evaluator output, evaluator `v2-audited`).

Selection rule (implemented in `scripts/export_failure_animations.py: locate()`):
- failure: `verdict != SUCCESS`
- flagged (verdict SUCCESS but over a threshold): release peak speed > 1.2 m/s, jaw-environment force > 5 N,
  or n_frames >= 890 (cap is 900)
- the same episode appearing in two files is exported once (`also_in`).

Result: 3 failures + 10 flagged.

| id | verdict | note |
|---|---|---|
| `high_mass_seed29` | FAILURE bounce_out | release 2.66 m/s |
| `low_friction_seed111` | FAILURE bounce_out | release 5.00 m/s |
| `layouts_seed1000_idx32` | FAILURE pick_miss | aborted no_clamp, 2 retries, jaw-env 23.1 N |
| 8 episodes | SUCCESS | jaw-env 5.3-17.2 N |
| 2 episodes | SUCCESS | 893 / 899 frames |

Episode id convention: `<condition>_seed<seed>` for matrix, `layouts_seed<layout_seed>_idx<NN>` for layouts.
Layouts are not stored: they are regenerated with `sample_layouts(60, layout_seed, top_cam)[idx]`
(deterministic from `layout_seed`).

## 2. How the animations were made

Tool: `scripts/export_failure_animations.py` (`--category failure|flagged|all`, `--only <id>`, `--list`).

1. Replay each episode with the same code as the benchmark: `bench_physics_expert.run_episode(cond, spec, top_cam, seed)`
   or `run_layout_episode(layout, spec, idx)`, with the `Cell` constructor patched to force `render_enabled=True`
   (the benchmark builds cells without a renderer). Rendering only reads state.
2. `PhysicsEpisode.step` and `ClosedLoopExpert.act` are monkeypatched to render after every 30 Hz frame and record
   the phase.
3. Frame = 2x2 grid of 640x480: dataset `top` camera, a fixed overview free camera, a chase camera following the brick,
   and the `wrist_cam`. Text overlay: frame, time, phase, brick speed/height, jaw normal forces, jaw-environment force;
   a red border when brick speed > 1.2 m/s. (There is no `front` camera in the compiled scene.)
4. Frames are streamed with OpenCV `VideoWriter` (mp4v), then transcoded with `ffmpeg` (libx264, yuv420p, crf 20).
5. Replay check: verdict, reason codes, frame count and release speed are compared with the recorded episode; all 13 matched
   exactly (the replay is bit-exact, so the animation is the recorded failure).

Outputs per episode in `results/failure-export/{failure,flagged}/`: `<id>.mp4`, `<id>.json` (recorded episode +
replay check + source file + flags), `<id>.npz` (per-frame telemetry: qpos, q_cmd, brick pose/velocity, pinch forces,
torques, phases, phase_names). Console log: `results/failure-export/export_run.log`.

## 3. Agent logs

Claude Code transcripts live outside the repo, in `C:\Users\Jiang\.claude\projects\E--digital-twin-armigo\`:
- main session `<session-id>.jsonl`; subagents in `<session-id>/subagents/agent-<id>.jsonl` with `.meta.json`
  (agentType, model, description) and `.prefix.json`.
- Session `904ca024-aa79-43d8-83e7-1fe27d5c9d21` holds the closed-loop expert work:
  `agent-acd5a06f78b65d5ca` (Opus 5.5, implementer, both rounds), `agent-a8fb67acf18aab1bc` (Sonnet 5.5, reviewer).
  The same session's `agent-a866a67219d42fab9` (snapshot store) and `agent-aced9b03a02df8d8e` (operation ledger)
  belong to stage two and were not exported.
- Identify the relevant agent by its `.meta.json` `description`.

Tool: `scripts/export_agent_logs.py --out <dir> NAME=<transcript.jsonl> ...` copies the raw `.jsonl` unchanged and writes
a readable `.md` (user/assistant text, tool calls, tool results cut to 4000 chars; thinking blocks in `<details>`).
Exported to `results/failure-export/agent_logs/` (`00_main_session_orchestrator`, `01_implementer_opus_...`,
`02_reviewer_sonnet_...`, plus `*.meta.json`). Raw jsonl also contains `api-request-blob` records (system prompts, tool
definitions); review before sharing outside the machine.

## 4. Physics diagnosis (follow-up)

`results/failure-export/physics_diagnosis/` has `REPORT.md`, the scripts and logs. Findings in brief: MuJoCo 4.0.1 native
box-vs-cylinder collision returns a 115 mm penetration for a brick on the 3 mm bowl-floor cylinder (high_mass_seed29);
a resting brick spins up and hops on the same floor (low_friction_seed111); in idx32 the moving jaw is blocked by a
bowl rim wall box. A box bowl floor gave 30/30 on those two conditions (not run on the rest). Read `REPORT.md` before
treating these as settled.

## 5. Suggested governance work

- A manifest (one row per exported item: id, category, flags, source JSON path + sha256, git commit / dirty state,
  MuJoCo and evaluator version, params hash, replay-check result, artifact paths + sha256, size).
- Record provenance of the replay code (the patches above are not in the repo's library code) and the dependency on
  `sample_layouts` determinism for layout ids.
- Decide retention/redaction for agent logs (full tool output, prompts, local paths, email in context) and whether
  the `.jsonl` or only the `.md` is kept.
- Everything is uncommitted; the repo has many other untracked stage-one files, so scope any commit to these paths only.
