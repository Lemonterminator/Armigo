# Armigo six-face inspection agent

The MuJoCo workflow runs a real friction grasp, obtains RGB observations, decides where to sort, verifies the motion
with the existing audited evaluator, and records demonstrations. Each episode uses the same task string. The goal
slot contains the inspection rule and the two bowl conventions.

## Run

From the repository root, with the Windows simulation venv and the separate anomaly venv provisioned:

```powershell
uv sync --extra mujoco --extra agent --extra twin --extra catalog
uv run --no-sync armigo-mcp serve
uv run --no-sync armigo-mcp dashboard --out data/catalog/agent
```

The default v6 scorer uses `results/anomaly/models-sup/sup.json` and the local
`E:/venvs/armigo-anomaly/Scripts/python.exe`. The committed spec stores its LoRA checkpoint as a path relative to
a worktree depth (`../../../../../../results/anomaly/sup-lora/head-lora.pt`), so from the main checkout it does not
resolve. Copy the spec with an absolute `checkpoint` path and pass it with `--detector`, or generate one with
`scripts/anomaly/export_sup.py`. Checkpoints and full rendered recordings are provisioned locally.

### MCP tools and jobs

| Tool | Purpose |
|---|---|
| `list_cases` | Opaque scene IDs. The operator can provide a private fixture list via `serve --suite fixtures.json`. |
| `start_episode` | Case ID, request ID, order, view and rewind budgets, `record`, `use_llm`. Returns a job ID. |
| `start_policy_search` | Successive-halving policy search (see below). Returns a job ID. |
| `job_status` | Status (`queued`, `running`, `complete`, `failed`, `cancelled`, `interrupted`), `kind` and `progress` (node, branch, frame, views). Simulator truth is private. |
| `wait_job` | Blocks up to `timeout_s` (max 600) and sends MCP progress notifications while the job runs. |
| `cancel_job` | Queued jobs are cancelled at once; running jobs stop cooperatively within 10 frames and close their trace. |
| `resume_job` | Restarts an `interrupted` (or failed) episode from its latest persisted checkpoint as a linked episode. |
| `get_observation` | The original decision crop or full camera PNG plus its metadata. |

Jobs live in `<out>/jobs.sqlite` (WAL). A repeated request ID with the same arguments returns its original job,
also after a restart. When the server starts, jobs left queued or running become `interrupted`. The inner graph
compiles with a LangGraph `SqliteSaver` in `<out>/graph.sqlite` (thread ID = job ID). Episodes write
`<out>/episodes/<job_id>`, including `task.json`, and `run --resume <episode dir> --out <new dir>` does the same
from the command line.

A resume restores the MuJoCo integration state, command filter, expert scalars, view counters, the live
checkpoint list and the checkpoint's views, plus `failures` from the saved graph thread, then continues at
`choose_view`. It does not carry over the parent's telemetry or video, so the health verdict and `sim_frames` cover
the resumed segment only. Failures after the last checkpoint and a node that was mid-flight are lost, a recovery
variant active in the parent is not re-applied, and resumed episodes are never `student_eligible`.

`--workers N` (on `serve` and `run`) runs episodes in N spawned processes. One detector process (about 590 MiB of
GPU memory) is owned by the parent and serves all workers through a queue, because vLLM leaves too little GPU
memory for one detector per worker. `run --workers N` writes the same `report.json` as `--workers 1`.
`serve --transport streamable-http --host 127.0.0.1 --port 8765` lets several clients share one server; stdio is
the default. The dashboard still runs as a separate process.

### Inner model: skill and recovery choices

Add `--llm-url http://127.0.0.1:8038/v1 --model qwen3.8-vllm` to enable `start_episode(use_llm=true)` and the
advisory failure analysis of the policy search. The inner model only picks one of the offered choices. It
receives a compact, leak-checked context (`TwinMujoco.decision_context`): covered and missing faces, each view's
camera, faces, score, threshold and margin, earlier failures with their reason codes, used and remaining budgets,
whether the brick is grasped, the brick position, the offered checkpoints and a one-line note per choice. The
visual mark geometry, mark seed, ground-truth label and face never enter it (`assert_no_defect_leakage` runs on
every context). A first decision is about 200 user tokens instead of 1,600 for the earlier full-scene dump.

`select_skill` constrains the answer with a JSON schema whose `skill` is an enum of the legal choices (vLLM
`response_format: json_schema`; a 4xx from the server degrades to `json_object` once). The prompt is counted
with the server's `/tokenize` before sending and the oldest views and failures are dropped if it would not fit.
An invalid answer gets one repair turn with the validation error. Otherwise the decision falls back to the first
choice, which is always the deterministic default, and records one reason: `http_error`, `timeout`,
`invalid_json`, `illegal_choice`, `context_too_long` or `empty_output`. Every `decision` event carries
`fallback_reason`, `prompt_tokens`, `latency_s` and `attempts`. Each episode result has an `llm` block, and the
run report adds `llm_consulted`, `llm_adopted`, `llm_adoption_rate`, `fallback_reasons` and token statistics.

Rewind choices are `<checkpoint>` and `<checkpoint>+<variant>`. Variants come from a frozen, bounded table
(`slow_scan`, `slow_carry`, `gentle_place`, `high_place`, `cautious`) that only slows motion or raises the
transfer height; they transform the first attempt's parameters, so they never stack. `rewind()` finds the start of
a saturated-torque run with the health check's own thresholds and attributes it to the skill running then. The
default choice is the latest checkpoint before that run, with the variant for that skill. A checkpoint after the
stall keeps it in the health history and is noted as such. On seed 442701 the stall was the wrist roll blocking
during the second wrist scan, so the old plain rewind to the latest checkpoint always failed again.

### Policy search

`start_policy_search` evaluates all 54 generic policies (`order` × `max_wrist_views` × `max_bottom_views` ×
`max_rewinds`) by successive halving on a generated development suite (seeds from 700001, balanced, disjoint
from the demo, dev4 and bench16 fixtures). Every policy in a round sees the same fixtures, and the default
`Policy()` is always kept to the final round. The best finalist must then beat the default on fresh fixtures:
an exact paired sign test on success, or equal success with a paired bootstrap 95% interval on simulator frames
that excludes zero. Otherwise the default is kept and the report says why. `budget` 1-8 caps the search at
50 × (budget + 1) episodes; small budgets usually keep the default. Episodes fan out over `--workers`. The LLM only
writes an advisory `llm_analysis` of the failure codes, which never affects selection. Search writes `report.json`
and a standalone `policy.json` under `<out>/search/<job_id>`. Freeze that policy before evaluation with a separate
operator suite:

```powershell
uv run --no-sync armigo-mcp run --policy <policy.json> --suite <held-out.json> --out data/catalog/held-out --headless
```

The fixture file contains rows of `[scene_seed, defective, defect_face]`. The detector receives only RGB crops.
Positive v6 evidence can terminate inspection early. A normal label requires scored negative coverage of all
six faces, with the bottom validated by the fixed external camera. If the view budget is exhausted safely,
`UNKNOWN` leaves the brick on the inspection table. A lost brick or unverified placement reports a failed
episode. `--bottom-profile` checks the camera profile and detector-spec digest before it changes the operating
point. The current pilot profile retains the original threshold of zero.

## Time travel and episode evidence

Recovery choices and variants are described under *Inner model* above.

The inner graph publishes checkpoints with the brick supported on the table and the gripper open. It saves the
MuJoCo integration vector, command filter state, control cursor, explicit skill parameters, prior valid view
references and simulator-frame index. A failed skill reports its reason to the graph and caller. The model can
pick an offered checkpoint or stop. Rewind preserves the failure trace and total episode budget and creates a
new branch. Decision evidence reverts to the selected checkpoint. A branch must recapture later views before
they can affect its sort. A rewind restores within the same run; `resume_job` restores the latest checkpoint after a restart. The files preserve
the checkpoint and trace for audit.

A recorded episode contains a separate time axis per branch with the common simulator frame at the fork.
`replay.mp4` shows the actual robot render, wrist and external RGB, and the active motion node.
`events.jsonl` preserves sequence order for nodes sharing one simulator frame. The dashboard can seek to
a decision, show the two original decision images, or select the earlier branch. FFV1 files archive front,
wrist, full external RGB and external depth. The original PNG and detector crop are also saved at each keyframe.
The v6 panel follows published inspection events while a run is active. Select **跟随最新 Episode** to switch
to new recordings automatically. On replay the panel follows the selected branch and playback position, and
clicking a timeline event opens its image and scores. The panel shows calibrated classification, segmentation
and fused scores, the effective threshold, and the score margin. These scores are in calibrated units, without
a probability scale. Previously recorded runs show the fused score when the per-head outputs were not collected.
Any RTX replay is a separate set of images.

## Motion smoothness

Each episode result carries a `smoothness` block, and the suite report aggregates it. Jerk is the third finite
difference of the arm joints (0-4) times FPS cubed, in rad/s³. It is computed for the commanded targets
(`jerk_rms_cmd`) and for the measured joint positions (`jerk_rms_meas`, `jerk_peak_meas`, `jerk_p99_meas`).
`jerk_peak_phase` and `worst_phases` name the phases with the highest measured jerk. `stops` counts moves that
come to rest and then start again (0.05 rad/s threshold, three-frame hysteresis). `clipped_frames` counts frames
where the expert's target had to be clipped to the joint range.

Every move is planned to the clipped target, so no joint stops dead at a limit halfway through a min-jerk path.
The return to rest is long enough to keep the peak joint speed under `return_peak_speed_rad_s` (4.4 rad/s, just
below what the wrist flex reaches under its torque limit), capped at one unchunked 1.4 s segment. On the 16-case
bench this cut the median measured jerk from 11.0 to 6.4 and the worst peak from 505 (in the return) to 112,
with identical sort outcomes. The stops that remain come from chunk pauses and holds that the torque-saturation
health rule requires.

## LeRobot

Use the environment that contains LeRobot 0.6 and PyAV.

```powershell
$env:PYTHONPATH = "src"
E:/il/lerobot/Scripts/python.exe -m digital_twin_armigo.agent.twin_record --out data/catalog/vla-six-face <episode-directory> <episode-directory>
```

On this Windows setup, load the result with `LeRobotDataset(..., video_backend="pyav")`.
The exporter checks branch eligibility and camera timestamps. It exports pre-action RGB and proprioception,
the following normalized joint command, and the shared instruction. `observation.images.top_full` preserves
the field of view used for classification in this camera profile, including areas outside the legacy top crop.
The legacy cropped RGB and depth are also included. Every successful rewound episode stays in the audit data.
Its action choices can depend on observations the student never received, even when the successful branch
reacquired a fresh view. The exporter accepts successful, continuous episodes only.
Use sequence history when training the VLA so the sort action can depend on the earlier inspection frames.

## Reproduce the gates

```powershell
uv run --no-sync python scripts/anomaly/evaluate_bottom.py --out data/catalog/bottom-pilot --calibration-n 8 --test-n 8
uv run --no-sync python scripts/anomaly/evaluate_qwen_bottom.py data/catalog/bottom-pilot/scores.json --out data/catalog/bottom-pilot/qwen.json
E:/il/v/Scripts/python.exe scripts/isaac/stage2_expert.py --no-cameras --out-dir data/catalog/isaac-probe --checkpoint-probe data/catalog/isaac-probe/checkpoint.json
```

Start the local vLLM service from WSL with `bash scripts/serve_armigo_qwen.sh` (port 8038, the client default).
It serves Qwen3.8-27B NVFP4 with a 4,096-token context, one image per request and NVFP4 KV cache. Defaults are
4 concurrent sequences (so parallel episode workers share the server) and prefix caching on (all decisions share
one system prompt), with eager execution kept as the memory-safe choice. Override with `ARMIGO_VLLM_MAX_LEN`,
`ARMIGO_VLLM_MAX_SEQS`, `ARMIGO_VLLM_GPU_UTIL`, `ARMIGO_VLLM_PREFIX_CACHE`, `ARMIGO_VLLM_EAGER` and
`ARMIGO_VLLM_EXTRA`. The earlier image test ran with one sequence and prefix caching disabled. A server
configured with `--language-model-only` can run skill selection and the failure analysis. Vision review requires
an image-enabled server. The current v6 path owns the sorting decision.

`scripts/bench_agent.py` measures the stack before and after a change: `episodes` (wall time, outcomes and LLM
decision statistics of a suite run), `vllm` (concurrent request throughput) and `context` (prompt tokens of the
first decision). The suites are in `config/agent_bench/`.

[Acceptance report](../results/agent-twin/REPORT.md)
