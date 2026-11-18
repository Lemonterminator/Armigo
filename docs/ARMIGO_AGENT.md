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
`E:/venvs/armigo-anomaly/Scripts/python.exe`. Read the detector spec's relative checkpoint path before using it in
another checkout. If needed, point `--detector` to a spec generated with `scripts/anomaly/export_sup.py` for that
checkout and its local LoRA weights. Checkpoints and full rendered recordings are provisioned locally.

Use an MCP stdio client to discover `list_cases`, `start_episode`, `start_policy_search` `get_observation` and `job_status`.
`list_cases` returns opaque IDs. The operator can provide a private fixture list via `serve --suite fixtures.json`.
`start_episode` accepts a case ID, a request ID, the order and the view and rewind budgets. Poll `job_status` with
the returned job ID. `get_observation` reads the original decision crop or full camera image. A repeated request ID with the same arguments returns its original job. This idempotency
scope is the current MCP process. The recorder writes artifacts to `<out>/episodes/<job_id>`.
Simulation and rendering run on a single worker, while the dashboard can follow the event log and live preview.

Add `--llm-url http://127.0.0.1:8038/v1 --model qwen3.8-vllm` to the MCP server to enable
`start_episode(use_llm=true)` and `start_policy_search`. The inner model receives the physical scene and
current observations as structured JSON, with world positions and orientations of all bodies, joint state and
limits, cameras, bowl locations, contact summaries, visible face coverage, v6 scores and legal skill names.
It can select an offered skill or an ancestor checkpoint. Low-level motions and the defect decision run in the
validated controller and scorer. The visual mark geometry, mark seed, ground-truth label and ground-truth face
stay outside the LLM and tool results. Recovery context contains the failure reason and earlier observed scores.

The outer model receives only aggregates from a fixed development set. It proposes a bounded JSON policy with
`order`, `max_wrist_views`, `max_bottom_views` and `max_rewinds`. LangGraph runs the candidate with real
MuJoCo motion. The selector ranks audited physical successes first, then labels correct, simulator frames,
and views. Search writes a `report.json` and standalone `policy.json` under `<out>/search/<job_id>`.
Freeze that policy before evaluation with a separate operator suite:

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

The inner graph publishes checkpoints with the brick supported on the table and the gripper open. It saves the
MuJoCo integration vector, command filter state, control cursor, explicit skill parameters, prior valid view
references and simulator-frame index. A failed skill reports its reason to the graph and caller. The model can
pick an offered checkpoint or stop. Rewind preserves the failure trace and total episode budget and creates a
new branch. Decision evidence reverts to the selected checkpoint. A branch must recapture later views before
they can affect its sort. The current skill checkpoints are restored within the same run. The files preserve
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

Start the local vLLM service from WSL with `bash scripts/serve_armigo_qwen.sh`. The image test used the local Qwen3.8-27B NVFP4 service with a 4,096-token context, one image per request,
NVFP4 KV cache, eager execution and prefix caching disabled. A server configured with
`--language-model-only` can run policy selection and policy search. Vision review requires an image-enabled
server. The current v6 path owns the sorting decision.

[Acceptance report](../results/agent-twin/REPORT.md)
