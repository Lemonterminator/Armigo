# Step-by-step MCP tools: omniscient and realistic

`armigo-mcp serve` runs whole episodes: the client submits a policy, and the LangGraph graph inside the server
drives the skills. `armigo-mcp step --profile omniscient|realistic` hands the orchestration to the client instead.
The client looks, picks the next skill, recovers from failures and decides the label
(`src/digital_twin_armigo/agent/step_mcp.py`).

## Tools

Common to both profiles:

| Tool | What it does |
|---|---|
| `list_cases` | opaque case ids, the task, the profile's tool list |
| `start_session(case_id)` | opens an episode; the arm goes to its ready pose (realistic: after hand-eye calibration and bowl localisation) |
| `session_state` | views with detector scores, cleared faces, the label the evidence supports, budgets, a pending failure and its recovery choices |
| `robot_state` | joint readings (`observation.state`, rad), gripper position, command and effort, gripper frame by forward kinematics |
| `capture(camera, kind)` | top or wrist image now: RGB PNG, or depth as a 16-bit PNG in mm (D415-like noise in realistic) |
| `view_image(view_id)` | detector crop or full frame of a scored view |
| `scan_wrist` | wrist-camera scan of the side and top faces (0-4) not yet cleared |
| `present_bottom` | grasp, then show the bottom face (5) to the top camera |
| `recover(choice)` | apply one of the offered recovery choices, or stop |
| `place(label)` | defective → pink bowl, normal → green bowl, unknown → leave it on the table |
| `finish` | end the episode and grade it against the hidden truth |

Only in `omniscient`: `scene_state` returns the brick pose and size, the bowl positions, the jaw contact forces and
the face pixels in both cameras.

Only in `realistic`: `camera_calibration` (intrinsics and the hand-eye-estimated pose in the robot frame),
`locate_brick` (from a fresh top depth image), `locate_bowls`.

How the profiles differ:

- **Skills.** `realistic` runs them with `TwinMujoco(sensing="realistic")` (see
  `results/agent-bench/perception/REPORT.md`):
  - believed brick, bowls from images;
  - gripper-reading contact;
  - camera placement check;
  - recovery from the current state.

  `omniscient` runs the same skills on simulator state and recovers by physics rewind.
- **Leak checks.** Every realistic payload is checked against `SIMULATOR_KEYS` and the defect leak check before it
  leaves the server. Both profiles keep the defect, its face and the true label private until `finish`.
- **Threading.** All simulation and rendering run on one server thread, since a MuJoCo GL context must stay on the
  thread that made it.

## Scripted client on bench16

`run_scripted` (`armigo-mcp step --scripted`) calls the same session methods the tools call. It follows the graph's
default policy: wrist scans first, then the bottom, then place the evidence label. It recovers with the first
offered choice. This checks that the tool set suffices for the task, and it is the baseline for an LLM client.

| Profile | Sorted | Labels | Recoveries | Tool calls per episode (median) | Simulated frames (median) |
|---|---|---|---|---|---|
| omniscient | 16/16 | 16/16 | 0 | 4 | 1638 |
| realistic | 16/16 | 16/16 | 0 | 4 | 1636 |

A median episode takes 4 skill calls: prepare (inside `start_session`), two wrist scans or one scan plus a bottom
view, then `place`. Both profiles match `armigo-mcp run` on bench16 (16/16), so the tool set suffices to do the
task. An LLM client remains to be tried.

Commands:

	uv run armigo-mcp step --profile realistic --scripted --detector results/agent-bench/sup-v7bot-local.json \
	  --suite config/agent_bench/bench16.json --out results/agent-bench/step-mcp/runs/bench16-realistic
	uv run armigo-mcp step --profile realistic --detector results/agent-bench/sup-v7bot-local.json \
	  --suite config/agent_bench/bench16.json   # stdio MCP server for an LLM client
