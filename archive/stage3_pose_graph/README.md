# Stage-3 show-pose graph (archived 2026-11-18)

This is the earlier LangGraph loop that chose extra wrist show poses from a measured pose menu
(`evaluate → build_context → propose → validate → execute`, with `interrupt`/`Command(resume=...)`
driving the MuJoCo expert's `next_pose` callback). It produced `results/inspection-policies/`.

The six-face agent (`src/digital_twin_armigo/agent/twin_graph.py`, served by `armigo-mcp`) replaced it.
The validate-and-repair pattern from this graph now lives in `twin_graph.select_skill`.
The shared pieces it used stay in the package: `agent/policies.py`, `agent/llm.py`.

The code here is kept runnable for reproduction but is not maintained or collected by the default test run:

```powershell
uv run --no-sync python -m pytest archive/stage3_pose_graph/test_graph.py
uv run --no-sync python archive/stage3_pose_graph/run_inspection_policies.py --help
```
