"""Round-spec driven three gates (scripts/experiments/loop/gates.py) on synthetic report rows."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "loop"))
import gates


def _row(batch, cam, case, ok, cause=None, frames=100):
	layout = case.split("-")[0]
	return {
		"path": f"runs/{batch}/{cam}/{case}",
		"case_id": case,
		"layout_id": layout,
		"batch": batch,
		"group": f"realistic-{cam}",
		"camera_condition": cam,
		"has_episode": True,
		"infra_failed": False,
		"physical_success": ok,
		"root_cause": "none" if ok else (cause or "grasp_or_path_failure"),
		"mechanism": None,
		"traj_id": f"agent:v2:{batch}/mujoco/realistic-{cam}/{case}",
		"sim_frames": frames,
		"wall_s": 10.0,
	}


def _write(root: Path, batch: str, rows: list[dict]) -> None:
	d = root / "reports" / batch
	d.mkdir(parents=True, exist_ok=True)
	(d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _world(root: Path, *, fix_bank=True, break_one=False):
	"""Old layouts L1-L20 (L1-L3 fail by grasp, L4 by bowl), fresh layouts L101-L130, two conditions."""
	cams = ("fixed", "drift")
	old = [f"L{i}-N" for i in range(1, 21)]
	fresh = [f"L{100 + i}-N" for i in range(1, 31)]
	fail_cause = {
		"L1-N": "grasp_or_path_failure",
		"L2-N": "recovery_failure",
		"L3-N": "grasp_or_path_failure",
		"L4-N": "bowl_localization",
	}
	base, rerun, fb, fc = [], [], [], []
	for cam in cams:
		for c in old:
			cause = fail_cause.get(c) if cam == "drift" else None
			base.append(_row("prev", cam, c, cause is None, cause))
			fixed = cause in ("grasp_or_path_failure", "recovery_failure") and fix_bank
			broken = break_one and cam == "fixed" and c in ("L10-N", "L11-N", "L12-N")
			rerun.append(_row("rerun", cam, c, (cause is None or fixed) and not broken, cause))
		for c in fresh:
			fb.append(_row("fb", cam, c, c != "L101-N"))
			fc.append(_row("fc", cam, c, True))
	for name, rows in (("prev", base), ("rerun", rerun), ("fb", fb), ("fc", fc)):
		_write(root, name, rows)


def _spec(**over):
	spec = {
		"tag": "round-test",
		"root": "unused",
		"conditions": ["fixed", "drift"],
		"base": ["prev"],
		"rerun": ["rerun"],
		"fresh_base": "fb",
		"fresh_candidate": "fc",
		"bank": {"root_causes": ["grasp_or_path_failure", "recovery_failure"]},
		"criteria": {
			"bank_min_fix_rate": 0.7,
			"regression_max_net_regression": 1,
			"regression_min_mcnemar_p": 0.2,
			"fresh_min_ci_lo": -0.02,
			"bootstrap_b": 200,
			"seed": 0,
			"metric": "success_rate",
		},
		"out": "verify-test",
	}
	spec.update(over)
	return spec


def _run(root, spec):
	return gates.run_gates(spec, gates.Loader(root, from_reports=True, dry=False, jobs=1))


def test_merge_with_root_cause_bank(tmp_path):
	_world(tmp_path)
	doc = _run(tmp_path, _spec())
	assert doc["_bank_ids"] == ["drift/L1-N", "drift/L2-N", "drift/L3-N"]
	assert len(doc["_reg_ids"]) == 36  # 40 old pairs minus the 4 base failures
	assert doc["gates"]["verdict"] == "merge"
	assert set(doc["gates_by_condition"]) == {"fixed", "drift"}
	assert doc["regressions"] == []


def test_regressions_reject(tmp_path):
	_world(tmp_path, break_one=True)
	doc = _run(tmp_path, _spec())
	assert doc["gates"]["verdict"] == "reject"
	assert {r["case"] for r in doc["regressions"]} == {"fixed/L10-N", "fixed/L11-N", "fixed/L12-N"}


def test_explicit_ids_and_condition_filter(tmp_path):
	_world(tmp_path)
	doc = _run(tmp_path, _spec(conditions=["drift"], bank={"ids": ["drift/L1-N", "drift/L2-N"]}))
	assert doc["_bank_ids"] == ["drift/L1-N", "drift/L2-N"]
	assert all(i.startswith("drift/") for i in doc["_reg_ids"])
	assert "gates_by_condition" not in doc


def test_unknown_bank_id_and_count_mismatch_stop(tmp_path):
	_world(tmp_path)
	with pytest.raises(SystemExit, match="not among the base rows"):
		_run(tmp_path, _spec(bank={"ids": ["drift/L99-N"]}))
	crit = {**_spec()["criteria"], "bank_records": 5}
	with pytest.raises(SystemExit, match="bank_records"):
		_run(tmp_path, _spec(criteria=crit))


def test_ledger_bank(tmp_path):
	_world(tmp_path)
	key = ["grasp", "NO_CLAMP", "grasp_or_path_failure"]
	ledger = tmp_path / "ledger" / "ledger.json"
	ledger.parent.mkdir()
	tids = [f"agent:v2:prev/mujoco/realistic-drift/{c}" for c in ("L1-N", "L3-N")]
	ledger.write_text(json.dumps({"clusters": [{"key": key, "examples": tids[:1]}]}), encoding="utf-8")
	(ledger.parent / "triage.jsonl").write_text(
		json.dumps({"traj_id": tids[1], "stage": key[0], "code": key[1], "root_cause": key[2]}) + "\n", encoding="utf-8"
	)
	doc = _run(tmp_path, _spec(bank={"ledger": str(ledger), "cluster": "grasp:NO_CLAMP:grasp_or_path_failure"}))
	assert doc["_bank_ids"] == ["drift/L1-N", "drift/L3-N"]


def test_main_writes_verify_and_bank(tmp_path, monkeypatch):
	_world(tmp_path)
	spec_path = tmp_path / "spec.json"
	spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
	monkeypatch.setattr(gates, "REPO", tmp_path / "repo")
	assert gates.main(["--round", str(spec_path), "--root", str(tmp_path), "--from-reports", "--write-bank"]) == 0
	verify = json.loads((tmp_path / "reports" / "round-test" / "verify-test.json").read_text(encoding="utf-8"))
	assert verify["gates"]["verdict"] == "merge" and not any(k.startswith("_") for k in verify)
	assert "**merge**" in (tmp_path / "reports" / "round-test" / "verify-test.md").read_text(encoding="utf-8")
	snap = json.loads((tmp_path / "repo" / "config" / "agent_bench" / "failure_bank" / "round-test.json").read_text())
	assert snap["bank_ids"] == ["drift/L1-N", "drift/L2-N", "drift/L3-N"] and len(snap["regression_ids"]) == 36


@pytest.mark.parametrize("path", sorted((REPO / "config" / "experiments" / "loop").glob("round_*.json")))
def test_repo_round_specs_load(path):
	spec = gates.load_spec(path)
	assert set(spec["bank"]) & {"ids", "root_causes", "ledger"}
	assert spec["criteria"]["bank_min_fix_rate"] > 0
