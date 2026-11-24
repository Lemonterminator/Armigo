"""Plan and summarize counterfactual oracle re-runs for failed realistic episodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

from digital_twin_armigo.experiments.counterfactual import VARIANTS, verdict

REPO = Path(__file__).resolve().parents[2]
_CASE_RE = re.compile(r"^L(\d+)-(?:N|D(\d+))$")


def _load_rows(path: Path) -> list[dict]:
	return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _case_from_row(row: dict) -> list | None:
	ep = row.get("path")
	if ep:
		task_file = Path(ep) / "task.json"
		if task_file.is_file():
			try:
				case = json.loads(task_file.read_text(encoding="utf-8-sig")).get("case")
				if isinstance(case, list) and len(case) >= 3:
					return [int(case[0]), bool(case[1]), int(case[2])]
			except (OSError, ValueError, TypeError):
				pass
	m = _CASE_RE.match(str(row.get("case_id") or ""))
	if m:
		return [int(m[1]), m[2] is not None, int(m[2]) if m[2] is not None else 0]
	return None


def _register_suite():
	script = Path(__file__).resolve().with_name("make_suites.py")
	if not script.is_file():
		script = REPO / "scripts" / "experiments" / "make_suites.py"
	spec = importlib.util.spec_from_file_location("make_suites", script)
	mod = importlib.util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(mod)
	return mod.register_suite


def _is_realistic_failure(row: dict) -> bool:
	group = str(row.get("group") or "")
	sensing = row.get("sensing") or ("realistic" if group.startswith("realistic") else "truth")
	return sensing == "realistic" and not bool(row.get("physical_success"))


def _variant_key(rows: list[dict], fallback: tuple[str, ...]) -> tuple[str, ...]:
	if not rows:
		return tuple(sorted(fallback))
	group = str(rows[0].get("group") or "")
	if "-oracle-" in group:
		raw = group.split("-oracle-", 1)[1]
		return tuple(sorted(p for p in raw.split("+") if p))
	return tuple(sorted(fallback))


def cmd_plan(a) -> int:
	rows = _load_rows(a.scores)
	cases = []
	seen = set()
	for r in rows:
		if not _is_realistic_failure(r):
			continue
		cid = r.get("case_id")
		if cid in seen:
			continue
		case = _case_from_row(r)
		if case is None:
			continue
		seen.add(cid)
		cases.append(case)
	a.out.parent.mkdir(parents=True, exist_ok=True)
	text = "[\n" + ",\n".join("  " + json.dumps(c) for c in cases) + "\n]\n"
	a.out.write_text(text, encoding="utf-8", newline="\n")
	manifest = a.manifest or (REPO / "config" / "agent_bench" / "v2" / "manifest.json")
	name = a.name or a.out.stem
	entry = _register_suite()(manifest, name, a.out, cases, sealed=False)
	print(json.dumps({"suite": name, "out": str(a.out), "manifest": str(manifest), **entry}))
	return 0


def cmd_summarize(a) -> int:
	if len(a.variants) != len(VARIANTS):
		raise SystemExit(f"expected {len(VARIANTS)} variant scores.jsonl files, got {len(a.variants)}")
	base_rows = {r["case_id"]: r for r in _load_rows(a.base) if "case_id" in r}
	by_variant: dict[tuple[str, ...], dict[str, dict]] = {}
	for path, fallback in zip(a.variants, VARIANTS, strict=True):
		vrows = _load_rows(path)
		vkey = _variant_key(vrows, fallback)
		by_variant[vkey] = {r["case_id"]: r for r in vrows if "case_id" in r}
	a.out.parent.mkdir(parents=True, exist_ok=True)
	cross: Counter = Counter()
	out_lines = []
	for cid, brow in sorted(base_rows.items()):
		vmap = {
			v: (
				None
				if cid not in by_variant.get(tuple(sorted(v)), {})
				else bool(by_variant[tuple(sorted(v))][cid].get("physical_success"))
			)
			for v in VARIANTS
		}
		res = verdict(bool(brow.get("physical_success")), vmap)
		mech = brow.get("mechanism") or "(none)"
		row = {
			"case_id": cid,
			"symptom": brow.get("root_cause"),
			"mechanism": brow.get("mechanism"),
			**res,
		}
		out_lines.append(json.dumps(row, ensure_ascii=False))
		cross[(mech, res["verdict"])] += 1
	a.out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8", newline="\n")
	print("## 机制假设 vs 反事实结论")
	for (mech, v), n in sorted(cross.items()):
		print(f"{mech} x {v}: {n}")
	return 0


def main(argv=None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	sub = p.add_subparsers(dest="cmd", required=True)
	pl = sub.add_parser("plan")
	pl.add_argument("--scores", type=Path, required=True)
	pl.add_argument("--out", type=Path, required=True)
	pl.add_argument("--manifest", type=Path)
	pl.add_argument("--name")
	sm = sub.add_parser("summarize")
	sm.add_argument("--base", type=Path, required=True)
	sm.add_argument("--variants", type=Path, nargs="+", required=True)
	sm.add_argument("--out", type=Path, required=True)
	a = p.parse_args(argv)
	return {"plan": cmd_plan, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
	raise SystemExit(main())
