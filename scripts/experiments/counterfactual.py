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
_ORACLE_RE = re.compile(r"-oracle-([a-z]+(?:\+[a-z]+)*)(?=$|-)")
_REALISTIC_GROUP_RE = re.compile(r"^realistic-([^-]+)-([A-Za-z0-9_]+)-([^-]+)(.*)$")
_TRUTH_GROUP_RE = re.compile(r"^truth-(?:([ABC])-)?([^-]+)(.*)$")
_MOTION_RE = re.compile(r"-m(open|w[1-4][sa])(?=$|-)")
_EVIDENCE_RE = re.compile(r"-e(hold|img|dense-10|dense-20)(?=$|-)")
_SEG_RE = re.compile(r"-s(lab|unet)(?=$|-)")
_CALIB_RE = re.compile(r"-c(\d+(?:\.\d+)?)(?=$|-)")


def _load_rows(path: Path) -> list[dict]:
	return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _normalize_case(case: object) -> list | None:
	if not isinstance(case, list) or len(case) < 3:
		return None
	base = [int(case[0]), bool(case[1]), int(case[2])]
	if len(case) >= 4 and isinstance(case[3], dict) and case[3]:
		return [*base, dict(case[3])]
	return base


def _case_from_row(row: dict) -> list | None:
	ep = row.get("path")
	if ep:
		task_file = Path(ep) / "task.json"
		if task_file.is_file():
			try:
				case = json.loads(task_file.read_text(encoding="utf-8-sig")).get("case")
				normalized = _normalize_case(case)
				if normalized is not None:
					return normalized
			except (OSError, ValueError, TypeError):
				pass
	raw_case = _normalize_case(row.get("case"))
	if raw_case is not None:
		return raw_case
	m = _CASE_RE.match(str(row.get("case_id") or ""))
	if m:
		return [int(m[1]), m[2] is not None, int(m[2]) if m[2] is not None else 0]
	return None


def _strip_oracle(group: str | None) -> str:
	if not group:
		return ""
	return _ORACLE_RE.sub("", str(group))


def _parse_group_meta(group: str | None) -> dict[str, str]:
	base = _strip_oracle(group)
	if not base:
		return {}
	meta: dict[str, str] = {}
	rest = ""
	m_real = _REALISTIC_GROUP_RE.match(base)
	if m_real:
		meta["camera_condition"] = m_real.group(1)
		meta["strategy"] = m_real.group(2)
		meta["detector"] = m_real.group(3)
		rest = m_real.group(4)
	else:
		m_truth = _TRUTH_GROUP_RE.match(base)
		if not m_truth:
			return {}
		meta["camera_condition"] = "fixed"
		meta["strategy"] = m_truth.group(1) or "A"
		meta["detector"] = m_truth.group(2)
		rest = m_truth.group(3)
	if rest:
		m_mot = _MOTION_RE.search(rest)
		if m_mot:
			meta["motion"] = m_mot.group(1)
		m_ev = _EVIDENCE_RE.search(rest)
		if m_ev:
			meta["evidence"] = m_ev.group(1)
		m_seg = _SEG_RE.search(rest)
		if m_seg:
			meta["segmenter"] = m_seg.group(1)
		m_cal = _CALIB_RE.search(rest)
		if m_cal:
			meta["calib_offset_mm"] = m_cal.group(1)
	return meta


def _norm_calib(val: object) -> str:
	if val is None or val == "":
		return "0"
	try:
		num = float(val)
	except (TypeError, ValueError):
		return str(val)
	return str(int(num)) if num.is_integer() else str(num)


def _row_meta(r: dict) -> dict[str, str]:
	base_group = _strip_oracle(r.get("group"))
	parsed = _parse_group_meta(base_group)
	return {
		"engine": str(r.get("engine") or "mujoco"),
		"camera_condition": str(r.get("camera_condition") or parsed.get("camera_condition") or "fixed"),
		"strategy": str(r.get("strategy") or parsed.get("strategy") or "A"),
		"motion": str(r.get("motion") or parsed.get("motion") or "open"),
		"evidence": str(r.get("evidence") or parsed.get("evidence") or "hold"),
		"segmenter": str(r.get("segmenter") or parsed.get("segmenter") or "lab"),
		"detector": str(r.get("detector") or parsed.get("detector") or "v7bot"),
		"calib_offset_mm": _norm_calib(
			r.get("calib_offset_mm") if r.get("calib_offset_mm") is not None else parsed.get("calib_offset_mm")
		),
		"custom_group": base_group if (base_group and not parsed) else "",
		"case_id": str(r["case_id"]),
	}


def _row_key(r: dict) -> tuple[str, ...]:
	"""Canonical composite key so differing camera_condition/strategy/motion/evidence/segmenter/group never collide."""
	m = _row_meta(r)
	return (
		m["engine"],
		m["camera_condition"],
		m["strategy"],
		m["motion"],
		m["evidence"],
		m["segmenter"],
		m["detector"],
		m["calib_offset_mm"],
		m["custom_group"],
		m["case_id"],
	)


def _index_rows(rows: list[dict], source: Path) -> dict[tuple[str, ...], dict]:
	indexed: dict[tuple[str, ...], dict] = {}
	for r in rows:
		if "case_id" not in r:
			continue
		key = _row_key(r)
		if key in indexed:
			raise ValueError(f"duplicate counterfactual composite key {key!r} in {source}")
		indexed[key] = r
	return indexed


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
	for row in rows:
		group = str(row.get("group") or "")
		m = _ORACLE_RE.search(group)
		if m:
			return tuple(sorted(p for p in m.group(1).split("+") if p))
	return tuple(sorted(fallback))


def cmd_plan(a) -> int:
	rows = _load_rows(a.scores)
	cases = []
	seen: set[str] = set()
	for r in rows:
		if not _is_realistic_failure(r):
			continue
		case = _case_from_row(r)
		if case is None:
			continue
		case_key = json.dumps(case, sort_keys=True)
		if case_key in seen:
			continue
		seen.add(case_key)
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
	base_rows = _index_rows(_load_rows(a.base), a.base)
	by_variant: dict[tuple[str, ...], dict[tuple[str, ...], dict]] = {}
	for path, fallback in zip(a.variants, VARIANTS, strict=True):
		vrows = _load_rows(path)
		vkey = _variant_key(vrows, fallback)
		if vkey in by_variant:
			raise ValueError(f"duplicate counterfactual variant {vkey!r} from {path}")
		by_variant[vkey] = _index_rows(vrows, path)
	a.out.parent.mkdir(parents=True, exist_ok=True)
	cross: Counter = Counter()
	out_lines = []
	for rkey, brow in sorted(base_rows.items()):
		vmap = {
			v: (
				None
				if rkey not in by_variant.get(tuple(sorted(v)), {})
				else bool(by_variant[tuple(sorted(v))][rkey].get("physical_success"))
			)
			for v in VARIANTS
		}
		res = verdict(bool(brow.get("physical_success")), vmap)
		mech = brow.get("mechanism") or "(none)"
		meta = _row_meta(brow)
		row = {
			"case_id": brow["case_id"],
			"engine": meta["engine"],
			"camera_condition": meta["camera_condition"],
			"strategy": meta["strategy"],
			"motion": meta["motion"],
			"evidence": meta["evidence"],
			"segmenter": meta["segmenter"],
			**({"group": brow["group"]} if brow.get("group") is not None else {}),
			**({"path": brow["path"]} if brow.get("path") is not None else {}),
			**({"traj_id": brow["traj_id"]} if brow.get("traj_id") is not None else {}),
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
