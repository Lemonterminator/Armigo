"""``armigo-inventory``: one table of every synthetic trajectory, scene and dataset, tagged by who produced it.

armigo-inventory scan [--root [LABEL=]PATH ...] [--out DIR]   # writes inventory.sqlite + CSV + SUMMARY.md
armigo-inventory summary [--out DIR]                           # the table in the terminal
armigo-inventory show <traj_id-or-path-fragment> [--out DIR]
armigo-inventory label --match GLOB --category CAT [--note TEXT] [--tag T ...]   # operator override, applied on scan
armigo-inventory rerun [--dry-run] [--limit N] [--run-group TEXT]   # re-run failed agent episodes; originals are kept
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
from pathlib import Path

from digital_twin_armigo.inventory import store, taxonomy


def _root_arg(text: str) -> tuple[str, Path]:
	label, sep, path = text.partition("=")
	if not sep:
		path, label = text, Path(text).name
	return label, Path(path).resolve()


def _default_out() -> Path:
	return store.project_root() / "data" / "inventory"


def cmd_scan(args) -> int:
	if args.root:
		roots = [_root_arg(r) for r in args.root]
	else:
		base = store.project_root()
		roots = [(p.name, p) for p in (base / "data" / "catalog", base / "results") if p.is_dir()]
	reruns = Path(args.out).resolve() / "reruns"  # where `armigo-inventory rerun` writes by default
	if reruns.is_dir() and not any(reruns == r or r in reruns.parents or reruns in r.parents for _, r in roots):
		roots.append(("reruns", reruns))
	if not roots:
		print("armigo-inventory: no scan roots (pass --root)", file=sys.stderr)
		return 2
	result = store.build(roots, Path(args.out), overrides=Path(args.overrides) if args.overrides else None)
	print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
	return 0


@contextlib.contextmanager
def _open(out: str):
	"""Read-only connection, closed on exit (an open handle blocks the next scan from replacing the file on Windows)."""
	db = Path(out) / "inventory.sqlite"
	if not db.is_file():
		raise SystemExit(f"armigo-inventory: {db} not found; run `armigo-inventory scan` first")
	con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
	con.row_factory = sqlite3.Row
	try:
		yield con
	finally:
		con.close()


def cmd_summary(args) -> int:
	with _open(args.out) as con:
		return _summary(con)


def _summary(con) -> int:
	print(f"{'category':<18}{'engine':<8}{'n(distinct)':>13}{'success':>9}{'basis(explicit/inferred/override)':>38}")
	rows = con.execute(
		"""select category, engine, count(*) n, sum(verdict='SUCCESS') ok,
		   sum(category_basis='explicit') e, sum(category_basis='inferred') i, sum(category_basis='override') o
		   from trajectories where same_as is null group by 1, 2 order by 1, 2"""
	).fetchall()
	for r in rows:
		print(f"{r['category']:<18}{r['engine']:<8}{r['n']:>13}{r['ok']:>9}{f'{r[4]}/{r[5]}/{r[6]}':>38}")
	n_scene = con.execute("select scene_kind, count(*) from scenes group by 1").fetchall()
	print("scenes:", ", ".join(f"{k}={n}" for k, n in n_scene))
	return 0


def cmd_show(args) -> int:
	like = f"%{args.query}%"
	with _open(args.out) as con:
		rows = con.execute(
			"select * from trajectories where traj_id like ? or path like ? limit ?", (like, like, args.limit)
		).fetchall()
	for r in rows:
		print(json.dumps({k: v for k, v in dict(r).items() if v is not None}, ensure_ascii=False, indent=1))
	if not rows:
		print("no match", file=sys.stderr)
		return 1
	return 0


def cmd_label(args) -> int:
	if args.category not in taxonomy.CATEGORIES:
		print(f"unknown category {args.category!r}; one of {', '.join(taxonomy.CATEGORIES)}", file=sys.stderr)
		return 2
	path = Path(args.overrides) if args.overrides else store.default_overrides()
	path.parent.mkdir(parents=True, exist_ok=True)
	rules = store.load_overrides(path)
	rule: dict = {"match": args.match, "set": {"category": args.category}}
	if args.control:
		rule["set"]["control"] = args.control
	if args.sensing:
		rule["set"]["sensing"] = args.sensing
	if args.note:
		rule["set"]["note"] = args.note
	if args.tag:
		rule["set"]["tags"] = args.tag
	rules.append(rule)
	path.write_text(json.dumps(rules, indent=1, ensure_ascii=False), encoding="utf-8")
	print(f"added override #{len(rules)} to {path}; run `armigo-inventory scan` to apply")
	return 0


def cmd_rerun(args) -> int:
	from digital_twin_armigo.inventory import rerun

	categories = tuple(args.category.split(",")) if args.category else rerun.RERUNNABLE
	bad = [c for c in categories if c not in taxonomy.CATEGORIES]
	if bad:
		print(f"unknown category {bad}; one of {', '.join(taxonomy.CATEGORIES)}", file=sys.stderr)
		return 2
	with _open(args.out):  # fail early when there is no inventory yet
		pass
	items, skipped = rerun.plan(
		Path(args.out), categories=categories, run_group=args.run_group, limit=args.limit, redo=args.redo
	)
	out_root = Path(args.out_root) if args.out_root else Path(args.out) / "reruns"
	detector = Path(args.detector) if args.detector else store.project_root() / "results/anomaly/models-sup/sup.json"
	print(f"{len(items)} failed episodes to re-run into {out_root}")
	for why, n in skipped.most_common():
		print(f"  skipped {n}: {why}")
	by_group = {}
	for item in items:
		by_group[item["run_group"]] = by_group.get(item["run_group"], 0) + 1
	for group, n in sorted(by_group.items(), key=lambda kv: -kv[1])[:15]:
		print(f"  {n:4d}  {group}")
	if args.dry_run or not items:
		return 0
	if not detector.is_file():
		print(f"detector {detector} not found (--detector)", file=sys.stderr)
		return 2
	results = rerun.execute(items, out_root, detector)
	finished = [r for r in results if "verdict" in r]
	errors = [r for r in results if "error" in r]
	passed = sum(r["verdict"] == "SUCCESS" for r in finished)
	print(f"done: {len(finished)} re-run, {passed} now SUCCESS, {len(errors)} crashed")
	print("run `armigo-inventory scan` to link them to the originals")
	return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		prog="armigo-inventory", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
	)
	p.add_argument("--out", default=str(_default_out()), help="inventory directory (default data/inventory)")
	sub = p.add_subparsers(dest="cmd", required=True)
	s = sub.add_parser("scan", help="scan roots and rebuild the tables")
	s.add_argument("--root", action="append", help="[LABEL=]PATH to scan; repeat. Default: data/catalog and results")
	s.add_argument("--overrides", help="override rules file (default config/inventory_overrides.json)")
	s.set_defaults(fn=cmd_scan)
	sub.add_parser("summary", help="counts by category and engine").set_defaults(fn=cmd_summary)
	w = sub.add_parser("show", help="print trajectory rows matching an id or path fragment")
	w.add_argument("query")
	w.add_argument("--limit", type=int, default=5)
	w.set_defaults(fn=cmd_show)
	lb = sub.add_parser("label", help="add an operator override (glob on path or id)")
	lb.add_argument("--match", required=True)
	lb.add_argument("--category", required=True)
	lb.add_argument("--control")
	lb.add_argument("--sensing")
	lb.add_argument("--note")
	lb.add_argument("--tag", action="append")
	lb.add_argument("--overrides", help="override rules file (default config/inventory_overrides.json)")
	lb.set_defaults(fn=cmd_label)
	rr = sub.add_parser("rerun", help="re-run failed agent episodes into new directories (originals are kept)")
	rr.add_argument("--category", help="comma list; default agent_omniscient,agent_realistic")
	rr.add_argument("--run-group", help="only run groups containing this text")
	rr.add_argument("--limit", type=int, help="at most N episodes")
	rr.add_argument("--redo", action="store_true", help="also originals that already have a re-run (another attempt)")
	rr.add_argument("--dry-run", action="store_true", help="show what would run")
	rr.add_argument("--out-root", help="where the new episodes go (default <inventory>/reruns, scanned automatically)")
	rr.add_argument("--detector", help="detector json (default results/anomaly/models-sup/sup.json)")
	rr.set_defaults(fn=cmd_rerun)
	args = p.parse_args(argv)
	return args.fn(args)


if __name__ == "__main__":
	raise SystemExit(main())
