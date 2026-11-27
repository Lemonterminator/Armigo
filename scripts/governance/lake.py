#!/usr/bin/env python
"""CLI for the bronze -> silver -> gold paired-trajectory lake."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from digital_twin_armigo.governance import bronze, gold, lake, silver

DEFAULT_DB = Path("E:/Armigo_data_v3/governance/lake.sqlite")


def _load_freeze(path: Path | None) -> dict | None:
	if path is None:
		return None
	return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _latest_silver_build(con) -> str:
	row = con.execute("select build_id from silver_episode order by rowid desc limit 1").fetchone()
	if row is None:
		raise SystemExit("no silver build found in lake; run `silver` first")
	return str(row[0])


def main(argv=None) -> int:
	root_common = argparse.ArgumentParser(add_help=False)
	root_common.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite lake path")
	sub_common = argparse.ArgumentParser(add_help=False)
	sub_common.add_argument("--db", type=Path, default=argparse.SUPPRESS, help="SQLite lake path")

	p = argparse.ArgumentParser(description=__doc__, parents=[root_common])
	sub = p.add_subparsers(dest="cmd", required=True)

	s_reg = sub.add_parser("register", parents=[sub_common], help="Register episode directories into bronze")
	s_reg.add_argument("--root", type=Path, required=True)
	s_reg.add_argument("--source", required=True)

	s_sil = sub.add_parser("silver", parents=[sub_common], help="Build silver_episode and silver_pair tables")
	s_sil.add_argument("--freeze", type=Path, default=None)
	s_sil.add_argument("--audit-path", type=Path, default=None)

	s_gold = sub.add_parser("gold", parents=[sub_common], help="Materialize a gold data product manifest and card")
	s_gold.add_argument("product", choices=sorted(gold.PRODUCTS))
	s_gold.add_argument("--freeze", type=Path, default=None)
	s_gold.add_argument("--silver-build", default=None)
	s_gold.add_argument("--out-root", type=Path, default=None)

	s_rep = sub.add_parser("report", parents=[sub_common], help="Print or write Markdown silver pair report")
	s_rep.add_argument("--silver-build", default=None)
	s_rep.add_argument("--out", type=Path, default=None)

	args = p.parse_args(argv)
	con = lake.connect(args.db)
	try:
		if args.cmd == "register":
			counts = bronze.register(con, args.root, source=args.source)
			print(json.dumps(counts))
		elif args.cmd == "silver":
			bid = silver.build_silver(con, freeze=_load_freeze(args.freeze), audit_path=args.audit_path)
			n_pairs = silver.build_pairs(con, bid)
			n_eps = con.execute("select count(*) from silver_episode where build_id = ?", (bid,)).fetchone()[0]
			print(json.dumps({"build_id": bid, "episodes": n_eps, "pairs": n_pairs}))
		elif args.cmd == "gold":
			bid = args.silver_build or _latest_silver_build(con)
			out_root = args.out_root or args.db.parent
			ver_dir = gold.build_product(
				con,
				args.product,
				silver_build=bid,
				out_root=out_root,
				freeze=_load_freeze(args.freeze),
			)
			print(str(ver_dir))
		elif args.cmd == "report":
			bid = args.silver_build or _latest_silver_build(con)
			md = silver.pair_report(con, bid)
			if args.out is not None:
				args.out.parent.mkdir(parents=True, exist_ok=True)
				args.out.write_text(md, encoding="utf-8", newline="\n")
			print(md, end="")
	finally:
		con.close()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
