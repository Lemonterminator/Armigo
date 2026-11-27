import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
	spec = importlib.util.spec_from_file_location("wrist_orbit_localise", ROOT / "scripts/vla/wrist_orbit_localise.py")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def test_summarize_pairs_fused_error_with_top_error():
	mod = _module()
	fix = {
		"poses": 8,
		"fixes": 8,
		"rejects": {},
		"first_err_mm": 2.0,
		"single_err_mm_median": 2.0,
		"mean_err_mm": 0.5,
		"median_err_mm": 0.4,
		"spread_mm": 2.0,
	}
	none = {"poses": 8, "fixes": 0, "rejects": {"NO_BLOB": 8}}
	rows = [
		{"seed": 1, "size_correct": True, "top_err_mm": 15.0, "variants": {"silhouette/tip40/aim": fix}},
		{"seed": 2, "size_correct": True, "top_err_mm": 9.0, "variants": {"silhouette/tip40/aim": none}},
		{"seed": 3, "error": "BRICK_NOT_FOUND"},
	]
	s = mod.summarize(rows)
	v = s["size_ok"]["silhouette/tip40/aim"]
	assert s["errors"] == ["BRICK_NOT_FOUND"]
	assert v["cases_with_fix"] == 1 and v["top_err_mm_same_cases"] == 15.0
	assert v["median_lt_5mm"] == 1 and v["better_than_top"] == 1
	assert "size_wrong" not in s
