"""Offline, blinded Qwen cross classification on paired bottom RGB crops."""

from __future__ import annotations

import argparse
import base64
import json
import random
import time
from pathlib import Path

import httpx


def main():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("scores", type=Path)
	p.add_argument("--url", default="http://127.0.0.1:8038/v1")
	p.add_argument("--model", default="qwen3.8-vllm")
	p.add_argument("--out", required=True, type=Path)
	a = p.parse_args()
	rows = [r for r in json.loads(a.scores.read_text(encoding="utf-8"))["test"] if r["crop"]]
	random.Random(23).shuffle(rows)
	results = []
	with httpx.Client(timeout=60) as client:
		for row in rows:
			image = base64.b64encode((a.scores.parent / row["crop"]).read_bytes()).decode()
			started = time.perf_counter()
			reply = client.post(
				a.url.rstrip("/") + "/chat/completions",
				json={
					"model": a.model,
					"temperature": 0,
					"max_tokens": 80,
					"chat_template_kwargs": {"enable_thinking": False},
					"response_format": {"type": "json_object"},
					"messages": [
						{
							"role": "user",
							"content": [
								{
									"type": "text",
									"text": (
										"Inspect the brick surface in this RGB crop for an ink X mark. "
										"Return JSON with one field 'label': 'cross_visible', 'no_cross_visible' or 'unreadable'. "
										"Use 'unreadable' if the surface is too small or obstructed."
									),
								},
								{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
							],
						}
					],
				},
			)
			reply.raise_for_status()
			raw = reply.json()["choices"][0]["message"]["content"]
			try:
				label = json.loads(raw)["label"]
			except (TypeError, KeyError, ValueError):
				label = "invalid"
			results.append(
				{
					"seed": row["seed"],
					"defective": row["defective"],
					"qwen": label,
					"v6_score": row["score"],
					"latency_s": round(time.perf_counter() - started, 3),
				}
			)
			print(json.dumps(results[-1]), flush=True)
	summary = {
		"model": a.model,
		"rows": results,
		"positives": sum(row["defective"] for row in results),
		"detects": sum(row["defective"] and row["qwen"] == "cross_visible" for row in results),
		"negatives": sum(not row["defective"] for row in results),
		"false_alarms": sum(not row["defective"] and row["qwen"] == "cross_visible" for row in results),
		"non_decisions": sum(row["qwen"] not in ("cross_visible", "no_cross_visible") for row in results),
	}
	a.out.parent.mkdir(parents=True, exist_ok=True)
	a.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
	main()
