"""Copy Claude Code session / subagent transcripts and render each as a readable Markdown file.

The raw ``.jsonl`` is copied unchanged (full tool output, request blobs); the Markdown keeps the conversation, the
tool calls and the tool results, with each tool result cut to ``--max-result-chars`` characters.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _blocks(content) -> list[dict]:
	if isinstance(content, str):
		return [{"type": "text", "text": content}]
	return [b for b in (content or []) if isinstance(b, dict)]


def _result_text(content) -> str:
	if isinstance(content, str):
		return content
	parts = []
	for b in content or []:
		if isinstance(b, dict):
			parts.append(b.get("text") or f"[{b.get('type', 'block')}]")
	return "\n".join(parts)


def to_markdown(src: Path, title: str, max_result: int) -> str:
	out = [f"# {title}", "", f"Source: `{src.name}`", ""]
	for line in src.read_text(encoding="utf-8").splitlines():
		try:
			rec = json.loads(line)
		except json.JSONDecodeError:
			continue
		if rec.get("type") not in ("user", "assistant"):
			continue
		msg = rec.get("message") or {}
		who = "USER / TOOL RESULT" if rec["type"] == "user" else "ASSISTANT"
		chunk = []
		for b in _blocks(msg.get("content")):
			kind = b.get("type")
			if kind == "text" and b.get("text", "").strip():
				chunk.append(b["text"].strip())
			elif kind == "thinking" and b.get("thinking", "").strip():
				chunk.append("<details><summary>thinking</summary>\n\n" + b["thinking"].strip() + "\n\n</details>")
			elif kind == "tool_use":
				args = json.dumps(b.get("input", {}), ensure_ascii=False, indent=1)
				if len(args) > max_result:
					args = args[:max_result] + f"\n... [{len(args) - max_result} more chars]"
				chunk.append(f"**tool_use** `{b.get('name')}`\n```json\n{args}\n```")
			elif kind == "tool_result":
				text = _result_text(b.get("content"))
				if len(text) > max_result:
					text = text[:max_result] + f"\n... [{len(text) - max_result} more chars, see raw jsonl]"
				chunk.append("**tool_result**\n```\n" + text + "\n```")
		if chunk:
			out += [f"## {who}  `{rec.get('timestamp', '')}`", "", "\n\n".join(chunk), ""]
	return "\n".join(out)


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--max-result-chars", type=int, default=4000)
	ap.add_argument("items", nargs="+", help="NAME=path/to/transcript.jsonl")
	args = ap.parse_args()
	args.out.mkdir(parents=True, exist_ok=True)
	for item in args.items:
		name, path = item.split("=", 1)
		src = Path(path)
		shutil.copy2(src, args.out / f"{name}.jsonl")
		(args.out / f"{name}.md").write_text(to_markdown(src, name, args.max_result_chars), encoding="utf-8")
		print(name, src.stat().st_size, "bytes")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
