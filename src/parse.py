"""Convert the Telegram export into a clean JSONL file (local only)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "result.json"
OUT = ROOT / "data" / "processed" / "messages.jsonl"


def flatten(text):
    if isinstance(text, str):
        return text
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in text)


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    chat = data["chats"]["list"][0] if "chats" in data else data
    names = {}
    rows = []
    for m in chat["messages"]:
        if m.get("type") != "message":
            continue
        text = flatten(m.get("text", "")).strip()
        if not text:
            continue
        rows.append({"id": m["id"], "t": m["date"], "who": m.get("from", "?"), "text": text})
        names[m.get("from")] = names.get(m.get("from"), 0) + 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} text messages -> {OUT}")
    print(names)


if __name__ == "__main__":
    sys.exit(main())
