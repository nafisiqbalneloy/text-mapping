"""Embed her messages locally (with surrounding context) and build a style profile."""
import json
import re
import sys
import collections
import numpy as np
import ollama

from config import ROOT, HER, EMBED_MODEL

MSGS = ROOT / "data" / "processed" / "messages.jsonl"
VEC = ROOT / "data" / "processed" / "her_vectors.npy"
META = ROOT / "data" / "processed" / "her_meta.json"
PROFILE = ROOT / "data" / "processed" / "style_profile.json"

MIN_CHARS = 12      # skip tiny messages for the retrieval index
BATCH = 64
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def load():
    return [json.loads(l) for l in open(MSGS, encoding="utf-8")]


def build_profile(her):
    lens = [len(m["text"]) for m in her]
    emo = collections.Counter(e for m in her for e in EMOJI.findall(m["text"]))
    words = collections.Counter(w for m in her for w in re.findall(r"\w+", m["text"].lower()))
    prof = {
        "messages": len(her),
        "median_chars": int(np.median(lens)),
        "pct_ends_with_period": round(100 * sum(m["text"].rstrip().endswith(".") for m in her) / len(her), 1),
        "pct_all_lowercase": round(100 * sum(m["text"] == m["text"].lower() for m in her) / len(her), 1),
        "pct_with_emoji": round(100 * sum(bool(EMOJI.search(m["text"])) for m in her) / len(her), 1),
        "pct_with_question": round(100 * sum("?" in m["text"] for m in her) / len(her), 1),
        "top_emoji": [e for e, _ in emo.most_common(10)],
        "top_words": [w for w, c in words.most_common(60) if len(w) > 2][:30],
    }
    json.dump(prof, open(PROFILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return prof


def main():
    msgs = load()
    her = [m for m in msgs if m["who"] == HER]
    print("her messages:", len(her))
    print("profile:", build_profile(her))

    idx = {m["id"]: i for i, m in enumerate(msgs)}
    items = [m for m in her if len(m["text"]) >= MIN_CHARS]
    print("indexing", len(items), "messages")
    # embed her message together with what she was replying to (previous message)
    texts = []
    for m in items:
        i = idx[m["id"]]
        prev = msgs[i - 1]["text"][:200] if i > 0 and msgs[i - 1]["who"] != HER else ""
        texts.append(f"{m['text']}" + (f"\n(replying to: {prev})" if prev else ""))

    vecs = []
    for s in range(0, len(texts), BATCH):
        r = ollama.embed(model=EMBED_MODEL, input=texts[s:s + BATCH])
        vecs.extend(r["embeddings"])
        if (s // BATCH) % 20 == 0:
            print(f"  {s + BATCH}/{len(texts)}", flush=True)
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    np.save(VEC, arr)
    json.dump([{"id": m["id"], "i": idx[m["id"]]} for m in items], open(META, "w"))
    print("done", arr.shape)


if __name__ == "__main__":
    sys.exit(main())
