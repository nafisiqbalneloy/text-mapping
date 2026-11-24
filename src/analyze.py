"""Interpret a new message from her, using recent context + her own past style. Fully local.

Usage:
  python src/analyze.py "the new message"
  python src/analyze.py "the new message" --context ctx.txt     # context as lines: "Name: text"
  (no --context: uses the last messages of the export as context)
"""
import argparse
import json
import re
import sys
import numpy as np
import ollama

from config import ROOT, HER, ME, EMBED_MODEL, LLM_MODEL, CONTEXT_MESSAGES, EXAMPLES

P = ROOT / "data" / "processed"

SYSTEM = f"""You help {ME} understand messages from his girlfriend ({HER}). You are given her typical writing style, \
examples of how she has written in similar situations, the recent conversation, and a new message.

Rules:
- You cannot read minds. Offer 2-3 plausible readings ranked by likelihood, each with a confidence (low/medium/high).
- Base every reading on evidence: quote her exact words, the conversation context, or how she usually writes. \
If her style differs from the usual (e.g. a normally warm, emoji-heavy writer becomes short and formal), say so.
- Consider: sarcasm, passive-aggression, genuine happiness, seeking reassurance, tiredness/stress, teasing, hurt, plain literal meaning.
- The plain literal meaning is often correct. Do not invent drama.
- State what is ambiguous and what single question or check would resolve it.
- Suggest 2-3 ways {ME} could respond; include asking her directly and kindly.
- Her messages are casual Bengali written in English letters (Banglish), mixed with English. Casual is her normal; \
do NOT call a message "formal".
- The STYLE CHECK is always correct. Mention a style difference ONLY if the check lists one. If it says nothing is \
unusual, do not talk about punctuation, length or style at all.
- If you do not understand the Banglish words, say so ("I can't tell what this means") instead of guessing a meaning.
- Always read the new message in light of the RECENT CONVERSATION: what was {ME} just saying or asking? A short reply \
usually answers that. Say what it is answering.
- The SIMILAR PAST MESSAGES show what happened right after each one ("then: ..."). Use that to judge how such \
messages usually go. Do not treat them as the answer.
- If the message is very short or the context is thin, say plainly that it is hard to read, and give only 1-2 \
cautious readings.
- Readings must not contradict each other (never list both "hurt" and "happy" unless truly equally likely).
- Write suggested replies in short casual Banglish like {ME} writes in the conversation, not formal English.
- Keep it short: at most 120 words in total. Use exactly this format:
  Most likely: <one sentence> (confidence: low/medium/high)
  Other possibility: <one sentence, or "none">
  Why: <one or two short points quoting her words or the context>
  Not sure about: <one line>
  Reply ideas:
  1) ...
  2) ...
  3) a kind, direct question asking her how she feels"""


def load():
    msgs = [json.loads(l) for l in open(P / "messages.jsonl", encoding="utf-8")]
    vecs = np.load(P / "her_vectors.npy")
    meta = json.load(open(P / "her_meta.json"))
    prof = json.load(open(P / "style_profile.json", encoding="utf-8"))
    return msgs, vecs, meta, prof


def retrieve(msg, context_text, msgs, vecs, meta):
    q = ollama.embed(model=EMBED_MODEL, input=[msg + (f"\n(replying to: {context_text[-200:]})" if context_text else "")])
    q = np.array(q["embeddings"][0], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    top = np.argsort(-(vecs @ q))[:EXAMPLES]
    out = []
    for k in top:
        i = meta[k]["i"]
        prev = msgs[i - 1] if i > 0 else None
        line = f'{HER}: "{msgs[i]["text"][:300]}"'
        if prev and prev["who"] != HER:
            line = f'{ME}: "{prev["text"][:150]}"  ->  ' + line
        # what happened right after: shows how the exchange actually went
        after = [f'{m["who"]}: "{m["text"][:80]}"' for m in msgs[i + 1:i + 3]]
        if after:
            line += "   [then: " + " / ".join(after) + "]"
        out.append(line)
    return out


EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def yn(b):
    return "YES" if b else "NO"


def facts(msg, prof):
    """Plain-language style check, computed by code (so the model never has to guess)."""
    t = msg.strip()
    n = len(t)
    med = prof["median_chars"]
    length = "about as long as usual" if med * 0.5 <= n <= med * 2 else ("much shorter than usual" if n < med * 0.5 else "much longer than usual")
    # Only report REAL deviations. Describing normal things ("no period") makes small models repeat them as if they were findings.
    notes = []
    if length != "about as long as usual":
        notes.append(f"It is {length} ({n} characters; her median is {med}).")
    if t.endswith("."):
        notes.append(f"It ends with a period, which is unusual for her (only {prof['pct_ends_with_period']}% do).")
    if t.endswith("...") or t.endswith("…"):
        notes.append("It trails off with dots.")
    if len(t) > 4 and t.isupper():
        notes.append("It is written in ALL CAPITAL LETTERS.")
    if EMOJI.search(t):
        notes.append("It contains an emoji.")
    return "\n".join(f"- {x}" for x in notes) or "- Nothing unusual: the style matches how she normally writes."


def analyze(message, ctx, model, data):
    """Yield the model's answer in chunks. `data` is the tuple returned by load()."""
    msgs, vecs, meta, prof = data
    examples = retrieve(message, ctx, msgs, vecs, meta)

    user = f"""HER USUAL STYLE: {json.dumps(prof, ensure_ascii=False)}

SIMILAR PAST MESSAGES FROM HER (for style reference only; their real meaning is not labeled):
{chr(10).join(examples)}

RECENT CONVERSATION:
{ctx}

NEW MESSAGE FROM {HER}: "{message}"
STYLE CHECK OF THIS MESSAGE (computed by code, always correct; use it silently, do not copy it into your answer):
{facts(message, prof)}

Analyze the new message."""
    for chunk in ollama.chat(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                             stream=True, options={"temperature": 0.3, "num_ctx": 4096}):
        yield chunk["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    ap.add_argument("--context", help="text file with recent conversation, one 'Name: text' per line")
    ap.add_argument("--model", default=LLM_MODEL)
    a = ap.parse_args()

    data = load()
    if a.context:
        ctx = open(a.context, encoding="utf-8").read().strip()
    else:
        ctx = "\n".join(f'{m["who"]}: {m["text"][:300]}' for m in data[0][-CONTEXT_MESSAGES:])
    for piece in analyze(a.message, ctx, a.model, data):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    sys.exit(main())
