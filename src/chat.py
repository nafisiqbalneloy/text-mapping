"""Interactive helper. Keep a live conversation and ask for a reading of her newest message.

Run:  run.bat        (or: python src/chat.py [--model qwen2.5:7b])

Type lines like this:
  me: kothay tumi?          -> adds YOUR message to the conversation (no analysis)
  her: ok thik ase          -> adds HER message and analyzes it
  ok thik ase               -> same as "her: ..." (default is her message)
  /show                     -> show the conversation so far
  /undo                     -> remove the last line
  /reset                    -> clear the conversation
  /model qwen2.5:7b         -> switch model
  /quit                     -> exit
Everything stays on this computer.
"""
import argparse
import sys

from analyze import analyze, load
from config import HER, ME, LLM_MODEL, CONTEXT_MESSAGES

MAX_LINE = 300
SEED = 20  # how many real recent messages to start with


def tag(line):
    """Return (speaker, text) from a typed line; speaker is None if no prefix."""
    low = line.lower()
    for key, name in (("me:", ME), ("her:", HER), (HER.lower() + ":", HER), (ME.lower() + ":", ME)):
        if low.startswith(key):
            return name, line[len(key):].strip()
    return None, line.strip()


def main():
    try:  # make sure Bengali / emoji print fine on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--no-seed", action="store_true", help="start with an empty conversation")
    a = ap.parse_args()
    model = a.model

    print("Loading her history ...", flush=True)
    data = load()
    convo = []  # list of (speaker, text)
    if not a.no_seed:  # start from the real, most recent messages of the export
        convo = [(m["who"], m["text"]) for m in data[0][-SEED:]]
    print(f"Ready. Model: {model}. Type /help for commands.")
    if convo:
        print(f"Started with the last {len(convo)} real messages from your export (latest: {data[0][-1]['t'][:16]}).")
        print("Add anything newer with 'me: ...' / 'her: ...', then paste her new message. /show to see the context.\n")
    else:
        print("Start by pasting recent messages (prefix yours with 'me:'), then her new one.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                print(__doc__)
            elif cmd == "/show":
                print("\n".join(f"{w}: {t}" for w, t in convo) or "(empty)")
            elif cmd == "/undo":
                if convo:
                    w, t = convo.pop()
                    print(f"removed: {w}: {t}")
            elif cmd == "/reset":
                convo.clear()
                print("conversation cleared")
            elif cmd == "/model":
                if arg.strip():
                    model = arg.strip()
                print("model:", model)
            else:
                print("unknown command, try /help")
            continue

        who, text = tag(line)
        if not text:
            continue
        if who == ME:
            convo.append((ME, text))
            print("(added)")
            continue

        # her message: analyze using the conversation BEFORE it, then add it
        who = HER
        ctx = "\n".join(f"{w}: {t[:MAX_LINE]}" for w, t in convo[-CONTEXT_MESSAGES:])
        if not ctx:
            print("(no conversation context yet, so the reading will be weak. "
                  "Add recent lines first with 'me: ...' / 'her: ...' for better results.)")
        print()
        try:
            for piece in analyze(text, ctx or "(none provided)", model, data):
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print("\n(stopped)")
        except Exception as e:  # model missing, ollama not running, etc.
            print(f"\nError: {e}")
        print("\n")
        convo.append((HER, text))


if __name__ == "__main__":
    main()
