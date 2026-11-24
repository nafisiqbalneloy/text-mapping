# Local chat-message interpreter (NLP / RAG)

A fully local tool that helps interpret a new message from one person, using your own Telegram chat history with them.
Nothing leaves your computer: embeddings and the language model both run through [Ollama](https://ollama.com).

## How it works

1. **Parse** (`src/parse.py`): converts a Telegram Desktop JSON export into clean text-only JSONL.
2. **Index** (`src/build_index.py`): embeds her messages (plus the message she replied to) with `nomic-embed-text`, and builds a simple style profile (message length, emoji use, punctuation habits).
3. **Analyze** (`src/analyze.py`, `src/chat.py`): for a new message it retrieves similar past messages (with what happened right after them), adds the recent conversation and a code-computed style check, and asks a local LLM for a short, cautious reading and reply ideas. This is retrieval-augmented generation (RAG); no model is trained or fine-tuned.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install ollama numpy
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

1. Export a chat from Telegram Desktop (JSON, text only is fine) and save it as `data/raw/result.json`.
2. Copy your sender names into `src/config_local.py` (git-ignored):
   ```python
   HER = "name as in the export"
   ME = "your name as in the export"
   ```
3. Build the data (set `PYTHONUTF8=1` on Windows so emoji print correctly):
   ```
   python src/parse.py
   python src/build_index.py
   ```
4. Use it: double-click `run.bat`, or `python src/chat.py`.

## Privacy

`data/`, logs, and `src/config_local.py` are git-ignored. Do not commit your chat export or the generated index.

## Limitations (honest)

- Small local models (3B) misread short messages and handle romanized Bengali ("Banglish") poorly. A 7B model is noticeably better but needs about 6 GB of free RAM.
- It cannot read minds. Treat the output as a rough second opinion, not a verdict. Short messages like "ok" are ambiguous for any model.
- Photos, voice and video messages are not analyzed (text only).
