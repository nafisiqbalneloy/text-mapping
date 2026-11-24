from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Sender names exactly as they appear in your Telegram export.
# Put your real ones in src/config_local.py (git-ignored) instead of editing these.
HER = "Her"
ME = "Me"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"  # CPU-friendly; a 7B model needs ~6 GB of free RAM
CONTEXT_MESSAGES = 20     # recent messages shown to the model
EXAMPLES = 8              # similar past messages retrieved

try:  # personal overrides, never committed
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
