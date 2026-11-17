"""Local educational lending platform based on Home Credit data."""

# Matplotlib is also imported by LightGBM's optional plotting compatibility code.
# Keep its writable cache inside this local project rather than the user's home.
import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / "data" / "cache" / "matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
