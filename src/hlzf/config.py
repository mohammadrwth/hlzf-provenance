"""Runtime configuration, read from the environment and an optional `.env` file.

Every value has a safe default so `uv run hlzf demo` works on a fresh clone without
an API key: without `ZAI_API_KEY` the pipeline runs offline from the committed cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Z.ai list prices in USD per 1M tokens: (input, cached input, output).
# Source: https://docs.z.ai/guides/overview/pricing.md, checked 2026-10-03.
# Update here if Z.ai changes prices; the cost log multiplies these by reported usage.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "glm-5.3": (1.4, 0.26, 4.4),
    "glm-5.3-flash": (0.15, 0.03, 0.50),
    "glm-5.3-flashx": (0.37, 0.075, 1.25),
    "glm-5.2": (1.4, 0.26, 4.4),
    "glm-ocr": (0.03, 0.03, 0.03),
}


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, no interpolation. Real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ('"', "'"):
            value = value[1:].split(value[0], 1)[0]
        else:
            value = value.split(" #", 1)[0].strip()  # inline comment
        os.environ.setdefault(key.strip(), value)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    api_key: str | None
    base_url: str
    extract_model: str
    check_model: str
    ocr_model: str
    reasoning_effort: str
    temperature: float
    budget_usd: float
    offline: bool
    resamples: int
    parallel: int
    reviewer: str
    autocorrect: bool
    corpus_file: Path
    upload_max_mb: float = 25.0
    upload_max_pages: int = 20
    extra: dict[str, str] = field(default_factory=dict)

    # Derived locations -------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hlzf.db"

    @property
    def synthetic_pdf_dir(self) -> Path:
        return self.data_dir / "synthetic"

    @property
    def live(self) -> bool:
        """Live LLM calls are allowed only with a key and without HLZF_OFFLINE=1."""
        return bool(self.api_key) and not self.offline

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.cache_dir, self.pages_dir, self.exports_dir,
                  self.synthetic_pdf_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings(root: Path | None = None, **overrides: object) -> Settings:
    root = Path(root) if root else Path(os.environ.get("HLZF_ROOT", PROJECT_ROOT))
    _load_dotenv(root / ".env")
    data_dir = Path(os.environ.get("HLZF_DATA_DIR", root / "data"))
    values: dict[str, object] = {
        "root": root,
        "data_dir": data_dir,
        "api_key": os.environ.get("ZAI_API_KEY") or None,
        "base_url": os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
        "extract_model": os.environ.get("GLM_EXTRACT_MODEL", "glm-5.3"),
        "check_model": os.environ.get("GLM_CHECK_MODEL", "glm-5.3-flash"),
        "ocr_model": os.environ.get("GLM_OCR_MODEL", "glm-ocr"),
        # GLM-5.3 always reasons; "low" keeps extraction cheap (docs: low|high|max).
        "reasoning_effort": os.environ.get("GLM_REASONING_EFFORT", "low"),
        # Z.ai documents temperature 1.0 for the reasoning models; determinism comes
        # from the response cache, not from temperature.
        "temperature": float(os.environ.get("GLM_TEMPERATURE", "1.0")),
        "budget_usd": float(os.environ.get("LLM_BUDGET_USD", "5")),
        "offline": _bool("HLZF_OFFLINE", False),
        "resamples": int(os.environ.get("HLZF_RESAMPLES", "5")),
        # Model calls in flight at once (text + vision, resamples, OCR). Z.ai rate limits are
        # per account; 429s are retried with backoff.
        "parallel": max(1, int(os.environ.get("HLZF_PARALLEL", "4"))),
        "reviewer": os.environ.get("HLZF_REVIEWER", "reviewer"),
        # Set a disputed table cell to the reading that GLM-OCR and the vision model agree on,
        # when every agreed value is printed in that cell (see pipeline.consensus_correct).
        "autocorrect": _bool("HLZF_AUTOCORRECT", True),
        "corpus_file": root / "corpus.yaml",
        # Uploaded PDFs: HLZF publications have 1-3 pages; the limits bound the cost of a
        # wrong file (every page goes to the text, vision and OCR models).
        "upload_max_mb": float(os.environ.get("HLZF_UPLOAD_MAX_MB", "25")),
        "upload_max_pages": int(os.environ.get("HLZF_UPLOAD_MAX_PAGES", "20")),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]
