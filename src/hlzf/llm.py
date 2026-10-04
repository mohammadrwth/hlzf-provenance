"""LLM access: Z.ai GLM through the OpenAI-compatible API, with a response cache,
cost log, budget guard and offline mode.

Verified against https://docs.z.ai (2026-10-03) and a first live run the same day:
* base URL `https://api.z.ai/api/paas/v4/`, Bearer auth, OpenAI SDK compatible;
* structured output is JSON mode only (`response_format={"type": "json_object"}`), no JSON
  schema enforcement, so every response is validated with pydantic and repaired once;
* GLM-5.3 and GLM-5.3-Flash always reason (`thinking.type` only accepts "enabled");
  GLM-5.3 is text-only, GLM-5.3-Flash accepts images;
* GLM-OCR runs on the separate `layout_parsing` endpoint, not chat completions. Live
  responses carry `bbox_2d` in pixels of the rendered page (not normalized as documented)
  and render tables as HTML inside `md_results`.

Reproducibility comes from the cache, not from temperature: identical requests are answered
from `data/cache/` and the cache is committed, so `HLZF_OFFLINE=1` replays a full run.

Calls may run in parallel threads (`Gateway.prefetch`): the cache writes atomically, database
access is serialized by a lock, and the budget guard reserves each call's worst-case cost
before it starts, so a parallel batch cannot overshoot `LLM_BUDGET_USD`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .config import PRICES_USD_PER_MTOK, Settings

RETRY_DELAYS_S = (2.0, 6.0, 15.0)  # backoff for rate limits and transient overload
_RETRYABLE = re.compile(r"\b(429|500|502|503|504|1302|1305)\b|rate limit|overload|timed? ?out",
                        re.IGNORECASE)


class OfflineCacheMiss(RuntimeError):
    """A response is needed that is not in the cache and live calls are disabled."""


class BudgetExceeded(RuntimeError):
    """The configured LLM_BUDGET_USD would be exceeded."""


class InvalidModelOutput(RuntimeError):
    pass


class ModelCallFailed(RuntimeError):
    """The API call itself failed (network, auth, rate limit, unsupported parameter)."""


@dataclass
class CallRecord:
    model: str
    purpose: str
    cache_key: str
    prompt_hash: str
    source: str  # live | cache | fixture
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw_text: str = ""
    repaired: bool = False
    retries: int = 0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    price = PRICES_USD_PER_MTOK.get(model)
    if price is None:
        return 0.0
    p_in, p_cached, p_out = price
    fresh = max(0, input_tokens - cached_tokens)
    return round((fresh * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1e6, 6)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating ```json fences."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            t = t[start:end + 1]
    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def is_retryable(err: BaseException) -> bool:
    status = getattr(err, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(_RETRYABLE.search(f"{type(err).__name__} {err}"))


class ResponseCache:
    """One JSON file per response under data/cache/<model>/<key>.json (committed to git)."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, model: str, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        return self.root / safe / f"{key}.json"

    def get(self, model: str, key: str) -> dict[str, Any] | None:
        p = self._path(model, key)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put(self, model: str, key: str, record: dict[str, Any]) -> None:
        p = self._path(model, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(p)  # atomic: a parallel reader never sees half a file


ChatFn = Callable[..., Any]


class LLMClient:
    def __init__(self, settings: Settings, conn: sqlite3.Connection | None = None,
                 chat_fn: ChatFn | None = None, http: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.s = settings
        self.conn = conn
        self.cache = ResponseCache(settings.cache_dir)
        self._chat_fn = chat_fn
        self._http = http
        self._sleep = sleep
        self._lock = threading.RLock()
        self._reserved = 0.0

    # --- plumbing --------------------------------------------------------------------
    def _chat(self, **kwargs: Any) -> Any:
        if self._chat_fn is None:
            with self._lock:
                if self._chat_fn is None:
                    from openai import OpenAI  # imported lazily: offline runs never need it

                    client = OpenAI(api_key=self.s.api_key, base_url=self.s.base_url,
                                    timeout=180, max_retries=0)
                    self._chat_fn = client.chat.completions.create
        return self._chat_fn(**kwargs)

    def spent_usd(self) -> float:
        if self.conn is None:
            return 0.0
        with self._lock:
            r = self.conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE "
                                  "source='live'").fetchone()
        return float(r[0])

    def _log(self, rec: CallRecord, document_id: str | None) -> None:
        if self.conn is None:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO llm_calls (ts, document_id, purpose, model, cache_key, prompt_hash, "
                "input_tokens, cached_tokens, output_tokens, cost_usd, latency_ms, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(timespec="seconds"), document_id, rec.purpose,
                 rec.model, rec.cache_key, rec.prompt_hash, rec.input_tokens,
                 rec.cached_tokens, rec.output_tokens, rec.cost_usd, rec.latency_ms,
                 rec.source))

    def _reserve(self, model: str, est_input_tokens: int) -> float:
        """Reserve a conservative cost estimate for one call, or refuse it."""
        price = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0, 0.0))
        # Conservative: full input price plus 8k output tokens of reasoning + JSON.
        estimate = (est_input_tokens * price[0] + 8000 * price[2]) / 1e6
        with self._lock:
            spent = self.spent_usd()
            if spent + self._reserved + estimate > self.s.budget_usd:
                raise BudgetExceeded(
                    f"LLM budget would be exceeded: spent ${spent:.4f}, in flight "
                    f"${self._reserved:.4f}, next call up to ${estimate:.4f}, limit "
                    f"${self.s.budget_usd:.2f} (LLM_BUDGET_USD)")
            self._reserved += estimate
        return estimate

    def _release(self, estimate: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - estimate)

    def _with_retries(self, fn: Callable[[], Any], rec: CallRecord) -> Any:
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            try:
                return fn()
            except Exception as err:
                if attempt == len(RETRY_DELAYS_S) or not is_retryable(err):
                    raise ModelCallFailed(
                        f"{rec.model}: {type(err).__name__}: {err}"[:400]) from err
                rec.retries += 1
                self._sleep(RETRY_DELAYS_S[attempt])
        raise AssertionError("unreachable")

    # --- chat completions in JSON mode -----------------------------------------------
    def chat_json(self, *, model: str, system: str, user: str | list[dict[str, Any]],
                  schema: type[BaseModel], purpose: str, prompt_version: str,
                  sample: int = 0, document_id: str | None = None,
                  temperature: float | None = None,
                  input_key: str | None = None) -> tuple[BaseModel, CallRecord]:
        # `input_key` lets callers key the cache on a stable identity of the input (e.g. the
        # PDF sha256 + page numbers for image input) instead of the raw payload bytes.
        input_digest = input_key or sha256_text(
            json.dumps(user, ensure_ascii=False, sort_keys=True))
        prompt_hash = sha256_text(system)[:16]
        key = sha256_text(json.dumps(
            {"model": model, "prompt_version": prompt_version, "prompt_hash": prompt_hash,
             "input": input_digest, "sample": sample}, sort_keys=True))[:32]

        cached = self.cache.get(model, key)
        if cached is not None:
            rec = CallRecord(model=model, purpose=purpose, cache_key=key,
                             prompt_hash=prompt_hash, source="cache",
                             input_tokens=cached.get("usage", {}).get("input_tokens", 0),
                             output_tokens=cached.get("usage", {}).get("output_tokens", 0),
                             # cost/latency of the original live call, for reporting;
                             # the budget guard only counts source='live' rows
                             cost_usd=float(cached.get("cost_usd", 0.0)),
                             latency_ms=int(cached.get("latency_ms", 0)),
                             raw_text=cached["response_text"],
                             repaired=cached.get("repaired", False))
            parsed = schema.model_validate(extract_json(cached["response_text"]))
            self._log(rec, document_id)
            return parsed, rec

        if not self.s.live:
            raise OfflineCacheMiss(
                f"No cached {model} response for {purpose} (key {key}). Set ZAI_API_KEY "
                "and unset HLZF_OFFLINE to call the API.")

        approx_tokens = len(json.dumps(user, ensure_ascii=False)) // 3 + len(system) // 3
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        rec = CallRecord(model=model, purpose=purpose, cache_key=key,
                         prompt_hash=prompt_hash, source="live")
        text = self._paid_completion(model, messages, rec, temperature, approx_tokens)
        try:
            parsed = schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as err:
            # One repair round: show the model its output and the validation error.
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "Your previous answer failed validation:\n"
                    f"{str(err)[:2000]}\n"
                    "Return the corrected JSON object only, same schema, no prose.")},
            ]
            try:
                text = self._paid_completion(model, messages, rec, temperature,
                                             approx_tokens)
            except (BudgetExceeded, ModelCallFailed):
                self._log(rec, document_id)  # the first call was paid for; count it
                raise
            rec.repaired = True
            try:
                parsed = schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as err2:
                self._log(rec, document_id)
                raise InvalidModelOutput(f"{model} output invalid after repair: {err2}") from err2

        rec.raw_text = text
        self.cache.put(model, key, {
            "key": key, "model": model, "purpose": purpose, "prompt_version": prompt_version,
            "prompt_hash": prompt_hash, "input_sha256": input_digest, "sample": sample,
            "response_text": text, "repaired": rec.repaired,
            "usage": {"input_tokens": rec.input_tokens, "cached_tokens": rec.cached_tokens,
                      "output_tokens": rec.output_tokens},
            "cost_usd": rec.cost_usd, "latency_ms": rec.latency_ms,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "source": "live",
        })
        self._log(rec, document_id)
        return parsed, rec

    def _paid_completion(self, model: str, messages: list[dict[str, Any]], rec: CallRecord,
                         temperature: float | None, approx_tokens: int) -> str:
        reserved = self._reserve(model, approx_tokens)
        try:
            return self._complete(model, messages, rec, temperature)
        finally:
            self._release(reserved)

    def _complete(self, model: str, messages: list[dict[str, Any]], rec: CallRecord,
                  temperature: float | None) -> str:
        t0 = time.monotonic()
        resp = self._with_retries(lambda: self._chat(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.s.temperature if temperature is None else temperature,
            extra_body={"thinking": {"type": "enabled"},
                        "reasoning_effort": self.s.reasoning_effort},
        ), rec)
        rec.latency_ms += int((time.monotonic() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            inp = int(getattr(usage, "prompt_tokens", 0) or 0)
            out = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
            rec.input_tokens += inp
            rec.output_tokens += out
            rec.cached_tokens += cached
            rec.cost_usd = round(rec.cost_usd + cost_usd(model, inp, cached, out), 6)
        return resp.choices[0].message.content or ""

    # --- GLM-OCR layout parsing ------------------------------------------------------
    def layout_parse(self, *, pdf_bytes: bytes, page_no: int, document_id: str | None = None
                     ) -> tuple[dict[str, Any], CallRecord]:
        """OCR one PDF page with GLM-OCR (`POST /layout_parsing`)."""
        model = self.s.ocr_model
        key = sha256_text(json.dumps({"model": model, "pdf": hashlib.sha256(pdf_bytes)
                                      .hexdigest(), "page": page_no}, sort_keys=True))[:32]
        rec = CallRecord(model=model, purpose="ocr", cache_key=key, prompt_hash="",
                         source="cache")
        cached = self.cache.get(model, key)
        if cached is not None:
            self._log(rec, document_id)
            return cached["response"], rec
        if not self.s.live:
            raise OfflineCacheMiss(f"No cached OCR for page {page_no} (key {key}).")
        reserved = self._reserve(model, 3000)
        try:
            data = self._layout_parse_live(model, rec, pdf_bytes, page_no)
        finally:
            self._release(reserved)
        self.cache.put(model, key, {"key": key, "model": model, "page": page_no,
                                    "response": data, "source": "live",
                                    "created_at": datetime.now(UTC).isoformat()})
        self._log(rec, document_id)
        return data, rec

    def _layout_parse_live(self, model: str, rec: CallRecord, pdf_bytes: bytes,
                           page_no: int) -> dict[str, Any]:
        rec.source = "live"
        http = self._http or httpx.Client(timeout=180)
        t0 = time.monotonic()
        b64 = base64.b64encode(pdf_bytes).decode("ascii")

        def post(file_value: str) -> httpx.Response:
            resp = http.post(
                self.s.base_url.rstrip("/") + "/layout_parsing",
                headers={"Authorization": f"Bearer {self.s.api_key}"},
                json={"model": model, "file": file_value,
                      "start_page_id": page_no, "end_page_id": page_no},
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()  # retried by _with_retries
            return resp

        resp = None
        # The docs accept "URL or base64" without fixing the base64 form; try raw base64
        # first, then a data URI.
        for file_value in (b64, f"data:application/pdf;base64,{b64}"):
            resp = self._with_retries(lambda v=file_value: post(v), rec)
            if resp.status_code != 400:
                break
        assert resp is not None
        if resp.status_code >= 400:
            raise ModelCallFailed(f"{model}: HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        rec.latency_ms = int((time.monotonic() - t0) * 1000)
        usage = data.get("usage") or {}
        rec.input_tokens = int(usage.get("prompt_tokens", 0))
        rec.output_tokens = int(usage.get("completion_tokens", 0))
        rec.cost_usd = cost_usd(model, rec.input_tokens, 0, rec.output_tokens)
        return data
