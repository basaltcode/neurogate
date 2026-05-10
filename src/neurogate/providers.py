from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx
from collections import OrderedDict
from google import genai
from google.genai import types

log = logging.getLogger(__name__)


# OpenAI-совместимые клиенты (cline, OpenAI SDK pydantic-модели и т.п.) при echo
# истории выкидывают неизвестные поля у tool_call, теряя `_gemini_thought_signature`.
# Gemini 3.x за это бьёт 400. Дублируем подпись в серверном LRU по id вызова и
# параллельно пишем в sqlite, чтобы пережить рестарт процесса (deploy переписывает
# код но не stats.db — следующий echo от клиента всё равно найдёт подпись).
_GEMINI_SIG_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_GEMINI_SIG_CACHE_MAX = 4096
_GEMINI_SIG_DB_RETAIN = 7 * 86400
_GEMINI_SIG_CONN: sqlite3.Connection | None = None
_GEMINI_SIG_CONN_TRIED = False


def _sig_db() -> sqlite3.Connection | None:
    global _GEMINI_SIG_CONN, _GEMINI_SIG_CONN_TRIED
    if _GEMINI_SIG_CONN is not None or _GEMINI_SIG_CONN_TRIED:
        return _GEMINI_SIG_CONN
    _GEMINI_SIG_CONN_TRIED = True
    db_path = os.getenv("NEUROGATE_STATS_DB", "stats.db")
    try:
        conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS gemini_signatures ("
            "call_id TEXT PRIMARY KEY, sig BLOB NOT NULL, ts INTEGER NOT NULL)"
        )
        conn.execute(
            "DELETE FROM gemini_signatures WHERE ts < ?",
            (int(time.time()) - _GEMINI_SIG_DB_RETAIN,),
        )
        _GEMINI_SIG_CONN = conn
    except Exception as exc:
        log.warning("gemini sig persistence disabled (%s): %s", db_path, exc)
    return _GEMINI_SIG_CONN


def _sig_cache_put(call_id: str | None, sig: bytes | None) -> None:
    if not call_id or not sig:
        return
    _GEMINI_SIG_CACHE[call_id] = sig
    _GEMINI_SIG_CACHE.move_to_end(call_id)
    while len(_GEMINI_SIG_CACHE) > _GEMINI_SIG_CACHE_MAX:
        _GEMINI_SIG_CACHE.popitem(last=False)
    conn = _sig_db()
    if conn is not None:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO gemini_signatures(call_id, sig, ts) VALUES(?, ?, ?)",
                (call_id, sig, int(time.time())),
            )
        except Exception as exc:
            log.warning("gemini sig persist write failed: %s", exc)


def _sig_cache_get(call_id: str | None) -> bytes | None:
    if not call_id:
        return None
    sig = _GEMINI_SIG_CACHE.get(call_id)
    if sig is not None:
        _GEMINI_SIG_CACHE.move_to_end(call_id)
        return sig
    conn = _sig_db()
    if conn is None:
        return None
    try:
        cur = conn.execute(
            "SELECT sig FROM gemini_signatures WHERE call_id = ?", (call_id,)
        )
        row = cur.fetchone()
    except Exception as exc:
        log.warning("gemini sig persist read failed: %s", exc)
        return None
    if row is None:
        return None
    sig = bytes(row[0])
    _GEMINI_SIG_CACHE[call_id] = sig
    _GEMINI_SIG_CACHE.move_to_end(call_id)
    while len(_GEMINI_SIG_CACHE) > _GEMINI_SIG_CACHE_MAX:
        _GEMINI_SIG_CACHE.popitem(last=False)
    return sig


class ProviderCallResult:
    """Shape returned by a provider: text + optional usage info + optional tool calls."""

    __slots__ = (
        "text", "prompt_tokens", "completion_tokens", "cached_tokens",
        "tool_calls", "finish_reason",
    )

    def __init__(
        self,
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str = "stop",
        cached_tokens: int = 0,
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason


class AudioTranscribeResult:
    """Shape returned by a transcription provider."""

    __slots__ = ("text", "duration_s", "language", "raw")

    def __init__(
        self,
        text: str,
        duration_s: float | None = None,
        language: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.duration_s = duration_s
        self.language = language
        self.raw = raw or {}


class AudioSpeechResult:
    """Shape returned by a TTS provider — raw audio bytes + media type."""

    __slots__ = ("audio", "content_type", "model", "voice", "raw")

    def __init__(
        self,
        audio: bytes,
        content_type: str = "audio/mpeg",
        model: str | None = None,
        voice: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.audio = audio
        self.content_type = content_type
        self.model = model
        self.voice = voice
        self.raw = raw or {}


class AudioGenerationResult:
    """Shape returned by a text-to-audio (SFX/ambient) provider — raw audio bytes
    + media type + optional duration / model id. Mirrors AudioSpeechResult but
    without `voice` and intended for non-speech generation (sound effects,
    ambience, foley) rather than speech synthesis."""

    __slots__ = ("audio", "content_type", "model", "duration_s", "raw")

    def __init__(
        self,
        audio: bytes,
        content_type: str = "audio/wav",
        model: str | None = None,
        duration_s: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.audio = audio
        self.content_type = content_type
        self.model = model
        self.duration_s = duration_s
        self.raw = raw or {}


class TranslationResult:
    """Shape returned by a translation provider — translated text + detected source
    language (if auto-detection was requested) + raw response for debugging."""

    __slots__ = ("text", "source_lang", "target_lang", "provider_model", "raw")

    def __init__(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
        provider_model: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.provider_model = provider_model
        self.raw = raw or {}


class ImageGenerationResult:
    """Shape returned by an image-generation provider. `images` is a list of
    dicts each shaped like OpenAI's image response items: {"url": ...} or
    {"b64_json": ...}. Extra per-image fields (seed, revised_prompt) are
    allowed and passed through unchanged."""

    __slots__ = ("images", "model", "raw")

    def __init__(
        self,
        images: list[dict[str, Any]],
        model: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.images = images
        self.model = model
        self.raw = raw or {}


class ModerationResult:
    """Shape returned by a moderation provider — список results (по одному на каждый
    input). Каждый result имеет `flagged`, `categories` (bool по каждой категории) и
    `category_scores` (float [0..1] либо None, если провайдер не возвращает scores —
    например, Llama Guard выдаёт только бинарный verdict + S-коды).

    Категории нормализуются под OpenAI omni-moderation схему:
      sexual, sexual/minors, harassment, harassment/threatening, hate,
      hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent,
      self-harm/instructions, violence, violence/graphic, prompt_injection,
      jailbreak.
    """

    __slots__ = ("results", "model", "raw")

    def __init__(
        self,
        results: list[dict[str, Any]],
        model: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.results = results
        self.model = model
        self.raw = raw or {}


class EmbeddingResult:
    """Shape returned by an embeddings provider. `vectors` — список list[float]
    в порядке входных текстов. `dim` — длина одного вектора. `prompt_tokens` —
    суммарное количество токенов входа (когда провайдер сообщает; иначе 0)."""

    __slots__ = ("vectors", "model", "dim", "prompt_tokens", "raw")

    def __init__(
        self,
        vectors: list[list[float]],
        model: str | None = None,
        dim: int | None = None,
        prompt_tokens: int = 0,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.vectors = vectors
        self.model = model
        self.dim = dim if dim is not None else (len(vectors[0]) if vectors else 0)
        self.prompt_tokens = prompt_tokens
        self.raw = raw or {}


class RerankResult:
    """Shape returned by a reranker. `results` — отсортированный по убыванию
    relevance_score список {index, relevance_score, document?}, где `index`
    указывает на позицию документа во входном списке. `top_n` ограничение
    применяет провайдер. `total_tokens` — токены billing-pool (Jina/Voyage),
    либо search_units для Cohere."""

    __slots__ = ("results", "model", "total_tokens", "raw")

    def __init__(
        self,
        results: list[dict[str, Any]],
        model: str | None = None,
        total_tokens: int = 0,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.results = results
        self.model = model
        self.total_tokens = total_tokens
        self.raw = raw or {}


class Provider(Protocol):
    name: str
    supports_tools: bool
    supports_vision: bool
    context_window: int | None

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> ProviderCallResult: ...

    def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> AsyncIterator[bytes]: ...


def _extract_cached_tokens(usage: Any) -> int:
    """Handle the three common shapes: OpenAI/DeepSeek prompt_tokens_details.cached_tokens,
    Anthropic cache_read_input_tokens, plain cached_tokens."""
    if not usage:
        return 0
    getter = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (
        lambda k: getattr(usage, k, 0) or 0
    )
    direct = getter("cached_tokens") or 0
    if direct:
        return direct
    anthropic = getter("cache_read_input_tokens") or 0
    if anthropic:
        return anthropic
    details = getter("prompt_tokens_details")
    if details:
        if isinstance(details, dict):
            return details.get("cached_tokens", 0) or 0
        return getattr(details, "cached_tokens", 0) or 0
    return 0


def _text_from_content(content: Any) -> str:
    """Flatten OpenAI content field (string or list of parts) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _tool_call_args_to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {"_value": raw}


def _messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str, list[Any]]:
    """Convert OpenAI-format messages to (system_instruction, gemini_contents).

    Handles four roles: system (lifted out), user/assistant (text + tool_calls),
    and tool (mapped to a `function_response` Part under role=user, per Gemini).
    """
    # Gemini требует FunctionResponse.name непустым, но OpenAI-клиенты (cline)
    # шлют tool-сообщение с tool_call_id, а имя функции — только в предыдущем
    # assistant.tool_calls. Строим мап id→name для подстановки.
    tool_name_by_id: dict[str, str] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            tc_id = tc.get("id")
            tc_name = (tc.get("function") or {}).get("name")
            if tc_id and tc_name:
                tool_name_by_id[tc_id] = tc_name

    system_parts: list[str] = []
    contents: list[Any] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            txt = _text_from_content(m.get("content"))
            if txt:
                system_parts.append(txt)
            continue

        if role == "tool":
            response_obj: dict[str, Any]
            raw = m.get("content")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    response_obj = parsed if isinstance(parsed, dict) else {"result": parsed}
                except json.JSONDecodeError:
                    response_obj = {"result": raw}
            elif isinstance(raw, dict):
                response_obj = raw
            else:
                response_obj = {"result": _text_from_content(raw)}
            tc_id = m.get("tool_call_id")
            fn_name = m.get("name") or tool_name_by_id.get(tc_id or "") or "tool_result"
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=tc_id,
                                name=fn_name,
                                response=response_obj,
                            )
                        )
                    ],
                )
            )
            continue

        if role == "assistant":
            parts: list[Any] = []
            txt = _text_from_content(m.get("content"))
            if txt:
                parts.append(types.Part(text=txt))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                sig_b64 = tc.get("_gemini_thought_signature")
                sig = base64.b64decode(sig_b64) if isinstance(sig_b64, str) and sig_b64 else None
                if sig is None:
                    sig = _sig_cache_get(tc.get("id"))
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=tc.get("id"),
                            name=fn.get("name", ""),
                            args=_tool_call_args_to_dict(fn.get("arguments")),
                        ),
                        thought_signature=sig,
                    )
                )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        # user / other → role=user with single text Part
        txt = _text_from_content(m.get("content")) or " "
        contents.append(types.Content(role="user", parts=[types.Part(text=txt)]))

    return "\n\n".join(system_parts), contents


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[Any]:
    decls = []
    for t in tools:
        fn = t.get("function") or {}
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        decls.append(
            types.FunctionDeclaration(
                name=fn.get("name", ""),
                description=fn.get("description") or "",
                parameters_json_schema=params,
            )
        )
    return [types.Tool(function_declarations=decls)] if decls else []


def _openai_tool_choice_to_gemini(tool_choice: Any) -> Any | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(tool_choice)
        if mode is None:
            return None
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=mode)
        )
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[name]
                )
            )
    return None


def _gemini_response_to_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    """Walk candidate parts so we can capture `thought_signature` alongside the call.

    Gemini 3.x rejects replayed function_call parts that are missing the signature,
    so we stash it on the outgoing tool_call as a non-standard `_gemini_thought_signature`
    (base64 of the original bytes) and restore it on the next turn.
    """
    tool_calls: list[dict[str, Any]] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        pending_sig: bytes | None = None
        for part in getattr(content, "parts", None) or []:
            part_sig = getattr(part, "thought_signature", None)
            fc = getattr(part, "function_call", None)
            if fc is None or not getattr(fc, "name", None):
                # thought / text part — запоминаем подпись для следующего fc
                if part_sig:
                    pending_sig = part_sig
                continue
            args = dict(fc.args) if getattr(fc, "args", None) else {}
            call_id = getattr(fc, "id", None) or f"call_{len(tool_calls)}"
            entry: dict[str, Any] = {
                "id": call_id,
                "type": "function",
                "function": {"name": fc.name, "arguments": json.dumps(args)},
            }
            sig = part_sig or pending_sig
            pending_sig = None
            if sig:
                entry["_gemini_thought_signature"] = base64.b64encode(sig).decode("ascii")
                _sig_cache_put(call_id, sig)
            tool_calls.append(entry)
    return tool_calls or None


class GeminiProvider:
    supports_tools = True
    supports_vision = True
    # Gemini 2.5+ и 3.x принимают inline audio как часть multimodal вызова.
    # Прямого /audio/transcriptions нет — транскрипция идёт через generate_content
    # с inline_data Blob. Дороже и медленнее Whisper, но работает как last-resort.
    supports_audio = True

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        *,
        rpd: int | None = None,
        rpm: int | None = None,
        context_window: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        reasoning: bool = False,
    ) -> None:
        self.name = name
        self._client = genai.Client(api_key=api_key)
        self._api_key = api_key
        self._model = model
        self.rpd = rpd
        self.rpm = rpm
        self.context_window = context_window
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.reasoning = reasoning

    # reasoning_effort → thinking_budget. Budgets kept small so they fit under
    # typical max_output_tokens; "high" still leaves room for a real answer.
    _THINKING_BUDGETS = {"none": 0, "low": 256, "medium": 1024, "high": 8192}

    def _build_config(
        self,
        *,
        system: str,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        reasoning_effort: str | None = None,
        web_search: bool = False,
    ) -> types.GenerateContentConfig | None:
        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if reasoning_effort in self._THINKING_BUDGETS:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._THINKING_BUDGETS[reasoning_effort]
            )
        gemini_tools: list[Any] = []
        if tools:
            gemini_tools.extend(_openai_tools_to_gemini(tools))
            tc = _openai_tool_choice_to_gemini(tool_choice)
            if tc is not None:
                config_kwargs["tool_config"] = tc
            # Disable Gemini's automatic tool loop — we want raw function_calls
            # back so the client can execute and send a follow-up turn.
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )
        # Native grounding: google_search tool (server-side, returns
        # grounding_metadata with citation URIs, no function-call roundtrip).
        if web_search:
            gemini_tools.append(types.Tool(google_search=types.GoogleSearch()))
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> ProviderCallResult:
        # request_extras is OpenAI-dialect (prompt_cache_key, response_format, etc.);
        # Gemini doesn't speak those, so we ignore silently — except `reasoning_effort`,
        # which we translate into a ThinkingConfig budget.
        system, contents = _messages_to_gemini(messages)
        re_effort = (request_extras or {}).get("reasoning_effort")
        config = self._build_config(
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=re_effort,
            web_search=web_search,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents if contents else [types.Content(role="user", parts=[types.Part(text=" ")])],
            config=config,
        )
        text = (response.text or "").strip() if response.text else ""
        tool_calls = _gemini_response_to_tool_calls(response)
        if not text and not tool_calls:
            raise RuntimeError(f"{self.name} empty response")
        usage = getattr(response, "usage_metadata", None)
        finish_reason = "tool_calls" if tool_calls else "stop"
        return ProviderCallResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
    ) -> AudioTranscribeResult:
        # response_format игнорируется кроме "json": Gemini не выдаёт SRT/VTT изначально.
        lang_hint = f" The audio is in {language}." if language else ""
        instruction = prompt or (
            "Transcribe the audio verbatim." + lang_hint
            + " Return only the transcription, no preamble or commentary."
        )
        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.0,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[types.Content(role="user", parts=[
                types.Part(text=instruction),
                types.Part(inline_data=types.Blob(mime_type=mime_type, data=audio)),
            ])],
            config=config,
        )
        text = (response.text or "").strip() if response.text else ""
        if not text:
            raise RuntimeError(f"{self.name} empty response")
        return AudioTranscribeResult(
            text=text,
            language=language,
            raw={"model": self._model, "provider": self.name},
        )

    async def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> AsyncIterator[bytes]:
        system, contents = _messages_to_gemini(messages)
        re_effort = (request_extras or {}).get("reasoning_effort")
        config = self._build_config(
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=re_effort,
            web_search=web_search,
        )
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        header = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self._model,
        }
        got_text = False
        tool_call_index = 0
        had_tool_calls = False
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents if contents else [types.Content(role="user", parts=[types.Part(text=" ")])],
            config=config,
        )
        pending_sig: bytes | None = None
        async for chunk in stream:
            candidates = getattr(chunk, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    text = getattr(part, "text", None) or ""
                    fc = getattr(part, "function_call", None)
                    part_sig = getattr(part, "thought_signature", None)
                    if text:
                        got_text = True
                        if part_sig and fc is None:
                            pending_sig = part_sig
                        payload = {
                            **header,
                            "choices": [
                                {"index": 0, "delta": {"content": text}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n".encode()
                    elif fc is None and part_sig:
                        pending_sig = part_sig
                    if fc is not None and getattr(fc, "name", None):
                        had_tool_calls = True
                        args = dict(fc.args) if getattr(fc, "args", None) else {}
                        call_id = getattr(fc, "id", None) or f"call_{tool_call_index}"
                        tc_entry: dict[str, Any] = {
                            "index": tool_call_index,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(args),
                            },
                        }
                        sig = part_sig or pending_sig
                        pending_sig = None
                        if sig:
                            tc_entry["_gemini_thought_signature"] = base64.b64encode(sig).decode("ascii")
                            _sig_cache_put(call_id, sig)
                        payload = {
                            **header,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [tc_entry]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n".encode()
                        tool_call_index += 1

        if not got_text and not had_tool_calls:
            raise RuntimeError(f"{self.name} empty response")
        finish_reason = "tool_calls" if had_tool_calls else "stop"
        final = {
            **header,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"


class OpenAICompatProvider:
    """Any provider speaking OpenAI /chat/completions: Groq, Cerebras, SambaNova, NVIDIA, OpenRouter, Z.ai, …"""

    supports_tools = True
    supports_audio = False
    supports_vision = False  # opt-in per-config via `vision: true` for vision-capable models

    def __init__(
        self,
        name: str,
        *,
        kind: str = "openai",
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        auth_scheme: str = "Bearer",
        timeout: float = 60.0,
        rpd: int | None = None,
        rpm: int | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        reasoning: bool = False,
        quota_limited: bool = False,
    ) -> None:
        self.name = name
        self._kind = kind
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or {}
        self._auth_scheme = auth_scheme
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.context_window = context_window
        self._max_output_tokens = max_output_tokens
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.reasoning = reasoning
        self.quota_limited = quota_limited

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip Anthropic-only `cache_control` markers and flatten text-only
        list-content to a plain string. OpenAI accepts list-of-parts widely,
        but Cerebras/Groq/SambaNova reject `cache_control` and reject list
        content in `system` outright.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            mm = dict(m)
            content = mm.get("content")
            if isinstance(content, list):
                cleaned_parts: list[Any] = []
                text_only = True
                for p in content:
                    if not isinstance(p, dict):
                        cleaned_parts.append(p)
                        text_only = False
                        continue
                    pp = {k: v for k, v in p.items() if k != "cache_control"}
                    if pp.get("type") not in (None, "text"):
                        text_only = False
                    cleaned_parts.append(pp)
                if text_only:
                    mm["content"] = "".join(
                        p.get("text", "") for p in cleaned_parts if isinstance(p, dict)
                    )
                else:
                    mm["content"] = cleaned_parts
            elif isinstance(content, dict):
                mm["content"] = {k: v for k, v in content.items() if k != "cache_control"}
            out.append(mm)
        return out

    def _normalize_extras(self, request_extras: dict[str, Any] | None) -> dict[str, Any]:
        """Translate our `reasoning_effort` flag into per-provider payload shape.

        - Z.AI uses `thinking: {type: enabled|disabled}` — map all effort levels.
        - OpenAI-compat reasoning models accept `reasoning_effort` natively ("low"/"medium"/"high").
          We strip "none" since there's no standard off-switch there; the model will think
          by default, but at least we don't send an invalid value.
        - Non-reasoning providers: strip `reasoning_effort` entirely to avoid 400 errors.
        """
        if not request_extras:
            return {}
        extras = dict(request_extras)
        # Defensive: claude-code-router can forward `reasoning` as a top-level
        # object (Anthropic thinking style). Gateway translates it upstream,
        # but if anything slips through, drop it so NVIDIA/Groq don't 400 on
        # `Unsupported parameter(s): reasoning`.
        extras.pop("reasoning", None)
        effort = extras.pop("reasoning_effort", None)
        if effort is None:
            return extras
        if self._kind == "zai":
            extras["thinking"] = {"type": "disabled" if effort == "none" else "enabled"}
        elif self._kind == "groq" and "qwen3" in self._model:
            # Groq's qwen3 only accepts `none`/`default` (rejects low/medium/high with 400).
            extras["reasoning_effort"] = "none" if effort == "none" else "default"
        elif self.reasoning and effort != "none":
            extras["reasoning_effort"] = effort
        # else: drop silently (non-reasoning provider, or "none" on a reasoning OpenAI-compat)
        return extras

    def _web_search_tool(self) -> dict[str, Any] | None:
        """Return the provider's native web-search tool payload, or None if
        there's nothing to inject (OR `:online` handles it server-side via
        the model-id suffix; others without native support return None)."""
        if self._kind == "groq" and "gpt-oss" in self._model:
            return {"type": "browser_search"}
        if self._kind == "zai":
            # Accepted by the API but GLM-4.5-Flash ignores it in practice
            # (bench_web2 2026-04-21); keep wired up in case higher tiers enable it.
            return {"type": "web_search", "web_search": {"enable": True, "search_engine": "search_std"}}
        return None

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> ProviderCallResult:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        if self._api_key:
            headers["Authorization"] = f"{self._auth_scheme} {self._api_key}"
        normalized = self._normalize_extras(request_extras)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._sanitize_messages(messages),
            **self._extra_body,
            **normalized,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            if self._max_output_tokens is not None:
                max_tokens = min(max_tokens, self._max_output_tokens)
            payload["max_tokens"] = max_tokens
        effective_tools = list(tools) if tools else []
        if web_search:
            wt = self._web_search_tool()
            if wt is not None:
                effective_tools.append(wt)
        if effective_tools:
            payload["tools"] = effective_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=payload
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"{self.name} provider error {code}: {msg}")

        try:
            choice = data["choices"][0]
            msg = choice["message"]
            # Cloudflare иногда отдаёт content как int (например, "3" для подсчётных
            # промптов) — приводим к str до .strip(), чтобы не получить AttributeError.
            content = msg.get("content")
            text = str(content).strip() if content not in (None, "") else ""
            raw_tool_calls = msg.get("tool_calls")
            finish_reason = choice.get("finish_reason") or "stop"
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"{self.name} unexpected response: {str(data)[:300]}") from exc

        tool_calls: list[dict[str, Any]] | None = None
        if raw_tool_calls:
            tool_calls = [
                {
                    "id": tc.get("id") or f"call_{i}",
                    "type": tc.get("type") or "function",
                    "function": {
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": (tc.get("function") or {}).get("arguments", "") or "",
                    },
                }
                for i, tc in enumerate(raw_tool_calls)
                if isinstance(tc, dict)
            ] or None

        if not text and not tool_calls:
            raise RuntimeError(f"{self.name} empty response")

        usage = data.get("usage") or {}
        return ProviderCallResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            cached_tokens=_extract_cached_tokens(usage),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> AsyncIterator[bytes]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self._extra_headers,
        }
        if self._api_key:
            headers["Authorization"] = f"{self._auth_scheme} {self._api_key}"
        normalized = self._normalize_extras(request_extras)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._sanitize_messages(messages),
            "stream": True,
            **self._extra_body,
            **normalized,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            if self._max_output_tokens is not None:
                max_tokens = min(max_tokens, self._max_output_tokens)
            payload["max_tokens"] = max_tokens
        effective_tools = list(tools) if tools else []
        if web_search:
            wt = self._web_search_tool()
            if wt is not None:
                effective_tools.append(wt)
        if effective_tools:
            payload["tools"] = effective_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"{self.name} HTTP {resp.status_code}: {body.decode(errors='replace')[:400]}"
                    )
                got_any = False
                async for raw in resp.aiter_bytes():
                    if raw:
                        got_any = True
                        yield raw
                if not got_any:
                    raise RuntimeError(f"{self.name} empty response")


class GigaChatProvider:
    """SberDevices GigaChat — OpenAI-compat /chat/completions с двухступенчатой авторизацией.

    Получаем короткоживущий (≈30 мин) Access token через POST /api/v2/oauth с Basic-auth
    Authorization key, затем дёргаем /api/v1/chat/completions как обычный OpenAI-compat.
    Сертификат ngw/gigachat.devices.sberbank.ru подписан Russian Trusted Root CA, которого
    нет в стандартных трастсторах; по умолчанию `verify_ssl=False`. Можно подложить
    путь к PEM-бандлу через verify_ssl: <path>.
    """

    supports_tools = True
    supports_audio = False
    supports_vision = False  # GigaChat-Pro Vision доступен отдельной моделью; default off

    OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def __init__(
        self,
        name: str,
        *,
        auth_key: str,
        model: str,
        scope: str = "GIGACHAT_API_PERS",
        base_url: str = "https://gigachat.devices.sberbank.ru/api/v1",
        timeout: float = 60.0,
        verify_ssl: bool | str = False,
        rpd: int | None = None,
        rpm: int | None = None,
        context_window: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        reasoning: bool = False,
        quota_limited: bool = True,
    ) -> None:
        self.name = name
        self._kind = "gigachat"
        self._auth_key = auth_key
        self._scope = scope
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._verify_ssl = verify_ssl
        self.rpd = rpd
        self.rpm = rpm
        self.context_window = context_window
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.reasoning = reasoning
        self.quota_limited = quota_limited
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    def _strip_extras(self, request_extras: dict[str, Any] | None) -> dict[str, Any]:
        if not request_extras:
            return {}
        extras = dict(request_extras)
        # GigaChat не понимает OpenAI reasoning_effort — выкидываем, чтобы не получить 400.
        extras.pop("reasoning_effort", None)
        return extras

    async def _ensure_token(self) -> str:
        # 60-секундный buffer, чтобы не отправить запрос с почти-просроченным токеном.
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        async with self._token_lock:
            if self._token and time.time() < self._token_expires_at - 60:
                return self._token
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self._auth_key}",
            }
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout), verify=self._verify_ssl
            ) as client:
                resp = await client.post(
                    self.OAUTH_URL, headers=headers, data={"scope": self._scope}
                )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"{self.name} oauth HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                payload = resp.json()
            tok = payload.get("access_token")
            if not tok:
                raise RuntimeError(f"{self.name} oauth missing access_token: {str(payload)[:200]}")
            self._token = tok
            # expires_at — unix-ms; если поле отсутствует — fallback 25 минут.
            ms = payload.get("expires_at")
            self._token_expires_at = (
                float(ms) / 1000.0 if isinstance(ms, (int, float)) and ms > 0
                else time.time() + 25 * 60
            )
            return self._token

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> ProviderCallResult:
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": str(uuid.uuid4()),
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            **self._strip_extras(request_extras),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout), verify=self._verify_ssl
        ) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=payload
            )
            if resp.status_code == 401:
                # Токен мог протухнуть на грани — форсируем рефреш и пробуем ещё раз.
                self._token = None
                token = await self._ensure_token()
                headers["Authorization"] = f"Bearer {token}"
                resp = await client.post(
                    f"{self._base_url}/chat/completions", headers=headers, json=payload
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        try:
            choice = data["choices"][0]
            msg = choice["message"]
            # Cloudflare иногда отдаёт content как int (например, "3" для подсчётных
            # промптов) — приводим к str до .strip(), чтобы не получить AttributeError.
            content = msg.get("content")
            text = str(content).strip() if content not in (None, "") else ""
            raw_tool_calls = msg.get("tool_calls")
            finish_reason = choice.get("finish_reason") or "stop"
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"{self.name} unexpected response: {str(data)[:300]}") from exc

        tool_calls: list[dict[str, Any]] | None = None
        if raw_tool_calls:
            tool_calls = [
                {
                    "id": tc.get("id") or f"call_{i}",
                    "type": tc.get("type") or "function",
                    "function": {
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": (tc.get("function") or {}).get("arguments", "") or "",
                    },
                }
                for i, tc in enumerate(raw_tool_calls)
                if isinstance(tc, dict)
            ] or None

        if not text and not tool_calls:
            raise RuntimeError(f"{self.name} empty response")

        usage = data.get("usage") or {}
        return ProviderCallResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            cached_tokens=usage.get("precached_prompt_tokens", 0) or 0,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> AsyncIterator[bytes]:
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-ID": str(uuid.uuid4()),
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            **self._strip_extras(request_extras),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout), verify=self._verify_ssl
        ) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"{self.name} HTTP {resp.status_code}: {body.decode(errors='replace')[:400]}"
                    )
                got_any = False
                async for raw in resp.aiter_bytes():
                    if raw:
                        got_any = True
                        yield raw
                if not got_any:
                    raise RuntimeError(f"{self.name} empty response")


class GigaChatImageProvider(GigaChatProvider):
    """Kandinsky 3.1 через GigaChat API.

    GigaChat не имеет отдельного image-эндпоинта: модель сама решает дёрнуть Kandinsky
    при наличии "draw"-намерения в промпте. Flow:
      1. POST /chat/completions с промптом + function_call="auto" → ответ ассистента
         содержит `<img src="<uuid>" fuse="true"/>` в тексте.
      2. GET /files/<uuid>/content → возвращает image/jpeg bytes (1024×1024).

    Тарификация: расход из 1М-токенового Freemium (1-3K токенов на картинку, ~300-500 шт/год).
    chat()/chat_stream() запрещены — провайдер только в image_gen цепочке.
    """

    supports_tools = False
    supports_images = True

    _IMG_RE = re.compile(r'<img\s+src=[\'"]([0-9a-f-]{36})[\'"]', re.IGNORECASE)

    async def chat(self, **_: Any) -> ProviderCallResult:  # type: ignore[override]
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:  # type: ignore[override]
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        token = await self._ensure_token()
        chat_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": str(uuid.uuid4()),
        }
        # GigaChat сам распознаёт draw-intent. Префиксуем "Нарисуй" чтобы поднять
        # вероятность функционального вызова Kandinsky на нейтральных промптах.
        draw_prompt = prompt if any(w in prompt.lower() for w in ("нарисуй", "draw", "сгенерируй", "изобрази")) else f"Нарисуй: {prompt}"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": draw_prompt}],
            "function_call": "auto",
        }

        async def _gen_one(client: httpx.AsyncClient) -> dict[str, Any]:
            resp = await client.post(
                f"{self._base_url}/chat/completions", headers=chat_headers, json=payload
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
            try:
                content = data["choices"][0]["message"].get("content", "") or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"{self.name} unexpected response: {str(data)[:300]}") from exc

            m = self._IMG_RE.search(content)
            if not m:
                # Модель отказалась рисовать (вернула обычный текст) — невыполнимо
                raise RuntimeError(
                    f"{self.name} no image returned (model declined): {content[:200]}"
                )
            file_uuid = m.group(1)

            file_resp = await client.get(
                f"{self._base_url}/files/{file_uuid}/content",
                headers={"Authorization": f"Bearer {token}", "Accept": "image/jpg"},
            )
            if file_resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} file fetch HTTP {file_resp.status_code}: {file_resp.text[:200]}"
                )
            img_bytes = file_resp.content
            if not img_bytes:
                raise RuntimeError(f"{self.name} empty image bytes")
            if response_format == "b64_json":
                return {"b64_json": base64.b64encode(img_bytes).decode("ascii")}
            return {"url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('ascii')}"}

        images: list[dict[str, Any]] = []
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout), verify=self._verify_ssl
        ) as client:
            for _ in range(max(1, min(n, 4))):
                images.append(await _gen_one(client))
        return ImageGenerationResult(images=images, model=self._model)


class GroqWhisperProvider:
    """Groq Whisper — transcription-only. OpenAI-compat /audio/transcriptions.

    Не поддерживает /chat/completions: включается только в цепочке `audio`. chat()
    специально бросает NotImplementedError, чтобы случайное размещение в chat-цепочке
    провалилось громко, а не молча вернуло пустоту.
    """

    supports_tools = False
    supports_audio = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 120.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is transcription-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is transcription-only")
        yield b""  # pragma: no cover — unreachable, keeps type as async generator

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
    ) -> AudioTranscribeResult:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        files = {"file": (filename, audio, mime_type)}
        data: dict[str, str] = {"model": self._model, "response_format": response_format}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = str(temperature)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")

        if response_format in ("json", "verbose_json"):
            j = resp.json()
            text = (j.get("text") or "").strip()
            if not text:
                raise RuntimeError(f"{self.name} empty response")
            return AudioTranscribeResult(
                text=text,
                duration_s=j.get("duration"),
                language=j.get("language") or language,
                raw=j,
            )
        text = resp.text.strip()
        if not text:
            raise RuntimeError(f"{self.name} empty response")
        return AudioTranscribeResult(text=text, language=language, raw={"text": text})


class PollinationsImageProvider:
    """Pollinations image generator — GET /prompt/{prompt} returns image bytes.

    Not OpenAI-compat on the wire (returns raw image/jpeg, not JSON), so this
    provider wraps the response into an OpenAI-style {data: [{b64_json: ...}]}
    shape. Auth is via `token=` query param; Bearer header is not honoured on
    this endpoint. The `referer` query param is required for any tier above
    anonymous to be recognised.

    Rate-limit model (APIDOCS.md 2026-04-22): anonymous = 1 req/15s, seed
    (registered) = 1 req/5s. Параллельные запросы одним токеном ловят 429
    мгновенно, поэтому n>1 сериализуется через asyncio.Lock + min_interval_s
    между соседними вызовами. На 429 делаем один ретрай с бэкоффом (2×interval).

    Included only in image-generation chains. chat()/chat_stream() raise
    NotImplementedError so accidental placement in a chat chain fails loudly.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://image.pollinations.ai",
        referer: str = "neurogate",
        timeout: float = 60.0,
        min_interval_s: float = 5.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        import asyncio as _asyncio
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._referer = referer
        self._timeout = timeout
        self._min_interval_s = max(0.0, min_interval_s)
        self._lock = _asyncio.Lock()
        self._last_call_ts: float = 0.0
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        width, height = _parse_size(size)
        params: dict[str, str] = {
            "model": self._model,
            "width": str(width),
            "height": str(height),
            "nologo": "true",
            "referer": self._referer,
        }
        if self._api_key:
            params["token"] = self._api_key
        if extra:
            for k, v in extra.items():
                if v is not None:
                    params[k] = str(v)

        import asyncio as _asyncio
        import secrets
        from urllib.parse import quote

        url = f"{self._base_url}/prompt/{quote(prompt, safe='')}"

        async def _fetch(client: httpx.AsyncClient, seed: int) -> dict[str, Any]:
            """Один запрос с ретраем на 429. Под self._lock чтобы гарантировать
            сериализацию на уровне провайдера и учёт min_interval_s между
            соседними вызовами (pollinations 429's on concurrent tokens)."""
            q = dict(params)
            q["seed"] = str(seed)
            attempts = 2  # первый + 1 retry после 429
            last_exc: Exception | None = None
            for attempt in range(attempts):
                async with self._lock:
                    if self._min_interval_s > 0 and self._last_call_ts > 0:
                        wait = self._min_interval_s - (time.monotonic() - self._last_call_ts)
                        if wait > 0:
                            await _asyncio.sleep(wait)
                    try:
                        resp = await client.get(url, params=q)
                    finally:
                        self._last_call_ts = time.monotonic()
                if resp.status_code == 429 and attempt < attempts - 1:
                    # Server просит подождать; 2×interval даёт запас против бёрстов
                    await _asyncio.sleep(self._min_interval_s * 2 if self._min_interval_s > 0 else 6.0)
                    continue
                if resp.status_code >= 400:
                    # Лог того, что МЫ реально отправили — Pollinations в error-body
                    # echo'ит requestParameters головной задачи в очереди, а не наши,
                    # поэтому без этого лога диагностика модели запутывается.
                    log.warning(
                        "%s HTTP %d sent_model=%s sent_url=%s",
                        self.name, resp.status_code, q.get("model"), str(resp.request.url)[:200],
                    )
                    last_exc = RuntimeError(
                        f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}"
                    )
                    raise last_exc
                ctype = resp.headers.get("content-type", "")
                if not ctype.startswith("image/"):
                    raise RuntimeError(
                        f"{self.name} non-image response (content-type={ctype}): "
                        f"{resp.text[:200]}"
                    )
                img_bytes = resp.content
                if not img_bytes:
                    raise RuntimeError(f"{self.name} empty image bytes")
                if response_format == "b64_json":
                    return {"b64_json": base64.b64encode(img_bytes).decode("ascii")}
                # response_format == "url": канонический публичный URL того же
                # эндпоинта — клиент может дёрнуть напрямую (Pollinations кэширует
                # по prompt+seed).
                return {"url": str(resp.url)}
            # attempts исчерпаны только если последний ответ был 429
            raise RuntimeError(
                f"{self.name} HTTP 429 after retry (rate-limited by Pollinations)"
            )

        seeds = [secrets.randbelow(2**31) for _ in range(max(1, min(n, 4)))]
        images: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            for s in seeds:
                images.append(await _fetch(client, s))
        return ImageGenerationResult(images=images, model=self._model)


class GeminiImageProvider:
    """Gemini image generation (Nano Banana, Imagen) через google-genai SDK.

    Для gemini-*-image моделей ответ приходит как inline_data в parts — байты
    извлекаются и оборачиваются в OpenAI-совместимый {data:[{b64_json}]}.
    n>1 сериализуется последовательными вызовами (SDK не умеет batch).

    chat()/chat_stream() бросают NotImplementedError — image-only провайдер.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        timeout: float = 90.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._client = genai.Client(api_key=api_key)
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        import asyncio as _asyncio

        # gemini-2.5-flash-image (Nano Banana) игнорирует width/height — размер
        # диктуется моделью. Для Imagen через aspect_ratio можно передавать,
        # но это не size-строка; просто прокидываем вниз через extra, если есть.
        config = types.GenerateContentConfig(response_modalities=["IMAGE"])

        def _call_once() -> dict[str, Any]:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            # Ищем первый part с inline_data (image/*). Gemini-image может
            # вернуть и text-part (описание), который мы игнорируем.
            candidates = getattr(resp, "candidates", None) or []
            for cand in candidates:
                parts = getattr(getattr(cand, "content", None), "parts", None) or []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline is None:
                        continue
                    raw = getattr(inline, "data", None)
                    if not raw:
                        continue
                    # SDK может вернуть уже bytes либо base64-string в зависимости
                    # от версии — нормализуем.
                    if isinstance(raw, bytes):
                        b64 = base64.b64encode(raw).decode("ascii")
                    else:
                        b64 = str(raw)
                    if response_format == "b64_json":
                        return {"b64_json": b64}
                    # URL-формат Gemini не отдаёт — отдаём data: URL как fallback
                    mime = getattr(inline, "mime_type", "image/png")
                    return {"url": f"data:{mime};base64,{b64}"}
            raise RuntimeError(f"{self.name} returned no image parts")

        images: list[dict[str, Any]] = []
        for _ in range(max(1, min(n, 4))):
            img = await _asyncio.to_thread(_call_once)
            images.append(img)
        return ImageGenerationResult(images=images, model=self._model)


class CloudflareImageProvider:
    """Cloudflare Workers AI text→image через `/ai/run/{model}`.

    Permanent free tier — 10 000 Neurons/день. FLUX-1 schnell возвращает
    JSON `{result: {image: "base64"}, success: true}`; SDXL/Lightning — raw
    PNG bytes. Нормализуем по content-type.

    Endpoint НЕ OpenAI-compat (`/v1/images/generations` у CF нет), поэтому
    обёртка сама строит ответ в форме {data:[{b64_json}]}.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        account_id: str,
        model: str,
        timeout: float = 60.0,
        steps: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._steps = steps
        self._base_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        )
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        width, height = _parse_size(size)
        payload: dict[str, Any] = {"prompt": prompt, "width": width, "height": height}
        if self._steps is not None:
            payload["num_steps"] = self._steps
        if extra:
            for k, v in extra.items():
                if v is not None and k not in payload:
                    payload[k] = v

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async def _fetch(client: httpx.AsyncClient) -> dict[str, Any]:
            resp = await client.post(self._base_url, json=payload, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}"
                )
            ctype = resp.headers.get("content-type", "")
            if ctype.startswith("image/"):
                # SDXL/Lightning — raw bytes
                img_bytes = resp.content
            elif ctype.startswith("application/json"):
                data = resp.json()
                if not data.get("success", True):
                    errs = data.get("errors") or data.get("messages") or data
                    raise RuntimeError(f"{self.name} CF error: {errs}")
                result = data.get("result") or {}
                b64 = result.get("image")
                if not b64:
                    raise RuntimeError(
                        f"{self.name} no image in result: {str(result)[:200]}"
                    )
                img_bytes = base64.b64decode(b64)
            else:
                raise RuntimeError(
                    f"{self.name} unexpected content-type {ctype!r}: {resp.text[:200]}"
                )
            if not img_bytes:
                raise RuntimeError(f"{self.name} empty image bytes")
            if response_format == "b64_json":
                return {"b64_json": base64.b64encode(img_bytes).decode("ascii")}
            # CF не даёт публичных URL → возвращаем data: URL
            mime = ctype if ctype.startswith("image/") else "image/png"
            b64s = base64.b64encode(img_bytes).decode("ascii")
            return {"url": f"data:{mime};base64,{b64s}"}

        images: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            for _ in range(max(1, min(n, 4))):
                images.append(await _fetch(client))
        return ImageGenerationResult(images=images, model=self._model)


class AIHordeImageProvider:
    """AI Horde (бывший Stable Horde) — community-distributed inference.

    Полностью бесплатно, anonymous-режим работает с apikey '0000000000'.
    Async job-based: POST /generate/async → poll /generate/check/{id} →
    GET /generate/status/{id}. Этот класс оборачивает в sync API: ждёт
    готовности до max_wait_s и скачивает финальный image.

    Median wait на anonymous: 30s-2min зависит от загрузки сети. Полезен
    как last-resort когда выгорели CF/Pollinations квоты.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str = "0000000000",
        model: str = "stable_diffusion",
        base_url: str = "https://aihorde.net/api/v2",
        client_agent: str = "neurogate:0.1:https://github.com/neurogate",
        steps: int = 20,
        sampler_name: str = "k_euler",
        cfg_scale: float = 7.0,
        max_wait_s: float = 180.0,
        poll_interval_s: float = 3.0,
        timeout: float = 240.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key or "0000000000"
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client_agent = client_agent
        self._steps = steps
        self._sampler_name = sampler_name
        self._cfg_scale = cfg_scale
        self._max_wait_s = max_wait_s
        self._poll_interval_s = poll_interval_s
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        import asyncio as _asyncio

        width, height = _parse_size(size)
        # AI Horde требует width/height кратные 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        params: dict[str, Any] = {
            "width": width,
            "height": height,
            "steps": self._steps,
            "sampler_name": self._sampler_name,
            "cfg_scale": self._cfg_scale,
            "n": max(1, min(n, 4)),
        }
        if extra:
            for k, v in extra.items():
                if v is not None and k not in {"width", "height"}:
                    params[k] = v

        payload: dict[str, Any] = {
            "prompt": prompt,
            "params": params,
            "models": [self._model],
            "nsfw": False,
            "r2": True,  # храним в Cloudflare R2 — быстрая раздача
            "trusted_workers": False,
        }
        headers = {
            "apikey": self._api_key,
            "Client-Agent": self._client_agent,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            # 1. Submit async job
            resp = await client.post(
                f"{self._base_url}/generate/async", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} submit HTTP {resp.status_code}: {resp.text[:300]}"
                )
            sub = resp.json()
            job_id = sub.get("id")
            if not job_id:
                raise RuntimeError(f"{self.name} no job id: {sub}")

            # 2. Poll until done
            start = time.monotonic()
            while True:
                if time.monotonic() - start > self._max_wait_s:
                    raise RuntimeError(
                        f"{self.name} job {job_id} timed out after {self._max_wait_s}s"
                    )
                await _asyncio.sleep(self._poll_interval_s)
                check = await client.get(
                    f"{self._base_url}/generate/check/{job_id}",
                    headers={"Client-Agent": self._client_agent},
                )
                if check.status_code >= 400:
                    raise RuntimeError(
                        f"{self.name} check HTTP {check.status_code}: {check.text[:200]}"
                    )
                cdata = check.json()
                if cdata.get("done"):
                    break
                if cdata.get("faulted"):
                    raise RuntimeError(f"{self.name} job faulted: {cdata}")

            # 3. Fetch result
            status = await client.get(
                f"{self._base_url}/generate/status/{job_id}",
                headers={"Client-Agent": self._client_agent},
            )
            if status.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} status HTTP {status.status_code}: {status.text[:200]}"
                )
            sdata = status.json()
            generations = sdata.get("generations") or []
            if not generations:
                raise RuntimeError(f"{self.name} no generations: {sdata}")

            # 4. Download images (img field — R2 URL когда r2=true; иначе base64)
            images: list[dict[str, Any]] = []
            for gen in generations:
                img_field = gen.get("img", "")
                if img_field.startswith("http"):
                    img_resp = await client.get(img_field)
                    if img_resp.status_code >= 400:
                        raise RuntimeError(
                            f"{self.name} img download HTTP {img_resp.status_code}"
                        )
                    img_bytes = img_resp.content
                else:
                    # Старый non-r2 путь — img уже base64
                    img_bytes = base64.b64decode(img_field)
                if not img_bytes:
                    raise RuntimeError(f"{self.name} empty image bytes")
                if response_format == "b64_json":
                    images.append({"b64_json": base64.b64encode(img_bytes).decode("ascii")})
                else:
                    # AI Horde R2 URL — публичный, можно отдать клиенту напрямую
                    if img_field.startswith("http"):
                        images.append({"url": img_field})
                    else:
                        b64s = base64.b64encode(img_bytes).decode("ascii")
                        images.append({"url": f"data:image/webp;base64,{b64s}"})
            return ImageGenerationResult(images=images, model=self._model)


class YandexARTImageProvider:
    """YandexART (text→image) через Yandex AI Studio async API.

    Двухэтапный flow: POST /foundationModels/v1/imageGenerationAsync → operation_id;
    дальше polling https://operation.api.cloud.yandex.net/operations/{id} до done=True.
    Картинка приходит base64 в `response.image`. Тарификация: 2,24 ₽/картинка
    (биллится из Yandex AI Studio баланса).

    chat()/chat_stream() бросают NotImplementedError — image-only.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    OP_BASE_URL = "https://operation.api.cloud.yandex.net/operations"

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        folder_id: str,
        model: str = "yandex-art/latest",
        base_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1",
        timeout: float = 120.0,
        max_wait_s: float = 90.0,
        poll_interval_s: float = 2.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        quota_limited: bool = True,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._folder_id = folder_id
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_wait_s = max_wait_s
        self._poll_interval_s = poll_interval_s
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.quota_limited = quota_limited

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        # YandexART принимает aspectRatio (widthRatio/heightRatio), а не WxH в пикселях.
        # OpenAI-style "1024x1024" → 1:1, "1024x576" → 16:9, и т.д.
        width, height = _parse_size(size)
        # Уменьшаем до простой дроби, чтобы API не ругался на большие числа.
        from math import gcd
        g = gcd(width, height) or 1
        ratio = {"widthRatio": width // g, "heightRatio": height // g}

        model_uri = self._model if self._model.startswith("art://") else f"art://{self._folder_id}/{self._model}"

        async def _generate_one(client: httpx.AsyncClient, seed: int) -> dict[str, Any]:
            headers = {
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
            }
            body: dict[str, Any] = {
                "modelUri": model_uri,
                "messages": [{"weight": 1, "text": prompt}],
                "generationOptions": {"seed": seed, "aspectRatio": ratio},
            }
            if extra:
                gen_opts = body["generationOptions"]
                for k, v in extra.items():
                    if v is not None:
                        gen_opts[k] = v
            resp = await client.post(
                f"{self._base_url}/imageGenerationAsync", headers=headers, json=body
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            op = resp.json()
            op_id = op.get("id")
            if not op_id:
                raise RuntimeError(f"{self.name} no operation id: {str(op)[:200]}")

            # Polling
            import asyncio as _asyncio
            deadline = time.monotonic() + self._max_wait_s
            while time.monotonic() < deadline:
                await _asyncio.sleep(self._poll_interval_s)
                op_resp = await client.get(
                    f"{self.OP_BASE_URL}/{op_id}", headers={"Authorization": f"Api-Key {self._api_key}"}
                )
                if op_resp.status_code >= 400:
                    raise RuntimeError(
                        f"{self.name} poll HTTP {op_resp.status_code}: {op_resp.text[:300]}"
                    )
                op_data = op_resp.json()
                if op_data.get("done"):
                    if "error" in op_data:
                        raise RuntimeError(f"{self.name} generation failed: {op_data['error']}")
                    img_b64 = (op_data.get("response") or {}).get("image")
                    if not img_b64:
                        raise RuntimeError(
                            f"{self.name} response missing image: {str(op_data)[:200]}"
                        )
                    if response_format == "b64_json":
                        return {"b64_json": img_b64}
                    return {"url": f"data:image/png;base64,{img_b64}"}
            raise RuntimeError(
                f"{self.name} timed out after {self._max_wait_s}s waiting for image"
            )

        import secrets
        seeds = [secrets.randbelow(2**31) for _ in range(max(1, min(n, 4)))]
        images: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            for s in seeds:
                images.append(await _generate_one(client, s))
        return ImageGenerationResult(images=images, model=self._model)


class TogetherImageProvider:
    """Together AI text→image через OpenAI-совместимый `/v1/images/generations`.

    Free endpoint: `black-forest-labs/FLUX.1-schnell-Free`. Ответ уже в OpenAI-
    формате `{data: [{b64_json|url}]}` — обёртка минимальная.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.together.xyz/v1",
        timeout: float = 60.0,
        steps: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._steps = steps
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        width, height = _parse_size(size)
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "n": max(1, min(n, 4)),
            "response_format": response_format,
        }
        if self._steps is not None:
            payload["steps"] = self._steps
        if extra:
            for k, v in extra.items():
                if v is not None and k not in payload:
                    payload[k] = v

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/images/generations", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}"
                )
            data = resp.json()
        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"{self.name} empty data: {str(data)[:200]}")
        images: list[dict[str, Any]] = []
        for it in items:
            if "b64_json" in it:
                images.append({"b64_json": it["b64_json"]})
            elif "url" in it:
                images.append({"url": it["url"]})
            else:
                raise RuntimeError(f"{self.name} item missing b64_json/url: {it}")
        return ImageGenerationResult(images=images, model=self._model)


class FreeTheAiImageProvider:
    """FreeTheAi (api.freetheai.xyz) text→image. OpenAI-compat endpoint, but the
    response shape varies by model:
      - `img/gpt-image-2`  → returns `b64_json` directly (~650KB PNG).
      - `vhr/*`            → returns `url` pointing to access.vheer.com.
        These URLs expire (404 within tens of minutes), so we MUST download +
        re-encode as `b64_json` before handing back to the client. Otherwise the
        client's saved link breaks shortly after delivery.
    Daily Discord /checkin required on the parent key — without it any request
    returns HTTP 403 daily_checkin_required and the chain falls through.
    """

    supports_tools = False
    supports_audio = False
    supports_images = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.freetheai.xyz/v1",
        timeout: float = 90.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        # Per FreeTheAi docs, edits are only supported on `img/gpt-image-2`.
        # vhr/* models are gen-only on Vheer's pipeline.
        self.supports_image_edit = model == "img/gpt-image-2"

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is image-generation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is image-generation-only")
        yield b""  # pragma: no cover

    async def edit_images(
        self,
        *,
        image: str,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        """Edit an existing image. `image` must be a `data:image/<fmt>;base64,<data>`
        URI per FreeTheAi docs. Returns b64_json (gpt-image-2 native format).
        """
        if not self.supports_image_edit:
            raise NotImplementedError(
                f"{self.name}: image edits supported only on img/gpt-image-2"
            )
        del n, size, response_format, extra
        if not image.startswith("data:image/"):
            raise RuntimeError(
                f"{self.name}: `image` must be a data URI like 'data:image/png;base64,...'"
            )
        payload = {"model": self._model, "prompt": prompt, "image": image}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/images/edits", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}"
                )
            data = resp.json()
            items = data.get("data") or []
            if not items:
                raise RuntimeError(f"{self.name} empty data: {str(data)[:200]}")
            images: list[dict[str, Any]] = []
            for it in items:
                if it.get("b64_json"):
                    images.append({"b64_json": it["b64_json"]})
                    continue
                url = it.get("url")
                if not url:
                    raise RuntimeError(f"{self.name} edit item missing b64_json/url: {it}")
                img_resp = await client.get(url)
                if img_resp.status_code >= 400:
                    raise RuntimeError(
                        f"{self.name} url-fetch HTTP {img_resp.status_code} for {url}"
                    )
                images.append({"b64_json": base64.b64encode(img_resp.content).decode("ascii")})
        return ImageGenerationResult(images=images, model=self._model)

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        # FreeTheAi docs accept only `model` and `prompt`. width/height/steps/n
        # are silently ignored upstream — we drop them rather than risk 400.
        del n, size, response_format, extra
        payload = {"model": self._model, "prompt": prompt}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/images/generations", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}"
                )
            data = resp.json()
            items = data.get("data") or []
            if not items:
                raise RuntimeError(f"{self.name} empty data: {str(data)[:200]}")

            images: list[dict[str, Any]] = []
            for it in items:
                if it.get("b64_json"):
                    images.append({"b64_json": it["b64_json"]})
                    continue
                url = it.get("url")
                if not url:
                    raise RuntimeError(f"{self.name} item missing b64_json/url: {it}")
                # vhr/* return ephemeral URLs on access.vheer.com — fetch within
                # the same client lifetime to convert into stable b64.
                img_resp = await client.get(url)
                if img_resp.status_code >= 400:
                    raise RuntimeError(
                        f"{self.name} url-fetch HTTP {img_resp.status_code} for {url}"
                    )
                images.append({"b64_json": base64.b64encode(img_resp.content).decode("ascii")})
        return ImageGenerationResult(images=images, model=self._model)


class EdgeTTSProvider:
    """Microsoft Edge TTS через неофициальный `edge-tts` пакет (rany2/edge-tts).

    Безлимит, без API-ключа — WebSocket к speech.platform.bing.com. 400+ голосов,
    включая русские `ru-RU-DmitryNeural` / `ru-RU-SvetlanaNeural` / `ru-RU-DariyaNeural`.
    Возвращает mp3 (audio/mpeg) — Edge не поддерживает другие форматы on-wire.

    Только speech-synthesis: chat()/chat_stream() бросают NotImplementedError,
    чтобы случайное размещение в chat-цепочке провалилось громко.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = True
    context_window = None
    reasoning = False

    # OpenAI TTS voice names → Edge equivalents (en-US fallbacks). Пользователь
    # всегда может передать нативное имя вроде `ru-RU-DmitryNeural` — оно уйдёт
    # на Edge как есть (lookup case-insensitive → not found → passthrough).
    _OPENAI_VOICE_MAP = {
        "alloy": "en-US-AriaNeural",
        "echo": "en-US-GuyNeural",
        "fable": "en-GB-SoniaNeural",
        "onyx": "en-US-AndrewNeural",
        "nova": "en-US-JennyNeural",
        "shimmer": "en-US-AvaNeural",
    }

    def __init__(
        self,
        name: str,
        *,
        model: str = "edge-tts",
        default_voice: str = "en-US-AriaNeural",
        timeout: float = 60.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._model = model
        self._default_voice = default_voice
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is tts-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is tts-only")
        yield b""  # pragma: no cover

    async def generate_speech(
        self,
        *,
        input_text: str,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        extra: dict[str, Any] | None = None,
    ) -> AudioSpeechResult:
        # Edge отдаёт только mp3. Другие форматы пришлось бы перекодировать через
        # ffmpeg — это вне scope бесплатного zero-dep TTS, поэтому отклоняем явно.
        if response_format != "mp3":
            raise NotImplementedError(
                f"{self.name} supports response_format=mp3 only (got {response_format!r})"
            )

        try:
            import edge_tts  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"{self.name}: edge-tts package not installed (pip install edge-tts)"
            ) from exc

        resolved_voice = self._resolve_voice(voice)
        rate = self._speed_to_rate(speed)

        kwargs: dict[str, Any] = {"rate": rate}
        if extra:
            # Прокидываем volume/pitch если клиент задал (extra payload).
            for k in ("volume", "pitch"):
                if k in extra and extra[k] is not None:
                    kwargs[k] = str(extra[k])

        try:
            communicate = edge_tts.Communicate(input_text, resolved_voice, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"{self.name} init failed: {exc}") from exc

        audio = bytearray()
        try:
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    data = chunk.get("data")
                    if data:
                        audio.extend(data)
        except Exception as exc:
            raise RuntimeError(f"{self.name} stream failed: {exc}") from exc

        if not audio:
            raise RuntimeError(f"{self.name} empty audio response")

        return AudioSpeechResult(
            audio=bytes(audio),
            content_type="audio/mpeg",
            model=self._model,
            voice=resolved_voice,
        )

    def _resolve_voice(self, voice: str | None) -> str:
        if not voice:
            return self._default_voice
        mapped = self._OPENAI_VOICE_MAP.get(voice.lower())
        return mapped or voice

    @staticmethod
    def _speed_to_rate(speed: float | None) -> str:
        """OpenAI speed (0.25..4.0, default 1.0) → Edge rate string ("+0%", "-50%").
        Clamps to Edge's practical range."""
        try:
            s = float(speed) if speed is not None else 1.0
        except (TypeError, ValueError):
            s = 1.0
        s = max(0.25, min(s, 4.0))
        pct = int(round((s - 1.0) * 100))
        sign = "+" if pct >= 0 else "-"
        return f"{sign}{abs(pct)}%"


class HFSpaceAudioProvider:
    """Text-to-audio (SFX / ambient) через публичные HuggingFace Spaces.

    Бесплатно, без HF Inference API квоты — Spaces работают на shared GPU/CPU.
    Платой служат латентность (cold-start ~30-60s, warm ~10-20s) и
    нестабильность (Space может «спать», менять схему API без анонса).

    Конфиг:
      space_id: "owner/space-name" (e.g. artificialguybr/Stable-Audio-Open-Zero)
      api_name: "/predict" — gradio endpoint (зависит от Space)
      prompt_field / duration_field — имена аргументов (если у Space kwargs);
        если None — позиционный вызов (prompt[, duration]).
      default_duration_s — что подставлять, если клиент не передал длину.
      hf_token (через api_key_env) — опционален; повышает rate limit, но
        Space всё равно бесплатен и работает анонимно.

    gradio_client синхронный → оборачиваем `predict()` в asyncio.to_thread,
    чтобы не блокировать event loop.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_sfx = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        space_id: str,
        api_name: str = "/predict",
        prompt_field: str | None = "prompt",
        duration_field: str | None = None,
        default_duration_s: float = 10.0,
        hf_token: str | None = None,
        timeout: float = 180.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._space_id = space_id
        self._api_name = api_name
        self._prompt_field = prompt_field
        self._duration_field = duration_field
        self._default_duration_s = default_duration_s
        self._hf_token = hf_token or None
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is sfx-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is sfx-only")
        yield b""  # pragma: no cover

    async def generate_sfx(
        self,
        *,
        prompt: str,
        duration_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AudioGenerationResult:
        try:
            from gradio_client import Client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"{self.name}: gradio_client not installed (pip install gradio-client)"
            ) from exc

        dur = float(duration_s) if duration_s is not None else self._default_duration_s

        def _call() -> Any:
            client = Client(self._space_id, token=self._hf_token, verbose=False)
            kwargs: dict[str, Any] = {}
            args: list[Any] = []
            if self._prompt_field:
                kwargs[self._prompt_field] = prompt
            else:
                args.append(prompt)
            if self._duration_field:
                kwargs[self._duration_field] = dur
            elif self._duration_field is None and not self._prompt_field:
                args.append(dur)
            if extra:
                for k, v in extra.items():
                    if v is not None:
                        kwargs[k] = v
            return client.predict(*args, api_name=self._api_name, **kwargs)

        try:
            result = await asyncio.wait_for(asyncio.to_thread(_call), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            # "timed out" ловится errors.classify() → ErrorCategory.TIMEOUT (retryable).
            raise RuntimeError(
                f"{self.name}: timed out after {self._timeout}s "
                f"(Space {self._space_id} likely cold/sleeping)"
            ) from exc
        except Exception as exc:
            # Помечаем как server_error — Space может быть в любом состоянии (sleeping,
            # gated, схема изменилась, разрыв соединения). Все эти случаи retryable
            # для router → перейти к следующему Space в цепочке.
            raise RuntimeError(
                f"{self.name}: HF Space unavailable: {exc}"
            ) from exc

        # Gradio Audio output — обычно путь к локальному tmp-файлу. Иногда —
        # tuple(sample_rate, np_array) или dict с 'value'/'path'. Нормализуем.
        audio_path = self._extract_audio_path(result)
        if not audio_path:
            raise RuntimeError(
                f"{self.name}: unrecognized gradio output shape: {type(result).__name__}"
            )

        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except OSError as exc:
            raise RuntimeError(f"{self.name}: failed to read output file: {exc}") from exc

        if not audio_bytes:
            raise RuntimeError(f"{self.name}: empty audio output")

        content_type = self._guess_content_type(audio_path)
        return AudioGenerationResult(
            audio=audio_bytes,
            content_type=content_type,
            model=self._space_id,
            duration_s=dur,
            raw={"space_id": self._space_id, "path": audio_path},
        )

    @staticmethod
    def _extract_audio_path(result: Any) -> str | None:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("path", "name", "value", "url"):
                v = result.get(key)
                if isinstance(v, str) and v:
                    return v
        if isinstance(result, (list, tuple)) and result:
            first = result[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return HFSpaceAudioProvider._extract_audio_path(first)
        return None

    @staticmethod
    def _guess_content_type(path: str) -> str:
        p = path.lower()
        if p.endswith(".mp3"):
            return "audio/mpeg"
        if p.endswith(".ogg") or p.endswith(".oga"):
            return "audio/ogg"
        if p.endswith(".flac"):
            return "audio/flac"
        if p.endswith(".m4a") or p.endswith(".aac"):
            return "audio/aac"
        return "audio/wav"


class LibreTranslateProvider:
    """LibreTranslate-compatible endpoint. Public mirrors (translate.fedilab.app,
    самохост) принимают POST /translate с {q, source, target, format}. Большинство
    mirror'ов API-ключ не требуют (но поле `api_key` в body опциональное, если захочется
    платный libretranslate.com).

    NB 2026-04-23: live-probe показал что libretranslate.com/.de стали платными,
    terraprint/vern-mirror умерли; translate.fedilab.app пока отвечает 200 OK.
    Community-зоопарк — поэтому timeout короткий и fallback на MyMemory/LLM обязателен.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        base_url: str = "https://translate.fedilab.app",
        api_key: str = "",
        timeout: float = 20.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is translation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is translation-only")
        yield b""  # pragma: no cover

    async def translate(
        self,
        *,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> TranslationResult:
        payload: dict[str, str] = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text",
        }
        if self._api_key:
            payload["api_key"] = self._api_key

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/translate",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        translated = (j.get("translatedText") or "").strip()
        if not translated:
            raise RuntimeError(f"{self.name} empty translatedText: {str(j)[:200]}")
        # LibreTranslate возвращает detectedLanguage.language если source=auto
        detected = source_lang
        det = j.get("detectedLanguage")
        if isinstance(det, dict) and det.get("language"):
            detected = det["language"]
        return TranslationResult(
            text=translated,
            target_lang=target_lang,
            source_lang=detected,
            provider_model="libretranslate",
            raw=j,
        )


class MyMemoryProvider:
    """MyMemory (translated.net) — бесплатный MT API с гибридной TM-базой.
    Anonymous tier: 5000 chars/day. С `?de=email` — 50000 chars/day (мы шлём если
    передан `contact_email`). GET /get?q=<text>&langpair=<src>|<tgt>, возвращает
    {responseData: {translatedText, match}, responseStatus: 200}.

    ⚠️ Не поддерживает source=auto: langpair требует оба языка. Если клиент передал
    source_lang='auto', провайдер бросает RuntimeError → router скипает на следующий.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        base_url: str = "https://api.mymemory.translated.net",
        contact_email: str = "",
        timeout: float = 15.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._contact_email = contact_email
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is translation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is translation-only")
        yield b""  # pragma: no cover

    async def translate(
        self,
        *,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> TranslationResult:
        if source_lang == "auto":
            raise RuntimeError(
                f"{self.name} requires explicit source_lang (MyMemory has no auto-detect)"
            )
        params: dict[str, str] = {
            "q": text,
            "langpair": f"{source_lang}|{target_lang}",
        }
        if self._contact_email:
            params["de"] = self._contact_email

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.get(f"{self._base_url}/get", params=params)
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        # MyMemory возвращает HTTP 200 даже на ошибки — смотрим responseStatus
        status = j.get("responseStatus")
        if status and status != 200 and status != "200":
            raise RuntimeError(f"{self.name} responseStatus {status}: {j.get('responseDetails', '')[:200]}")
        data = j.get("responseData") or {}
        translated = (data.get("translatedText") or "").strip()
        if not translated:
            raise RuntimeError(f"{self.name} empty translatedText: {str(j)[:200]}")
        return TranslationResult(
            text=translated,
            target_lang=target_lang,
            source_lang=source_lang,
            provider_model="mymemory",
            raw=j,
        )


class YandexTranslateProvider:
    """Yandex Cloud Translate через AI Studio. Dedicated MT, scan-00 рейтинг ★★★★★ RU,
    ★★★★ AR. Free tier 1M chars/мес для физлиц (по scan-06) / 1M grant для юрлиц,
    потом платно $~5/1M.

    Endpoint OpenAPI-style: POST /translate/v2/translate с folderId в body + header
    `Authorization: Api-Key <key>`. Поддерживает batch (texts[]), auto-detect
    (если sourceLanguageCode не указан — Yandex детектит, возвращает
    detectedLanguageCode).

    API-ключ должен быть создан со scope `yc.ai.translate.execute` и назначен
    сервисному аккаунту с ролью `ai.translate.user` на нужный каталог.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        folder_id: str,
        base_url: str = "https://translate.api.cloud.yandex.net",
        timeout: float = 30.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        if not folder_id:
            raise ValueError(f"{name}: folder_id is required (set YANDEX_FOLDER_ID or pass in config)")
        self.name = name
        self._api_key = api_key
        self._folder_id = folder_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is translation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is translation-only")
        yield b""  # pragma: no cover

    async def translate(
        self,
        *,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> TranslationResult:
        payload: dict[str, Any] = {
            "folderId": self._folder_id,
            "texts": [text],
            "targetLanguageCode": target_lang,
        }
        # `auto` для нас = «не указывать sourceLanguageCode, пусть Yandex детектит».
        # Yandex возвращает detectedLanguageCode в ответе.
        if source_lang and source_lang != "auto":
            payload["sourceLanguageCode"] = source_lang

        headers = {
            "Authorization": f"Api-Key {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/translate/v2/translate",
                headers=headers,
                json=payload,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        translations = j.get("translations") or []
        if not translations:
            raise RuntimeError(f"{self.name} empty translations: {str(j)[:200]}")
        first = translations[0]
        translated = (first.get("text") or "").strip()
        if not translated:
            raise RuntimeError(f"{self.name} empty text in translation: {str(j)[:200]}")
        detected = first.get("detectedLanguageCode") or (
            source_lang if source_lang != "auto" else None
        )
        return TranslationResult(
            text=translated,
            target_lang=target_lang,
            source_lang=detected,
            provider_model="yandex:translate",
            raw=j,
        )


class CohereTranslateProvider:
    """Cohere Aya Expanse — multilingual LLM (102 языка, специально обучен для
    Global South / арабского / индийских языков). Endpoint: /v2/chat с translate-
    промптом. Не OpenAI-compat по shape (response.message.content — список блоков
    [{type:'text', text:...}]), поэтому отдельный класс.

    Trial tier: 1000 calls/мес, 20 RPM. TOS запрещает production — только
    личные/pet-project нагрузки. Для production апгрейд до Production key.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = True
    context_window = None
    reasoning = False

    # Langname map для translate-prompt (та же что в router._LANG_NAMES — делаем свою
    # копию чтобы не тянуть circular import; Aya понимает и сырые коды языков, но
    # явные имена дают чуть лучше precision на редких парах).
    _LANG_NAMES = {
        "en": "English", "ru": "Russian", "ar": "Arabic", "de": "German",
        "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
        "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "tr": "Turkish",
        "pl": "Polish", "uk": "Ukrainian", "nl": "Dutch", "sv": "Swedish",
        "cs": "Czech", "he": "Hebrew", "hi": "Hindi", "vi": "Vietnamese",
        "id": "Indonesian", "th": "Thai", "el": "Greek", "fi": "Finnish",
        "da": "Danish", "no": "Norwegian", "ro": "Romanian", "hu": "Hungarian",
        "bg": "Bulgarian", "fa": "Persian", "sw": "Swahili", "ur": "Urdu",
        "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ms": "Malay",
    }

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str = "c4ai-aya-expanse-32b",
        base_url: str = "https://api.cohere.com/v2",
        timeout: float = 60.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is translation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is translation-only")
        yield b""  # pragma: no cover

    async def translate(
        self,
        *,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> TranslationResult:
        tgt = self._LANG_NAMES.get(target_lang, target_lang)
        if source_lang == "auto":
            sys_prompt = (
                f"You are a professional translator. Translate the text to {tgt}. "
                "Detect the source language automatically. Output ONLY the translated "
                "text — no explanations, no quotes, no source text, no language tags."
            )
        else:
            src = self._LANG_NAMES.get(source_lang, source_lang)
            sys_prompt = (
                f"You are a professional translator. Translate from {src} to {tgt}. "
                "Output ONLY the translated text — no explanations, no quotes, no "
                "source text, no language tags."
            )

        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            # Грубая оценка: перевод редко длиннее исходника × 3. Cohere имеет
            # лимит на max_tokens (по модели), 4096 — хороший потолок для 32B.
            "max_tokens": max(256, min(len(text) * 3, 4096)),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/chat",
                headers=headers,
                json=body,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        # Shape: {message: {content: [{type:'text', text:'...'}]}, finish_reason, usage}
        msg = j.get("message") or {}
        content_blocks = msg.get("content") or []
        if not isinstance(content_blocks, list):
            raise RuntimeError(f"{self.name} unexpected content shape: {str(j)[:200]}")
        text_out = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        if not text_out:
            raise RuntimeError(f"{self.name} empty translation: {str(j)[:200]}")
        # Cohere любит оборачивать в кавычки/бэктики несмотря на инструкцию — чистим
        text_out = text_out.strip('"\'`')
        if text_out.startswith("```") and text_out.endswith("```"):
            text_out = text_out[3:-3].strip()
        return TranslationResult(
            text=text_out,
            target_lang=target_lang,
            source_lang=source_lang if source_lang != "auto" else None,
            provider_model=f"cohere:{self._model}",
            raw=j,
        )


class CohereChatProvider:
    """Cohere v2 /chat для Command-R / R+ / R7B (general chat, не translate).

    Endpoint POST {base_url}/chat. Request body совместим с OpenAI-style
    messages, но response shape собственный: {message: {content: [{type:'text', text:...}]}}.

    Trial tier (2026-04): 20 RPM на модель, 1000 calls/мес. Документация Cohere
    не уточняет shared/per-model для месячного квотума — на каждой записи
    rpd=33 как safe-floor (1000/30).

    Streaming эмулируется: один OpenAI-compat SSE chunk из полного ответа
    (Cohere v2 SSE имеет свой формат — дешевле fake-stream, чем переводить).
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.cohere.com/v2",
        timeout: float = 60.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return body

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> ProviderCallResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = self._build_payload(messages, temperature, max_tokens)
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(f"{self._base_url}/chat", headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        msg = data.get("message") or {}
        blocks = msg.get("content") or []
        if not isinstance(blocks, list):
            raise RuntimeError(f"{self.name} unexpected content shape: {str(data)[:200]}")
        text = "".join(
            b.get("text", "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        if not text:
            raise RuntimeError(f"{self.name} empty response: {str(data)[:200]}")
        usage = (data.get("usage") or {}).get("tokens") or {}
        return ProviderCallResult(
            text=text,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            finish_reason=data.get("finish_reason") or "stop",
        )

    async def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> AsyncIterator[bytes]:
        result = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        chunk = {
            "id": f"cohere-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "model": self._model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": result.text},
                "finish_reason": None,
            }],
        }
        done = {
            "id": chunk["id"],
            "object": "chat.completion.chunk",
            "model": self._model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield f"data: {json.dumps(done)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def _parse_size(size: str | None) -> tuple[int, int]:
    """Parse OpenAI-style size '1024x1024' → (1024, 1024). Defaults 1024×1024.
    Clamps to [64, 2048] per-dimension to avoid provider 4xx."""
    if not size:
        return 1024, 1024
    try:
        w_s, h_s = size.lower().split("x", 1)
        w, h = int(w_s), int(h_s)
    except Exception:
        return 1024, 1024
    w = max(64, min(w, 2048))
    h = max(64, min(h, 2048))
    return w, h


def _normalize_input(input_data: str | list[str]) -> list[str]:
    """OpenAI-compat /embeddings принимает как одиночный string, так и list[str].
    Внутри хранится list — единичный вход оборачиваем."""
    if isinstance(input_data, str):
        return [input_data]
    if not isinstance(input_data, list):
        raise ValueError(f"input must be str or list[str], got {type(input_data).__name__}")
    out: list[str] = []
    for i, item in enumerate(input_data):
        if not isinstance(item, str):
            raise ValueError(f"input[{i}] must be a string, got {type(item).__name__}")
        out.append(item)
    return out


class OpenAIEmbedProvider:
    """Универсальный embedding-провайдер для всех OpenAI-compat /v1/embeddings:
    Voyage, Jina, Mistral, NVIDIA NIM, GitHub Models. Шейп идентичный:
    POST {base_url}/embeddings с {model, input}, ответ
    {data:[{embedding:[...], index:N}, ...], usage:{total_tokens, prompt_tokens?}}.

    Особенности отдельных провайдеров инжектируются через `extra_body` в config:
    - Voyage: `input_type: 'document'|'query'` (опционально, задаёт оптимизацию)
    - Jina v3+: `task: 'retrieval.query'|'retrieval.passage'|...`, `normalized: true`
    - Все: `dimensions: N` для MRL-truncation (Voyage, Jina v5)
    Per-request override берётся из `extra` (req body) и перекрывает `extra_body`.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        kind: str = "openai_embed",
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        auth_scheme: str = "Bearer",
        timeout: float = 60.0,
        dim: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        quota_limited: bool = False,
    ) -> None:
        self.name = name
        self._kind = kind
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or {}
        self._auth_scheme = auth_scheme
        self._timeout = timeout
        self.dim = dim
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.quota_limited = quota_limited

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is embeddings-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is embeddings-only")
        yield b""  # pragma: no cover

    async def embed(
        self,
        *,
        input_texts: list[str],
        extra: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        headers = {
            "Authorization": f"{self._auth_scheme} {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "input": input_texts,
            **self._extra_body,
        }
        if extra:
            payload.update(extra)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings", headers=headers, json=payload
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"{self.name} provider error: {msg}")

        items = data.get("data") if isinstance(data, dict) else None
        if not items or not isinstance(items, list):
            raise RuntimeError(f"{self.name} unexpected response: {str(data)[:300]}")
        # Сохраняем порядок по `index` если он задан, иначе по позиции в массиве.
        ordered = sorted(items, key=lambda x: x.get("index", 0)) if all(
            isinstance(x, dict) and "index" in x for x in items
        ) else items
        vectors: list[list[float]] = []
        for i, item in enumerate(ordered):
            if not isinstance(item, dict):
                raise RuntimeError(f"{self.name} item[{i}] not a dict")
            emb = item.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise RuntimeError(f"{self.name} item[{i}] empty embedding")
            vectors.append([float(x) for x in emb])

        if len(vectors) != len(input_texts):
            raise RuntimeError(
                f"{self.name} returned {len(vectors)} vectors for {len(input_texts)} inputs"
            )

        usage = data.get("usage") or {}
        # OpenAI/Voyage: total_tokens. Jina/Mistral: prompt_tokens. Берём что есть.
        prompt_tokens = (
            usage.get("prompt_tokens") or usage.get("total_tokens") or 0
        ) if isinstance(usage, dict) else 0
        return EmbeddingResult(
            vectors=vectors,
            model=data.get("model") or self._model,
            dim=len(vectors[0]) if vectors else 0,
            prompt_tokens=int(prompt_tokens),
            raw={"usage": usage},
        )


class GeminiEmbedProvider:
    """Gemini embeddings через нативный google.genai SDK.

    `client.models.embed_content(model, contents, config)` — единый вызов на
    список входов. Параметр `task_type` (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT
    / SEMANTIC_SIMILARITY / CLASSIFICATION / CLUSTERING) пробрасывается из
    `extra` запроса через EmbedContentConfig.

    Note: Gemini-эмбеддинги делят 1500 RPD-бакет с chat-Gemini того же ключа —
    на индексации легко выжечь chat. Поэтому в default `embed`-цепочке Gemini
    стоит после Voyage (200M tok permanent независимый бакет).
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        *,
        dim: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._client = genai.Client(api_key=api_key)
        self._api_key = api_key
        self._model = model
        self.dim = dim
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is embeddings-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is embeddings-only")
        yield b""  # pragma: no cover

    async def embed(
        self,
        *,
        input_texts: list[str],
        extra: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        config_kwargs: dict[str, Any] = {}
        if extra:
            # task_type: RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT / SEMANTIC_SIMILARITY / ...
            task_type = extra.get("task_type") or extra.get("task")
            if task_type:
                config_kwargs["task_type"] = str(task_type).upper()
            # output_dimensionality: MRL-truncation (3072 → 768/1536/3072 для embedding-001)
            dim_override = extra.get("dimensions") or extra.get("output_dimensionality")
            if dim_override:
                config_kwargs["output_dimensionality"] = int(dim_override)
            title = extra.get("title")
            if title:
                config_kwargs["title"] = str(title)
        config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=input_texts,
            config=config,
        )
        # response.embeddings — список ContentEmbedding с .values list[float]
        raw_embs = getattr(response, "embeddings", None) or []
        vectors: list[list[float]] = []
        for i, emb in enumerate(raw_embs):
            values = getattr(emb, "values", None)
            if not values:
                raise RuntimeError(f"{self.name} item[{i}] empty embedding")
            vectors.append([float(x) for x in values])

        if len(vectors) != len(input_texts):
            raise RuntimeError(
                f"{self.name} returned {len(vectors)} vectors for {len(input_texts)} inputs"
            )

        usage = getattr(response, "metadata", None)
        prompt_tokens = getattr(usage, "billable_character_count", 0) or 0 if usage else 0
        return EmbeddingResult(
            vectors=vectors,
            model=self._model,
            dim=len(vectors[0]) if vectors else 0,
            prompt_tokens=int(prompt_tokens),
        )


class CohereEmbedProvider:
    """Cohere v2 /embed — отдельный шейп от OpenAI:
    POST /v2/embed с {model, texts, input_type, embedding_types:['float']}.
    Ответ: {embeddings: {float: [[...], ...]}, billed_units: {input_tokens: N}}.

    `input_type` обязателен у Cohere v2: 'search_document' | 'search_query' |
    'classification' | 'clustering'. По умолчанию 'search_document' (для
    индексации). Если клиент явно указал в extra — оверрайд.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.cohere.com/v2",
        timeout: float = 60.0,
        dim: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        quota_limited: bool = False,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.dim = dim
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.quota_limited = quota_limited

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is embeddings-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is embeddings-only")
        yield b""  # pragma: no cover

    async def embed(
        self,
        *,
        input_texts: list[str],
        extra: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        input_type = (extra or {}).get("input_type") or "search_document"
        payload: dict[str, Any] = {
            "model": self._model,
            "texts": input_texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        # Optional: dimensions (для embed-v4 — 256/512/1024/1536)
        if extra and (dims := extra.get("dimensions") or extra.get("output_dimension")):
            payload["output_dimension"] = int(dims)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/embed", headers=headers, json=payload
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        if isinstance(data, dict) and data.get("message") and not data.get("embeddings"):
            raise RuntimeError(f"{self.name} provider error: {data.get('message')}")

        embeddings_obj = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings_obj, dict):
            raise RuntimeError(f"{self.name} unexpected response: {str(data)[:300]}")
        floats = embeddings_obj.get("float")
        if not isinstance(floats, list) or not floats:
            raise RuntimeError(f"{self.name} no float embeddings in response: {str(data)[:300]}")

        vectors: list[list[float]] = []
        for i, vec in enumerate(floats):
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(f"{self.name} item[{i}] empty embedding")
            vectors.append([float(x) for x in vec])

        if len(vectors) != len(input_texts):
            raise RuntimeError(
                f"{self.name} returned {len(vectors)} vectors for {len(input_texts)} inputs"
            )

        billed = data.get("billed_units") or {}
        prompt_tokens = (
            billed.get("input_tokens") if isinstance(billed, dict) else None
        ) or 0
        return EmbeddingResult(
            vectors=vectors,
            model=self._model,
            dim=len(vectors[0]) if vectors else 0,
            prompt_tokens=int(prompt_tokens),
            raw={"billed_units": billed},
        )


class CloudflareEmbedProvider:
    """Cloudflare Workers AI embeddings — POST /accounts/{id}/ai/run/@cf/baai/bge-m3
    с {text: list[str]} (≤100 items). Ответ: {result: {data: [[...], ...]}, success: true}.

    Один вызов CF = 1 neuron независимо от батча, шейп ответа отличается от OpenAI.
    """

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        account_id: str,
        model: str,
        timeout: float = 60.0,
        dim: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._account_id = account_id
        self._model = model.lstrip("/")
        self._timeout = timeout
        self.dim = dim
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is embeddings-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is embeddings-only")
        yield b""  # pragma: no cover

    async def embed(
        self,
        *,
        input_texts: list[str],
        extra: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"text": input_texts}
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{self._model}"
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        if not isinstance(data, dict) or not data.get("success"):
            errs = data.get("errors") if isinstance(data, dict) else None
            raise RuntimeError(f"{self.name} CF error: {errs or str(data)[:300]}")

        result = data.get("result") or {}
        floats = result.get("data") if isinstance(result, dict) else None
        if not isinstance(floats, list) or not floats:
            raise RuntimeError(f"{self.name} no embeddings in result: {str(data)[:300]}")

        vectors: list[list[float]] = []
        for i, vec in enumerate(floats):
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(f"{self.name} item[{i}] empty embedding")
            vectors.append([float(x) for x in vec])

        if len(vectors) != len(input_texts):
            raise RuntimeError(
                f"{self.name} returned {len(vectors)} vectors for {len(input_texts)} inputs"
            )

        return EmbeddingResult(
            vectors=vectors,
            model=self._model,
            dim=len(vectors[0]) if vectors else 0,
            prompt_tokens=0,
            raw={"shape": result.get("shape")},
        )


# ============================================================================
# Moderation providers
# ============================================================================
# Output shape единый — ModerationResult.results = list[dict]:
#   [{flagged: bool, categories: {<openai_cat>: bool, ...},
#     category_scores: {<openai_cat>: float, ...} | None}]
# Категории нормализованы под OpenAI omni-moderation схему.

# Полный список OpenAI omni-moderation категорий — `False` дефолтный.
_OPENAI_MOD_CATEGORIES = (
    "sexual",
    "sexual/minors",
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "violence",
    "violence/graphic",
)


def _empty_categories() -> dict[str, bool]:
    return {k: False for k in _OPENAI_MOD_CATEGORIES}


def _coerce_image_input(image: Any) -> str:
    """Принимает url-string ИЛИ data: URI ИЛИ raw base64 → возвращает URL/data-URI
    в формате, который понимают OpenAI/Llama Guard vision-моделям."""
    if not isinstance(image, str) or not image:
        raise ValueError("image input must be a non-empty string (URL or data: URI)")
    if image.startswith("http://") or image.startswith("https://") or image.startswith("data:"):
        return image
    # Голый base64 без data-prefix — оборачиваем как PNG (наиболее частый случай).
    return f"data:image/png;base64,{image}"


class OpenAIModerationProvider:
    """OpenAI omni-moderation-latest — multimodal classifier (text + image),
    13 OpenAI native категорий с category_scores. POST /v1/moderations,
    Bearer auth. Бесплатно навсегда (без CC), даже если на ключе нет credits
    для chat-моделей.

    Шейп ответа уже OpenAI-native — пропускаем через себя без нормализации.
    Поддерживает обе модальности (одна модель, разный input)."""

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = False
    supports_moderation_text = True
    supports_moderation_image = True
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str = "omni-moderation-latest",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is moderation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is moderation-only")
        yield b""  # pragma: no cover

    async def _post_moderations(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/moderations", headers=headers, json=payload
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        if not isinstance(j, dict) or "results" not in j:
            raise RuntimeError(f"{self.name} unexpected response: {str(j)[:300]}")
        return j

    async def moderate_text(
        self, *, input_texts: list[str]
    ) -> ModerationResult:
        # OpenAI moderations принимает либо одиночный string, либо list[str].
        body_input: Any = input_texts[0] if len(input_texts) == 1 else input_texts
        j = await self._post_moderations({"model": self._model, "input": body_input})
        return ModerationResult(
            results=list(j.get("results") or []),
            model=j.get("model") or self._model,
            raw={"id": j.get("id")},
        )

    async def moderate_image(
        self, *, images: list[str], context_text: str | None = None
    ) -> ModerationResult:
        # omni-moderation мультимодальный: input — это list of content parts.
        # OpenAI рекомендует сделать ОДИН запрос на изображение (модель не делает
        # batch по нескольким images), но шейп позволяет multipart per-input. Мы
        # делаем по одному запросу на каждое изображение и склеиваем results.
        all_results: list[dict[str, Any]] = []
        last_raw: dict[str, Any] = {}
        for img in images:
            url = _coerce_image_input(img)
            parts: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": url}}
            ]
            if context_text:
                parts.insert(0, {"type": "text", "text": context_text})
            j = await self._post_moderations(
                {"model": self._model, "input": parts}
            )
            results = j.get("results") or []
            all_results.extend(results)
            last_raw = {"id": j.get("id")}
        return ModerationResult(
            results=all_results,
            model=self._model,
            raw=last_raw,
        )


# Mistral возвращает свой набор категорий — нормализуем в OpenAI-схему.
# Источник: docs.mistral.ai/capabilities/guardrailing/ (2026-04 snapshot).
_MISTRAL_TO_OPENAI = {
    "sexual": ["sexual"],
    "hate_and_discrimination": ["hate"],
    "violence_and_threats": ["violence", "harassment/threatening"],
    "dangerous_and_criminal_content": ["illicit", "illicit/violent"],
    "selfharm": ["self-harm"],
    # health/financial/law/pii — у OpenAI нет аналогов, оставляем в `extra_*`
    "health": [],
    "financial": [],
    "law": [],
    "pii": [],
}


class MistralModerationProvider:
    """Mistral mistral-moderation-latest. text-only через POST /v1/moderations.
    Возвращает свои 9 категорий — конвертируем в OpenAI native шейп; «лишние»
    категории (health/financial/law/pii) попадают в `extra_categories` /
    `extra_category_scores` results-элемента."""

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = False
    supports_moderation_text = True
    supports_moderation_image = False
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        model: str = "mistral-moderation-latest",
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 30.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is moderation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is moderation-only")
        yield b""  # pragma: no cover

    @staticmethod
    def _normalize_one(item: dict[str, Any]) -> dict[str, Any]:
        cats_raw = item.get("categories") or {}
        scores_raw = item.get("category_scores") or {}
        out_cats = _empty_categories()
        out_scores: dict[str, float] = {k: 0.0 for k in _OPENAI_MOD_CATEGORIES}
        extra_cats: dict[str, bool] = {}
        extra_scores: dict[str, float] = {}
        for mistral_key, mapped in _MISTRAL_TO_OPENAI.items():
            v = bool(cats_raw.get(mistral_key, False))
            s = float(scores_raw.get(mistral_key, 0.0) or 0.0)
            if not mapped:
                extra_cats[mistral_key] = v
                extra_scores[mistral_key] = s
                continue
            for ok in mapped:
                if v:
                    out_cats[ok] = True
                if s > out_scores[ok]:
                    out_scores[ok] = s
        flagged = bool(item.get("flagged")) or any(out_cats.values()) or any(extra_cats.values())
        return {
            "flagged": flagged,
            "categories": out_cats,
            "category_scores": out_scores,
            "extra_categories": extra_cats,
            "extra_category_scores": extra_scores,
        }

    async def moderate_text(
        self, *, input_texts: list[str]
    ) -> ModerationResult:
        body_input: Any = input_texts[0] if len(input_texts) == 1 else input_texts
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/moderations",
                headers=headers,
                json={"model": self._model, "input": body_input},
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        items = j.get("results") or []
        if not isinstance(items, list):
            raise RuntimeError(f"{self.name} unexpected results shape: {str(j)[:300]}")
        normalized = [self._normalize_one(it) for it in items if isinstance(it, dict)]
        return ModerationResult(
            results=normalized,
            model=j.get("model") or self._model,
            raw={"id": j.get("id")},
        )

    async def moderate_image(self, **_: Any) -> ModerationResult:
        raise NotImplementedError(f"{self.name} is text-only")


# Llama Guard 4 taxonomy (S1..S14). Источник: model card meta-llama/Llama-Guard-4-12B.
_LLAMA_GUARD_S_TO_OPENAI = {
    "S1": ["violence"],
    "S2": ["illicit"],
    "S3": ["sexual/minors"],
    "S4": ["sexual/minors"],
    "S5": ["harassment"],
    "S6": ["illicit"],
    "S7": ["harassment"],
    "S8": ["illicit"],
    "S9": ["illicit/violent"],
    "S10": ["hate"],
    "S11": ["self-harm"],
    "S12": ["sexual"],
    "S13": ["illicit"],
    "S14": ["illicit"],
}

_S_CODE_RE = re.compile(r"\bS(?:[1-9]|1[0-4])\b")


class LlamaGuardProvider:
    """OpenAI-compat chat-completions wrapper для Llama Guard / Prompt Guard /
    Qwen3-Guard семейства (Groq, Together, OpenRouter, Cloudflare). Модель
    возвращает text content вида `safe` или `unsafe\\nS1,S5,S10` — парсим в
    OpenAI moderation шейп.

    `multimodal=True` в config → провайдер объявляет `supports_moderation_image`
    и при moderate_image() шлёт `[{type:"text"}, {type:"image_url"}]` content.
    Llama Guard 4 12B — нативно multimodal; 3-8B и Prompt Guard — text-only."""

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = False
    context_window = None
    reasoning = False

    def __init__(
        self,
        name: str,
        *,
        kind: str = "llama_guard",
        base_url: str,
        api_key: str,
        model: str,
        multimodal: bool = False,
        extra_headers: dict[str, str] | None = None,
        auth_scheme: str = "Bearer",
        timeout: float = 30.0,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        quota_limited: bool = False,
    ) -> None:
        self.name = name
        self._kind = kind
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._extra_headers = extra_headers or {}
        self._auth_scheme = auth_scheme
        self._timeout = timeout
        self.supports_moderation_text = True
        self.supports_moderation_image = bool(multimodal)
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.quota_limited = quota_limited

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is moderation-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is moderation-only")
        yield b""  # pragma: no cover

    @staticmethod
    def _parse_verdict(text: str) -> dict[str, Any]:
        """Унифицированный парсер для трёх форматов output:

        1. `safe` / `unsafe\\nS1,S5` — Llama Guard 3/4, Qwen3-Guard.
           Маппим S-коды → OpenAI-категории.
        2. Plain float "0.99..." — Llama Prompt Guard 2 (вероятность того
           что вход = prompt-injection). Threshold 0.5; результат пишем в
           extra_categories.jailbreak (вне OpenAI-таксономии).
        3. Что-то другое — флажок если есть слово "unsafe", иначе safe.
        """
        out_cats = _empty_categories()
        if not text:
            return {"flagged": False, "categories": out_cats, "category_scores": None}
        stripped = text.strip()

        # Format 2: prompt-guard-2 — голый float [0..1].
        try:
            score = float(stripped)
            if 0.0 <= score <= 1.0:
                flagged = score >= 0.5
                return {
                    "flagged": flagged,
                    "categories": out_cats,
                    "category_scores": None,
                    "extra_categories": {"jailbreak": flagged},
                    "extra_category_scores": {"jailbreak": score},
                }
        except ValueError:
            pass

        lowered = stripped.lower()
        if lowered.startswith("safe"):
            return {"flagged": False, "categories": out_cats, "category_scores": None}
        codes = _S_CODE_RE.findall(stripped)
        if not codes:
            return {
                "flagged": "unsafe" in lowered,
                "categories": out_cats,
                "category_scores": None,
            }
        seen: set[str] = set()
        for code in codes:
            for ok in _LLAMA_GUARD_S_TO_OPENAI.get(code, []):
                seen.add(ok)
        for ok in seen:
            out_cats[ok] = True
        return {
            "flagged": True,
            "categories": out_cats,
            "category_scores": None,
        }

    async def _classify(
        self,
        *,
        user_content: Any,
    ) -> dict[str, Any]:
        """Send a single classify request and return parsed verdict + raw text."""
        headers = {
            "Authorization": f"{self._auth_scheme} {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=payload
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
        j = resp.json()
        choices = j.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.name} no choices: {str(j)[:300]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        text_out = _text_from_content(content)
        verdict = self._parse_verdict(text_out)
        verdict["raw_verdict"] = text_out.strip()
        return verdict

    async def moderate_text(
        self, *, input_texts: list[str]
    ) -> ModerationResult:
        results: list[dict[str, Any]] = []
        for txt in input_texts:
            results.append(await self._classify(user_content=txt))
        return ModerationResult(
            results=results,
            model=self._model,
            raw={"kind": self._kind},
        )

    async def moderate_image(
        self, *, images: list[str], context_text: str | None = None
    ) -> ModerationResult:
        if not self.supports_moderation_image:
            raise NotImplementedError(f"{self.name} is text-only (multimodal=False)")
        results: list[dict[str, Any]] = []
        for img in images:
            url = _coerce_image_input(img)
            content_blocks: list[dict[str, Any]] = []
            if context_text:
                content_blocks.append({"type": "text", "text": context_text})
            else:
                # Llama Guard 4 ожидает хоть какой-то текстовый якорь.
                content_blocks.append({"type": "text", "text": "Classify this image."})
            content_blocks.append(
                {"type": "image_url", "image_url": {"url": url}}
            )
            results.append(await self._classify(user_content=content_blocks))
        return ModerationResult(
            results=results,
            model=self._model,
            raw={"kind": self._kind},
        )


# ============================================================================
# Reranker providers
# ============================================================================
# Единый класс на 3 шейпа (Jina/Cohere/Voyage). Все три — POST /rerank на разный
# host, Bearer auth, request почти идентичен:
#   {model, query, documents:[str], top_n|top_k, return_documents?}
# Разница в имени поля top_N и разборе ответа:
#   Jina   v1: results=[{index, relevance_score, document?}], usage.total_tokens
#   Cohere v2: results=[{index, relevance_score}],            meta.billed_units.search_units
#   Voyage v1: data=[{index, relevance_score, document?}],    usage.total_tokens
class RerankProvider:
    """OpenAI-style унификация Jina v1 / Cohere v2 / Voyage v1 reranker API.
    Все три принимают `query + documents[]`, возвращают отсортированный список
    `{index, relevance_score}`. Класс выбирает шейп через `kind`."""

    supports_tools = False
    supports_audio = False
    supports_images = False
    supports_speech = False
    supports_translation = False
    supports_embed = False
    supports_rerank = True
    context_window = None
    reasoning = False

    _SHAPES = {
        "jina_rerank": {
            "path": "/rerank",
            "top_field": "top_n",
            "results_key": "results",
            "tokens_path": ("usage", "total_tokens"),
        },
        "cohere_rerank": {
            "path": "/rerank",
            "top_field": "top_n",
            "results_key": "results",
            "tokens_path": ("meta", "billed_units", "search_units"),
        },
        "voyage_rerank": {
            "path": "/rerank",
            "top_field": "top_k",
            "results_key": "data",
            "tokens_path": ("usage", "total_tokens"),
        },
    }

    def __init__(
        self,
        name: str,
        *,
        kind: str,
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        max_documents: int | None = None,
        max_query_chars: int | None = None,
        rpd: int | None = None,
        rpm: int | None = None,
        quality: int | None = None,
        latency_s: float | None = None,
        ru: int | None = None,
        quota_limited: bool = False,
    ) -> None:
        if kind not in self._SHAPES:
            raise ValueError(f"unknown rerank kind: {kind}")
        self.name = name
        self._kind = kind
        self._shape = self._SHAPES[kind]
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or {}
        self._timeout = timeout
        self._max_documents = max_documents
        self._max_query_chars = max_query_chars
        self.rpd = rpd
        self.rpm = rpm
        self.quality = quality
        self.latency_s = latency_s
        self.ru = ru
        self.quota_limited = quota_limited

    async def chat(self, **_: Any) -> ProviderCallResult:
        raise NotImplementedError(f"{self.name} is rerank-only")

    async def chat_stream(self, **_: Any) -> AsyncIterator[bytes]:
        raise NotImplementedError(f"{self.name} is rerank-only")
        yield b""  # pragma: no cover

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> RerankResult:
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        if not documents or not isinstance(documents, list):
            raise ValueError("documents must be a non-empty list")
        if any(not isinstance(d, str) for d in documents):
            raise ValueError("documents must be list[str]")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            **self._extra_body,
        }
        if top_n is not None:
            payload[self._shape["top_field"]] = int(top_n)
        if return_documents:
            payload["return_documents"] = True
        if extra:
            payload.update(extra)

        url = f"{self._base_url}{self._shape['path']}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()

        if not isinstance(data, dict):
            raise RuntimeError(f"{self.name} unexpected response: {str(data)[:300]}")
        if data.get("error"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"{self.name} provider error: {msg}")

        items = data.get(self._shape["results_key"])
        if not isinstance(items, list):
            raise RuntimeError(
                f"{self.name} no '{self._shape['results_key']}' in response: {str(data)[:300]}"
            )

        results: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f"{self.name} result[{i}] not a dict")
            if "index" not in item or "relevance_score" not in item:
                raise RuntimeError(f"{self.name} result[{i}] missing index/relevance_score")
            entry: dict[str, Any] = {
                "index": int(item["index"]),
                "relevance_score": float(item["relevance_score"]),
            }
            doc = item.get("document")
            if doc is not None:
                # Jina/Voyage могут возвращать {"text": "..."} — нормализуем в plain str.
                if isinstance(doc, dict):
                    text = doc.get("text")
                    if isinstance(text, str):
                        entry["document"] = text
                elif isinstance(doc, str):
                    entry["document"] = doc
            results.append(entry)

        # Шейп API уже отсортирован по убыванию score, но фиксируем инвариант на случай
        # сюрпризов от провайдера.
        results.sort(key=lambda r: r["relevance_score"], reverse=True)

        # Token / search-unit usage.
        total_tokens = 0
        cursor: Any = data
        for key in self._shape["tokens_path"]:
            if isinstance(cursor, dict):
                cursor = cursor.get(key)
            else:
                cursor = None
                break
        if isinstance(cursor, (int, float)):
            total_tokens = int(cursor)

        return RerankResult(
            results=results,
            model=data.get("model") or self._model,
            total_tokens=total_tokens,
            raw={"kind": self._kind},
        )
