"""Anthropic Messages API compatibility layer.

Translates between Anthropic's `/v1/messages` shape and OpenAI's
`/v1/chat/completions` — the format every provider in this gateway already speaks.

Lets Claude Code, the Anthropic SDK, and any Anthropic-native client hit this
gateway by pointing `ANTHROPIC_BASE_URL` at it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel


# ---------- Anthropic request model (permissive) ----------

class MessagesRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


# ---------- finish_reason ↔ stop_reason ----------

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    "stop_sequence": "stop_sequence",
}


def _finish_to_stop(reason: str | None) -> str:
    return _FINISH_TO_STOP.get(reason or "stop", "end_turn")


# ---------- request: Anthropic → OpenAI ----------

def _system_to_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n\n".join(p for p in parts if p)
    return str(system)


def _anthropic_content_to_openai(content: Any) -> tuple[str, list[dict[str, Any]]]:
    """Return (text, tool_calls) from an assistant-side Anthropic content array."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    return "".join(text_parts), tool_calls


def _tool_result_blocks_to_openai(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Each Anthropic tool_result block becomes a separate OpenAI `tool` message."""
    msgs: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        raw = block.get("content")
        if isinstance(raw, list):
            text = "\n".join(
                p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = raw if isinstance(raw, str) else json.dumps(raw) if raw is not None else ""
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": text,
            }
        )
    return msgs


def _user_content_to_openai(content: Any) -> list[dict[str, Any]]:
    """A single Anthropic user message may contain text blocks + tool_result blocks.

    Tool results must become separate OpenAI `tool` messages (OpenAI has no
    `tool_result` inside a user message), so we may emit multiple messages.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": str(content)}]

    tool_msgs = _tool_result_blocks_to_openai(content)
    text_parts = [
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    out: list[dict[str, Any]] = list(tool_msgs)
    text = "".join(text_parts)
    if text:
        out.append({"role": "user", "content": text})
    return out


def _tool_choice_to_openai(tc: dict[str, Any] | None) -> Any:
    if tc is None:
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        name = tc.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


def _tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def request_to_openai(req: MessagesRequest) -> dict[str, Any]:
    """Return kwargs suitable for `LLMRouter.chat(...)`."""
    openai_messages: list[dict[str, Any]] = []
    system_text = _system_to_text(req.system)
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    for m in req.messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            openai_messages.extend(_user_content_to_openai(content))
        elif role == "assistant":
            text, tool_calls = _anthropic_content_to_openai(content)
            out: dict[str, Any] = {"role": "assistant"}
            if text:
                out["content"] = text
            if tool_calls:
                out["tool_calls"] = tool_calls
            # assistant must have something — content or tool_calls
            if "content" not in out and "tool_calls" not in out:
                out["content"] = ""
            openai_messages.append(out)
        # Anthropic "system" inside messages is uncommon; silently drop — it goes in req.system.

    return {
        "messages": openai_messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "tools": _tools_to_openai(req.tools),
        "tool_choice": _tool_choice_to_openai(req.tool_choice),
    }


# ---------- response: OpenAI → Anthropic ----------

def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def result_to_anthropic(
    *,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed = {"_raw": raw_args}
        block: dict[str, Any] = {
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": parsed if isinstance(parsed, dict) else {"_value": parsed},
        }
        # Preserve Gemini's thought signature if present, so a replay works.
        sig = tc.get("_gemini_thought_signature")
        if sig:
            block["_gemini_thought_signature"] = sig
        blocks.append(block)
    if not blocks:
        blocks = [{"type": "text", "text": ""}]

    usage: dict[str, Any] = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }
    if cached_tokens:
        usage["cache_read_input_tokens"] = cached_tokens

    return {
        "id": _new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": _finish_to_stop(finish_reason),
        "stop_sequence": None,
        "usage": usage,
    }


# ---------- streaming translator: OpenAI SSE → Anthropic SSE ----------

def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def translate_stream(
    openai_stream: AsyncIterator[bytes],
    *,
    model: str,
) -> AsyncIterator[bytes]:
    """Wrap an OpenAI-format SSE byte stream as an Anthropic-format SSE byte stream.

    Upstream may split events across chunks — we buffer until we see a blank line.
    """
    msg_id = _new_message_id()
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    # Content block state.
    text_block_index: int | None = None  # anthropic index for active text block
    tool_index_map: dict[int, int] = {}  # openai tool index → anthropic index
    next_anthropic_index = 0
    stop_reason = "end_turn"
    output_tokens = 0  # we don't always know, fall back to 0

    buf = ""
    done = False

    async for raw in openai_stream:
        if done:
            break
        buf += raw.decode("utf-8", errors="replace")
        while "\n\n" in buf:
            event, buf = buf.split("\n\n", 1)
            data_line = None
            for line in event.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data_line = line[len("data: "):]
                    break
            if data_line is None:
                continue
            if data_line == "[DONE]":
                done = True
                break
            try:
                payload = json.loads(data_line)
            except json.JSONDecodeError:
                continue

            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason")

            content_delta = delta.get("content")
            if content_delta:
                if text_block_index is None:
                    text_block_index = next_anthropic_index
                    next_anthropic_index += 1
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": content_delta},
                    },
                )

            for tc in delta.get("tool_calls") or []:
                oi = tc.get("index", 0)
                fn = tc.get("function") or {}
                if oi not in tool_index_map:
                    # Close text block before switching to tool blocks.
                    if text_block_index is not None:
                        yield _sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": text_block_index},
                        )
                        text_block_index = None
                    ai = next_anthropic_index
                    next_anthropic_index += 1
                    tool_index_map[oi] = ai
                    start_block: dict[str, Any] = {
                        "type": "tool_use",
                        "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                        "name": fn.get("name", ""),
                        "input": {},
                    }
                    sig = tc.get("_gemini_thought_signature")
                    if sig:
                        start_block["_gemini_thought_signature"] = sig
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": ai,
                            "content_block": start_block,
                        },
                    )
                ai = tool_index_map[oi]
                args_frag = fn.get("arguments")
                if args_frag:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": ai,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_frag,
                            },
                        },
                    )

            if finish:
                stop_reason = _finish_to_stop(finish)
                for ai in sorted(tool_index_map.values()):
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": ai},
                    )
                tool_index_map.clear()
                if text_block_index is not None:
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_block_index},
                    )
                    text_block_index = None
                done = True
                break

    # Defensive: close anything still open if upstream dropped.
    for ai in sorted(tool_index_map.values()):
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": ai})
    if text_block_index is not None:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})
