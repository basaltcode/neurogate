# Tool calling (function calling)

neurogate пробрасывает OpenAI-совместимые поля `tools` и `tool_choice` в upstream и возвращает `tool_calls` / `finish_reason: "tool_calls"` как есть. Можно использовать neurogate как backend для агентов, работающих по протоколу OpenAI function calling.

## Как это работает в прокси

- `tools` и `tool_choice` из запроса пересылаются в upstream без изменений.
- Роутер **автоматически пропускает** провайдеров без поддержки tool calling, если `tools` переданы — сейчас это только **Gemini native** (его SDK требует отдельной конвертации схемы, она пока не реализована). Все OpenAI-совместимые провайдеры (Groq, Cerebras, SambaNova, NVIDIA, OpenRouter, Z.ai, GitHub Models, Mistral, Cloudflare, собственный OpenAI) поддерживают tools.
- Клиентские сообщения с `role: "assistant"` + `tool_calls` и `role: "tool"` + `tool_call_id` проксируются as-is.
- При `finish_reason: "tool_calls"` поле `content` в ответе будет `null` (как в OpenAI API).
- При любой retryable-ошибке (429/5xx/quota/timeout) прокси переходит к следующему провайдеру, сохраняя тот же список tools.

## Пример (Python, OpenAI SDK)

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="sk-noop")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather by city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "What's the weather in Berlin?"}]

# round 1: модель решает вызвать инструмент
r = client.chat.completions.create(model="auto", messages=messages, tools=tools)
assistant_msg = r.choices[0].message

if r.choices[0].finish_reason == "tool_calls":
    messages.append(assistant_msg.model_dump(exclude_none=True))
    for call in assistant_msg.tool_calls:
        args = json.loads(call.function.arguments)
        # исполняешь функцию САМ
        result = f"+18C sunny in {args['city']}"
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })
    # round 2: отдаём результат обратно модели
    r2 = client.chat.completions.create(model="auto", messages=messages, tools=tools)
    print(r2.choices[0].message.content)
```

## Интеграционный брифинг для AI-агента

Памятка для модели / агента, которого подключают к neurogate:

- **Endpoint**: `POST <base_url>/v1/chat/completions`. Протокол полностью совместим с OpenAI Chat Completions API.
- **Имя модели в запросе**: одно из значений — см. [chains.md](chains.md) для полного списка. Короче: `"chat"` (default), `"code"`, `"latency"`, `"quality"`, `"chat_en"`, `"unlimited"`, `"image"` (vision), `"web"` (веб-поиск), `"reasoning_quality"` / `"reasoning_deep"` (thinking), `"paid"` (Claude Opus), `"moa"` (ансамбль 25 моделей), `"sc"` (N сэмплов одной модели), `"debate"` (N моделей × R раундов inter-agent revision), `"deep_search"` (research-агент), `"auto"` (= default). Имя фактически отработавшего провайдера — в `model` и `provider` ответа, имя выбранной цепочки — в `chain`. Для ансамблей/research дополнительные метаданные лежат в полях `moa` / `sc` / `debate` / `deep_search` соответственно.
- **Поддерживаемые поля запроса**, которые реально доезжают до upstream: `messages`, `temperature`, `max_tokens` / `max_completion_tokens`, `tools`, `tool_choice`. Остальные (`top_p`, `stop`, `presence_penalty`, `parallel_tool_calls`, …) принимаются схемой, но на этой версии в upstream не пересылаются.
- **Стриминг поддерживается**: `stream: true` → SSE. Fallback на следующего провайдера работает только до первого чанка; после — обрыв соединения при ошибке.
- **Формат tool definition** — OpenAI-стиль:
  ```json
  {"type":"function","function":{"name":"...","description":"...","parameters":{ /* JSON Schema */ }}}
  ```
- **Формат ответа при вызове инструмента**:
  ```json
  {
    "choices":[{
      "message":{
        "role":"assistant",
        "content":null,
        "tool_calls":[
          {"id":"call_0","type":"function","function":{"name":"...","arguments":"<JSON-строка аргументов>"}}
        ]
      },
      "finish_reason":"tool_calls"
    }],
    "provider":"groq:llama-3.3-70b"
  }
  ```
  `arguments` — это именно строка с JSON, её нужно `json.loads()` на клиенте.
- **Как отдавать результат инструмента обратно**: добавь в `messages` сначала весь assistant-message (с `tool_calls`), затем по одному message на каждый вызов:
  ```json
  {"role":"tool","tool_call_id":"call_0","content":"<строковый результат>"}
  ```
  `content` должен быть строкой. Если хочешь передать структуру — сериализуй в JSON сам.
- **Фоллбэк прозрачен**: если upstream вернул 429/5xx/пустой ответ — повтор с тем же списком tools уйдёт на следующего провайдера. Поле `provider` в ответе покажет, кто реально отработал.
- **Аутентификация**: если на сервере выставлен `NEUROGATE_API_TOKEN`, шли его в `Authorization: Bearer <token>` (в OpenAI SDK — через `api_key`). Без токена (локальный режим) подойдёт любая dummy-строка вроде `sk-noop`.
- **Ошибки**: если все провайдеры упали / исчерпаны — HTTP 502 с телом `{"error":{"type":"upstream_exhausted","message":"..."}}`. Ретрай возможен, но уместен только если время исправит квоты.
- **Качество моделей**: цепочка (см. [config.yaml.example](../config.yaml.example)) рассчитана на то, что первые в списке — самые «умные» свободные. Если агенту нужна стабильная модель — зафиксируй её на клиенте, не через `auto`.
