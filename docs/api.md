# API reference

Полный список эндпоинтов, авторизация, формат запросов/ответов, фоллбэк-логика, ad-hoc провайдеры, dashboard editor.

## Авторизация

Если `NEUROGATE_API_TOKEN` задан, все защищённые эндпоинты требуют `Authorization: Bearer <token>`. Если пуст — авторизации нет (только для локальной разработки на `127.0.0.1`).

## Эндпоинты

### OpenAI-совместимые (drop-in для существующих клиентов)

| Путь | Метод | Auth | Назначение |
|---|---|---|---|
| `/v1/chat/completions` | POST | **Bearer** | основной chat (поддерживает все цепочки + ad-hoc) |
| `/v1/models` | GET | **Bearer** | список виртуальных моделей и провайдеров |
| `/v1/embeddings` | POST | **Bearer** | embeddings через `embed` / `embed_code` chain |
| `/v1/audio/speech` | POST | **Bearer** | TTS (цепочка `tts` → Edge TTS) |
| `/v1/audio/transcriptions` | POST | **Bearer** | speech-to-text (цепочка `audio` → Whisper/Gemini) |
| `/v1/audio/sfx` | POST | **Bearer** | SFX/ambient generation (цепочка `sfx` → HF Space) |
| `/v1/images/generations` | POST | **Bearer** | text-to-image (цепочка `image_gen`) |
| `/v1/images/edits` | POST | **Bearer** | image editing (цепочка `image_edit`) |
| `/v1/moderations` | POST | **Bearer** | text moderation (цепочка `moderation`) |
| `/v1/translate` | POST | **Bearer** | перевод (цепочка `translation` — дешёвые MT-движки в приоритете) |
| `/v1/messages` | POST | **Bearer** | Anthropic Messages API совместимость |

### Управление и обзор

| Путь | Метод | Auth | Назначение |
|---|---|---|---|
| `/health` | GET | нет | минимальный liveness (`{"ok":true}`) |
| `/dashboard` | GET | нет (HTML) | веб-UI: цепочки, лимиты, исходы, латентность, тестовый чат |
| `/metrics` | GET | **Bearer** | Prometheus метрики (RPS, latency, исходы по провайдерам) |
| `/v1/health` | GET | **Bearer** | detailed: default chain + список цепочек |
| `/v1/stats` | GET | **Bearer** | usage по провайдерам (rpd/rpm + счётчики) |
| `/v1/metrics.json` | GET | **Bearer** | JSON-версия Prometheus счётчиков + средняя латентность |
| `/v1/calls` | GET | **Bearer** | история запросов для дашборда |
| `/v1/chains/edit` | GET | **Bearer** | снимок цепочек + пул провайдеров для редактора |
| `/v1/chains` | PUT | **Bearer** | переписать `chains:`/`default_chain:` + hot reload без рестарта |
| `/v1/audit/run` | POST | **Bearer** | ручной запуск еженедельного аудита моделей |
| `/v1/audit/{date}` | GET | **Bearer** | результаты конкретного аудита |

### Что передавать в `model`

- Любая цепочка: `chat` (default), `chat_fast`, `chat_en`, `code`, `code_fast`, `latency`, `quality`, `unlimited`, `quota`, `image`, `web`, `reasoning_quality`, `reasoning_deep`, `paid`, `translation`, `translate_adaptive`, `moa`, `sc`, `debate`, `deep_search`, `image_gen`, `image_edit`, `audio`, `tts`, `sfx`, `embed`, `embed_code`, `rerank`, `moderation`, `moderation_image`, `moderation_jailbreak`, `moderation_ru`.
- `auto` — эвристический роутинг (см. [chains.md](chains.md#авто-роутинг-auto)).
- Имя конкретного провайдера из `/v1/models`.
- Ad-hoc `kind:model_id` (см. ниже).

Полный текущий состав каждой цепочки — `GET /v1/health` или dashboard.

## OpenAI-совместимость

`/v1/chat/completions` и `/v1/models` — drop-in замена OpenAI API. Подходит любой OpenAI-клиент (Python SDK, Vercel AI SDK, Cursor, curl). Ставишь `base_url = http://<server>:8765/v1` и `api_key = <NEUROGATE_API_TOKEN>` — работает как есть.

**Request body** (стандартный OpenAI `ChatCompletionRequest`):

```json
{
  "model": "chat",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 512,
  "max_completion_tokens": 512,
  "stop": ["\n\n"],
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "n": 1,
  "user": "user-id",
  "tools": [{"type":"function","function":{"name":"...","parameters":{...}}}],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "stream": false
}
```

Любые дополнительные поля принимаются и прокидываются провайдеру.

**Response** — стандартный OpenAI `chat.completion` + дополнительные поля `provider` (какой upstream ответил) и `chain` (какая цепочка использовалась):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1776510642,
  "model": "mistral:large",
  "choices": [{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage": {"prompt_tokens":9,"completion_tokens":7,"total_tokens":16},
  "provider": "mistral:large",
  "chain": "chat"
}
```

**Что поддерживается:** system/user/assistant/tool roles, tool calling (function calling), multi-turn, `n`, `stop`, `temperature`, `top_p`, penalty-поля, `tools`/`tool_choice`/`parallel_tool_calls`, **streaming** (`stream: true` — SSE как в OpenAI, включая Gemini через конвертацию native-чанков), **image input** (OpenAI vision-формат с `{"type":"image_url",...}` — работает на цепочке `image`), **web-search** (цепочка `web`).

**Нюанс стриминга**: fallback между провайдерами работает только **до первого чанка**. Как только upstream начал стримить — мы закоммичены; mid-stream ошибка придёт клиенту как обрыв соединения, а не как прозрачное переподключение (проще и совпадает с поведением самого OpenAI). Цепочки `moa`/`sc`/`debate`/`deep_search` **не поддерживают streaming** — они агрегирующие.

**Что НЕ поддерживается:**
- Logprobs — не пересылаются.
- `tools` в цепочках `moa`/`sc`/`debate`/`deep_search`.

## Как работает фоллбэк

При каждом запросе роутер пробует провайдеров по порядку. Переход к следующему происходит при:

- HTTP 429, 500, 502, 503, 504
- `resource_exhausted`, `quota`, `rate limit`
- `unavailable`, `overloaded`, `timeout`
- `empty response` (модель вернула пустоту)
- `model_not_found`, `decommissioned` (модель убрали у провайдера)
- `error` поле в JSON-теле с HTTP 200 (OpenRouter так делает при исчерпанной квоте)

Любая другая ошибка (400 bad request, невалидный ключ) — НЕ фоллбэкается, возвращается клиенту как 502.

Если **все** провайдеры цепочки провалились — HTTP 502 с телом `{"error":{"type":"upstream_exhausted","message":"..."}}`.

## Ad-hoc модели (любая `kind:model_id`)

Если в `model` приходит строка с разделителем `:` или `/`, которой нет ни среди цепочек, ни среди провайдеров — например `openai:gpt-5-foo`, `groq/llama-99b` — neurogate парсит префикс как `kind`, забирает дефолтный `base_url` из `PROVIDER_KIND_DEFAULTS` и серверный env-ключ, собирает `OpenAICompatProvider` на лету и шлёт запрос. Никакого UI/storage для ключа, никаких хедеров от клиента — единственный источник правды это env на сервере.

Поддерживаемые kinds (env → kind):

| env-переменная сервера | kind | дефолтный base_url |
|---|---|---|
| `OPENAI_API_KEY` | `openai` | `https://api.openai.com/v1` |
| `GROQ_API_KEY` | `groq` | `https://api.groq.com/openai/v1` |
| `CEREBRAS_API_KEY` | `cerebras` | `https://api.cerebras.ai/v1` |
| `SAMBANOVA_API_KEY` | `sambanova` | `https://api.sambanova.ai/v1` |
| `NVIDIA_API_KEY` | `nvidia` | `https://integrate.api.nvidia.com/v1` |
| `OPENROUTER_API_KEY` | `openrouter` | `https://openrouter.ai/api/v1` |
| `MISTRAL_API_KEY` | `mistral` | `https://api.mistral.ai/v1` |
| `GITHUB_MODELS_TOKEN` | `github` | `https://models.github.ai/inference` |
| `HF_TOKEN` | `huggingface` | `https://router.huggingface.co/v1` |
| `ZAI_API_KEY` | `zai` | `https://api.z.ai/api/paas/v4` |
| `POLLINATIONS_API_KEY` | `pollinations` | `https://text.pollinations.ai/v1` |
| `TOGETHER_API_KEY` | `together` | `https://api.together.xyz/v1` |

Поведение на ошибки:

- неизвестный kind → 400 `kind 'wat' is not supported for ad-hoc models (need: ...)`
- env-ключ не задан → 400 `kind 'openai' requires OPENAI_API_KEY on the server, which is not set`
- строка без `:` / `/` → silent fallback на default chain (как и раньше для неизвестных model)

Кеш ad-hoc провайдеров живёт в памяти роутера: первый запрос строит HTTP-клиента, повторные используют закешированный. Кеш сбрасывается при горячем релоаде через `PUT /v1/chains`. Gemini / GigaChat / Yandex / Cloudflare / LlamaGuard и прочие kinds, требующие `folder_id` / `account_id` / нестандартного auth — не поддерживаются ad-hoc, добавляйте их явно в `config.yaml`.

В дашборде есть текстовое поле «custom» рядом с селектом модели — туда можно вписать `kind:model_id`, оно переопределяет dropdown.

## Dashboard editor (drag-n-drop редактор цепочек)

В дашборде, во вью «Провайдеры», есть кнопка «✎ редактировать цепочки». Включает редактор с drag-n-drop через SortableJS:

- Слева — список цепочек: переключение, переименовать, удалить, отметить как default (★).
- В центре — провайдеры активной цепочки: перетаскивание для reorder, ✕ для удаления.
- Справа — пул всех провайдеров (с фильтром): drag в любую цепочку (один и тот же провайдер может быть в нескольких цепочках).
- Кнопка «сохранить» → `PUT /v1/chains`. Сервер валидирует, переписывает `chains:` и `default_chain:` в `config.yaml` (комментарии и секция `providers:` сохраняются), пишет бэкап `config.yaml.bak`, делает hot-reload роутера в памяти. Следующий запрос идёт по новой схеме без рестарта процесса.

Безопасность: эндпоинт под тем же `NEUROGATE_API_TOKEN`. Если у тебя `config.yaml` синкается с локального файла на прод (через rsync / CI), помни: правки через дашборд на сервере перезапишутся следующим деплоем.

## Конфиг

`config.yaml` — список провайдеров в порядке фоллбэка. Каждый провайдер:

```yaml
providers:
  - name: groq:llama-3.3-70b       # любое имя для логов
    kind: groq                      # см. таблицу kinds
    model: llama-3.3-70b-versatile  # имя модели на стороне провайдера
    api_key_env: GROQ_API_KEY       # откуда брать ключ
    rpm: 30                          # опционально: локальный кап запросов/мин
    rpd: 1000                        # опционально: локальный кап запросов/сут
    # extra_body: { thinking: { type: disabled } }   # для Z.ai и подобных
    # extra_headers: { X-My-Header: value }
    # timeout: 60

chains:
  chat:
    - groq:llama-3.3-70b
    - cerebras:qwen-3-235b
  code:
    - nvidia:qwen3-coder-480b
    - gemini:flash-latest

default_chain: chat
```

Меняешь порядок → меняешь приоритет фоллбэка. Добавить провайдера — дописать блок.

### Multi-key pool (несколько ключей одного провайдера)

Если у тебя несколько ключей от Groq / Gemini / OpenRouter — впиши их через запятую в env, замени `api_key_env` на `api_keys_env`:

```yaml
- name: groq:llama-3.3-70b
  kind: groq
  model: llama-3.3-70b-versatile
  api_keys_env: GROQ_API_KEYS_POOL  # GROQ_API_KEYS_POOL=gsk_aaa,gsk_bbb,gsk_ccc
```

Provider развернётся в `groq:llama-3.3-70b#1`, `#2`, `#3` — каждый со своим ключом, общая квота умножается.

### Локальный rate-tracking

Если у провайдера задан `rpm` и/или `rpd`, прокси ведёт собственный счётчик попыток в SQLite (`rate_events`) и при достижении потолка **скипает провайдера без сетевого вызова** — переходит к следующему в цепочке. Окна rolling: последние 60 секунд для `rpm`, последние 24 часа для `rpd`. Журнал переживает рестарт.

Поля опциональны — если не указать, проверки нет и провайдер вызывается пока сам не отдаст 429 (который уже ловится фоллбэком). Разумные значения уже проставлены в [config.yaml.example](../config.yaml.example).

## Наблюдаемость

- **`/metrics`** — Prometheus-endpoint. Метрики:
  - `neurogate_requests_total{provider,outcome}` — счётчик запросов по исходам: `success` / `rate_limit` / `server_error` / `empty` / `timeout` / `decommissioned` / `rate_capped` / `other`
  - `neurogate_request_duration_seconds{provider,outcome}` — гистограмма латентности (Grafana считает p50/p95 через `histogram_quantile()`)
- **`/v1/stats`** — текущая загрузка rate-tracker'а: сколько запросов к каждому провайдеру за последние 1m/24h и их caps.
- **SQLite** `stats.db` — хранит `rate_events` для per-provider rate tracking.

Метрики Prometheus — **в памяти**, сбрасываются при рестарте. Долгоживущее хранение — задача TSDB скрапера.

## Безопасность

Трафик по умолчанию — HTTP без TLS. Токен и содержимое запросов идут в открытом виде. Для прода — заверни через nginx + Let's Encrypt на поддомене (см. [README — server install](../README.md#на-сервере-production)).

`/health` и `/metrics` — открыты анонимно: `/health` для liveness, `/metrics` для Prometheus-скраперов. `/metrics` показывает имена провайдеров и счётчики, но не содержимое запросов или ключи. Если хочешь спрятать и их — закрой 8765 в файрволе и проксируй через nginx с auth.
