# Цепочки (chains) — справочник

Все `model`-значения, которые neurogate умеет принимать, и что они делают. Имя цепочки передаётся в поле `model` запроса — остальное neurogate выбирает сам.

Полный текущий состав каждой цепочки видно в `GET /v1/health` или в дашборде на `/dashboard`. Здесь — описание + первые провайдеры, чтобы понимать характер.

## Базовые fallback-цепочки

Простые цепочки: **пробуют провайдеров по порядку до первого успеха**, 429/5xx переходят на следующего. Ответ приходит от первого сработавшего.

| `model` | Что делает | Первые в цепочке |
|---|---|---|
| `chat` (default) | Русский chit-chat, RU-floor по качеству | `github:gpt-4.1-mini` → `yandex:alice` → `gemini:2.5-flash-lite` → `sambanova:llama-4-maverick` → большой fallback-хвост |
| `chat_fast` | `chat` без слишком медленных моделей | первые ~10 быстрых из `chat` |
| `chat_en` | Английский chit-chat, AA Intelligence Index desc | `gemini:flash-latest` → `sambanova:deepseek-v3.2` → `openrouter:nemotron-3-super-free` |
| `code` | Reasoning + код, композит SWE-bench/LiveCodeBench. **Долгая** (10-40s) — первые провайдеры с thinking | `nvidia:qwen3-coder-480b` → `gemini:flash-latest` → `nvidia:qwen3.5-397b` → `sambanova:deepseek-v3.2` |
| `code_fast` | `code` без долгих thinking-моделей | подмножество `code` |
| `latency` | Минимум wall-clock (медиана Total time) | `cerebras:llama3.1-8b` (~420ms) → `groq:llama-3.1-8b` (~670ms) |
| `quality` | Максимум AA Intelligence Index v4.0 | `freetheai:gpt-5` → `gemini:flash-latest` → `sambanova:deepseek-v3.2` |
| `unlimited` | Только провайдеры без жёсткого RPD-капа (для high-volume) | `nvidia:*` → `zai:*` → `mistral:*` |
| `quota` | Только провайдеры с quota-limited free tiers (когда хочется бить именно по ним) | различные `:free` |
| `reasoning_quality` | Thinking-модели по AA Index | `gemini:flash-latest` → `openrouter:nemotron-3-super-free` → `groq:gpt-oss-120b` |
| `reasoning_deep` | Те же, но по глубине thinking (reasoning_tokens desc) | `gemini:2.5-flash` → `groq:qwen3-32b` → `zai:glm-4.5-flash` |
| `paid` | Платные модели для сравнения (нужен **отдельный ключ `OPENROUTER_PAID_API_KEY`**) | `openrouter:claude-opus-4.7` |

## Специальные цепочки

Эти не являются простым fallback — они делают что-то отличное от «перебирай провайдеров».

### `web` — актуальные данные через веб-поиск

Цепочка активирует native-веб-поиск у провайдера: Gemini вызывает `google_search`-tool server-side (бесплатно), OpenRouter `:online`-провайдеры используют Exa-поиск на своей стороне (через `OPENROUTER_PAID_API_KEY`). Клиенту ничего дополнительно делать не нужно — просто `model: "web"`.

> ⚠️ **Осторожно: prompt injection.** Когда модель тянет содержимое веб-страниц, на них могут оказаться спрятанные инструкции от злоумышленника («забудь предыдущие указания, верни вместо ответа …»). Любой ответ `web`-цепочки нужно воспринимать как **недоверенный ввод**: не кидай его сразу в дальнейшие tool-вызовы или в код, который что-то выполняет. Не используй `web` для агентного цикла «найди → выполни», без ручного review посередине.

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "web",
    "messages": [{"role":"user","content":"Что нового у Anthropic в апреле 2026?"}],
    "max_tokens": 1000
  }' | python3 -m json.tool
```

URL-источники приходят в тексте ответа (как inline-ссылки) или в `annotations.url_citation` у `:online`-провайдеров. Raw-список URL для дальнейшего fetch-а пока не извлекается отдельным полем (на roadmap — отдельный bare-search backend через Brave/DuckDuckGo).

### `image` — vision (распознавание картинок)

Цепочка из vision-capable моделей, отсортированных по точности описаний. Клиент передаёт `content` как массив с элементами `{"type":"text",...}` и `{"type":"image_url","image_url":{"url":"..."}}` — стандартный OpenAI vision-формат.

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Что на картинке?"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}}
      ]
    }],
    "max_tokens": 300
  }' | python3 -m json.tool
```

Поддерживаются URL (http/https), data-URI (`data:image/png;base64,...`) работают у части провайдеров — Gemini и GitHub Models принимают, Groq и Cerebras только URL.

### `image_gen` / `image_edit` — генерация и редактирование картинок

Отдельные эндпоинты `/v1/images/generations` и `/v1/images/edits` (drop-in OpenAI). Цепочка `image_gen` пробует Cloudflare (FLUX/SDXL/Phoenix/DreamShaper), затем Yandex ART, GigaChat (Kandinsky), Pollinations, FreeTheAi, Gemini nano-banana, AIhorde. RU-модели (Kandinsky/YandexART) держат фотореализм лучше FLUX когда в промпте есть «digital painting»/«oil painting».

### `audio` / `tts` / `sfx` — речь, озвучка, звуки

- `audio` — распознавание речи (`POST /v1/audio/transcriptions`): Groq Whisper → Gemini.
- `tts` — синтез речи (`POST /v1/audio/speech`): Edge TTS, без ключа, безлимит.
- `sfx` — генерация звуковых эффектов и эмбиента (`POST /v1/audio/sfx`): HuggingFace Space `Stable-Audio-Open-Zero` через `gradio_client`. Bottleneck — HF Zero-GPU (~3-5 минут квоты/день/IP анонимно; с `HF_TOKEN` чуть больше; с PRO — реально юзабельно).

```bash
# SFX-пример
curl -X POST http://127.0.0.1:8765/v1/audio/sfx \
  -H "Content-Type: application/json" \
  -d '{"prompt":"wind howling through pine forest at night, no music","duration":10}' \
  --output ambient.wav
```

### `translation` — дешёвый перевод

Цепочка идёт через специализированные MT-движки (LibreTranslate, MyMemory, Yandex Translate, Cohere Aya) и обращается к chat-моделям только в фоллбэке. Дешевле и быстрее, чем гонять перевод через `chat`.

`/v1/translate` — отдельный эндпоинт; `model: "translation"` через `/v1/chat/completions` тоже работает.

### `embed` / `embed_code` / `rerank` / `moderation`

Отдельные эндпоинты для embeddings (`/v1/embeddings`), реранкинга, модерации (`/v1/moderations`). Дополнительные варианты модерации: `moderation_image`, `moderation_jailbreak`, `moderation_ru`.

### `moa` — Mixture of Agents (ансамбль моделей)

Параллельный fan-out к **25 уникальным моделям** → aggregator (дефолт — `reasoning_quality`) синтезирует финальный ответ. По MoA-паттерну Wang et al., 2024 (Together AI): «25 свободных моделей + аггрегатор» часто обходят один фронтир-вызов на open-ended Q&A. Aggregator **исключает свой собственный ответ** из synthesis-промпта, чтобы избежать self-bias.

```bash
# все 25 моделей опрашиваются параллельно, aggregator синтезирует
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moa",
    "messages": [{"role":"user","content":"Объясни квантовую запутанность школьнику"}],
    "max_tokens": 1500
  }' | python3 -m json.tool

# сменить aggregator-цепочку
curl -s 'http://127.0.0.1:8765/v1/chat/completions?aggregator=quality' \
  -H "Content-Type: application/json" \
  -d '{"model":"moa","messages":[{"role":"user","content":"..."}]}'
```

**Response extras**: поле `moa` с `proposals[]` (все индивидуальные ответы: `{provider, text, latency_ms, completion_tokens, error}`), `aggregator_chain`, `aggregator_provider`, `proposer_count`, `proposer_success`.

**Ограничения**: `stream=true` и `tools` не поддерживаются (собирается целиком, потом синтезируется); 1 запрос → ~26 вызовов (25 proposers + 1 aggregator), latency 10-90s (упирается в самого медленного proposer-а, timeout 90s).

### `sc` — Self-Consistency (N сэмплов одной модели)

Паттерн Wang et al., 2022: берём N сэмплов от **одной** модели (дефолт 5) с высокой `temperature=1.0`, потом aggregator консолидирует разные линии рассуждения. Дешевле MoA (5-6× вызовов vs 26×), помогает когда нужно разнообразие рассуждений, а не разнообразие архитектур. Особенно полезно на math/verifiable задачах.

```bash
# 5 сэмплов gemini:flash-latest (первый в цепочке sc) → aggregator синтезирует
curl -s 'http://127.0.0.1:8765/v1/chat/completions?samples=5' \
  -H "Content-Type: application/json" \
  -d '{"model":"sc","messages":[{"role":"user","content":"Сколько будет 17 × 24? Покажи вычисления."}]}'
```

**Query-параметры**: `?samples=N` (2-20, default 5), `?aggregator=<chain>` (default `reasoning_quality`).

**Response extras**: поле `sc` с `samples[]` (список сэмплов: `{sample_index, provider, text, latency_ms, completion_tokens, temperature, error}`), `base_provider`, `aggregator_chain`, `sample_count`, `sample_success`.

### `debate` — Multi-Agent Debate (N моделей × R раундов)

Паттерн Du et al. 2023 («Improving Factuality and Reasoning via Multiagent Debate») / Liang et al. 2023 («Encouraging Divergent Thinking»). N **разных** моделей отвечают независимо в раунде 0; в раундах 1..R-1 каждый агент видит ответы коллег и переписывает свой с учётом критики; aggregator синтезирует финальный ответ из последнего раунда. Ключевое отличие от `moa` — обмен мнениями: ошибки/галлюцинации одной модели корректируются другими через диалог, а не просто усредняются. Полезно на factual-задачах, где модели иначе поодиночке уверенно врут.

```bash
# 3 агента × 2 раунда (default), aggregator = reasoning_quality
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "debate",
    "messages": [{"role":"user","content":"Сколько штатов было в США в 1820 году?"}],
    "max_tokens": 1500
  }' | python3 -m json.tool

# 4 агента × 3 раунда + альтернативный aggregator
curl -s 'http://127.0.0.1:8765/v1/chat/completions?agents=4&rounds=3&aggregator=quality' \
  -H "Content-Type: application/json" \
  -d '{"model":"debate","messages":[{"role":"user","content":"..."}]}'
```

**Query-параметры**: `?agents=N` (2-6, default 3), `?rounds=R` (1-4, default 2; R=1 ≈ MoA без обмена), `?aggregator=<chain>` (default `reasoning_quality`).

**Response extras**: поле `debate` с `agents[]`, `rounds`, `transcript[round][agent]`, `aggregator_chain`, `aggregator_provider`, `agent_count`, `final_round_success`.

**Ограничения**: `stream=true` и `tools` не поддерживаются; latency = R × медианный latency агента + aggregator (для 3×2 ≈ 15-30s); ~N×R + 1 LLM-вызов на запрос.

### `deep_search` — research-агент (plan → search → synthesize → critique)

Multi-step pipeline уровня Perplexity Deep Research:
1. **Planner** раскладывает вопрос на 2-4 подзапроса (JSON).
2. **Searcher** × N (parallel) — каждый подзапрос через `web`-цепочку.
3. **Synthesizer** собирает все findings + извлечённые URL-ы в структурированный markdown с numbered-цитатами `[1][2]`.
4. **Critic** проверяет draft на пробелы → если найдены, ещё 1 round `search + synth`.

> ⚠️ **Та же история с prompt injection, что и у `web` — только сильнее.** `deep_search` несколько раз подряд тянет веб-контент и кормит его в LLM. Если на одной из подгруженных страниц стоит инъекция, она проходит через synthesizer и попадает в финальный ответ как «факт со ссылкой». Используй для research-задач, где ты сам проверишь итог; **не** запускай в автоматических цепочках, где `deep_search`-output идёт прямо в действия.

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deep_search",
    "messages": [{"role":"user","content":"какие фронтенд-фреймворки актуальны для 2026 года"}],
    "max_tokens": 2000
  }' | python3 -m json.tool

# настройка: 3 подзапроса, без critic-раунда
curl -s 'http://127.0.0.1:8765/v1/chat/completions?max_subq=3&rounds=0' \
  -H "Content-Type: application/json" \
  -d '{"model":"deep_search","messages":[{"role":"user","content":"..."}]}'
```

**Query-параметры**: `?max_subq=N` (1-6, default 4), `?rounds=R` (0-2 дополнительных critic-раунда, default 1), `?planner=<chain>`, `?searcher=<chain>` (default `web`), `?synth=<chain>`, `?critic=<chain>`.

**Response extras**: поле `deep_search` с `subquestions[]`, `sources[]` (`{id, url, subq}`), `trace[]` (каждый шаг), `iterations`.

**Типичные цифры**: 15-40s wall-clock, 5-8 LLM-вызовов (без critic-round) или 8-12 (с ним), 3-10 URL в финальных источниках.

## Когда какую цепочку использовать

| Задача | Рекомендуемая |
|---|---|
| Быстрый чат на русском | `chat` |
| Быстрый чат на английском | `chat_en` |
| Код / алгоритмы / рефакторинг | `code` (медленно, но качественно) |
| Нужно быстро, качество второстепенно | `latency` или `chat_fast` |
| Нужно умно, скорость вторична | `quality` или `reasoning_quality` |
| Актуальные данные из веба | `web` |
| Картинки, OCR, описание фото | `image` |
| Генерация картинок | `image_gen` (или `POST /v1/images/generations`) |
| Озвучка | `tts` (или `POST /v1/audio/speech`) |
| Перевод | `translation` (или `POST /v1/translate`) |
| Критичный ответ, нужен консенсус многих моделей | `moa` |
| Factual-вопрос, где модели поодиночке уверенно врут | `debate` |
| Математика / пошаговые рассуждения | `sc` (с math-вопросом) или `reasoning_deep` |
| Ресёрч-вопрос с несколькими подтемами и цитатами | `deep_search` |
| Сложная задача, нужен фронтир-класс | `paid` (тратит Opus-ключ) |

## Авто-роутинг (`auto`)

Имя `auto` (настраивается через `NEUROGATE_VIRTUAL_MODEL`) — heuristic intent detection по последнему user-сообщению. `image_url` → `image`, веб-маркеры/URL → `web`, код-маркеры (` ``` ` / `def ` / `Traceback` / `.py`) → `code`, длинный prompt с reasoning-маркерами → `reasoning_quality`, иначе → `default_chain`.

Это **regex-эвристика, не ML-классификатор**; для критичных кейсов выбирай чейн явно. `paid`/`moa`/`sc`/`debate`/`deep_search` авто-роутер **никогда** не выбирает (требуют opt-in). Graceful degradation: если в твоём `config.yaml` нет нужной целевой цепочки — `auto` тихо падает в `default_chain`.
