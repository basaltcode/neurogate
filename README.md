# llmgate

> ⚠️ **Early access (ранний доступ).** Баги ожидаемы. Если что-то отвалилось — открой [issue](../../issues/new?template=bug_report.md), посмотрим. Шаблон формы и подсказка какие данные приложить — `docs/bug-report.md`.

> **English**: free-tier LLM multiplexer — one OpenAI-compatible endpoint on top of 10+ providers (Gemini, Groq, Cerebras, SambaNova, OpenRouter, Cloudflare, GitHub Models, Mistral, NVIDIA, Z.AI, GigaChat, HuggingFace) with automatic fallback on 429/5xx/quota, web search (`model: "web"`), vision (`model: "image"`), and ensembles (`moa` / `sc` / `debate` / `deep_search`). Drop-in replacement for the OpenAI API — point your SDK at `http://127.0.0.1:8765/v1` and use one of 16 chain names as `model`. Self-hosted, $0/month. Config and docs below are in Russian; the code, config keys, and HTTP API are English. License: MIT.

**Бесплатный мультиплексор LLM-провайдеров: один OpenAI-совместимый endpoint поверх 10+ free-тиров с автоматическим фоллбэком, веб-поиском, распознаванием картинок и ансамблями моделей.**

## Содержание

- [Quick start (5 минут)](#quick-start-5-минут)
- [Что это даёт](#что-это-даёт)
- [Field notes — наблюдения с практики](#field-notes--наблюдения-с-практики)
- [Без ключей — что работает](#без-ключей--что-работает)
- [Чем отличается от других](#чем-отличается-от-других)
- [Поддерживаются из коробки](#поддерживаются-из-коробки)
- [Цепочки](#цепочки)
- [Конфиг](#конфиг)
- [Деплой](#деплой)
- [Безопасность](#безопасность)

## Quick start (5 минут)

Минимум — один бесплатный ключ от Groq. Регистрация на console.groq.com → API Keys → Create. Дальше:

```bash
git clone https://github.com/<your-fork>/llmgate && cd llmgate
uv sync

cp .env.example .env
# открой .env и впиши:  GROQ_API_KEY=gsk_...
cp config.yaml.example config.yaml

uv run neurogate
# запустится на http://127.0.0.1:8765
# в консоли увидишь: какие провайдеры активны, какие ключи можно ещё добавить
```

Тест — обычный OpenAI-совместимый запрос:

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"скажи привет"}]}'
```

Хочешь больше моделей и квоты — добавь ещё ключей в `.env` (Gemini, Cerebras, OpenRouter, …). Гайд по получению — [docs/providers-setup.md](docs/providers-setup.md). Без ключа провайдер автоматически пропускается; стартап-репорт подскажет какие env-vars выставить, чтобы оживить нужные цепочки.

Dashboard со статистикой по чейнам и провайдерам — `http://127.0.0.1:8765/dashboard`.

Клиент шлёт обычный `POST /v1/chat/completions` с одним из 16 `model`-значений:
- **Базовые fallback-цепочки**: `chat` (русский chit-chat, default), `code` (frontier reasoning + код), `latency` (минимум wall-clock), `quality` (максимальный AA-индекс), `chat_en` (английский), `unlimited` (без RPD-капа), `reasoning_quality` / `reasoning_deep` (thinking-модели), `paid` (Claude Opus 4.7).
- **Специальные режимы**: `web` (веб-поиск через Gemini `google_search` / OpenRouter `:online`), `image` (vision: принимает `image_url`), `moa` (Mixture of Agents: fan-out к 25 моделям + aggregator-синтез), `sc` (Self-Consistency: N сэмплов одной модели), `debate` (Multi-Agent Debate: N моделей × R раундов inter-agent revision), `deep_search` (research-агент: plan→search→synthesize→critique).
- **Авто-роутинг**: `auto` (имя настраивается через `NEUROGATE_VIRTUAL_MODEL`) — heuristic intent detection по последнему user-сообщению. `image_url` → `image`, веб-маркеры/URL → `web`, код-маркеры (```/`def `/`Traceback`/`.py`) → `code`, длинный prompt с reasoning-маркерами → `reasoning_quality`, иначе → `default_chain`. Это **regex-эвристика, не ML-классификатор**; для критичных кейсов выбирай чейн явно. `paid`/`moa`/`sc`/`debate`/`deep_search` авто-роутер **никогда** не выбирает (требуют opt-in). Graceful degradation: если в твоём `config.yaml` нет нужной целевой цепочки (например, не описаны `image`/`web`/`code`) — `auto` тихо падает в `default_chain`. Так же, как и любой неизвестный/не указанный `model` — он мапится на `default_chain` (см. шапку [config.yaml.example](config.yaml.example): `"chat" / "auto" / не задано → chat`).

Всего до 32+ моделей у 10 провайдеров. При 429 / 5xx / quota / empty-response роутер переходит к следующему. Для клиента всё выглядит как один стабильный OpenAI endpoint.

## Что это даёт

- **~20-25k сообщений/сутки** на verified hard caps, до **~110k** с NVIDIA-теоретикой, **~60M токенов/сутки** по явным TPD — суммарно со всех провайдеров (RPM/RPD-лимиты по каждому провайдеру проставлены в [config.yaml.example](config.yaml.example))
- **Фактически токены не упираются в 60M**: цифра собрана только из провайдеров, которые отдают TPD-хедеры (Mistral, Gemini, Groq, Cerebras, GitHub, Cloudflare, OpenRouter). Три провайдера — **Mistral (RPM-only), Z.AI (concurrency-only), NVIDIA (RPM-only, без RPD)** — TPD не декларируют, т.е. сверху они ограничены только пропускной способностью и временем. В практическом смысле, если долбить их 24/7 на длинном контексте, потолок чейна становится **условно безлимитным** по токенам — реальный ceiling зависит только от того, как долго эти три провайдера держат заявленные RPM без throttling.
- **Drop-in замена OpenAI API**: любой SDK (Python, JS, curl, ChatGPT-обёртки) просто меняет `base_url` — код не трогается
- **Надёжность**: если Gemini режет квоту без предупреждения, а OpenRouter flaky — прокси молча переходит на Groq/Cerebras/SambaNova
- **Веб-поиск и vision из коробки**: `model: "web"` — актуальные данные через Gemini `google_search` / OpenRouter `:online`; `model: "image"` — vision-capable провайдеры для изображений (`image_url` во входе)
- **Продвинутые ансамбли**: `moa` (25 моделей параллельно + aggregator), `sc` (N сэмплов одной модели + majority synth), `deep_search` (multi-step research с плановщиком и критиком)
- **Локально по умолчанию** (127.0.0.1, без auth), но готов к деплою на VPS с Bearer-токеном

## Field notes — наблюдения с практики

Делаем ранний доступ — это не бенчмарк, а заметки о том, как работают конкретные цепочки и провайдеры на наших задачах. Если у тебя картина другая — кидай в issues, обновим.

- **`chat` chain** — стабильна. Фоллбэк-логика отрабатывает молча, на пользовательской стороне не заметно когда провайдер падает.
- **`image_gen`** — рабочая, без сюрпризов. RU-модели (`Kandinsky`, `YandexART`) удерживают фотореализм лучше FLUX когда в промпте есть «digital painting»/«oil painting»: меньше «пластика», больше живой текстуры. FLUX отлично на абстрактном/иллюстративном.
- **`code` chain** — средне. Иногда отваливается по timeout / quota — фоллбэк-следующий-провайдер срабатывает, но ответ скачет по качеству от запроса к запросу. Стабильнее работает если ключей побольше (Cerebras, SambaNova, Groq, OpenRouter), хуже — на одном-двух.
- **Русский язык** — `Yandex Alice` (через kind `yandex_foundation`) даёт хороший русский, рекомендуется для RU-сценариев. **GigaChat** на наших промптах слабее — обходят даже отдельные китайские модели (Qwen, GLM). При возможности — Алиса первой, GigaChat в фоллбэк-хвост.
- **Скорость отклика** — NVIDIA NIM медленный (TTFB заметный), уместен в `unlimited` где скорость не критична. Cerebras и Groq — моментальные. SambaNova — посередине.
- **`web` chain** — Gemini с `google_search` качественнее OpenRouter `:online` на актуальных запросах. OpenRouter — хороший фоллбэк когда у Gemini свежая квота закончилась.

## Без ключей — что работает

Если запустить без `.env` или с пустыми ключами — сервер всё равно стартует, но в обрезанном режиме:

- **Работают (без ключей):** `translation` (LibreTranslate, MyMemory), `tts` (Edge TTS), `image_gen` (AIhorde anonymous), плюс OVHcloud-эндпоинт. Это ~6-7 провайдеров с публичным/анонимным доступом.
- **Не работают:** `chat`, `code`, `quality`, `image`, `web` и большинство остальных — они нуждаются хотя бы в одном LLM-ключе. Эти цепочки автоматически выбрасываются на старте, в логах будет понятный список «set X env var to enable».
- **Стартап-репорт** на консоли покажет: сколько провайдеров активно, сколько пропущено, и какие ровно env-vars надо добавить чтобы оживить нужные цепочки.

Минимум для полноценного `chat`/`code` — один из: `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY`. Лучше два-три (фоллбэк работает).

## Зачем это нужно

Каждый free-тир по отдельности капризный:
- **Gemini** режет квоту без предупреждения и без причины
- **OpenRouter free** делит 50 RPD на всех пользователей планеты
- **Groq** даёт жирные 1000 RPD, но только на конкретную 70B-модель
- **Cerebras/SambaNova** без суточного потолка, но узкий выбор моделей

По отдельности ни один не годится для бота 24/7 — ручного ресетить придётся каждые пару часов. Вместе через llmgate — получается коммерчески-надёжный endpoint за $0/месяц.

**Типичные use-cases**:
- Персональный Telegram/Discord-бот с auto-reply
- Локальные эксперименты с LLM без привязки к Anthropic/OpenAI
- Backup-endpoint для проектов, где OpenAI-биллинг — блокер
- Sandbox для сравнения моделей на одной и той же задаче

## Чем отличается от других

Похожих проектов хватает, но у каждого своя ниша:

| Проект | Чем отличается от llmgate |
|---|---|
| **[LiteLLM](https://github.com/BerriAI/litellm)** (15K⭐) | Тяжёлый Python-framework с 100+ провайдеров, load balancing, бюджеты, Postgres. Мощно для команд — overkill для одного юзера. |
| **[OpenRouter](https://openrouter.ai)** | Коммерческий сервис-агрегатор: ты платишь им, они маршрутизируют. У llmgate — self-hosted за $0. |
| **[one-api / new-api](https://github.com/songquanpeng/one-api)** | Китайский OpenAI-gateway с web-UI для SaaS-бизнесов. 80% фич для продажи API-ключей, не для себя. |
| **[Portkey](https://portkey.ai)** | Коммерческий gateway с observability и guardrails. |

**Ниша llmgate**:
- **Заточен под free-тиры** (а не под балансировку платных API между командами)
- **Компактный** — ~1000 строк Python против 50k+ у LiteLLM
- **15 готовых цепочек** (chat / code / latency / quality / chat_en / unlimited / image / web / reasoning_quality / reasoning_deep / paid / moa / sc / debate / deep_search) из 32+ моделей в [config.yaml.example](config.yaml.example) — не абстрактный фреймворк, а рецепт «скопировал и работает»
- **Ансамбли и research-агент встроены** — MoA/SC/debate/deep_search идут из коробки, а не строятся поверх LangChain/LangGraph
- **RPM/RPD-капы по каждому провайдеру** проставлены прямо в [config.yaml.example](config.yaml.example) — у аналогов их обычно надо вычитывать из доков

Если нужен production-gateway для команды — бери LiteLLM. Если нужен личный бесплатный endpoint, который просто работает, — это llmgate.

## Поддерживаются из коробки

| kind | endpoint | особенность |
|---|---|---|
| `gemini` | native SDK | 1M контекст |
| `groq` | `api.groq.com` | очень быстрый (300-1000+ т/с) |
| `cerebras` | `api.cerebras.ai` | 1400+ т/с на 235B |
| `sambanova` | `api.sambanova.ai` | нет суточного лимита |
| `nvidia` | `integrate.api.nvidia.com` | нет суточного лимита |
| `zai` | `api.z.ai` | permanent free, privacy-ok |
| `openrouter` | `openrouter.ai` | агрегатор |
| `cloudflare` | Workers AI | edge |
| `github` | GitHub Models | PAT-auth |
| `gigachat` | SberDevices GigaChat | Basic→OAuth, Russian Trusted Root CA |
| `openai` | любой OpenAI-compat | твой base_url |

15 готовых цепочек с 32+ проверенными моделями лежат в [config.yaml.example](config.yaml.example). Полный разбор ниже в [§ Цепочки](#цепочки).

### Контроль расхода (платные/grant-провайдеры)

Большинство провайдеров — permanent free, но у пары есть биллинг или grant-квота, которую полезно мониторить:

- **GigaChat (Сбер)** — Freemium 1М токенов/30 дней, после — billed. Проекты, лицевой счёт и расход: [developers.sber.ru/studio](https://developers.sber.ru/studio/)
- **Yandex Cloud** (YandexGPT/YandexART/Translate, AI Studio Search/Vision/OCR) — расход, гранты и бюджет-алерты: [center.yandex.cloud/billing/accounts](https://center.yandex.cloud/billing/accounts/). На текущем аккаунте активны два гранта: **4 000 ₽ до 22.06.2026** (все сервисы, кроме GPU Compute / Marketplace / Support / Postbox) и **6 000 ₽ до 22.10.2026** (включая Yandex AI Studio + Search/Vision/OCR).
- **OpenRouter Opus** (`paid` chain) — отдельный ключ `OPENROUTER_PAID_API_KEY`, остаток баланса виден в OR dashboard.

## Проверка и примеры

Health и каталог моделей:

```bash
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/models
curl -s http://127.0.0.1:8765/metrics  # Prometheus-формат
```

Отправить сообщение в конкретную цепочку (любой OpenAI-клиент подойдёт — значения `chat` / `code` / `latency` / `quality` / `auto` выбирают цепочку):

```bash
# русский chat (default)
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role":"user","content":"Привет, как дела?"}],
    "temperature": 0.7,
    "max_tokens": 200
  }' | python3 -m json.tool

# code / reasoning
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "code",
    "messages": [{"role":"user","content":"Докажи теорему о 4 красках"}]
  }' | python3 -m json.tool
```

## Цепочки

Все 15 `model`-значений и что они делают. Имя цепочки передаётся в поле `model` запроса — остальное llmgate выбирает сам.

### Базовые fallback-цепочки

Простые цепочки: **пробуют провайдеров по порядку до первого успеха**, 429/5xx переходят на следующего. Ответ приходит от первого сработавшего.

| `model` | Что делает | Первые в цепочке |
|---|---|---|
| `chat` (default) | Русский chit-chat, RU-floor по качеству | `nvidia:qwen3.5-397b` (RU 92) → `cerebras:qwen-3-235b` → `sambanova:deepseek-v3.2` |
| `chat_en` | Английский chit-chat, AA Intelligence Index desc | `gemini:flash-latest` → `sambanova:deepseek-v3.2` → `openrouter:nemotron-3-super-free` |
| `code` | Reasoning + код, композит SWE-bench/LiveCodeBench | `gemini:flash-latest` → `nvidia:qwen3.5-397b` → `sambanova:deepseek-v3.2` |
| `latency` | Минимум wall-clock (медиана Total time) | `cerebras:llama3.1-8b` (421ms) → `groq:llama-3.1-8b` (668ms) → `groq:gpt-oss-20b` (714ms) |
| `quality` | Максимум AA Intelligence Index v4.0 | `gemini:flash-latest` (AA 46) → `sambanova:deepseek-v3.2` (AA 46) → `nvidia:qwen3.5-397b` (AA 40) |
| `unlimited` | Только провайдеры без жёсткого RPD-капа (для high-volume) | `sambanova:*` → `nvidia:*` → `zai:*` |
| `reasoning_quality` | 9 thinking-моделей, порядок по AA | `gemini:flash-latest` → `openrouter:nemotron-3-super-free` → `groq:gpt-oss-120b` |
| `reasoning_deep` | Те же, но по глубине thinking (reasoning_tokens desc) | `gemini:2.5-flash` → `groq:qwen3-32b` → `zai:glm-4.5-flash` |
| `paid` | Единственная платная (для тестов — **отдельный ключ `OPENROUTER_PAID_API_KEY`**) | `openrouter:claude-opus-4.7` |

### Специальные цепочки

Эти не являются простым fallback — они делают что-то отличное от «перебирай провайдеров».

#### `web` — актуальные данные через веб-поиск

Цепочка активирует native-веб-поиск у провайдера: Gemini вызывает `google_search`-tool server-side (бесплатно), OpenRouter `:online`-провайдеры (elephant-alpha, nemotron-3-super, glm-4.5-air) используют Exa-поиск на своей стороне (через `OPENROUTER_PAID_API_KEY`). Клиенту ничего дополнительно делать не нужно — просто `model: "web"`.

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

#### `image` — vision (распознавание картинок)

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

Поддерживаются URL (http/https), data-URI (`data:image/png;base64,...`) работают у части провайдеров — Gemini и GitHub Models принимают, Groq и Cerebras только URL. Первые в цепочке: `sambanova:llama-4-maverick` (детальный, узнаёт водяные знаки), `gemini:flash-lite-latest` (быстрый + точный), `github:gpt-4.1-mini` (средней детальности).

#### `moa` — Mixture of Agents (ансамбль моделей)

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

#### `sc` — Self-Consistency (N сэмплов одной модели)

Паттерн Wang et al., 2022: берём N сэмплов от **одной** модели (дефолт 5) с высокой `temperature=1.0`, потом aggregator консолидирует разные линии рассуждения. Дешевле MoA (5-6× вызовов vs 26×), помогает когда нужно разнообразие рассуждений, а не разнообразие архитектур. Особенно полезно на math/verifiable задачах.

```bash
# 5 сэмплов gemini:flash-latest (первый в цепочке sc) → aggregator синтезирует
curl -s 'http://127.0.0.1:8765/v1/chat/completions?samples=5' \
  -H "Content-Type: application/json" \
  -d '{"model":"sc","messages":[{"role":"user","content":"Сколько будет 17 × 24? Покажи вычисления."}]}'
```

**Query-параметры**: `?samples=N` (2-20, default 5), `?aggregator=<chain>` (default `reasoning_quality`).

**Response extras**: поле `sc` с `samples[]` (список сэмплов: `{sample_index, provider, text, latency_ms, completion_tokens, temperature, error}`), `base_provider`, `aggregator_chain`, `sample_count`, `sample_success`.

#### `debate` — Multi-Agent Debate (N моделей × R раундов inter-agent revision)

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

**Query-параметры**: `?agents=N` (2-6, default 3), `?rounds=R` (1-4, default 2; R=1 ≈ MoA без обмена), `?aggregator=<chain>` (default `reasoning_quality`). Агенты берутся как первые N eligible провайдеров из цепочки `debate` — порядок задаёт состав, идеал — разные семьи моделей (Gemini / DeepSeek / Qwen / GPT-OSS) для диверсификации.

**Response extras**: поле `debate` с:
- `agents[]` — имена выбранных N моделей
- `rounds` — фактическое число пройденных раундов
- `transcript[round][agent]` — каждый агент-раунд: `{round, agent_index, provider, text, latency_ms, completion_tokens, error}`
- `aggregator_chain`, `aggregator_provider`, `agent_count`, `final_round_success`

**Ограничения**: `stream=true` и `tools` не поддерживаются; latency = R × медианный latency агента + aggregator (для 3×2 ≈ 15-30s); ~N×R + 1 LLM-вызов на запрос (8 вызовов для 3×2 + agg vs 26 у MoA — дешевле и обычно качественнее на factual-задачах).

#### `deep_search` — research-агент (plan → search → synthesize → critique)

Multi-step pipeline уровня Perplexity Deep Research:
1. **Planner** раскладывает вопрос на 2-4 подзапроса (JSON).
2. **Searcher** × N (parallel) — каждый подзапрос через `web`-цепочку.
3. **Synthesizer** собирает все findings + извлечённые URL-ы в структурированный markdown с numbered-цитатами `[1][2]`.
4. **Critic** проверяет draft на пробелы → если найдены, ещё 1 round `search + synth`.

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deep_search",
    "messages": [{"role":"user","content":"Какие фреймворки актуальны для построения ReAct-агентов в 2026?"}],
    "max_tokens": 2000
  }' | python3 -m json.tool

# настройка: 3 подзапроса, без critic-раунда
curl -s 'http://127.0.0.1:8765/v1/chat/completions?max_subq=3&rounds=0' \
  -H "Content-Type: application/json" \
  -d '{"model":"deep_search","messages":[{"role":"user","content":"..."}]}'
```

**Query-параметры**: `?max_subq=N` (1-6, default 4), `?rounds=R` (0-2 дополнительных critic-раунда, default 1), `?planner=<chain>`, `?searcher=<chain>` (default `web`), `?synth=<chain>`, `?critic=<chain>`.

**Response extras**: поле `deep_search` с:
- `subquestions[]` — что planner решил искать
- `sources[]` — `{id, url, subq}` все уникальные URL, найденные в searcher-ответах (дедупликация, порядковые id для `[N]`-цитат в тексте)
- `trace[]` — каждый шаг с `{step, provider, latency_ms, ...step-specific}`
- `iterations` — сколько раз прошли synthesize (1 без critic, 2 если critic нашёл пробелы)

**Типичные цифры**: 15-40s wall-clock, 5-8 LLM-вызовов (без critic-round) или 8-12 (с ним), 3-10 URL в финальных источниках.

### Когда какую цепочку использовать

| Задача | Рекомендуемая |
|---|---|
| Быстрый чат на русском | `chat` |
| Быстрый чат на английском | `chat_en` |
| Код / алгоритмы / рефакторинг | `code` |
| Нужно быстро, качество второстепенно | `latency` |
| Нужно умно, скорость вторична | `quality` или `reasoning_quality` |
| Актуальные данные из веба | `web` |
| Картинки, OCR, описание фото | `image` |
| Критичный ответ, нужен консенсус многих моделей | `moa` |
| Factual-вопрос, где модели поодиночке уверенно врут | `debate` |
| Математика / пошаговые рассуждения | `sc` (с math-вопросом) или `reasoning_deep` |
| Ресёрч-вопрос с несколькими подтемами и цитатами | `deep_search` |
| Сложная задача, нужен фронтир-класс | `paid` (тратит Opus-ключ) |

### `sfx` — звуки и эмбиент для игр (text→audio)

Отдельный endpoint `POST /v1/audio/sfx` для генерации звуковых эффектов и эмбиента (не музыки и не голоса). Цепочка `sfx` зовёт публичный HuggingFace Space `artificialguybr/Stable-Audio-Open-Zero` через `gradio_client` — без HF Inference API квоты. Bottleneck — **HF Zero-GPU**: каждый вызов резервирует ~120с GPU, анонимная квота ~3-5 минут/день/IP, то есть **1-2 вызова/сутки**. С `HF_TOKEN` (free) квота per-account и чуть выше; с HF PRO (~$9/мес) — 25× больше, реально юзабельно. Альтернативные Space (AudioLDM2, Tango2) держал в цепочке раньше — оба сейчас 503 на стороне HF, можно вернуть в [config.yaml](config.yaml) когда оживут.

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/sfx \
  -H "Content-Type: application/json" \
  -d '{"prompt":"wind howling through pine forest at night, no music","duration":10}' \
  --output ambient.wav
```

Body: `{prompt: str, duration?: 1-30s, model?: "sfx" | "hfspace:..."}`. Возвращает raw audio (`audio/wav` по умолчанию). Headers ответа: `X-Neurogate-Provider`, `X-Neurogate-Chain`, `X-Neurogate-Duration`. Для синтеза голоса используйте `POST /v1/audio/speech` (цепочка `tts`); для распознавания речи — `POST /v1/audio/transcriptions` (цепочка `audio`).

## Использование из OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="sk-noop",  # или токен из NEUROGATE_API_TOKEN
)

r = client.chat.completions.create(
    model="chat",   # или "code" / "latency" / "quality"
    messages=[{"role": "user", "content": "hi"}],
)
print(r.choices[0].message.content)
print("served by:", r.model)  # имя фактического провайдера
```

## Конфиг

`config.yaml` — список провайдеров в порядке фоллбэка. Каждый провайдер:

```yaml
providers:
  - name: groq:llama-3.3-70b       # любое имя для логов
    kind: groq                      # см. таблицу выше
    model: llama-3.3-70b-versatile  # имя модели на стороне провайдера
    api_key_env: GROQ_API_KEY       # откуда брать ключ
    rpm: 30                          # опционально: локальный кап запросов/мин
    rpd: 1000                        # опционально: локальный кап запросов/сут
    # extra_body: { thinking: { type: disabled } }   # для Z.ai и подобных
    # extra_headers: { X-My-Header: value }
    # timeout: 60
```

Меняешь порядок → меняешь приоритет фоллбэка. Добавить провайдера — дописать блок.

### Локальный rate-tracking

Если у провайдера задан `rpm` и/или `rpd`, прокси ведёт собственный счётчик попыток в SQLite (`rate_events`) и при достижении потолка **скипает провайдера без сетевого вызова** — переходит к следующему в цепочке. Окна rolling: последние 60 секунд для `rpm`, последние 24 часа для `rpd`. Журнал переживает рестарт: если процесс упал после того как Gemini отдал 249/250 запросов, после подъёма прокси всё ещё помнит и не будет долбиться в тот же 429.

Поля опциональны — если не указать, проверки нет и провайдер вызывается пока сам не отдаст 429 (который уже ловится фоллбэком). Разумные значения уже проставлены в [config.yaml.example](config.yaml.example).

## Авторизация и как пользоваться

Механизм: если `NEUROGATE_API_TOKEN` задан, все защищённые эндпоинты требуют `Authorization: Bearer <token>`. Если пуст — авторизации нет (только для локальной разработки на `127.0.0.1`).

### Эндпоинты

| Путь | Метод | Auth | Назначение |
|---|---|---|---|
| `/health` | GET | нет | минимальный liveness (`{"ok":true}`) |
| `/dashboard` | GET | нет (HTML) | веб-UI: цепочки, лимиты, исходы, латентность |
| `/metrics` | GET | **Bearer** | Prometheus метрики (RPS, latency, исходы по провайдерам) |
| `/v1/health` | GET | **Bearer** | detailed: default chain + список цепочек |
| `/v1/models` | GET | **Bearer** | список виртуальных моделей и провайдеров |
| `/v1/stats` | GET | **Bearer** | usage по провайдерам (rpd/rpm + счётчики) |
| `/v1/metrics.json` | GET | **Bearer** | JSON-версия Prometheus счётчиков + средняя латентность |
| `/v1/chat/completions` | POST | **Bearer** | основной chat-эндпоинт (OpenAI-совместимый) |
| `/v1/chains/edit` | GET | **Bearer** | снимок цепочек + пул провайдеров для редактора |
| `/v1/chains` | PUT | **Bearer** | переписать `chains:`/`default_chain:` в config.yaml + hot reload |

`model` в запросе: `chat` (default), `code`, `latency`, `quality`, `chat_en`, `unlimited`, `image`, `web`, `reasoning_quality`, `reasoning_deep`, `paid`, `moa`, `sc`, `debate`, `deep_search`, `auto`, либо конкретный провайдер из `/v1/models`, либо ad-hoc `kind:model_id` (см. ниже).

### Ad-hoc модели (любая `kind:model_id`)

Если в `model` приходит строка с разделителем `:` или `/`, которой нет ни среди цепочек, ни среди провайдеров — например `openai:gpt-5-foo`, `groq/llama-99b` — llmgate парсит префикс как `kind`, забирает дефолтный `base_url` из `PROVIDER_KIND_DEFAULTS` и серверный env-ключ, собирает `OpenAICompatProvider` на лету и шлёт запрос. Никакого UI/storage для ключа, никаких хедеров от клиента — единственный источник правды это env на сервере.

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

### Drag-n-drop редактор цепочек (дашборд)

В дашборде, во вью «Провайдеры», есть кнопка «✎ редактировать цепочки». Включает редактор с drag-n-drop через SortableJS:

- Слева — список цепочек: переключение, переименовать, удалить, отметить как default (★).
- В центре — провайдеры активной цепочки: перетаскивание для reorder, ✕ для удаления.
- Справа — пул всех провайдеров из `cfg.all_providers` (с фильтром): drag в любую цепочку (clone — один и тот же провайдер может быть в нескольких цепочках, как и в YAML).
- Кнопка «сохранить» → `PUT /v1/chains`. Сервер валидирует, переписывает `chains:` и `default_chain:` в `config.yaml` (комментарии и секция `providers:` сохраняются — replace через regex), пишет бэкап `config.yaml.bak`, делает hot-reload роутера в памяти. Следующий запрос идёт по новой схеме без рестарта процесса.

Безопасность: эндпоинт под тем же `NEUROGATE_API_TOKEN`. На проде `config.yaml` синкается через rsync (см. [deploy.md](deploy.md)) — если редактируешь конфиг прямо на сервере через дашборд, имей в виду что следующий деплой с локального файла перезапишет правки. Для постоянных изменений правь локальный `config.yaml` и деплой как обычно.

### OpenAI-совместимость

`/v1/chat/completions` и `/v1/models` — drop-in замена OpenAI API. Подходит любой OpenAI-клиент (Python SDK, LangChain, Vercel AI SDK, Cursor, curl). Ставишь `base_url = http://<server>:8765/v1` и `api_key = <NEUROGATE_API_TOKEN>` — работает как есть.

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

**Что поддерживается:** system/user/assistant/tool roles, tool calling (function calling), multi-turn, `n`, `stop`, `temperature`, `top_p`, penalty-поля, `tools`/`tool_choice`/`parallel_tool_calls`, **streaming** (`stream: true` — SSE как в OpenAI, включая Gemini через конвертацию native-чанков), **image input** (OpenAI vision-формат с `{"type":"image_url",...}` — работает на цепочке `image`), **web-search** (цепочка `web` — native tool у Gemini / сервер-сайд у OpenRouter `:online`).

**Нюанс стриминга**: fallback между провайдерами работает только **до первого чанка**. Как только upstream начал стримить — мы закоммичены; mid-stream ошибка придёт клиенту как обрыв соединения, а не как прозрачное переподключение (проще и совпадает с поведением самого OpenAI). Цепочки `moa`/`sc`/`debate`/`deep_search` **не поддерживают streaming** — они агрегирующие, собирают результаты целиком перед синтезом.

**Что НЕ поддерживается:**
- Logprobs — не пересылаются.
- `tools` в цепочках `moa`/`sc`/`debate`/`deep_search` — пока нет (вернётся 400 с пояснением).

### curl

```bash
BASE=http://<your-server>:8765
TOKEN=<your-llmgate-token>

# chat (русский, default)
curl -sS "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role":"user","content":"Привет"}],
    "max_tokens": 200
  }'

# code / reasoning
curl -sS "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"code","messages":[{"role":"user","content":"Докажи теорему о 4 красках"}]}'

# список моделей
curl -sS -H "Authorization: Bearer $TOKEN" "$BASE/v1/models"

# usage по провайдерам
curl -sS -H "Authorization: Bearer $TOKEN" "$BASE/v1/stats"

# Prometheus метрики
curl -sS -H "Authorization: Bearer $TOKEN" "$BASE/metrics"
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<your-server>:8765/v1",
    api_key="<your-llmgate-token>",
)

r = client.chat.completions.create(
    model="chat",
    messages=[{"role": "user", "content": "hi"}],
)
print(r.choices[0].message.content)
print("served by:", r.model)
```

### Безопасность

Трафик по HTTP, без TLS — токен и содержимое запросов идут в открытом виде. `/health` и `/metrics` открыты анонимно: `/metrics` показывает, какие провайдеры срабатывали и с какой латентностью, но не утекает содержимое запросов или ключи. Для чувствительного трафика — завернуть через nginx с TLS на поддомене.

## Деплой

Через systemd + uv (без Docker — меньше RAM и проще дебаг на маленьком VPS).

```bash
# на сервере
git clone <repo> /opt/llmgate && cd /opt/llmgate
uv sync
cp config.yaml.example config.yaml  # отредактируй
cp .env.example .env                # заполни ключи (гайд: docs/providers-setup.md)
sudo tee /etc/systemd/system/llmgate.service <<'EOF'
[Unit]
Description=llmgate
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/llmgate
EnvironmentFile=/opt/llmgate/.env
ExecStart=/opt/llmgate/.venv/bin/neurogate
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now llmgate
```

Обновление: `git pull && uv sync && sudo systemctl restart llmgate`.
Логи: `journalctl -u llmgate -f`.

## Как работает фоллбэк

При каждом запросе роутер пробует провайдеров по порядку. Переход к следующему происходит при:

- HTTP 429, 500, 502, 503, 504
- `resource_exhausted`, `quota`, `rate limit`
- `unavailable`, `overloaded`, `timeout`
- `empty response` (модель вернула пустоту)
- `model_not_found`, `decommissioned` (модель убрали у провайдера)
- `error` поле в JSON-теле с HTTP 200 (OpenRouter так делает при исчерпанной квоте)

Любая другая ошибка (400 bad request, невалидный ключ) — НЕ фоллбэкается, возвращается клиенту как 502.

## Наблюдаемость

- **`/metrics`** — Prometheus-endpoint (без auth, стандарт для скраперов). Метрики:
  - `neurogate_requests_total{provider,outcome}` — счётчик запросов по исходам: `success` / `rate_limit` / `server_error` / `empty` / `timeout` / `decommissioned` / `rate_capped` / `other`
  - `neurogate_request_duration_seconds{provider,outcome}` — гистограмма латентности (Grafana считает p50/p95 через `histogram_quantile()`)
- **`/v1/stats`** (auth) — текущая загрузка rate-tracker'а: сколько запросов к каждому провайдеру за последние 1m/24h и их caps
- **SQLite** `stats.db` — хранит `rate_events` для фичи Per-provider rate tracking (не метрики, а реальные скипы провайдеров при достижении cap'а)

Метрики Prometheus — **в памяти**, сбрасываются при рестарте. Долгоживущее хранение — задача TSDB скрапера.

## Tool calling (function calling)

llmgate пробрасывает OpenAI-совместимые поля `tools` и `tool_choice` в upstream и возвращает `tool_calls` / `finish_reason: "tool_calls"` как есть. Можно использовать llmgate как backend для агентов, работающих по протоколу OpenAI function calling.

**Как это работает в прокси**:
- `tools` и `tool_choice` из запроса пересылаются в upstream без изменений.
- Роутер **автоматически пропускает** провайдеров без поддержки tool calling, если `tools` переданы — сейчас это только **Gemini native** (его SDK требует отдельной конвертации схемы, она пока не реализована). Все OpenAI-совместимые провайдеры (Groq, Cerebras, SambaNova, NVIDIA, OpenRouter, Z.ai, GitHub Models, Mistral, Cloudflare, собственный OpenAI) поддерживают tools.
- Клиентские сообщения с `role: "assistant"` + `tool_calls` и `role: "tool"` + `tool_call_id` проксируются as-is.
- При `finish_reason: "tool_calls"` поле `content` в ответе будет `null` (как в OpenAI API).
- При любой retryable-ошибке (429/5xx/quota/timeout) прокси переходит к следующему провайдеру, сохраняя тот же список tools.

### Пример (Python, OpenAI SDK)

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

### Интеграционный брифинг для AI-агента

Памятка для модели / агента, которого подключают к llmgate:

- **Endpoint**: `POST <base_url>/v1/chat/completions`. Протокол полностью совместим с OpenAI Chat Completions API.
- **Имя модели в запросе**: одно из 15 значений — см. [§ Цепочки](#цепочки) для подробного разбора. Короче: `"chat"` (default), `"code"`, `"latency"`, `"quality"`, `"chat_en"`, `"unlimited"`, `"image"` (vision), `"web"` (веб-поиск), `"reasoning_quality"` / `"reasoning_deep"` (thinking), `"paid"` (Claude Opus), `"moa"` (ансамбль 25 моделей), `"sc"` (N сэмплов одной модели), `"debate"` (N моделей × R раундов inter-agent revision), `"deep_search"` (research-агент), `"auto"` (= default). Имя фактически отработавшего провайдера — в `model` и `provider` ответа, имя выбранной цепочки — в `chain`. Для ансамблей/research дополнительные метаданные лежат в полях `moa` / `sc` / `debate` / `deep_search` соответственно.
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
- **Качество моделей**: цепочка (см. [config.yaml.example](config.yaml.example)) рассчитана на то, что первые в списке — самые «умные» свободные. Если агенту нужна стабильная модель — зафиксируй её на клиенте, не через `auto`.

## Лицензия и разработка

- **Лицензия**: [MIT](LICENSE).
- **Тесты не публикуются.** Внутренние ru-bench / latency-бенчи живут в приватной директории и поддерживаются под мою конфигурацию провайдеров — выкладывать их в публичную репу нет смысла. CI ([ci.yml](.github/workflows/ci.yml)) делает только smoke-load конфигов; функциональные регрессии я ловлю руками. Если шлёшь PR — приложи короткий repro, я прогоню локально.
- **Issues / PR** приветствуются: новые провайдеры, баги в фоллбэке, опечатки в [docs/providers-setup.md](docs/providers-setup.md).

## Ограничения текущей версии

- Gemini native игнорируется при запросах с `tools` (нет конвертера схемы; fallback автоматический)
- Нет кэша (prompt caching) — каждый запрос отправляется заново
- Fallback между провайдерами работает только **до первого чанка стрима** (после — обрыв соединения)
- Цепочки `moa` / `sc` / `debate` / `deep_search` не поддерживают `stream=true` и `tools`
- `deep_search` опирается на `web`-цепочку (OpenRouter `:online` + Gemini `google_search`); отдельный bare-search backend (Brave / DuckDuckGo) пока не подключён

## Roadmap

- [x] Streaming через SSE
- [x] Tool calling passthrough (OpenAI-compat провайдеры)
- [ ] Gemini native tool calling (конвертация схемы)
- [x] Отдельные цепочки для разных задач — `chat` / `code` / `latency` / `quality` / `chat_en` / `unlimited` / `reasoning_quality` / `reasoning_deep` / `paid`
- [x] `image`-цепочка (vision, OpenAI-совместимый `image_url`)
- [x] `web`-цепочка (native web-search у Gemini + OpenRouter `:online`)
- [x] `moa` — Mixture of Agents (25 моделей параллельно + aggregator с self-bias защитой)
- [x] `sc` — Self-Consistency (N сэмплов одной модели + aggregator)
- [x] `deep_search` — research-агент (plan → search → synthesize → critique, с iterations)
- [x] `debate` — Multi-Agent Debate (N моделей × R раундов с inter-agent revision)
- [ ] Отдельный search backend (Brave / DuckDuckGo) для raw URL-ов до reader-шага
- [x] Per-provider rate tracking (локальный SQLite с RPM/RPD)
- [x] Prometheus `/metrics`

## Зачем бесплатные модели?

**Сила — в объёме и скорости, а не в IQ.**

Llama 3.3 70B, Qwen 2.5 72B, Gemini Flash — это модели уровня GPT-4-класса 2023 года. По бенчмаркам они отстают от Claude 4.7 / GPT-5 на 15-25%, но:

- **Бесплатны** и в сумме дают ~60M токенов/сутки
- **Безумно быстрые** на Groq/Cerebras/SambaNova (300-1400 т/с против ~80 у Claude)
- **Надёжны в формате** — JSON/tool calls не хуже frontier-моделей

### Насколько они умны vs Claude и арена

> _Данные на апрель 2026 — Elo на LMArena и расклад по моделям быстро дрейфуют, через пару месяцев цифры уедут и модели обновятся._

**TL;DR.** Reasoning-чейн llmgate — примерно уровень Claude Sonnet 4 / GPT-4.1, местами подбирается к Opus 4 2024-го года, но заметно отстаёт от Opus 4.5–4.7 и GPT-5.x. В LMArena верхушка этих моделей сидит в районе **1400–1425 Elo**, фронтир — **1480–1500+**. Разница ≈60–100 Elo = ~60-65% побед у фронтира в прямом сравнении.

Если переводить в практический язык: ты получаешь **~90% качества Sonnet 4.6** и **~70–75% качества Opus 4.6/4.7** — за $0/мес. Для массовых и интерактивных задач этого хватает с запасом; для агентных цепочек и прод-кода — идёшь к фронтиру напрямую (см. ниже).

**Для сверки с индексом умности (`quality` 0-100 в дашборде).** Шкала откалибрована по **Artificial Analysis Intelligence Index v4.0** (10 бенчмарков — GDPval-AA, τ²-Bench, Terminal-Bench Hard, SciCode, AA-LCR, AA-Omniscience, IFBench, HLE, GPQA Diamond, CritPt; шкала 0–57 у AA). Формула: `quality = round(AA × 100 / 57)`, где 57 = фронтир (Gemini 3.1 Pro, GPT-5.4, Claude Opus 4.7). Текущий топ бесплатного чейна: `gemini:flash-latest` = 81 (AA 46), `nvidia:qwen3.5-397b` = 79 (AA 45), `openrouter:nemotron-3-super-free` = 63 (AA 36), `groq:gpt-oss-120b` = 58 (AA 33), `sambanova:deepseek-v3.2` = 56 (AA 32). Frontier для сверки: **Claude Sonnet 4.6 ≈ 91** (AA 52), **Claude Opus 4.6 ≈ 93** (AA 53), **Claude Opus 4.7 / GPT-5.4 / Gemini 3.1 Pro = 100** (AA 57) — тройка делит первое место. Между топом бесплатного чейна (81) и фронтиром (100) — разрыв ~19 пунктов.

### По задачам — llmgate vs платное API

**Паритет (90–100% от Claude Sonnet):**
- Русский chit-chat (DeepSeek V3.2 силён в ru, Qwen3-235B тоже)
- Стандартная генерация кода (Python, JS, простые Vue-компоненты)
- RAG-суммаризация, парафраз, редактура
- Function calling / structured output
- Обычные переводы, классификация, извлечение данных

**Middle-tier (70–85%):**
- Сложные рефакторинги в большой кодовой базе (Claude Opus «понимает всё», стек llmgate — «понимает, но дроппает детали»)
- Длинный контекст >100k (у DeepSeek V3.2 — 163k, но качество на длинном контексте у Opus 4.6 заметно выше)
- Многоходовое агентное планирование

**Заметно слабее (50–70%):**
- Агентный coding уровня SWE-bench Verified — здесь Opus 4.6 доминирует, чейн не догоняет
- Сложное math / olympiad reasoning (DeepSeek R1 тянул бы, но в стеке V3.2-exp без thinking)
- Инструктируемость на тонких нюансах длинных system-промптов
- Мультимодальность (стек чисто текстовый)

### Где они реально выигрывают

**Массовые задачи с простой единицей работы:**
- Классификация / тегирование / модерация (миллионы записей)
- Извлечение структуры из текста (email → JSON с полями)
- Нормализация данных, дедупликация, чистка
- Перевод, рерайт, стилистика
- Суммаризация статей / логов / тикетов

**Первый проход в двухэтапном пайплайне:**
- Отфильтровал 10k кандидатов дешёвыми → на Opus ушло 200 сложных
- Черновик от llmgate → frontier-модель на финальную правку

**Интерактив, где скорость важнее глубины:**
- Telegram-бот (пользователь не заметит разницу с GPT-5 на «привет, как дела?»)
- Автодополнение комментариев, подсказки по тексту
- Голосовой ассистент (latency решает)

**Синтетика и эксперименты:**
- Генерация датасетов (reasoning traces, Q&A пары)
- Аугментация для обучения маленьких моделей
- A/B-тесты промптов на тысячах примеров

### Где они сливают — не берись

- **Агенты с длинной цепочкой шагов** — разваливаются через 4-5 tool calls
- **Архитектурные решения в коде** — не видят системный контекст
- **Математика / логика олимпиадного уровня**
- **Длинный контекст с глубоким пониманием** (>50k токенов, где надо связывать факты из начала и конца)
- **Прод-код без ревью** — для этого Opus/Sonnet

### Правило большого пальца

> Задача сводится к «прочитай кусок → выдай структурированный ответ» — бери llmgate.
> Задача требует «подумать и спланировать» — бери Claude/GPT напрямую.