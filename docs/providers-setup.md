# Как получить API-ключи провайдеров

Пошаговая инструкция для всех провайдеров из [config.yaml.example](../config.yaml.example) — где зарегистрироваться, где взять ключ, какая переменная в [.env.example](../.env.example), важные нюансы.

**Минимум для старта**: достаточно 1-2 ключей — llmgate пропустит позиции, у которых ключ пустой, и пойдёт дальше по цепочке. Стартап-репорт на консоли подскажет, какие именно env-vars не заполнены и какие провайдеры это включит. Начать проще всего с **Groq** (самый быстрый onboarding) + **Gemini** (самая жирная квота на 1 ключ).

## Free quota at a glance

Сводная таблица — сколько просят, что дают на бесплатном тире. Цифры отражают потолки на момент cmu confg-среза, провайдеры периодически режут/щедрят — `/v1/models` и dashboard покажут реальную картину.

| Провайдер | ENV-переменная | RPM | RPD / day | Monthly | Phone | Card | Особенность / quirk |
|---|---|---|---|---|---|---|---|
| **Groq** | `GROQ_API_KEY` | 30 | 1k–14.4k | — | нет | нет | Самый быстрый onboarding, скорость 300-1000+ т/с. Дубликаты ловят. |
| **Gemini** | `GEMINI_API_KEY` | 10 | 250-1500 | — | нет | нет | Per-project квота: можно завести несколько GCP-проектов = ×N. Vision + tools. |
| **Cerebras** | `CEREBRAS_API_KEY` | 5–30 | без RPD | — | да | нет | Узкий выбор моделей, но без суточного потолка. |
| **SambaNova** | `SAMBANOVA_API_KEY` | без RPM-cap | 20/model | — | нет | нет | RPD на модель, 4 модели = 80 RPD на ключ. |
| **NVIDIA NIM** | `NVIDIA_API_KEY` | 40 | без RPD | — | нет | нет | TTFB заметный (медленный). Использовать в `unlimited`/фоллбэк. |
| **Z.ai** | `ZAI_API_KEY` | без cap | concurrency-only | — | нет | нет | Permanent free на GLM-4.5-Flash. China upstream — учти при чувствительном трафике. |
| **OpenRouter (free)** | `OPENROUTER_API_KEY` | 20 | 50 (1000 с $10) | — | нет | нет* | $10 депозит → 1000 RPD на `:free`. Без депозита — 50 RPD общих. |
| **Cloudflare Workers AI** | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | varies | 300 (Llama 70B) | — | нет | нет | Edge inference, низкая латентность. |
| **GitHub Models** | `GITHUB_MODELS_TOKEN` | varies | 50–150 | — | нет | нет | gpt-5-mini в free tier. Auth через GitHub PAT. |
| **Mistral** | `MISTRAL_API_KEY` | ~4 | — | — | да | нет | Experiment plan. Без RPD-cap, но узкий RPM. |
| **HuggingFace** | `HF_TOKEN` | varies | varies | ~$0.10 | нет | нет | Общий кредит на paid sub-providers. PRO ($9/mo) = $2 кредит. |
| **GigaChat (Sber)** | `GIGACHAT_AUTH_KEY` | varies | varies | 1M tok / 30 дней | да (РФ) | нет | Freemium 1M токенов на 30 дней. |
| **FreeTheAi** | `FREETHEAI_API_KEY` | 10 | varies | — | нет (Discord) | нет | **Требует daily Discord `/checkin`** — иначе 403. |
| **DeepSeek** | `DEEPSEEK_API_KEY` | без cap | — | 5M tok signup | да | нет | После signup-кредитов — PAYG. Off-peak 50–75% скидка. |
| **Alibaba DashScope Intl** | `DASHSCOPE_API_KEY` | varies | — | 1M tok / model × 90д | да | да | Singapore region, 1M context на флагманах. |
| **Cohere** | `COHERE_API_KEY` | 20 | — | 1000 calls | нет | нет | **Non-commercial only** на free tier (ToS). |
| **OVHcloud** | — (anonymous) | 2 | — | — | нет | нет | Без ключа. EU-инфра. |
| **LibreTranslate** | — (anonymous) | varies | — | — | нет | нет | Public mirrors. Перевод. |
| **MyMemory** | `MYMEMORY_CONTACT_EMAIL` (опц) | — | 5k chars/IP | (50k chars с email) | нет | нет | Перевод. Email увеличивает квоту. |
| **Edge TTS** | — (anonymous) | — | — | — | нет | нет | Microsoft TTS, безлимит. |
| **AIhorde** | — (anonymous) | — | — | — | нет | нет | Image gen, community-distributed inference. |

`*` OpenRouter не требует карту для базового free tier, но $10 однократный депозит поднимает лимиты в 20 раз.

### Quick suggestion: какие 2-3 ключа добавить первыми

- **Самый быстрый старт:** Groq (без верификаций) → даёт мгновенный отклик и приличную квоту.
- **Долгий контекст / vision:** Gemini.
- **Бэкап без RPD-капа:** Cerebras (одно подтверждение телефона) или SambaNova.
- **Русский:** Yandex Alice (`yandex_foundation` kind) — лучше для RU чем GigaChat по нашим наблюдениям.

## Провайдеры

### 1. Gemini (Google)

- **Регистрация**: любой Google-аккаунт
- **Получить ключ**: https://aistudio.google.com/apikey → *Create API key*
- **ENV**: `GEMINI_API_KEY`
- **Нюанс**: квота считается **per-project**, не per-key. Если завести 2-3 GCP-проекта на том же Google-аккаунте и сгенерить по ключу в каждом, получишь 2-3× квоту легально (TOS это разрешает — дубликаты запрещены только на уровне аккаунтов). Дополнительные ключи можно прописать как `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` и завести дубликаты провайдеров в `config.yaml` с разными `api_key_env`.

### 2. Groq

- **Регистрация**: https://console.groq.com (Google / GitHub / email)
- **Получить ключ**: *API Keys* → *Create API Key*
- **ENV**: `GROQ_API_KEY`
- **Нюанс**: самый быстрый провайдер в цепочке (300-1000+ токенов/сек). Free tier даёт 1000 RPD на `llama-3.3-70b-versatile` и 14400 RPD на `llama-3.1-8b-instant`. Дубликаты аккаунтов Groq ловит и банит оба — **не регистрируй второй аккаунт** с того же IP/телефона/карты.

### 3. Cerebras

- **Регистрация**: https://cloud.cerebras.ai (требует подтверждение телефона)
- **Получить ключ**: *API Keys* в dashboard
- **ENV**: `CEREBRAS_API_KEY`
- **Нюанс**: нет суточного лимита, но free tier сейчас узкий — реально работают только `qwen-3-235b-a22b-instruct-2507` (5 RPM) и `llama3.1-8b` (30 RPM). Более крупные модели (llama-3.3-70b, gpt-oss-120b) депрекейтнуты или paid-only.

### 4. SambaNova

- **Регистрация**: https://cloud.sambanova.ai (Google / email)
- **Получить ключ**: *API Keys* в dashboard
- **ENV**: `SAMBANOVA_API_KEY`
- **Нюанс**: 20 RPD **на каждую модель**, а не на аккаунт — 4 SambaNova-позиции в цепочке дают суммарно 80 RPD на один ключ. Лимит читается из заголовка `x-ratelimit-limit-requests-day` в ответе.

### 5. NVIDIA NIM

- **Регистрация**: https://build.nvidia.com (email, developer-программа)
- **Получить ключ**: профиль справа вверху → *Get API Key* → выдают ключ вида `nvapi-...`
- **ENV**: `NVIDIA_API_KEY`
- **Нюанс**: формально **только для research/dev/test** — production TOS требует AI Enterprise license (~$4500/GPU/год). Для личного бота не критично. Нет суточного лимита, только RPM (40 на большинстве моделей).

### 6. Z.ai (GLM)

- **Регистрация**: https://z.ai (international) или https://open.bigmodel.cn (Китай — для аккаунтов из РФ/СНГ)
- **Получить ключ**: *API Keys* в dashboard
- **ENV**: `ZAI_API_KEY`
- **Нюанс**: `glm-4.5-flash` — **permanent free**, без RPD-лимита. В конфиге стоит `extra_body: { thinking: { type: disabled } }` — это принудительно отключает reasoning-режим (по умолчанию модель тратит токены на chain-of-thought, что режет скорость). Z.ai официально заявляет «no training on API data» — ок для чувствительного трафика.

### 7. OpenRouter

- **Регистрация**: https://openrouter.ai (Google / GitHub / MetaMask)
- **Получить ключ**: https://openrouter.ai/keys → *Create Key*
- **ENV**: `OPENROUTER_API_KEY`
- **Нюанс**: `:free` модели делят глобальный пул 50 RPD на всех пользователей планеты — часто упирается в лимит к вечеру UTC. **Однократный депозит $10** поднимает RPD → 1000 на все `:free` модели (сами $10 не сгорают, лежат на балансе). Это единственное действие с 20× ROI на лимиты — см. «Платные расширения».

### 8. Cloudflare Workers AI

- **Регистрация**: https://dash.cloudflare.com (email)
- **Получить ключ**: *My Profile* (правый верх) → *API Tokens* → *Create Token* → шаблон *Workers AI* (или custom: permission `Account → Workers AI → Read` + `Edit`)
- **Получить Account ID**: на главной странице dashboard в правом сайдбаре — строка *Account ID*
- **ENV**: `CLOUDFLARE_API_KEY` (токен) + `CLOUDFLARE_ACCOUNT_ID`
- **Нюанс**: единственный провайдер, которому нужны **две** переменные. Account ID нельзя передать как env — Cloudflare вшивает его в URL запроса. 300 RPD на Llama 70B fp8-fast.

### 9. GitHub Models

- **Регистрация**: любой GitHub-аккаунт
- **Получить токен**: https://github.com/settings/personal-access-tokens → *Generate new token (fine-grained)* → Account permissions → **Models: Read-only**
- Классический PAT (`settings/tokens`) тоже работает, но fine-grained безопаснее — скоуп только на Models, не трогает репы.
- **ENV**: `GITHUB_MODELS_TOKEN`
- **Нюанс**: endpoint `https://models.github.ai/inference`, жёсткий потолок 8K токенов на запрос (input+output). Лимиты 50-150 RPD по моделям, считаются на токен.

### 10. Mistral La Plateforme

- **Регистрация**: https://console.mistral.ai (email + SMS-верификация)
- **Получить ключ**: *API Keys*
- **План**: выбрать **Experiment** (бесплатный) — даёт доступ к `mistral-large-latest` и остальным флагманам
- **ENV**: `MISTRAL_API_KEY`
- **Нюанс**: **обязательно зайти** в *Admin* → *Privacy* и **выключить опцию использования данных для обучения** — по умолчанию она включена. Без этого твои запросы уходят на training. Лимит ~4 RPM (заголовок `x-ratelimit-limit-req-minute`), не 60 как в старых доках.

### 11. HuggingFace Inference Providers Router

- **Регистрация**: любой HuggingFace-аккаунт https://huggingface.co
- **Получить токен**: *Settings* → *Access Tokens* → *Create new token* → тип **Read**
- **ENV**: `HF_TOKEN`
- **Нюанс**: HF проксирует запросы к paid sub-providers (Novita, Cerebras, Together, …) через единый OpenAI-совместимый endpoint. Free tier даёт **~$0.10/мес общего кредита** на все sub-providers — этого хватает на эксперименты, не на постоянный трафик. **PRO-подписка ($9/мес)** даёт $2 кредита/мес. Модель указывается как `<repo>:<sub-provider>`, например `meta-llama/Llama-3.3-70B-Instruct:novita`.

### 12. GigaChat (SberDevices)

- **Регистрация**: https://developers.sber.ru/studio (требует подтверждённый аккаунт Сбера, телефон РФ)
- **Получить ключ**: *Личное пространство* → *Мой GigaChat API* → *Получить новый ключ* → выдают `Authorization key` (это уже base64 от `client_id:client_secret`, готовый для `Basic`-авторизации)
- **ENV**: `GIGACHAT_AUTH_KEY` (значение строки *Authorization key* как есть)
- **Scope**: `GIGACHAT_API_PERS` — **Freemium** для физлиц, 1 млн токенов на 30 дней (включая Pro/Max-модели). После исчерпания квоты — billed.
- **Нюанс 1**: GigaChat отдаёт сертификат, подписанный *Russian Trusted Root CA* — стандартные системные truststore (на macOS/Linux вне РФ) его не знают. В конфиге для `kind: gigachat` стоит `verify_ssl: false`. Для прода — добавить Russian Trusted Sub CA в системный truststore.
- **Нюанс 2**: авторизация двухступенчатая — провайдер обменивает `Authorization key` на короткоживущий OAuth access-token и кэширует его. Этот flow реализован внутри `kind: gigachat`, ничего вручную делать не нужно.
- **Биллинг и расход**: https://developers.sber.ru/studio (раздел *Проекты* → *Лицевой счёт*).

### 13. FreeTheAi

- **Регистрация**: только через Discord. Зайти на сервер → https://discord.gg/secrets → команда `/signup` в канале #api-keys (https://discord.com/channels/1461555807731585158/1473159205048553705) → бот пришлёт ключ вида `sta_…` в DM. Если потерял — `/resetkey`.
- **ENV**: `FREETHEAI_API_KEY`
- **Лимиты**: 10 RPM + **1 concurrent** на ключ — последовательные вызовы, никакого parallel.
- **Каталог**: ~16 252 алиасов. Префиксы: `cat/*` (gpt-5/claude/gemini frontier через прокси), `bbg/*` (DeepSeek/GLM/Kimi/MiniMax reasoning), `bbl/*` (Gemini/GPT non-reasoning), `or/*` (OpenRouter passthrough), `vhr/*` (image-gen scrap из Vheer), `img/gpt-image-2` (image gen+edit b64), `fth/*` (16k случайных HF моделей — почти всегда мусор). Не все алиасы реально работают — `glm/glm-5.1` (#1 в leaderboard) и `cat/gemini-3-1-pro` возвращают 400/timeout. Перед использованием — пробовать.
- **⚠️ Главный нюанс — DAILY /checkin**: ключ блокируется **каждые 24 часа**. Чтобы разлочить — заходишь в Discord, в тот же канал #api-keys пишешь `/checkin`, бот возвращает «ключ активен на 24h». Без этого все запросы получают `HTTP 403 daily_checkin_required`.
- **Автоматизировать /checkin НЕЛЬЗЯ**:
  1. Discord ToS запрещает self-bot'ы (использование своего user-токена для скриптов) — ловят и банят аккаунт.
  2. Обычные боты Discord не могут вызывать slash-команды других ботов (by design платформы).
  3. Browser-automation поверх web-Discord — та же категория self-bot, ban guarantied.
  4. FreeTheAi специально поставили /checkin как защиту от abuse — обходить = бан и от них тоже.
- **Когда использовать**:
  - Ad-hoc-запросы для тестов (`model: "freetheai:cat/gpt-5"`) — 24h cooldown терпим, если не забыл сделать /checkin утром.
  - Не для продовых chains (без daily-checkin сервис ляжет).
- **Bait-and-switch**: проверка identity показала, что `cat/claude-4-6-sonnet` отвечает «built by Tiny» — это НЕ Claude. `cat/gpt-5` идентифицируется как OpenAI, но без гарантий. `bbg/deepseek-v4-pro` выдаёт gibberish. Перед интеграцией каждой модели в логику — прогнать identity-test.
- **Image-gen**: `vhr/*` модели возвращают URL'ы на `access.vheer.com` — это прокси на бесплатный Vheer.com сервис, файлы хранятся у них. `img/gpt-image-2` отдаёт b64 (~650KB на запрос) — единственная модель, поддерживающая edits.

### 14. DeepSeek (direct API)

**Пошагово (получить ключ):**

1. Открой https://platform.deepseek.com → *Sign Up* (Google или email)
2. Phone verify — российские номера принимает (МТС/Мегафон/Билайн/Tele2). Если по какой-то причине не принимает твой номер — попробуй через Google-аккаунт.
3. После входа: левое меню → *API Keys* → *Create new API key* → дать любое имя → скопировать ключ (вид `sk-xxxxx...`). Ключ показывается **один раз** — потом только regenerate.
4. В `.env` локально и на проде:
   ```
   DEEPSEEK_API_KEY=sk-xxxxx...
   ```

**Что получаешь сразу (бесплатно):**

- **5M токенов signup credits** + 7-14 дней без RPM-лимитов. Этого хватит на много экспериментов.
- Модели: `deepseek-chat` (V3.2) и `deepseek-reasoner` (R1/R2).

**Когда signup credits закончатся (опционально):**

- $2 топ-ап ≈ 170 ₽ → месяцы PAYG-работы (`deepseek-chat` $0.28/$0.42 за 1M, `deepseek-reasoner` сопоставимо).
- Off-peak 16:30–00:30 UTC — скидка 50% на V3, 75% на R1.
- Prefix caching: cache hit в 4–10× дешевле miss. Структурируй промпты так, чтобы стабильный префикс (system + RAG) шёл первым, юзер-ввод — последним.
- ⚠️ **Не включай auto-recharge** — карта будет молча докидывать каждый раз когда баланс падает.
- ⚠️ Российские карты Visa/MC топ-ап **не примет**. Варианты: карта дружественной страны (Казахстан/Армения/Грузия), Wise/Revolut, виртуалка от иностранного fintech.

**Подводные камни:**

- **Privacy**: данные хранятся в Китае, политика допускает использование для обучения → **не использовать с PII**. Если через шлюз идут чувствительные данные клиентов — либо удали DeepSeek из chain, либо ставь его последним позади cohere/openrouter/sambanova.
- **Из России без VPN**: иногда нестабильные 5xx-ответы. Держи fallback в той же chain ниже — он сработает автоматически.

**kind**: `deepseek`. Provider в config.yaml — простая запись с `model: deepseek-chat` или `model: deepseek-reasoner`. Уже добавлено в `chat`/`code`/`quality`/`latency` chains в `config.yaml.example`.

### 15. Alibaba DashScope International (Qwen)

**⚠️ Самый муторный из четырёх. Делай утром, на это уйдёт 20-30 минут.**

**Пошагово (получить ключ):**

1. Открой https://accountclient.alibabacloud.com/intl → *Sign Up*
2. **Регион — обязательно Singapore.** US/HK/EU/China версии free quota НЕ дают. Если на этапе onboarding спрашивает регион — выбирай Singapore. Если уже зарегистрировался в другом регионе — придётся завести отдельный аккаунт.
3. Phone verify (российские номера принимает).
4. **Карта обязательна** — для Real-name Verification. С карты ничего не списывается, но без привязки free quota не активируется.
   - ⚠️ Российские карты Visa/MC **не пройдут**.
   - Варианты:
     - Карта дружественной страны (Казахстан / Армения / Грузия)
     - Wise / Revolut если есть
     - Виртуальная карта от иностранного fintech (Pyypl, Genome, etc.)
5. После Real-name Verification → открыть https://bailian.console.alibabacloud.com (Model Studio, регион Singapore)
6. Левое меню → *API-KEY* → *Create* → скопировать ключ (вид `sk-xxxxx...`). Ключ показывается полностью один раз.
7. В `.env` локально и на проде:
   ```
   DASHSCOPE_API_KEY=sk-xxxxx...
   ```

**Что получаешь сразу (бесплатно):**

- **1M токенов на каждую модель** × **90 дней** с момента активации. Десятки моделей в каталоге → десятки миллионов токенов суммарно.
- Флагманы: `qwen3.6-plus` (топ-3 на OpenRouter trafficу, 1M context, всегда-вкл CoT), `qwen3.6-max-preview`, `qwen3-coder-480b-a35b-instruct`, `qwen3-next-80b-a3b-instruct/thinking`, `qwen3-vl-plus`.

**После 90 дней**: paid-only. Поставь напоминание `/schedule` за неделю до истечения — посмотрим использование и решим продлять или уйти на DashScope-paid.

**Подводные камни:**

- **Регион Singapore ОБЯЗАТЕЛЬНО**. Endpoint в config.py уже забит: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. Если у тебя аккаунт в другом регионе — ключ будет работать, но free quota = 0.
- **Privacy**: Alibaba явно заявляет «will never use your data for model training», данные в Сингапуре → **можно использовать с PII** в отличие от китайского endpoint DeepSeek.
- **Real-name Verification может зависнуть на сутки** — в первый раз. Если статус «pending» больше 24ч — открыть тикет в support.

**kind**: `dashscope`. Provider в config.yaml — простая запись с `model: qwen3.6-plus` или другой моделью. Уже добавлено в `chat`/`code`/`quality`/`latency` chains в `config.yaml.example`.

### 16. Cohere (Command-A / c4ai-aya — для русского)

- **Регистрация**: https://dashboard.cohere.com (Google / GitHub / email)
- **Получить ключ**: *API Keys* → *Generate Trial Key*
- **ENV**: `COHERE_API_KEY`
- **Free tier**: 20 RPM на модель, 1000 calls/месяц (≈33 RPD safe-floor).
- **Модели**: `c4ai-aya-expanse-32b/8b` — multilingual, **сильный русский**. `c4ai-aya-vision-32b/8b` — vision. `command-a-03-2025` — флагман общего назначения. `command-a-reasoning-08-2025` — reasoning. `command-a-translate-08-2025` — translation.
- **ToS**: free tier — **non-commercial only**. Для open-source / личных проектов ок. Если кто-то форкнет коммерчески — нарушение. Помечай в README.
- **Особенность реализации**: Cohere v2 /chat — собственный response shape (`{message: {content: [{type:'text', text:...}]}}`), не OpenAI-compat. Streaming эмулируется одним SSE chunk'ом из полного ответа (Cohere v2 SSE имеет свой формат, дешевле fake-stream чем переводить).
- **kind**: `cohere_chat` (для chat) или `cohere` (для translate-only). Эмбеддинги отдельно — `cohere_embed`.

### 17. OVHcloud AI Endpoints (anonymous, без ключа)

- **Регистрация**: НЕ ТРЕБУЕТСЯ. Endpoint анонимный.
- **ENV**: НЕТ. В config.yaml запись `kind: ovhcloud` без `api_key_env`.
- **Endpoint**: `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` — OpenAI-совместимый.
- **Лимит**: 2 RPM **на IP** на модель. Без ключа — прозрачный rate-limit, при превышении 429.
- **Каталог**: 40+ open-моделей, EU-host (Франция/Германия). Llama 3.x, Mistral, Qwen, Mixtral, gpt-oss-120b, и т.д.
- **Когда использовать**: исключительно как **deep fallback / last-resort** в конце chain — «лучше отдать что-то, чем 503». В основной chain ставить нет смысла из-за 2 RPM на IP.
- **Реализация**: `OpenAICompatProvider` пропускает `Authorization` header при пустом `api_key` (см. providers.py — изменение из этого PR).
- **kind**: `ovhcloud`.

## Платные расширения (опционально)

Ниже — одноразовые или recurring платные действия с непропорционально большим профитом на лимиты.

### OpenRouter — $10 депозит → 20× квоты

**Что делает**: однократный пополнение на $10 поднимает 50 RPD → **1000 RPD** на все `:free` модели (DeepSeek R1/V3.2, Llama 4 Maverick/Scout, Qwen3 235B, GLM-5 `:free`, Kimi K2 `:free`, MiniMax M2 `:free`, Qwen3-Coder 480B `:free`).

**Как**: *Settings* → *Credits* → пополнить любую сумму ≥ $10. Сами $10 остаются на балансе и тратятся только если ты явно вызовешь платную модель. Для `:free` — это просто unlock лимита.

**После депозита**: в `config.yaml` обновить `rpd: 50` → `rpd: 1000` на всех позициях с `kind: openrouter`.

### xAI Grok — $25 signup + $150/мес recurring

- **Регистрация**: https://console.x.ai (phone verify)
- **Signup**: $25 кредитов на 30 дней
- **Data-sharing program**: opt-in на sharing → $150/мес recurring (обновляется пока включено, требует минимум $5 spend за период)
- **Зачем**: Grok 4.1 Fast — уникальный **2M context window**, никто больше такого не даёт
- **В конфиг**: нужен `kind: xai` провайдер (OpenAI-compat endpoint `https://api.x.ai/v1`). **Пока не реализован** в llmgate.

### Alibaba Qwen Singapore — 1M токенов × десятки моделей на 90 дней

- **Регистрация**: https://accountclient.alibabacloud.com/intl (требует телефон и карту, +966 работает)
- **Регион критичен**: Singapore. Endpoint `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. US/HK/EU версии free quota не дают.
- **Квота**: 1M токенов **на каждую модель** на 90 дней → десятки моделей × 1M = десятки миллионов токенов суммарно. Флагманы: `qwen3-max`, `qwen-plus`, `qwen-flash`, `qwen3-coder-plus`, `qwen3.5-plus`.
- **Privacy**: официально «will never use your data for model training», данные в Сингапуре.
- **После 90 дней**: paid-only.
- **В конфиг**: `kind: dashscope` — нативно поддерживается, см. секцию #15 выше для деталей.

### DeepSeek direct — 5M signup credits + off-peak 50-75%

- **Регистрация**: https://platform.deepseek.com (phone verify)
- **Signup**: 5M токенов + 7-14 дней без rate limits
- **Off-peak**: 16:30-00:30 UTC — **50% скидка на V3, 75% на R1**. Годится для batch-задач, не для realtime-чата.
- **Privacy-блокер**: данные хранятся в Китае, политика допускает использование для обучения → **не использовать с PII**.
- **В конфиг**: `kind: deepseek` — нативно поддерживается, см. секцию #14 выше для деталей.

### Anthropic Claude direct — $5 signup

- **Регистрация**: https://console.anthropic.com (phone verify)
- **Signup**: $5 кредитов на 30 дней после phone verify
- **Зачем**: единственный способ добавить frontier Claude Sonnet/Opus в цепочку без Bedrock/Databricks
- **Нюанс**: Anthropic API не OpenAI-compat — нужен отдельный `kind: anthropic`. В llmgate **пока не реализован**.

### OpenAI direct — единоразово $5 ради бесплатной модерации (text + image)

**TL;DR — зачем подключать OpenAI вообще:** ради **бесплатной модерации**. Текстовая (`text-moderation-latest`) и multimodal text+image (`omni-moderation-latest`) у OpenAI **бесплатны навсегда** и качественнее всех альтернатив. Платные модели ($0.05–$10 per 1M токенов) — опционально, если иногда понадобится высокое качество в `paid` chain. **На бесплатные chat / embed / image-gen цепочки OpenAI подключать не нужно** — у Groq / Gemini / Z.ai лимиты щедрее и без депозита.

**Регистрация:**

- https://platform.openai.com (Google / Microsoft / email)
- https://platform.openai.com/api-keys → *Create new secret key*
- **ENV**: `OPENAI_API_KEY` (для бесплатной модерации) и опционально `OPENAI_API_KEY_PAID` (для paid-моделей; можно тот же ключ — разделение нужно лишь чтобы случайно не сжечь baseline-баланс)

**Критично — почему без $5 ничего не работает:**

На Free tier (без привязанной карты) OpenAI режет **все** запросы через `429 Too Many Requests` с `type: invalid_request_error` — **включая бесплатную модерацию**. Это не rate-limit моды, это аккаунт-уровневая блокировка. В ответе нет ни одного `x-ratelimit-*` хедера — это диагностический признак, что блок именно account-level, а не лимит модели. Лечится только депозитом.

**Депозит — единоразово $5:**

1. https://platform.openai.com/settings/organization/billing/overview → *Add payment details*
2. Введи **`5`** в *Initial credit purchase* (минимум).
3. ⚠️ **Снять галочку *Automatically add credits when your balance runs low***. По умолчанию она включена. Если оставить — карта будет молча докидывать $5 каждый раз когда баланс упадёт. Это главная ловушка для тех, кто хотел «только бесплатно».
4. Подтвердить (банк может попросить 3DS).
5. За 1-5 минут Tier поднимается `Free` → `Tier 1`, лимиты на moderation вырастают с 3 RPM до ~500 RPM.

**Где следить за лимитами и расходом:**

- **Rate limits per model** (RPM/RPD/TPM): https://platform.openai.com/settings/organization/limits — таблица с текущим тиром и лимитами по каждой модели. Полезно проверить после оплаты, что Tier действительно поднялся до 1.
- **Usage (расход токенов и запросов):** https://platform.openai.com/usage — графики по дням/моделям, что именно сжигало баланс. Здесь же видно, что moderation идёт нулевой строкой.
- **Billing & balance:** https://platform.openai.com/settings/organization/billing/overview — текущий остаток депозита, история платежей, настройки auto-recharge.

**Стратегия использования $5:**

- **Основной use-case (бесплатно):** модерация — `omni-moderation-latest` (text+image, 13 категорий) и `text-moderation-latest` (legacy text-only, 7 категорий) **не списываются с баланса**. $5 просто лежат бессрочно.
- **Опционально (платно):** редкие вызовы `paid` chain через `gpt-5-mini` для случаев когда бесплатные провайдеры упёрлись или не справляются с качеством. При умеренном трафике $5 хватает на месяцы.
- **Не подключать:** chat / embed / image_gen цепочки — у бесплатных провайдеров (Groq / Gemini / Z.ai / Cohere / Voyage / Cloudflare / Pollinations) лимиты на порядки больше.

**Бесплатно после депозита:**

| Модель | Что делает | Цена |
|---|---|---|
| `omni-moderation-latest` | Text + image moderation, 13 категорий, native scores | $0 / $0 |
| `text-moderation-latest` | Legacy text-only, 7 категорий | $0 / $0 |

В llmgate реализованы как `kind: openai_moderation` в [config.py:118](../src/llmgate/config.py#L118), провайдер [`OpenAIModerationProvider`](../src/llmgate/providers.py#L3205). Обе уже в дефолтных цепочках `moderation` и `moderation_image`.

**Платные модели — опциональный subset для `paid` chain (используя `OPENAI_API_KEY_PAID`):**

| Модель | Цена in/out (per 1M) | Когда нужна |
|---|---|---|
| `gpt-5-mini` | $0.25 / $2.00 | Основной paid-fallback. Уже добавлен во вторую позицию `paid` после Claude Opus. |
| `gpt-5-nano` | $0.05 / $0.40 | Если нужен ультра-дешёвый paid-fallback. На простых запросах хватает. |
| `gpt-4.1-mini` | $0.40 / $1.60 | **1M контекст** — для длинных документов / RAG. Подключать только если упираешься в context window других моделей. |
| `whisper-1` | $0.006/мин | STT-fallback после Groq Whisper. Только если делаешь транскрипцию серьёзных объёмов. |

**Не брать:** `o3` (есть `o4-mini` в 4× дешевле), `dall-e-3` (устарел против `gpt-image-1`), `gpt-4o` / `gpt-4o-mini` (`gpt-5-mini`/`nano` дешевле и не хуже), `text-embedding-3-small/large` (Voyage даёт 200M tok permanent free на каждую модель — OpenAI embeddings подключать смысла мало), `gpt-image-1` (Kandinsky / YandexART лучше на фотореализме, см. отдельную заметку).

**В конфиге:** OpenAI-нативный endpoint `https://api.openai.com/v1` — `kind: openai` для chat (`gpt-5-mini` и др.), `kind: openai_moderation` для модерации. Конкретные провайдеры см. в [config.yaml](../config.yaml) (`openai:moderation`, `openai:text-moderation`, `openai:gpt-5-mini`).

## После получения ключей

1. Скопировать: `cp .env.example .env`
2. Вписать ключи в `.env` (пустые строки llmgate просто пропустит — провайдеры без ключа исключаются из цепочки)
3. Запустить: `uv run llmgate`
4. Проверить, какие провайдеры реально доступны: `curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/v1/models`

Если провайдер упирается в 429 — llmgate автоматически перейдёт к следующему. Если все упали — вернётся HTTP 502 `upstream_exhausted` (на free-тирах в сумме с 1 ключа это ~20-25k сообщений/сутки, так что до потолка надо постараться).

## Если что-то не работает — как сообщить о баге

Без следующих данных мы не можем отделить баг шлюза от проблемы провайдера / твоего соединения / клиента, и фикс затягивается на дни. Скинь всё одним сообщением:

### Чек-лист (минимум)

- [ ] **Что произошло** — одна-две фразы. Что должно было быть, что получилось вместо.
- [ ] **Скриншот или видео** — обязательно. На скрине должно быть видно: само сообщение об ошибке, время, и (если применимо) URL/endpoint.
- [ ] **Текст ошибки целиком** — копи-паст, не пересказ. Если приложение режет текст — скрин консоли / network tab из DevTools.
- [ ] **Когда произошло** — точное время с часовым поясом (например, `2026-05-09 14:32 МСК`). Это нужно чтобы найти запись в логах.
- [ ] **Что делал до ошибки** — последовательность действий. Например: «открыл Cursor → задал вопрос про код → пошёл стрим, на 3-й секунде оборвался». Или: «отправил длинный документ через Continue → сразу 502». **Воспроизводится повторно или один раз?**
- [ ] **Тип соединения**:
  - Wi-Fi (домашний роутер / офис / публичный)
  - мобильный интернет (4G/5G)
  - смешанно (когда переключался с одного на другое — отметь)
- [ ] **Провайдер/оператор**:
  - для мобильного: МТС / Мегафон / Билайн / Tele2 / Yota / другой
  - для домашнего: Ростелеком / МТС / Билайн / ТТК / Дом.ру / другой
  - если за VPN — **обязательно укажи**, какой VPN и какая страна выхода
- [ ] **Город** (грубо — Москва / СПб / регион / зарубежье). Не для слежки, а потому что некоторые провайдеры блокируют upstream-эндпоинты только в части регионов.
- [ ] **Какой клиент** и его версия:
  - Cursor / Claude Code / Continue / Cline / Aider / curl / собственный скрипт / браузер
  - версия (например, `Cursor 1.4.2`)
- [ ] **Какая модель / chain** была запрошена — посмотри в настройках клиента, поле `model:`. Например: `auto`, `chat`, `deepseek:reasoner`, `paid`.

### Желательно (если можешь)

- **Request ID** — если ошибка вернулась с шлюза (HTTP-ответ), там в headers должен быть `x-request-id`. Скинь его — найдём конкретный запрос в логах за миллисекунды.
- **Скрин Network tab** браузера или вывод `curl -v` — видны заголовки и точный путь.
- **Размер промпта** — короткий вопрос или длинный документ на 50K токенов? Длинный context часто выявляет баги, которых нет на коротком.
- **Stream или non-stream** — обычно стрим по умолчанию, но некоторые клиенты выключают.

### Шаблон сообщения

Скопируй и заполни:

```
Что произошло:
Скриншот/видео:  (приложить файл)
Текст ошибки:
   <вставить текст>
Когда: 2026-05-09  HH:MM МСК
Что делал до:
Воспроизводится: да / нет / иногда
Соединение: Wi-Fi / мобильный
Оператор:
VPN: нет / да (страна: ...)
Город:
Клиент: ... версия ...
Модель/chain:
Request ID:
```

### Почему мы спрашиваем про оператора

llmgate за собой держит 15+ upstream-провайдеров (Groq, Cerebras, SambaNova, Gemini, OpenRouter, DeepSeek, …). Часть из них **блокируется или режется на уровне отдельных российских операторов** — эта блокировка не от шлюза, а от твоего ISP. Признаки: connection reset / timeout / SSL handshake fail на конкретный домен. На разных операторах живут разные подмножества upstream'ов:

- МТС иногда режет HuggingFace и часть Cloudflare-эндпоинтов
- Мегафон блокирует часть Google IP-блоков → Gemini нестабильна
- Ростелеком блокирует часть международного трафика по DPI
- Из мобильного интернета бывает странный SNI-фильтр

Без информации про оператора мы будем час крутить логи в поисках бага в коде, которого нет.
