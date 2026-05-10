# NeuroGate

> ⚠️ **Status: alpha / early access (v0.1).** Проект на активной разработке, API и конфиг могут меняться без особых церемоний до v1.0. Баги ожидаемы. Поломалось — открой [issue](../../issues/new?template=bug_report.md), посмотрим (шаблон отчёта — `docs/bug-report.md`). Звёздочка на репе и репорты помогают приоритетам.

> **English**: free-tier LLM multiplexer — one OpenAI-compatible endpoint on top of 20+ providers (Gemini, Groq, Cerebras, SambaNova, OpenRouter, Cloudflare, GitHub Models, Mistral, NVIDIA, Z.AI, GigaChat, HuggingFace, Yandex, DeepSeek, Cohere, DashScope, FreeTheAi, Edge TTS, AIhorde, …) with automatic fallback on 429/5xx/quota, web search (`model: "web"`), vision (`model: "image"`), image generation (`image_gen`), TTS (`tts`), and ensembles (`moa` / `sc` / `debate` / `deep_search`). Drop-in replacement for the OpenAI API — point your SDK at `http://127.0.0.1:8765/v1` and use one of 30+ chain names as `model`. Self-hosted, $0/month. Config and docs below are in Russian; the code, config keys, and HTTP API are English. License: MIT.

**Бесплатный мультиплексор LLM-провайдеров: один OpenAI-совместимый endpoint поверх 10+ free-тиров с автоматическим фоллбэком, веб-поиском, распознаванием картинок и ансамблями моделей.**

## Содержание

- [Что с этим можно делать](#что-с-этим-можно-делать)
- [Установка — два пути](#установка--два-пути)
  - [Путь 1: попросить AI сделать всё за тебя](#путь-1-попросить-ai-сделать-всё-за-тебя-)
  - [Путь 2: руками](#путь-2-руками-если-хочешь-понять-что-происходит)
- [Что это даёт](#что-это-даёт)
- [Field notes — наблюдения с практики](#field-notes--наблюдения-с-практики)
- [Без ключей — что работает](#без-ключей--что-работает)
- [Поддерживаются из коробки](#поддерживаются-из-коробки)
- [Цепочки](#цепочки)
- [Конфиг](#конфиг)
- [Безопасность](#безопасность)

## Что с этим можно делать

neurogate даёт **OpenAI-совместимый API**, поэтому используй как угодно. На вход — обычный POST на `/v1/chat/completions` (как у OpenAI). Что внутри — куча free-провайдеров с автоматическим фоллбэком. Главные сценарии:

- 🌐 **Свой сайт / приложение** — хочешь подключить ИИ к своему сервису, боту, плагину или внутреннему инструменту бесплатно? Это оно. На любом языке (Python, JS, Go, что угодно) ставишь `base_url` на свой neurogate и `api_key` — свой токен. Если уже использовал OpenAI SDK — тем более: в существующем коде меняется только две строчки.
- 🛠️ **API доступ для вайбкодинга** — Claude Code, Cursor, Codex, OpenCode, Cline и т.д. — любой клиент, к которому есть доступ. В настройках: `base_url = http://your-neurogate:8765/v1`. Платный API заменяется на твой бесплатный.
- 💬 **Чат прямо в браузере без кода** — встроенный dashboard на `/dashboard`: тестовый чат по любой цепочке (включая `moa` и `deep_search`), отдельный таб **Debate** для мультиагентных дебатов, таб **Вызовы** с историей всех API-запросов (какая модель, цепочка, провайдер, токены, латентность), таб **Провайдеры** со статусом каждого + drag-n-drop редактор цепочек с hot-reload (без рестарта сервера).
- 🤖 **Бот для Telegram** — обёртка над OpenAI работает как есть, просто перенаправь её на neurogate.
- 📊 **Скрипты автоматизации** — суммаризация писем, перевод документов, классификация тикетов, генерация постов.
- 🎨 **Генерация картинок через API** — `POST /v1/images/generations` (или `model: "image_gen"` через chat-эндпоинт). FLUX, Kandinsky, SDXL — бесплатно. Подключай к любому приложению, которому нужны картинки.
- 🔊 **Озвучка текста через API** — `POST /v1/audio/speech` (или `model: "tts"`). Edge TTS, безлимит. Для голосовых ботов, читалок, озвучки контента.
- 🎙️ **Распознавание речи через API** — `POST /v1/audio/transcriptions` (Whisper / Gemini). Транскрипция голосовых сообщений, подкастов.
- 🌍 **Перевод через API** — `POST /v1/translate` (или `model: "translation"`). Цепочка идёт через специализированные дешёвые переводчики (LibreTranslate, MyMemory, Yandex Translate, Cohere Aya) и обращается к chat-моделям только в фоллбэке. Дешевле и быстрее, чем гонять перевод через `chat` chain.
- 🔍 **Веб-поиск через API** — `model: "web"` (Gemini google_search + OpenRouter `:online`). Запросы с актуальными данными прямо из своего приложения.
- 🧠 **Reasoning-агенты через API** — встроенные `moa` / `sc` / `debate` / `deep_search`, ничего вручную собирать не надо. Ставишь `model: "moa"` — получаешь ансамбль из 25 моделей, ставишь `deep_search` — research-агент с web-search.

Включай фантазию — у тебя теперь есть бесплатный OpenAI-совместимый API.

## Установка — два пути

> Раздел написан максимально подробно — рассчитан на то, что запустить сможет даже человек, незнакомый с программированием. Если ты разработчик и каждый шаг очевиден — листай к нужной команде, мы всё равно остановились на стандартных `git clone` / `uv sync` / systemd.

Выбирай по уверенности в командной строке. Оба пути закончатся одинаково — рабочий neurogate либо локально, либо на сервере.

### Путь 1: попросить AI сделать всё за тебя

Самый простой. Нужен «терминальный AI» — CLI-инструмент с ИИ-агентом, который сам читает твои файлы и выполняет команды. Подойдёт **любой**, к которому у тебя есть доступ: Claude Code, Cursor, Codex, OpenCode и т.д. Это не веб-чат, а программа в терминале с правами на чтение/запись твоих файлов.

#### Локально (на своём компе)

Открой терминал в любой папке, запусти выбранный AI-агент и скажи примерно так:

> Склонируй репозиторий `https://github.com/basaltcode/neurogate` в папку `~/neurogate`. Установи `uv` если его нет, выполни `uv sync`, скопируй `.env.example` в `.env` и `config.yaml.example` в `config.yaml`. Я добавлю API-ключи в `.env` сам — подскажи когда. После этого запусти сервер через `uv run neurogate` и открой dashboard в браузере.

Когда AI попросит ключи — открой [docs/providers-setup.md](docs/providers-setup.md) (там для каждого провайдера написано где зарегаться, где взять ключ, какая квота бесплатно). Можно начать с одного `GROQ_API_KEY` — этого хватит для проверки. Чем больше ключей — тем стабильнее работает.

#### На сервере (24/7-работа)

**1. Купи VPS** (любой провайдер с Ubuntu 24.04 и SSH):

| Провайдер | Цена | Где IP | Нюанс |
|---|---|---|---|
| **[Hetzner](https://www.hetzner.com/cloud)** CX23 | €6/мес, 8GB RAM | EU (Falkenstein/Helsinki) | Лучший выбор — нейтральный IP, OpenAI/Anthropic не блокируют. |
| **[DigitalOcean](https://www.digitalocean.com)** | $6/мес, 1GB | мировой выбор | Стандарт индустрии. |
| **[Timeweb Cloud](https://timeweb.cloud)**, **[Beget](https://beget.com)** | от ~200₽/мес | РФ | Ок для GigaChat/Yandex; OpenAI/Anthropic будут блокировать по IP — RU-ключи останутся живы, остальные не построятся. |

При покупке выбирай Ubuntu 24.04, добавь свой SSH-ключ (или возьми временный пароль).

**2. Дай AI задачу** (в локальном AI-агенте на твоём компе):

> У меня VPS `<IP>`, root-доступ по SSH (паролем `<пароль>` или своим ключом — посмотри какой у меня настроен в `~/.ssh/`). Разверни на нём `https://github.com/basaltcode/neurogate` в `/opt/neurogate`. Поставь `uv`, выполни `uv sync`, скопируй конфиги из `.example`, помоги мне вписать ключи в `.env` (см. `docs/providers-setup.md`). Сгенерируй случайный `NEUROGATE_API_TOKEN` и поставь `NEUROGATE_HOST=0.0.0.0`. Создай systemd unit `neurogate.service`, открой 22 и 8765 в ufw, запусти. После запуска проверь `http://<IP>:8765/health` и покажи мне URL дашборда.

AI всё сделает. Если что-то не так — он сам спросит или починит.

### Путь 2: руками (если хочешь понять что происходит — или если нет доступа к терминальному ИИ)

#### Локально (на своём компе)

**1. Поставь зависимости:**

```bash
# uv — менеджер Python-зависимостей. Нужен потому что neurogate написан
# на Python и тащит за собой ~30 пакетов; uv их подтягивает в изолированное
# окружение, чтобы они не мешали другому Python-софту на компе. Альтернатива
# — `pip + venv` руками, но uv в 10× быстрее и сам ставит нужный Python.
curl -LsSf https://astral.sh/uv/install.sh | sh

# git — для скачивания кода с GitHub. Если уже есть, пропусти.
# macOS:    brew install git
# Ubuntu:   sudo apt install -y git
# Windows:  https://git-scm.com/download/win
```

**2. Склонируй и установи:**

```bash
git clone https://github.com/basaltcode/neurogate.git neurogate
cd neurogate

# uv sync — читает pyproject.toml, скачивает все нужные библиотеки в .venv/
# (изолированная папка внутри проекта). Делается один раз после клонирования
# и каждый раз после `git pull`, если зависимости изменились.
uv sync
```

**3. Настрой ключи:**

```bash
cp .env.example .env
cp config.yaml.example config.yaml
```

Открой `.env` в любом редакторе и впиши хотя бы один API-ключ от провайдера. Полный гайд — где у каждого провайдера зарегистрироваться, где взять ключ, какая бесплатная квота: **[docs/providers-setup.md](docs/providers-setup.md)**. Самый простой старт — Groq (без верификаций, регистрация за минуту).

`NEUROGATE_API_TOKEN` локально оставь пустым — на `127.0.0.1` никто кроме тебя достучаться не может, защита не нужна. Он становится обязательным, когда сервер выходит наружу (на VPS).

**4. Запусти:**

```bash
uv run neurogate
```

В консоли — баннер со счётчиком активных провайдеров. Открой `http://127.0.0.1:8765/dashboard` — там встроенный чат и метрики.

**5. Используй из своего кода:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="sk-noop")
r = client.chat.completions.create(model="auto", messages=[{"role":"user","content":"привет"}])
print(r.choices[0].message.content)
```

#### На сервере (production)

**1. Купи VPS** (см. таблицу выше). Запиши: IP-адрес, root-пароль или путь к SSH-ключу.

**2. Подключись:**

```bash
ssh root@<server-ip>
```

**3. Установи систему:**

```bash
# git+curl — для скачивания, ufw — простой файрвол
apt update && apt install -y git curl ufw

# uv — менеджер Python-зависимостей (тот же, что и для локальной установки —
# тащит ~30 пакетов в изолированное .venv/ внутри проекта)
curl -LsSf https://astral.sh/uv/install.sh | sh
# подгрузи свежий PATH (там теперь есть путь к uv)
source ~/.bashrc
```

**4. Склонируй и настрой:**

```bash
git clone https://github.com/basaltcode/neurogate.git /opt/neurogate
cd /opt/neurogate

# uv sync — создаёт .venv/ и ставит туда зависимости из pyproject.toml
uv sync

# Шаблоны конфига — копия example в рабочий, который ты будешь редактировать
cp config.yaml.example config.yaml
cp .env.example .env

# В отличие от локалки, здесь токен ОБЯЗАТЕЛЕН — иначе любой кто
# найдёт твой IP сможет жечь твою бесплатную квоту. Генерим случайный:
echo "NEUROGATE_API_TOKEN=neurogate_$(openssl rand -hex 24)" >> .env

# По умолчанию neurogate слушает только 127.0.0.1. На сервере надо
# слушать все интерфейсы, иначе извне до него не достучаться.
echo "NEUROGATE_HOST=0.0.0.0" >> .env

# Открой .env и впиши провайдерские ключи (см. docs/providers-setup.md)
nano .env

# Защити .env от чтения другими пользователями (только root читает)
chmod 600 .env
```

**5. Создай systemd unit (чтобы запускалось автоматически):**

```bash
cat > /etc/systemd/system/neurogate.service <<'EOF'
[Unit]
Description=neurogate
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/neurogate
EnvironmentFile=/opt/neurogate/.env
ExecStart=/opt/neurogate/.venv/bin/neurogate
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now neurogate
systemctl status neurogate     # должен быть active (running)
journalctl -u neurogate -n 30  # покажет стартап-баннер
```

**6. Открой порт в файрволе:**

```bash
ufw allow 22/tcp
ufw allow 8765/tcp
ufw --force enable
```

**7. Проверь, что работает:**

```bash
curl http://localhost:8765/health
# {"ok": true}

# С любой машины:
curl -H "Authorization: Bearer <NEUROGATE_API_TOKEN из .env>" \
     http://<server-ip>:8765/v1/models
```

**8. (Рекомендуется) HTTPS через nginx + Let's Encrypt:**

Без TLS токен и запросы летят открытым текстом — кто угодно по дороге может их подсмотреть. Если у тебя есть домен:

```bash
apt install -y nginx certbot python3-certbot-nginx
# направь A-запись домена на IP сервера, подожди пару минут DNS, потом:
certbot --nginx -d your-domain.com
# certbot сам настроит nginx и обновление сертификата
```

После HTTPS — закрой 8765 в ufw (`ufw delete allow 8765/tcp`), доступ только через 443 → nginx → localhost:8765.

**9. Используй из своих приложений:**

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://your-domain.com/v1",  # или http://<ip>:8765/v1если без HTTPS
    api_key="<NEUROGATE_API_TOKEN из .env>",
)
r = client.chat.completions.create(model="auto", messages=[{"role":"user","content":"hi"}])
```

**Обновление сервера** (когда выйдет новая версия):

```bash
cd /opt/neurogate && git pull && uv sync && systemctl restart neurogate
```

---

Не получилось — открой [issue](../../issues/new?template=bug_report.md) и приложи `journalctl -u neurogate -n 100`.

## Подробности по цепочкам и роутингу

Клиент шлёт обычный `POST /v1/chat/completions` с одним из `model`-значений:
- **Базовые fallback-цепочки**: `chat` (русский chit-chat, default), `code` (frontier reasoning + код), `latency` (минимум wall-clock), `quality` (максимальный AA-индекс), `chat_en` (английский), `unlimited` (без RPD-капа), `reasoning_quality` / `reasoning_deep` (thinking-модели), `paid` (Claude Opus 4.7).
- **Специальные режимы**: `web` (веб-поиск через Gemini `google_search` / OpenRouter `:online`), `image` (vision: принимает `image_url`), `moa` (Mixture of Agents: fan-out к 25 моделям + aggregator-синтез), `sc` (Self-Consistency: N сэмплов одной модели), `debate` (Multi-Agent Debate: N моделей × R раундов inter-agent revision), `deep_search` (research-агент: plan→search→synthesize→critique).
- **Авто-роутинг**: `auto` (имя настраивается через `NEUROGATE_VIRTUAL_MODEL`) — heuristic intent detection по последнему user-сообщению. `image_url` → `image`, веб-маркеры/URL → `web`, код-маркеры (```/`def `/`Traceback`/`.py`) → `code`, длинный prompt с reasoning-маркерами → `reasoning_quality`, иначе → `default_chain`. Это **regex-эвристика, не ML-классификатор**; для критичных кейсов выбирай чейн явно. `paid`/`moa`/`sc`/`debate`/`deep_search` авто-роутер **никогда** не выбирает (требуют opt-in). Graceful degradation: если в твоём `config.yaml` нет нужной целевой цепочки (например, не описаны `image`/`web`/`code`) — `auto` тихо падает в `default_chain`. Так же, как и любой неизвестный/не указанный `model` — он мапится на `default_chain` (см. шапку [config.yaml.example](config.yaml.example): `"chat" / "auto" / не задано → chat`).

Реально 100+ model-entries из 20+ провайдеров (от Groq и Gemini до Yandex Alice и AIhorde). При 429 / 5xx / quota / empty-response роутер переходит к следующему. Для клиента всё выглядит как один стабильный OpenAI endpoint.

## Что это даёт

- **~20-25k сообщений/сутки** на verified hard caps, до **~110k** с NVIDIA-теоретикой, **~60M токенов/сутки** по явным TPD — суммарно со всех провайдеров (RPM/RPD-лимиты по каждому провайдеру проставлены в [config.yaml.example](config.yaml.example))
- **Фактически токены не упираются в 60M**: цифра собрана только из провайдеров, которые отдают TPD-хедеры (Mistral, Gemini, Groq, Cerebras, GitHub, Cloudflare, OpenRouter). Три провайдера — **Mistral (RPM-only), Z.AI (concurrency-only), NVIDIA (RPM-only, без RPD)** — TPD не декларируют, т.е. сверху они ограничены только пропускной способностью и временем. В практическом смысле, если долбить их 24/7 на длинном контексте, потолок чейна становится **условно безлимитным** по токенам — реальный ceiling зависит только от того, как долго эти три провайдера держат заявленные RPM без throttling.
- **Drop-in замена OpenAI API**: любой SDK (Python, JS, curl, ChatGPT-обёртки) просто меняет `base_url` — код не трогается
- **Надёжность**: если Gemini режет квоту без предупреждения, а OpenRouter flaky — прокси молча переходит на Groq/Cerebras/SambaNova
- **Веб-поиск и vision из коробки**: `model: "web"` — актуальные данные через Gemini `google_search` / OpenRouter `:online`; `model: "image"` — vision-capable провайдеры для изображений (`image_url` во входе)
- **Продвинутые ансамбли**: `moa` (25 моделей параллельно + aggregator), `sc` (N сэмплов одной модели + majority synth), `deep_search` (multi-step research с плановщиком и критиком)
- **Локально по умолчанию** (127.0.0.1, без auth), но готов к деплою на VPS с Bearer-токеном

## Field notes — наблюдения с практики

Не бенчмарк, а заметки о том, как работают конкретные цепочки и провайдеры на наших задачах. Если у тебя картина другая — кидай в issues, обновим.

- **`chat` chain** — стабильна. Фоллбэк-логика отрабатывает молча, на пользовательской стороне не заметно когда провайдер падает.
- **`image_gen`** — рабочая, без сюрпризов. RU-модели (`Kandinsky`, `YandexART`) удерживают фотореализм лучше FLUX когда в промпте есть «digital painting»/«oil painting»: меньше «пластика», больше живой текстуры. FLUX отлично на абстрактном/иллюстративном.
- **`code` chain** — средне и **может быть долгой**: первые провайдеры — reasoning-модели с thinking-режимом, ответ обычно идёт 10-40 секунд, при фоллбэке ещё дольше. Иногда отваливается по timeout / quota — фоллбэк срабатывает, но ответ скачет по качеству от запроса к запросу. Стабильнее с большим числом ключей (Cerebras, SambaNova, Groq, OpenRouter); если нужна скорость, а не глубина — используй `code_fast` или `chat`.
- **Русский язык** — `Yandex Alice` (через kind `yandex_foundation`) даёт хороший русский, рекомендуется для RU-сценариев из бесплатных. Новый **Gemini** тоже подтянулся и держит русский ровно. **Claude Opus** платный, но на сложном русском заметно лучше Алисы. **GigaChat** на наших промптах слабее всех перечисленных — обходят даже отдельные китайские модели (Qwen, GLM). При возможности — Алиса первой, GigaChat в фоллбэк-хвост.
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

По отдельности ни один не годится для бота 24/7 — ручного ресетить придётся каждые пару часов. Вместе через neurogate — получается коммерчески-надёжный endpoint за $0/месяц.

**Типичные use-cases**:
- Персональный Telegram/Discord-бот с auto-reply
- Локальные эксперименты с LLM без привязки к Anthropic/OpenAI
- Backup-endpoint для проектов, где OpenAI-биллинг — блокер
- Sandbox для сравнения моделей на одной и той же задаче

## Поддерживаются из коробки

**Chat / reasoning / vision:**

| kind | endpoint | особенность |
|---|---|---|
| `gemini` | native SDK | 1M контекст, vision, web-search tool |
| `groq` | `api.groq.com` | очень быстрый (300-1000+ т/с) |
| `cerebras` | `api.cerebras.ai` | 1400+ т/с на 235B |
| `sambanova` | `api.sambanova.ai` | нет суточного лимита |
| `nvidia` | `integrate.api.nvidia.com` | нет суточного лимита |
| `zai` | `api.z.ai` | permanent free GLM-4.5-Flash |
| `openrouter` | `openrouter.ai` | агрегатор + `:online` web search |
| `cloudflare` | Workers AI | edge inference |
| `github` | GitHub Models | gpt-5-mini, PAT-auth |
| `mistral` | `api.mistral.ai` | Experiment plan |
| `huggingface` | HF Inference Router | прокси к paid sub-providers |
| `dashscope` | Alibaba International | Qwen flagships, 1M context |
| `deepseek` | `api.deepseek.com` | DeepSeek V3.2/R1 |
| `cohere` | `api.cohere.com` | Command-R, Aya для русского |
| `freetheai` | `api.freetheai.xyz` | proxy к gpt-5/claude/gemini, daily checkin |
| `gigachat` | SberDevices | Basic→OAuth, RU Trusted Root CA |
| `yandex_foundation` | Yandex Cloud | Alice — лучший RU из бесплатных |
| `pollinations` | `pollinations.ai` | image+text без ключа |
| `ovhcloud` | OVHcloud AI Endpoints | EU без ключа |
| `openai` | любой OpenAI-compat | твой base_url для exotic |

**Image generation:** `cloudflare` (FLUX/SDXL/Phoenix/DreamShaper), `gemini` (nano-banana), `gigachat` (Kandinsky), `yandex_art` (YandexART), `pollinations` (Flux/Sana/Turbo), `freetheai_image` (gpt-image-2, Seedream-v4 etc.), `aihorde` (community).

**Audio:** `edge_tts` (Microsoft Edge TTS, безлимит), `groq_whisper` (распознавание), `gemini` (распознавание + sound classification), `hf_space_audio` (Stable-Audio-Open для SFX).

**Translation:** `libretranslate`, `mymemory`, `yandex_translate`, `cohere_translate` (Aya).

**Embeddings / rerank / moderation:** `voyage_embed`, `jina_embed`, `cohere_embed`, `gemini_embed`, `mistral_embed`, `cloudflare_embed` / `voyage_rerank`, `jina_rerank`, `cohere_rerank` / `openai_moderation`, `mistral_moderation`, `llama_guard` (Cloudflare).

Полный список chains, providers и моделей — в [config.yaml.example](config.yaml.example). Разбор основных цепочек — ниже в [§ Цепочки](#цепочки). Гайд по получению ключей у каждого провайдера — [docs/providers-setup.md](docs/providers-setup.md).

### Контроль расхода (платные/grant-провайдеры)

Большинство провайдеров — permanent free, но у пары есть биллинг или grant-квота, которую полезно мониторить:

- **GigaChat (Сбер)** — Freemium квота, после — billed. Проекты, лицевой счёт и расход: [developers.sber.ru/studio](https://developers.sber.ru/studio/).
- **Yandex Cloud** (YandexGPT/YandexART/Translate, AI Studio Search/Vision/OCR) — новым аккаунтам обычно дают grant-квоту в рублях на 6-12 месяцев. Расход, гранты и бюджет-алерты: [center.yandex.cloud/billing/accounts](https://center.yandex.cloud/billing/accounts/).
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

Все 15 `model`-значений и что они делают. Имя цепочки передаётся в поле `model` запроса — остальное neurogate выбирает сам.

### Базовые fallback-цепочки

Простые цепочки: **пробуют провайдеров по порядку до первого успеха**, 429/5xx переходят на следующего. Ответ приходит от первого сработавшего.

| `model` | Что делает | Первые в цепочке |
|---|---|---|
| `chat` (default) | Русский chit-chat, RU-floor по качеству | `github:gpt-4.1-mini` → `yandex:alice` → `gemini:2.5-flash-lite` → `sambanova:llama-4-maverick` → дальше большой fallback-хвост |
| `chat_en` | Английский chit-chat, AA Intelligence Index desc | `gemini:flash-latest` → `sambanova:deepseek-v3.2` → `openrouter:nemotron-3-super-free` |
| `code` | Reasoning + код, композит SWE-bench/LiveCodeBench | `nvidia:qwen3-coder-480b` → `gemini:flash-latest` → `nvidia:qwen3.5-397b` → `sambanova:deepseek-v3.2` |
| `latency` | Минимум wall-clock (медиана Total time) | `cerebras:llama3.1-8b` (421ms) → `groq:llama-3.1-8b` (668ms) → `groq:gpt-oss-20b` (714ms) |
| `quality` | Максимум AA Intelligence Index v4.0 | `freetheai:gpt-5` → `gemini:flash-latest` → `sambanova:deepseek-v3.2` → `nvidia:qwen3.5-397b` |
| `unlimited` | Только провайдеры без жёсткого RPD-капа (для high-volume) | `sambanova:*` → `nvidia:*` → `zai:*` |
| `reasoning_quality` | 9 thinking-моделей, порядок по AA | `gemini:flash-latest` → `openrouter:nemotron-3-super-free` → `groq:gpt-oss-120b` |
| `reasoning_deep` | Те же, но по глубине thinking (reasoning_tokens desc) | `gemini:2.5-flash` → `groq:qwen3-32b` → `zai:glm-4.5-flash` |
| `paid` | Единственная платная (для тестов — **отдельный ключ `OPENROUTER_PAID_API_KEY`**) | `openrouter:claude-opus-4.7` |

### Специальные цепочки

Эти не являются простым fallback — они делают что-то отличное от «перебирай провайдеров».

#### `web` — актуальные данные через веб-поиск

Цепочка активирует native-веб-поиск у провайдера: Gemini вызывает `google_search`-tool server-side (бесплатно), OpenRouter `:online`-провайдеры (elephant-alpha, nemotron-3-super, glm-4.5-air) используют Exa-поиск на своей стороне (через `OPENROUTER_PAID_API_KEY`). Клиенту ничего дополнительно делать не нужно — просто `model: "web"`.

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

> ⚠️ **Та же история с prompt injection, что и у `web` — только сильнее.** `deep_search` несколько раз подряд тянет веб-контент и кормит его в LLM. Если на одной из подгруженных страниц стоит инъекция, она проходит через synthesizer и попадает в финальный ответ как «факт со ссылкой». Используй для research-задач, где ты сам проверишь итог; **не** запускай в автоматических цепочках, где `deep_search`-output идёт прямо в действия (создать файл, сделать запрос, переслать кому-то).

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

OpenAI-совместимые (drop-in для существующих клиентов):

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

Управление и обзор:

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

`model` в запросе: `chat` (default), `chat_fast`, `chat_en`, `code`, `code_fast`, `latency`, `quality`, `unlimited`, `quota`, `image`, `web`, `reasoning_quality`, `reasoning_deep`, `paid`, `translation`, `translate_adaptive`, `moa`, `sc`, `debate`, `deep_search`, `image_gen`, `image_edit`, `audio`, `tts`, `sfx`, `embed`, `embed_code`, `rerank`, `moderation`, `moderation_image`, `moderation_jailbreak`, `moderation_ru`, `auto`, либо конкретный провайдер из `/v1/models`, либо ad-hoc `kind:model_id` (см. ниже). Полный список и текущий состав каждой — `GET /v1/health` или dashboard.

### Ad-hoc модели (любая `kind:model_id`)

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

### Drag-n-drop редактор цепочек (дашборд)

В дашборде, во вью «Провайдеры», есть кнопка «✎ редактировать цепочки». Включает редактор с drag-n-drop через SortableJS:

- Слева — список цепочек: переключение, переименовать, удалить, отметить как default (★).
- В центре — провайдеры активной цепочки: перетаскивание для reorder, ✕ для удаления.
- Справа — пул всех провайдеров из `cfg.all_providers` (с фильтром): drag в любую цепочку (clone — один и тот же провайдер может быть в нескольких цепочках, как и в YAML).
- Кнопка «сохранить» → `PUT /v1/chains`. Сервер валидирует, переписывает `chains:` и `default_chain:` в `config.yaml` (комментарии и секция `providers:` сохраняются — replace через regex), пишет бэкап `config.yaml.bak`, делает hot-reload роутера в памяти. Следующий запрос идёт по новой схеме без рестарта процесса.

Безопасность: эндпоинт под тем же `NEUROGATE_API_TOKEN`. Если у тебя `config.yaml` синкается с локального файла на прод (через rsync / CI), помни: правки через дашборд на сервере перезапишутся следующим деплоем. Для постоянных изменений — редактируй локальный `config.yaml` и деплой как обычно.

### OpenAI-совместимость

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

**Что поддерживается:** system/user/assistant/tool roles, tool calling (function calling), multi-turn, `n`, `stop`, `temperature`, `top_p`, penalty-поля, `tools`/`tool_choice`/`parallel_tool_calls`, **streaming** (`stream: true` — SSE как в OpenAI, включая Gemini через конвертацию native-чанков), **image input** (OpenAI vision-формат с `{"type":"image_url",...}` — работает на цепочке `image`), **web-search** (цепочка `web` — native tool у Gemini / сервер-сайд у OpenRouter `:online`).

**Нюанс стриминга**: fallback между провайдерами работает только **до первого чанка**. Как только upstream начал стримить — мы закоммичены; mid-stream ошибка придёт клиенту как обрыв соединения, а не как прозрачное переподключение (проще и совпадает с поведением самого OpenAI). Цепочки `moa`/`sc`/`debate`/`deep_search` **не поддерживают streaming** — они агрегирующие, собирают результаты целиком перед синтезом.

**Что НЕ поддерживается:**
- Logprobs — не пересылаются.
- `tools` в цепочках `moa`/`sc`/`debate`/`deep_search` — пока нет (вернётся 400 с пояснением).

### curl

```bash
BASE=http://<your-server>:8765
TOKEN=<your-neurogate-token>

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
    api_key="<your-neurogate-token>",
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

neurogate пробрасывает OpenAI-совместимые поля `tools` и `tool_choice` в upstream и возвращает `tool_calls` / `finish_reason: "tool_calls"` как есть. Можно использовать neurogate как backend для агентов, работающих по протоколу OpenAI function calling.

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

Памятка для модели / агента, которого подключают к neurogate:

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
- **Issues / PR** приветствуются: новые провайдеры, баги в фоллбэке, неточности в документации, идеи по дизайну.

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

## Когда брать neurogate, когда не брать

**Сила — в объёме и скорости, а не в IQ.** Llama 3.3 70B, Qwen3-235B, Gemini Flash, GPT-OSS — это модели не-фронтирного уровня (примерно как GPT-4 класс / Sonnet 4): по AA Intelligence Index сидят в районе 33–46, фронтир (Opus 4.7 / GPT-5 / Gemini 3 Pro) — 53–57. В прямом сравнении на сложных задачах фронтир выигрывает ~60-65% case-by-case. Но за $0 это очень неплохо — и в форматных задачах (JSON, tool calls, структурированный вывод) разрыв почти не виден.

**Бери neurogate, когда:**
- задача сводится к «прочитай кусок → выдай структурированный ответ» (классификация, извлечение, перевод, рерайт, суммаризация);
- нужна **скорость** на массовых задачах (Groq/Cerebras держат 300-1400 т/с — фронтир ~80);
- двухэтапный пайплайн: дешёвые модели прогоняют 10k кандидатов, фронтир добивает 200 сложных;
- интерактив, где пользователь не заметит разницу — Telegram-боты, автодополнение, voice;
- синтетика, аугментация датасетов, A/B-тесты промптов.

**Иди к фронтиру (Claude / GPT / Gemini Pro) напрямую, когда:**
- агент с длинной цепочкой шагов (>5 tool calls подряд) — стек neurogate начинает разваливаться;
- архитектурные решения в большой кодовой базе;
- сложное math / olympiad reasoning;
- длинный контекст >50k токенов с глубоким пониманием связей;
- прод-код без ревью.

В дашборде у каждого провайдера показано поле `quality 0-100` — оно откалибровано по [AA Intelligence Index v4.0](https://artificialanalysis.ai/), где 100 = фронтир. Топ бесплатной модели сейчас в районе 75-81; ниже 60 — массово-форматные задачи.
