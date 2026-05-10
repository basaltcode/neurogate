# Auto-Audit & Add Models — план автоматизации

Дата составления: 2026-04-26 (ревизия 2026-04-29).
Контекст: ручной аудит «новых free AI API» на 26.04 показал, что 3 из 13 пунктов — галлюцинации (`gemini:3.1-flash`, `github:grok-3`, `openrouter:ling-2.6-1t:free` уже добавлен). Нужен pipeline, который ловит враньё **до** правки `config.yaml`.

## Цель

Автоматически раз в неделю собирать новые free-модели у вендоров и добавлять их в `config.yaml` через PR с измеренными метриками. **Никогда** не мержить и не деплоить автоматически.

---

## 4-стадийный pipeline

### Stage 1 — CATALOG DIFF (cron, weekly, БЕЗ LLM)

- Скрипт `scans/audit_catalog.py`.
- Опрашивает `/v1/models` у OpenAI-compat вендоров: groq, openrouter, cerebras, sambanova, nvidia, mistral, zai. (Gemini/GitHub/Cohere/HF — нестандартные шейпы, добавим по необходимости.)
- Сохраняет полный snapshot в `scans/catalog-snapshot.json`.
- На втором и последующих запусках — эмитит **только delta**: модели, которых не было в предыдущем snapshot И не было в `config.yaml`.
- Фильтрует не-чат модели по подстрокам (`whisper`, `embed`, `rerank`, `guard`, `image`, `tts`, `parakeet`, `cosmos` и т.д.).
- **Free-detection** (`--free-only` флаг):
  - **OpenRouter**: парсит поле `pricing.prompt`/`pricing.completion` из `/v1/models`. Free = оба значения `"0"`. Это programmatic source of truth.
  - **Mistral, Z.AI**: deny-list подстрок в `scans/paid_models_blocklist.yaml` (`mistral-large-*`, `pixtral-large-*`, `codestral-*`, `glm-5*`, etc). Поддержка ~раз в месяц.
  - **Groq, Cerebras, SambaNova**: ничего не делаем — у них нет paid-only моделей, все доступны на free-tier с rate-limits.
  - **NVIDIA NIM**: `--free-only` пропускает все модели (не блокирует). NIM — единый billable starter-credit пул, а не free vs paid модели. См. open question #9 про balance monitor.
- Output: `scans/audit-YYYY-MM-DD-candidates.json` — массив `{name, kind, model, api_key_env}`, сразу пригоден для Stage 2.

**Почему не LLM-поиск.** Изначально планировался WebSearch + LLM-агент, но это та же галлюцинационная воронка, против которой строится Stage 2 (см. эмпирику 2026-04-29: gemini:flash-latest выдал 13 «новинок», 5 уже были в config, 1 был silent-fallback NIM, остальные — несуществующие id). Каталог вендора — детерминированный источник правды: можно увидеть только то, что вендор реально хостит.

**Bootstrap режим.** Первый запуск (нет prev snapshot) сохраняет snapshot и эмитит пустой candidates — иначе на старте полетели бы сотни smoke-запросов. Со второй недели pipeline видит реальные изменения (обычно 0-3 модели в неделю).

### Stage 2 — VERIFY (cron, после Stage 1)

**Stage 2 строится на трёх deterministic gate'ах — без LLM-консенсуса.** Эмпирика 2026-04-29 (см. ниже) показала, что dedup + smoke + identity ловят 100% галлюцинаций на боевом аудите. LLM-судьи галлюцинируют синхронно с автором аудита (правдоподобные version-bumps вроде `gemini-3.1-flash` поддержит вся тройка), поэтому LLM-триангуляция выкинута из pipeline до момента, когда deterministic-gate'ы что-то пропустят.

Порядок шагов (каждый — gate, обрыв при первом fail):

1. **`_dedup_check()`** — сравнение `name` и `model` с уже существующим `config.yaml`. Дубликат → `rejected_dedup`. **Выполняется первым**, чтобы не жечь smoke- и identity-квоту на уже добавленные модели.
2. **`_smoke_test()`** — реальный POST `/v1/chat/completions` с `max_tokens=10`. HTTP != 200 → `rejected_http`.
3. **`_identity_probe()`** — после HTTP 200 короткий промпт «What is your exact model name and version? Answer in one short sentence. Do not roleplay.» с `max_tokens=120`. Эвристика: family-токен (`deepseek`/`grok`/`gemma`/`glm`/`qwen`/…) + claimed major-version должны присутствовать в ответе. Mismatch → `rejected_identity`. Без этого gate NVIDIA NIM пропускал выдуманный `deepseek-v4-flash` с HTTP 200, роутя на DeepSeek-V2.5 (см. safety #11).

Финальные статусы:
- dedup OK + smoke 200 + identity match → `confirmed`.
- dedup OK + smoke 200 + identity ambiguous (модель не назвала версию или семейство) → `confirmed_unverified` (manual review перед Stage 3).
- dedup OK + smoke 200 + identity mismatch → `rejected_identity` (silent fallback или галлюцинация).
- smoke != 200 → `rejected_http` (с кодом).
- name/model уже в config → `rejected_dedup`.

Output: `scans/audit-YYYY-MM-DD-verified.json` — `[{name, kind, model, status, http_code, detail, identity_reply}]`.

**Эмпирика на аудите 2026-04-28** (проверено 2026-04-29 на 13 кандидатах от `gemini:flash-latest`): dedup отсёк 5, smoke (HTTP 4xx/timeout/empty) — 6, identity-probe — 1 (`nvidia:deepseek-v4-flash` представился DeepSeek-V2.5). Confirmed = 0. Это и есть основание выкинуть LLM-триангуляцию: три deterministic gate'а покрыли все галлюцинации без LLM-вызовов.

**Когда возвращать LLM-консенсус:** если в реальном использовании появится класс ошибок, который ни dedup, ни smoke, ни identity не ловят — например, модель честно отвечает на identity, но claimed RPM/RPD из аудита расходятся с заголовками вендора. Тогда добавить LLM как evidence-обогатитель, но всё равно не как primary gate.

### Stage 3 — BENCH (manual trigger или cron, в worktree)

- `tests/bench_new.py --only-from scans/audit-*-verified.json --status confirmed`.
- Работает в **отдельном git worktree** (физическая изоляция от основной ветки).
- Добавляет провайдеры в `config.yaml` ТОЛЬКО в секцию `providers:`, **не в `chains:`**.
- Прогоняет `fetchtest/bench_latency.py` + `tests/ru_bench.py` фильтром только по новым.
- Заполняет реальные `quality / latency_s / ru` в YAML.
- Открывает PR в ветку `audit/auto-YYYY-MM-DD`.

### Stage 4 — REVIEW & MERGE (только вручную)

- Никаких auto-merge. Только ты после code review.
- Решение о включении в чейны (`chat`, `code`, `quality`, etc.) — твоё, не агента.
- Деплой на прод — отдельная команда вручную, никогда не автоматически.

---

## Safety guards (защита от поломок)

| # | Угроза | Защита |
|---|---|---|
| 1 | Hallucinated model name | Stage 2 deterministic gate'ы: dedup → smoke (HTTP) → identity-probe. Эмпирика 2026-04-29: 13/13 галлюцинаций пойманы без LLM. |
| 2 | Дубликат уже в `config.yaml` | `_dedup_check()` по `name` И `model`-полю — выполняется **первым** в Stage 2, чтобы не жечь smoke/identity-квоту на уже добавленные модели. |
| 3 | Сломанный YAML после правки | После Stage 3: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` + `from neurogate.config import load_config; load_config('config.yaml')`. Если падает — PR не открывается, ветка остаётся. |
| 4 | Платная модель в free-тестах | `paid_models_blocklist.yaml` — явный deny-list (`gpt-5*`, `o3*`, `o4-mini*`, `claude-opus-*`, `gemini-3.1-pro*`, `deep-research-*`). Stage 2 отбрасывает совпадения. |
| 5 | Модель попадает в чейн «не на своё место» | Stage 3 НЕ редактирует секцию `chains:` вообще. Chain placement — только review. |
| 6 | Жжём квоту на бенч | Stage 3 — только `bench_latency` (5 запросов × 2 варианта = 10) + `ru_bench` (20 запросов). Итого ≤30 запросов на модель. RPD-кэп ставим консервативный. |
| 7 | Auto-merge / auto-deploy | PR создаётся, не мержится. SSH-ключи прод-сервера — никогда в pipeline. |
| 8 | Утечка API-ключей в PR | Pre-commit hook: `grep -rE "sk-[a-zA-Z0-9]{20,}\|nvapi-\|gho_" --include='*.yaml' --include='*.py'`. CI блокирует PR. |
| 9 | Stage 1 пишет на устаревшую дату | Все `# === ... 2026-04-XX ===` в config — генерятся из `today`, не хардкодятся. |
| 10 | Агент молча перезаписывает existing entry | Stage 3 использует `Edit` (не `Write`) на `config.yaml`. При конфликте — PR fail. |
| 11 | Silent fallback провайдера на родственную модель | NVIDIA NIM на 2026-04-29 принимал выдуманный `deepseek-ai/deepseek-v4-flash` с HTTP 200 и роутил на DeepSeek-V2.5. Smoke-тест один не ловит. **Identity-probe** в Stage 2: после HTTP 200 запрос «What is your exact model name and version?» (max_tokens=120). Эвристика: family-токен (deepseek/grok/glm/…) + claimed major-version должны быть в ответе. Mismatch → `rejected_identity`. |

---

## CI/CD layout

**Решение: Local cron + git worktree (бывший Вариант B).**

Вариант A (GitHub Actions) отклонён:
- `WebSearch`/`WebFetch` — инструменты Claude Code CLI, в Actions runner недоступны. Stage 1 пришлось бы переписывать на Anthropic SDK с web_search tool (платно, противоречит free-духу) или на голый scraping changelog'ов.
- API-ключи уже в локальном `.env`, дублирование в `repo > Settings > Secrets` не требуется.
- Local cron не имеет SSH-доступа к прод-серверу — это соответствует правилу «не деплоить без явного OK».

### Local cron + git worktree

Минус: надо держать локальную машину живой (Mac не должен спать в момент запуска cron).
Плюс: WebSearch/WebFetch работают, ключи не дублируются, прод физически недостижим.

```bash
# crontab -e
0 10 * * 1 cd /Users/niko/Desktop/llmgate && bash scans/run_weekly_audit.sh
```

`scans/run_weekly_audit.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd /Users/niko/Desktop/llmgate
WT="../llmgate-audit-$(date +%F)"
git worktree add "$WT" -b "audit/auto-$(date +%F)"
cd "$WT"
uv run python scans/audit_runner.py
uv run python scans/audit_verifier.py
uv run python tests/bench_new.py --only-from scans/audit-*-verified.json
git add config.yaml scans/audit-*
git commit -m "audit: new free models $(date +%F)"
git push -u origin "audit/auto-$(date +%F)"
gh pr create --base main --title "audit: new free models $(date +%F)" --body-file scans/audit-*-verified.md
# worktree оставляем — после merge PR чистится вручную:
#   git worktree remove ../llmgate-audit-YYYY-MM-DD
# Автоматическая очистка не делается, чтобы не потерять незамерженные правки.
```

---

## Файлы

| # | Файл | Статус |
|---|---|---|
| 1 | `scans/audit_catalog.py` | **готово** (Stage 1, catalog diff, snapshot mode) |
| 2 | `scans/audit_verifier.py` | **готово** (Stage 2, dedup + smoke + identity-probe) |
| 3 | `scans/run_weekly_audit.sh` | **готово** (driver для cron) |
| 4 | `scans/catalog-snapshot.json` | **bootstrap saved** 2026-04-29 |
| 5 | `tests/bench_new.py` | TODO (Stage 3, обёртка над bench_latency + ru_bench) |
| 6 | `scans/paid_models_blocklist.yaml` | **готово** (mistral/zai substrings + openrouter `:online`/`:nitro`) |
| 7 | `.github/PULL_REQUEST_TEMPLATE/audit.md` | TODO (checklist для review) |

---

## Open questions (нужно решить до реализации)

1. ~~**GitHub Actions или local cron?**~~ — **resolved 2026-04-29**: local cron, см. секцию CI/CD layout.
2. **Хранить ли verified.json в git?** Да — даёт audit trail, но раздувает репо. Альтернатива — gist.
3. **Что делать с `confirmed_disputed` моделями?** Сейчас — пропускаем в Stage 3, но с пометкой в PR. Можно: открывать отдельный issue с тегом `audit-disputed` для ручного решения до bench.
4. **Cooldown для повторно-провалившихся.** Если модель `rejected` 3 недели подряд — добавить в `audit_blocklist.yaml`, чтобы не тратить квоту.
5. **Замер `quality` AA Index v4.0.** Stage 3 ставит `quality: 0` или `null`? AA не публикует API — придётся либо консервативная оценка, либо ставить `null` и заполнять вручную.
6. ~~**Anthropic API в CI**~~ — **resolved 2026-04-29**: Stage 1 переделан на catalog diff (без LLM вообще), вопрос отпал.
7. **GitHub Models / Gemini / Cohere catalog**. У них нестандартные `/models` шейпы — пока не покрыты Stage 1. Можно добавить отдельные парсеры либо принять, что новые модели у этих вендоров находим вручную.
8. ~~**paid_models_blocklist.yaml.**~~ — **resolved 2026-04-29**: реализовано pricing-парсинг для openrouter (поле `pricing.prompt/completion` из `/v1/models`) + substring-blocklist в `scans/paid_models_blocklist.yaml` для mistral/zai/openrouter (`:online`/`:nitro` теги). Groq/cerebras/sambanova не нуждаются (нет paid-only моделей).
9. **NVIDIA NIM balance monitor.** NIM — единый $1000-starter-credit пул на все модели. После исчерпания все запросы 402. Сейчас pipeline это не отслеживает. Варианты: (1) опросить `/v1/credits` если NVIDIA публикует endpoint (нужно проверить docs); (2) считать число 402 в Stage 2 smoke-test'ах и алертить при >0; (3) ничего не делать, ждать пока сломается. Реализация — отдельный скрипт `scans/nvidia_balance.py`, не блокирует основной audit-pipeline.

---

## Итог

- **Stage 1+2 — полностью автономно.** Это ловит галлюцинации.
- **Stage 3 — автономно, но в worktree, без чейнов.** Это безопасно правит config.
- **Stage 4 — только ты.** Это финальный gate.
- **Никаких** auto-merge, auto-deploy, auto-chain-edit, auto-prod-restart.
