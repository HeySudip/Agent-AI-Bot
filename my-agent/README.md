# Agent AI Bot 🤖

A fully conversational Telegram AI agent — no slash commands needed. Just chat
naturally. The bot reasons with an LLM, calls tools to do real work
(GitHub, web search, URL summarization, math, PDFs, …), and persists
per-user memory in SQLite.

## Highlights

- **LLM fallback chain**: Gemini 2.5 Flash → Gemini 2.0 → Claude (auto on rate
  limit / overload / 5xx).
- **Web search**: DuckDuckGo (free, always on) + Tavily (optional, higher
  quality).
- **Video → PDF tool** with five modes — `summary`, `qa`, `screenshots`,
  `subtitles`, `full` — that pulls multilingual transcripts, extracts
  evenly-spaced frames via yt-dlp + ffmpeg, and renders a clean PDF with
  metadata, summary, embedded screenshots, and the full transcript.
- **GitHub tools** (26): create / fork / delete repos, manage issues, PRs,
  branches, files, gists, commits.
- **URL summarizer** with **SSRF protection** — refuses loopback, private,
  link-local, multicast, reserved, cloud-metadata addresses, and non-HTTP
  schemes.
- **Safe calculator** — AST-walker evaluator that allows only arithmetic and
  an allow-list of `math` functions. No `eval`.
- **Persistent memory**: SQLite-backed conversation history per user.
- **Rate limiting**: per-user burst + sustained limits.
- **Structured JSON logs** with automatic redaction of common secret formats
  (OpenAI/Anthropic/GitHub/Slack/etc.).
- **Production packaging**: Dockerfile (multi-stage, non-root user),
  docker-compose, Makefile, pyproject.toml, GitHub Actions CI.

## Quick start

```bash
git clone https://github.com/HeySudip/Agent-AI-Bot.git
cd Agent-AI-Bot/my-agent
cp .env.example .env             # fill in TELEGRAM_BOT_TOKEN
make install
make run
```

Then in your Telegram chat, paste your provider key directly — the bot
auto-detects and saves it:

| Key prefix      | Unlocks                |
|-----------------|------------------------|
| `AIzaSy…`       | Gemini AI (primary)    |
| `sk-ant-…`      | Anthropic Claude       |
| `ghp_…`         | GitHub operations      |
| `tvly-…`        | Tavily premium search  |
| `gsk_…`         | Groq                   |
| `xai-…`         | xAI / Grok             |
| `hf_…`          | HuggingFace            |
| `sk-or-v1-…`    | OpenRouter             |

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

The compose file mounts `./data` and `./config` so SQLite state and the JSON
config persist across container restarts. The container runs as a
non-root user (`uid=10001`), uses JSON-formatted logs, and has rotating log
files.

## Architecture

```
my-agent/
├── bot.py              # Telegram app entrypoint
├── agent.py            # LLM orchestration + provider fallback
├── settings.py         # typed env-driven settings
├── config.py           # disk-backed JSON config (chat-supplied keys)
├── exceptions.py       # structured exception hierarchy
├── logging_config.py   # structlog + secret redaction processor
├── handlers/           # Telegram command + message handlers
├── tools/              # LangChain tools
│   ├── pdf_builder.py  # shared, structured PDF renderer
│   ├── video.py        # YouTube → PDF with summary/QA/screenshots/subtitles
│   ├── youtube_pdf.py  # research_and_create_pdf, generate_text_to_pdf
│   ├── github.py
│   ├── search.py
│   ├── url.py
│   └── utilities.py
├── memory/             # SQLite conversation + user stats store
├── safety/             # safe_eval, ssrf_guard, secrets_redactor
├── utils/              # rate limiter, async helpers, formatting
└── tests/              # unit tests for safety primitives + rate limiter
```

## Configuration

All runtime knobs come from environment variables. Provider API keys are
normally pasted in chat and stored in `config/config.json`. See
[`.env.example`](./.env.example) for the full list with defaults and
descriptions.

| Variable                          | Default          | Description                               |
|-----------------------------------|------------------|-------------------------------------------|
| `TELEGRAM_BOT_TOKEN`              | — (required)     | Token from @BotFather                     |
| `ENVIRONMENT`                     | `dev`            | `dev` / `staging` / `prod`                |
| `LOG_LEVEL`                       | `INFO`           | DEBUG / INFO / WARNING / ERROR            |
| `LOG_FORMAT`                      | auto             | `json` (prod) / `console` (dev)           |
| `MAX_TOOL_ITERATIONS`             | `10`             | Hard cap per user turn                    |
| `SHORT_TERM_MEMORY_SIZE`          | `30`             | Messages of context per user              |
| `MAX_INPUT_CHARS`                 | `8000`           | Reject longer inputs                      |
| `RATE_LIMIT_BURST`                | `5`              | Messages allowed per burst window         |
| `RATE_LIMIT_BURST_WINDOW_S`       | `10`             | Burst window seconds                      |
| `RATE_LIMIT_SUSTAINED`            | `40`             | Messages allowed per sustained window     |
| `RATE_LIMIT_SUSTAINED_WINDOW_S`   | `300`            | Sustained window seconds                  |
| `HTTP_TIMEOUT_S`                  | `20`             | Outbound HTTP timeout                     |
| `FETCH_URL_MAX_CHARS`             | `10000`          | Max characters returned by fetch_url      |
| `DATABASE_PATH`                   | `data.db`        | SQLite path                               |
| `CONFIG_PATH`                     | `config/config.json` | JSON config for chat-supplied keys    |

## Safety & security model

- **No `eval`**. The calculator tool uses
  [`safety/safe_eval.py`](./safety/safe_eval.py), an AST-walking evaluator
  that allows only numeric literals, the operators `+ - * / // % **` and
  unary `+/-`, the named constants `pi`, `e`, `tau`, `inf`, and a fixed
  allow-list of `math` functions.
- **SSRF protection**. Every outbound URL fetch is checked by
  [`safety/ssrf_guard.py`](./safety/ssrf_guard.py), which:
    - refuses any non-`http(s)` scheme,
    - resolves the hostname and rejects loopback / private /
      link-local / multicast / reserved / unspecified addresses,
    - blocks the AWS/GCP cloud-metadata endpoints,
    - re-validates the target after every redirect (max 5).
- **Secret redaction in logs**. The structlog pipeline runs
  [`safety/secrets_redactor.py`](./safety/secrets_redactor.py) on every event
  before serialization. It redacts known formats — OpenAI/Anthropic/Groq/xAI/
  GitHub/Slack/Tavily/HuggingFace/Google AI/AWS/JWT/Telegram bot tokens —
  and scrubs values whose key looks sensitive (`token`, `secret`, `password`,
  `api_key`, `authorization`, etc.).
- **Tightened key auto-detection**. Provider-prefix patterns only — the
  previous catch-all `[A-Za-z0-9+/=]{80,100}` pattern that matched arbitrary
  base64 has been removed.

## Video tool

The `video_to_pdf` tool turns a YouTube URL (or a free-text search query)
into a clean PDF. Pick a mode:

| mode          | what's in the PDF                                                  |
|---------------|--------------------------------------------------------------------|
| `summary`     | Overview, key points, quotes, takeaways (LLM-generated)            |
| `qa`          | Answers to questions you pass in `questions=...` (LLM, transcript) |
| `screenshots` | N evenly-spaced frames extracted via yt-dlp + ffmpeg               |
| `subtitles`   | Just the multilingual transcript (with translation fallback)       |
| `full`        | Metadata + summary + screenshots + transcript (default)            |

Frame extraction uses `yt-dlp` to download the lowest-quality video and
`imageio-ffmpeg`'s vendored ffmpeg binary to grab frames — no system
packages required. If either is missing, the tool falls back to the
high-resolution YouTube thumbnail and records the degradation in the
PDF's "Processing notes" section.

Transcript fetching tries, in order: a manual transcript in a preferred
language, an auto-generated transcript in a preferred language, anything
in any language (translated to English when possible). Any failure is
surfaced explicitly rather than silently producing a blank PDF.

## Testing

```bash
make test          # run unit tests
make cov           # with coverage report
make lint          # ruff + mypy
```

Test suites cover: safe_eval (arithmetic + every rejection path),
ssrf_guard (every blocked range + DNS failure), secrets_redactor (every
provider format + structlog processor), rate limiter (burst + sustained +
isolation between users), and the calculator tool wired through `safe_eval`.

## Model fallback chain

```
gemini-flash-latest → gemini-2.5-flash → gemini-2.5-flash-lite
  → gemini-3.5-flash → Claude (if key set)
```

On 429 (rate limit), 503 (overload), or 404 (model unavailable) the bot
silently tries the next model. Multiple Gemini keys are cycled to spread
the rate-limit budget.

## Requirements

- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Google AI Studio key (free): [aistudio.google.com](https://aistudio.google.com/app/apikey)
- Optional: any of the other provider keys listed above

## License

MIT — see [LICENSE](../LICENSE) at the repo root.
