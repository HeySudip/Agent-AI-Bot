# Agent AI Bot 🤖

A fully conversational Telegram AI agent — no slash commands needed. Just chat naturally.

## Features

- **AI Models**: Gemini 2.5 Flash (primary, free) → auto-falls back through Gemini 2.0 → Claude on errors/rate limits
- **Web Search**: DuckDuckGo (free, always on) + Tavily (optional, higher quality)
- **GitHub Tools**: 26 operations — create repos, open/close issues, manage PRs, read/write files, and more
- **URL Summarizer**: Paste any link and get a summary
- **Utilities**: Calculator, unit converter, text tools
- **Persistent Memory**: SQLite-backed conversation history per user
- **Rate Limiting**: Per-user request throttling
- **Admin Panel**: Broadcast messages, manage users

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/HeySudip/Agent-AI-Bot.git
cd Agent-AI-Bot/my-agent
pip install -r requirements.txt
```

### 2. Set environment variable

```bash
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
```

Or copy `.env.example` to `.env` and fill it in.

### 3. Run

```bash
python bot.py
```

### 4. Configure API keys via chat

Just paste your keys directly in the Telegram chat — the bot auto-detects and saves them:

| Key prefix | What it unlocks |
|---|---|
| `AIzaSy...` | Gemini AI (primary LLM) |
| `sk-ant-...` | Anthropic Claude (fallback LLM) |
| `ghp_...` | GitHub operations |
| `tvly-...` | Tavily premium search |

## Architecture

```
my-agent/
├── bot.py              # Telegram app entry point
├── agent.py            # LLM orchestration + model fallback logic
├── config.py           # Config persistence (config.json)
├── handlers/
│   ├── commands.py     # /start, /help, /clear, /stats, /admin
│   └── messages.py     # Conversational message handler + file uploads
├── tools/
│   ├── github.py       # 26 GitHub tools via PyGithub
│   ├── search.py       # DuckDuckGo + Tavily search
│   ├── url.py          # URL fetching & summarization
│   └── utilities.py    # Calculator, converter, text utilities
├── memory/
│   └── store.py        # SQLite conversation + user stats store
└── utils/
    ├── formatting.py   # Telegram markdown helpers
    └── rate_limiter.py # Per-user rate limiting
```

## Model Fallback Chain

```
gemini-2.5-flash → gemini-2.0-flash → gemini-2.0-flash-lite
  → gemini-flash-latest → gemini-flash-lite-latest → Claude (if key set)
```

On 429 (rate limit), 503 (overload), or 404 (model unavailable) the bot silently tries the next model.

## Requirements

- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Google AI Studio key (free): [aistudio.google.com](https://aistudio.google.com/app/apikey)
