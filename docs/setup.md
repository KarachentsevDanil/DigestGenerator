# Setup Guide

How to set up and run the Smart Digest Generator from scratch.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Required for modern type hints |
| UV | 0.8+ | Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/)) |
| Ollama | latest | Local SLM runtime ([install](https://ollama.com/download)) |
| Telegram account | — | For Telethon userbot (reads channels) |
| Telegram bot | — | Created via [@BotFather](https://t.me/BotFather) |

---

## 1. Clone and Install Dependencies

```bash
git clone <repo-url> DigestGenerator
cd DigestGenerator

# Pin Python version (UV will auto-download if needed)
uv python pin 3.12

# Install all dependencies + create .venv
uv sync
```

This reads `pyproject.toml`, resolves all dependencies, creates `uv.lock`, and installs everything into `.venv/`.

---

## 2. Configure Environment

### Create `.env` from template

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Telegram MTProto API (from https://my.telegram.org/apps)
TELEGRAM_API_ID=12345
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890

# Telegram Bot (from @BotFather)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz

# Webhook URL for bot (optional for local dev)
WEBHOOK_URL=https://your-domain.com
```

### How to get Telegram credentials

**API ID and Hash (for Telethon):**
1. Go to https://my.telegram.org/apps
2. Log in with your phone number
3. Create a new application (any name/platform)
4. Copy `api_id` and `api_hash`

**Bot Token (for python-telegram-bot):**
1. Open Telegram, search for `@BotFather`
2. Send `/newbot`, follow prompts
3. Copy the bot token

### Review `config.yaml`

The default `config.yaml` has sensible defaults. Adjust if needed:

```yaml
ollama:
  base_url: http://localhost:11434
  model: gemma4:e4        # or gemma4:e2 for smaller GPU
  timeout: 60
  temperature: 0.1

embeddings:
  model: all-MiniLM-L6-v2

dedup:
  minhash_threshold: 0.7
  minhash_num_perm: 128
  cosine_definite_threshold: 0.88
  cosine_borderline_threshold: 0.80
  window_hours: 72

scrape:
  messages_per_channel: 100
  delay_between_channels_seconds: 1

digest:
  default_top_k: 5
  daily_relevance_floor: 0.2
  weekly_relevance_floor: 0.4
```

---

## 3. Set Up Ollama

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Gemma 4 model
ollama pull gemma4:e4

# Verify it's running
curl http://localhost:11434/api/tags
```

The sentence-transformers embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use.

---

## 4. Initialize Database

```bash
# Create data directories
mkdir -p data/chromadb data/sessions

# Run Alembic migrations (creates data/digest.db)
uv run alembic upgrade head
```

---

## 5. Authenticate Telethon

First-time only. Creates a session file for headless operation.

```bash
uv run python scripts/setup_session.py
```

This will:
1. Prompt for your phone number
2. Send a Telegram login code
3. Ask for the code (and 2FA password if enabled)
4. Save session to `data/sessions/digest_session.session`

All subsequent runs use the session file automatically.

---

## 6. Start the Application

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/health
# {"status": "ok", "database": "ok", "ollama": "ok", "chromadb": "ok"}
```

---

## 7. Initial Bot Setup

1. Open Telegram, find your bot by username
2. Send `/start` to register yourself
3. Add categories:
   ```
   /addcat AI/ML | Artificial intelligence, LLMs, ML research
   /addcat Crypto | Cryptocurrency, blockchain, DeFi
   /addcat Tech | Big tech news, startups, funding
   ```
4. Add sources (Telegram channels):
   ```
   /addsource @ai_news_channel
   /addsource @crypto_daily
   ```
5. Trigger a manual test:
   ```
   /digest
   ```

---

## 8. Set Up Cron Jobs

Add to your crontab (`crontab -e`):

```bash
# Scrape + dedup + classify every 4 hours
0 */4 * * *  curl -sf -X POST http://localhost:8000/pipeline/run-all

# Check if any user needs a daily digest (hourly, handles timezones)
0 * * * *    curl -sf -X POST http://localhost:8000/generate-digest?type=daily

# Check if any user needs a weekly digest (hourly on Sundays)
0 * * * 0    curl -sf -X POST http://localhost:8000/generate-digest?type=weekly
```

---

## Running in Production

### Systemd Service

Create `/etc/systemd/system/smart-digest.service`:

```ini
[Unit]
Description=Smart Digest Generator
After=network.target

[Service]
Type=simple
User=digest
WorkingDirectory=/path/to/DigestGenerator
ExecStart=/path/to/DigestGenerator/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PATH=/path/to/DigestGenerator/.venv/bin

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable smart-digest
sudo systemctl start smart-digest
```

### Webhook for Bot (Production)

For production, set `WEBHOOK_URL` in `.env` to your public-facing URL:

```env
WEBHOOK_URL=https://digest.yourdomain.com
```

The app will register `https://digest.yourdomain.com/webhook` with Telegram on startup.

For local development without a public URL, you can use a tunnel (e.g., ngrok, Cloudflare Tunnel).

---

## Troubleshooting

### "Ollama unavailable" in /health
- Check Ollama is running: `systemctl status ollama` or `ollama serve`
- Verify model is pulled: `ollama list` should show `gemma4:e4`
- Check base_url in `config.yaml` matches Ollama's address

### "FloodWaitError" during scrape
- Telethon hit Telegram's rate limit
- The scraper auto-waits and retries
- Reduce `scrape.messages_per_channel` or increase `delay_between_channels_seconds`

### Telethon session expired
- Delete `data/sessions/digest_session.session`
- Re-run `uv run python scripts/setup_session.py`

### Database migration issues
- Check current revision: `uv run alembic current`
- View migration history: `uv run alembic history`
- Re-create from scratch: delete `data/digest.db`, run `uv run alembic upgrade head`

### Bot not responding to commands
- Verify `TELEGRAM_BOT_TOKEN` is correct
- Check webhook is registered: `curl https://api.telegram.org/bot<TOKEN>/getWebhookInfo`
- Check app logs for webhook processing errors

---

## Development

### Running Tests

```bash
uv run pytest                          # All tests
uv run pytest tests/test_dedup.py      # Specific test file
uv run pytest -v --tb=short            # Verbose with short tracebacks
uv run pytest --cov=src                # With coverage
```

### Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Database Migrations

After changing models in `src/db/models.py`:

```bash
uv run alembic revision --autogenerate -m "description of change"
uv run alembic upgrade head
```

### Manual Pipeline Trigger

```bash
# Run individual stages
curl -X POST http://localhost:8000/scrape
curl -X POST http://localhost:8000/deduplicate
curl -X POST http://localhost:8000/classify
curl -X POST http://localhost:8000/generate-digest?type=daily

# Run all stages sequentially
curl -X POST http://localhost:8000/pipeline/run-all

# Retry failed messages
curl -X POST http://localhost:8000/pipeline/retry-failed

# Check pipeline history
curl http://localhost:8000/pipeline/runs

# View system stats
curl http://localhost:8000/stats
```
