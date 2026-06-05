# lead-gen-system

A Python 3.11 multi-agent lead generation system.

> **Phase 0 — setup only.** This repository currently contains the project
> scaffold, configuration, database schema, and verification tooling. No agents
> or data-source clients are wired up yet.

## Requirements

- Python 3.11+

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

   On Windows (cmd):

   ```cmd
   python -m venv .venv & .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and paste in your API keys.

4. Place your GCP service account JSON at:

   ```
   secrets/gcp-service-account.json
   ```

5. Initialize the database:

   ```bash
   python scripts/init_db.py
   ```

6. Verify the setup (no external API calls are made):

   ```bash
   python scripts/verify_setup.py
   ```

## Project layout

```
lead-gen-system/
├── .github/workflows/   # CI workflows (empty for now)
├── agents/              # Agent implementations (Phase 1+)
├── orchestrator/        # LangGraph orchestration (Phase 1+)
├── sources/             # Data source clients (Phase 1+)
├── sinks/               # Output sinks: Sheets, Telegram (Phase 7)
├── prompts/             # Prompt templates
├── config/              # Settings + logging configuration
├── data/                # SQLite database (gitignored *.db)
├── dashboard/           # Streamlit dashboard
├── tests/               # Test suite
├── scripts/             # init_db.py, verify_setup.py
├── secrets/             # Service account JSON (gitignored)
└── logs/                # Runtime logs (gitignored)
```

## Configuration

All configuration is loaded from `.env` via `config/settings.py`
(`pydantic-settings`). See `.env.example` for the full list of variables.

## Testing

```bash
pytest tests/test_phase0.py -v
```
