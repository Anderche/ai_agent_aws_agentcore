# AgentCore + LangGraph FAQ POC

Compact README for the internal proof-of-concept. Keep credentials and identifiers in private channels only.

## Overview
A chat assistant for operations and compliance teams. It remembers the conversation, looks things up, and can take a few actions. It runs on Amazon Bedrock (AWS's hosted AI models) through AgentCore (AWS's runtime for this kind of agent).

**What it can look up**
- `query_faq` — answers from a saved FAQ file, like a searchable help desk.
- `lookup_sec_filings` — recent SEC filings for a company, by ticker, form type (10-K, 10-Q, and so on), and year.
- Filing chat — ask questions about one filing after it has been split into searchable chunks.

**What it can do**
- Session memory — the chat keeps earlier messages, so you do not start over each turn.
- `submit_ticket` — files a support ticket.
- `send_slack_notification` — posts a message to Slack.
- `initiate_sec_inquiry` — starts a review request and opens a prefilled Google Form.
- `download_reference_document` — pulls a document from your computer, a web link, or S3 (AWS file storage).

You talk to it in a browser or a terminal. A small web server (Starlette) serves both.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.web:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` for the web UI or run `python -m app.app` for the terminal CLI.

## Configuration
Create a `.env` file or export variables before starting the server:
- `APP_ENV`: `development` for local testing, `production` for hosted deployments.
- `AWS_REGION`: AWS region for Bedrock.
- `BEDROCK_MODEL_ID`: Bedrock model identifier.
- `ENABLE_NETWORK_TOOLS`: `false` to stay offline, `true` to enable external calls.
- `FAQ_PATH`: Path to the FAQ JSON file (defaults to `data/faq.json`).
- Optional integrations: Slack webhook, Google Form URLs, SEC inquiry storage paths.

## Local Data
- Place FAQ data in `data/faq.json` (copy from `demo_files/faq.json` as a baseline).
- Vectorstore artifacts live in `data/vectorstores/`; they are read on startup when present.

## Deployment Notes
- **Railway**: Add `Procfile` and `Railway.toml` (already included) and set `APP_ENV=production` plus required secrets. Railway injects `PORT`; the app binds automatically. Production demo lives at <https://aiagentawsagentcore-production.up.railway.app>.
- **AWS AgentCore**: Package the repo, run `agentcore configure -e main.py`, provide the same environment variables, then `agentcore launch`.
- **Containers**: `docker build -t agentcore-faq .` followed by any runtime command (e.g., `uvicorn app.web:app --host 0.0.0.0 --port 8080`).

## Testing & Troubleshooting
- Unit-level checks can import helpers from `app.tools` and `app.graph`.
- If Bedrock access fails, verify IAM permissions and model availability in the chosen region.
- When `ENABLE_NETWORK_TOOLS=false`, external integrations return safe fallbacks; toggle to `true` only when credentials are configured.

## Repository Map
- `app/`: core runtime, graph definition, and tool implementations.
- `frontend/`: static assets for the browser UI.
- `demo_files/`: sample content and scripts preserved for reference.
- `scripts/`: helper scripts such as container build/publish.
