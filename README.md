# AI Agent for Corporate Filings

## AI Agent Overview
Find and query filings from publically traded companies sourced from the SEC's Edgar database. This app is a chat assistant for operations and compliance teams, built with AgentCore and LangGraph. It remembers the conversation, looks things up, and can take a few actions. It runs on Amazon Bedrock (AWS's hosted AI models) using AgentCore (AWS's runtime for this kind of agent).

**What it can look up**
- `query_faq` — answers from a saved FAQ file, like a searchable help desk.
- `lookup_sec_filings` — recent SEC filings for a company, by ticker, form type (10-K, 10-Q, and so on), and year.
- Filing chat — ask questions about one filing after it has been split into searchable chunks.

**What it can do**
- Session memory — the chat keeps earlier messages, so you do not start over each turn.
- `submit_ticket` — files a support ticket.
- `send_slack_notification` — posts a message to Slack.
- `initiate_sec_inquiry` — starts a review request and opens a prefilled Google Form.
- `download_reference_document` — pulls a document from a public `https://` link or an allowlisted S3 (AWS file storage) location. Local files and private network addresses are refused.

You talk to it in a browser or a terminal. A small web server (Starlette) serves both.

## Quick Start
Requires [uv](https://docs.astral.sh/uv/). Then:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
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
- `MAX_EMBEDDED_FILINGS`: shared cap on filings embedded into the local vectorstore (default `3`).
- `MAX_FILING_CHAT_PREPARES`: shared cap on filing-chat sessions prepared with Bedrock embeddings (default `2`).
- `MAX_BEDROCK_REPLIES`: shared cap on assistant replies that call Bedrock (default `20`). SEC lookups and menus do not count.
- `COST_GUARDS_DISABLED`: `true` skips every cap. Ignored when `APP_ENV=production`.
- `COST_GUARDS_RESET`: `true` on startup zeros the filing-chat and reply counters and allows the next embed cap on top of vectorstore files already on disk. Ignored when `APP_ENV=production`; to reset a hosted budget, delete `data/cost_guards.json` (a fresh deploy without a volume starts at zero).
- `CORS_ALLOWED_ORIGINS`: comma-separated origins allowed to call the API from another site (e.g. `https://example.com`). Leave unset when the bundled frontend is served by this app; cross-origin calls are then blocked.
- `REFERENCE_S3_ALLOWLIST`: comma-separated `bucket` or `bucket/prefix` entries that `download_reference_document` may read. Unset means no S3 reads.
- `FILING_VECTOR_TABLE_NAME`: DynamoDB table for filing chat (default `agentcore_filing_vectors`). The app does not create it; run `scripts/create_filing_vector_table.sh` once, then grant the app role only `dynamodb:BatchWriteItem`, `dynamodb:PutItem`, and `dynamodb:Query` on that table.

### Secrets
- `SLACK_WEBHOOK_URL` is a credential: anyone holding it can post to your channel. Never put it in a committed file.
  - Local: keep it in the untracked `.env`.
  - Railway: add it under the service's **Variables** tab (optionally a sealed variable).
  - AWS AgentCore / ECS: store it in AWS Secrets Manager or SSM Parameter Store (`SecureString`) and inject it as an environment variable at runtime.
- The app never echoes the webhook URL in replies or logs; if it leaks anyway, revoke it in Slack and create a new one.
- AWS credentials should come from the runtime role (IAM role, `aws configure`, or SSO), never from `.env`.

## Local Data
- Place FAQ data in `data/faq.json` (copy from `demo_files/faq.json` as a baseline).
- Vectorstore artifacts are generated at runtime in `data/vectorstores/` and are not committed; they are read on startup when present.

## Deployment Notes
- **Railway**: Add `Procfile` and `Railway.toml` (already included) and set `APP_ENV=production` plus required secrets. Railway injects `PORT`; the app binds automatically. Production demo lives at <https://aiagentawsagentcore-production.up.railway.app>.
- **AWS AgentCore**: Package the repo, run `agentcore configure -e main.py`, provide the same environment variables, then `agentcore launch`.
- **Containers**: `docker build -t agentcore-faq .` followed by any runtime command (e.g., `uvicorn app.web:app --host 0.0.0.0 --port 8080`).

## Testing & Troubleshooting
- Unit-level checks can import helpers from `app.tools` and `app.graph`.
- If Bedrock access fails, verify IAM permissions and model availability in the chosen region.
- When `ENABLE_NETWORK_TOOLS=false`, external integrations return safe fallbacks; toggle to `true` only when credentials are configured.
- User-facing errors are intentionally generic; check the server logs for the full exception.

## Repository Map
- `app/`: core runtime, graph definition, and tool implementations.
- `frontend/`: static assets for the browser UI.
- `demo_files/`: sample content and scripts preserved for reference.
- `scripts/`: helper scripts such as container build/publish.
