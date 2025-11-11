# AgentCore + LangGraph FAQ POC

This proof-of-concept demonstrates how a small internal support team can field frequently asked questions using AWS AgentCore, LangGraph, and Amazon Bedrock. It keeps infrastructure costs to an absolute minimum while remaining deployable on AgentCore.

## Features
- LangGraph stateful agent with Bedrock Claude 3.5 Sonnet.
- Static FAQ lookup with lightweight fuzzy matching.
- SEC EDGAR filings lookup tool for company/form queries.
- Embedded filings explorer backed by the local vectorstore.
- Browser-based frontend that replaces the CLI for demos.
- Optional tools for Slack alerts, form-based ticket creation, and document retrieval.
- Configurable to disable all paid integrations by default.
- Works locally (web UI) or via AgentCore runtime.

## Project Layout
- `app/config.py` – runtime settings and environment detection.
- `app/graph.py` – LangGraph definition and memory integration.
- `app/tools.py` – tool implementations with cost-safe defaults.
- `app/app.py` – Bedrock AgentCore app entrypoint.
- `app/web.py` – Starlette application serving the browser UI and REST APIs.
- `main.py` – script referenced by `agentcore configure`.
- `frontend/` – static assets for the proof-of-concept interface.
- `data/faq.json` – static FAQ data loaded at startup.
- `data/sec_inquiries/` – local storage for SEC inquiry metadata and attachments.
- `demo_files/` – legacy demo assets retained for reference.

## Requirements
- Python 3.10+
- AWS account with AgentCore and Bedrock access (for production deployment).
- `uv` or `pip` for dependency installation.

Install dependencies:
```bash
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

## Configuration

Set environment variables in a `.env` file or your shell.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Controls defaults (development disables paid tools). |
| `AWS_REGION` | `us-west-2` | AWS region for Bedrock and AgentCore. |
| `BEDROCK_MODEL_ID` | `qwen.qwen3-coder-30b-a3b-v1:0` | Bedrock model to invoke. |
| `BEDROCK_AGENTCORE_MEMORY_ID` | _unset_ | AgentCore managed memory store. Leave empty to use in-memory saver. |
| `DEFAULT_ACTOR_ID` | `poc-user` | Actor identifier used in graph config. |
| `FAQ_PATH` | `data/faq.json` | Path to FAQ JSON file. |
| `ENABLE_NETWORK_TOOLS` | `false` in development, `true` in production | Allows outbound requests for tools. |
| `SLACK_WEBHOOK_URL` | _unset_ | Slack webhook for notifications (requires `ENABLE_NETWORK_TOOLS=true`). |
| `SLACK_DEFAULT_CHANNEL` | `#alerts` | Default Slack channel. |
| `TICKET_FORM_URL` | _unset_ | Google Form endpoint for ticket submission. |
| `SEC_INQUIRY_FORM_BASE_URL` | _unset_ | Base Google Form URL used for prefilled SEC inquiries. |
| `SEC_INQUIRY_FIELD_COMPANY` | _unset_ | Google Form field ID for the company name. |
| `SEC_INQUIRY_FIELD_CIK` | _unset_ | Google Form field ID for the CIK value. |
| `SEC_INQUIRY_FIELD_FORM` | _unset_ | Google Form field ID for the filing form type. |
| `SEC_INQUIRY_FIELD_CONTEXT` | _unset_ | Google Form field ID for additional context. |
| `SEC_INQUIRY_UPLOAD_DIR` | `data/sec_inquiries/attachments` | Directory for archived SEC inquiry attachments. |
| `SEC_INQUIRY_DB_PATH` | `data/sec_inquiries/inquiries.db` | SQLite database path for SEC inquiry records. |
| `SEC_INQUIRY_MAX_IMAGE_MB` | `5` | Maximum size (in MB) for uploaded SEC inquiry images. |
| `HTTP_TIMEOUT` | `8` | Request timeout in seconds. |
| `OBSERVABILITY_SERVICE_NAME` | `agentcore-faq-agent` | Service name used for logs and traces. |
| `ENABLE_XRAY` | `false` | Enable AWS X-Ray tracing when `true`. |

### Development Defaults
- `APP_ENV=development`
- `ENABLE_NETWORK_TOOLS=false`
- Memory falls back to in-process LangGraph saver.
- Safe responses returned for network tools so no paid services run.

### Production Defaults
- `APP_ENV=production`
- `ENABLE_NETWORK_TOOLS` automatically true (override to disable).
- Configure `BEDROCK_AGENTCORE_MEMORY_ID`, Slack webhook, and ticket form URL to enable full toolkit.

## Local Development
1. Copy `demo_files/faq.json` into `data/faq.json` or edit to match your FAQs.
2. Export or load environment variables.
3. Launch the proof-of-concept frontend:
   ```bash
   uvicorn app.web:app --reload --host 0.0.0.0 --port 8000
   ```
   Visit `http://localhost:8000` to explore embedded filings and chat with the assistant.

   The legacy CLI runner is still available if you prefer the terminal experience:
   ```bash
   python -m app.app
   ```
   Type `exit` to quit.

You can also run a one-off prompt:
```python
from app.app import invoke
response = invoke({"prompt": "What are business hours?"}, context=None)
print(response)
```

## Deploying with AWS AgentCore
1. Authenticate with AWS and install the AgentCore CLI.
2. Configure the application:
   ```bash
   agentcore configure -e main.py
   ```
3. Review generated environment template and fill required variables (Bedrock model, region, memory ID, Slack webhook if desired).
4. Launch the environment:
   ```bash
   agentcore launch
   ```
5. Invoke for testing:
   ```bash
   agentcore invoke '{"prompt": "How do I reset my password?"}'
   ```
6. Tear down when finished to avoid charges:
   ```bash
   agentcore destroy
   ```

## Container Image & ECR Workflow
1. Build the linux/amd64 container locally:
   ```bash
   docker build --platform linux/amd64 -t agentcore-faq .
   ```
2. Push to Amazon ECR (requires authenticated AWS CLI and Docker):
   ```bash
   export AWS_REGION=us-west-2
   export IMAGE_NAME=agentcore-faq
   ./scripts/build_and_push_ecr.sh
   ```
   - The script creates the repository if missing, logs in to ECR, tags with the current git commit, and pushes the image.
   - Override `IMAGE_TAG` for custom versioning (e.g., CI build number).
3. Use the resulting image URI (e.g., `123456789012.dkr.ecr.us-west-2.amazonaws.com/agentcore-faq:abc1234`) when configuring AgentCore environments that consume container artifacts.

For ad-hoc smoke tests, run the gateway locally:
```bash
docker run --rm -p 8080:8080 agentcore-faq
curl -sS -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are business hours?"}'
```

To serve the full web experience from a container, swap the CMD (or override at runtime) with:
```bash
uvicorn app.web:app --host 0.0.0.0 --port 8080
```

## Cost Control Tips
- Keep `APP_ENV=development` and `ENABLE_NETWORK_TOOLS=false` when running locally to avoid external API calls.
- Use the static FAQ tool for most answers before enabling network integrations.
- SEC filings lookup uses data.sec.gov REST endpoints directly and requires no additional contact information.
- Configure `SEC_INQUIRY_FORM_*` variables only when you intend to drive the SEC inquiry Google Form workflow.
- Deploy to a single AgentCore environment and destroy it after demos.
- Monitor usage via AgentCore's observability dashboards to ensure minimal Bedrock invocations.

## Testing
- Unit tests can target individual tools by importing from `app.tools`.
- For end-to-end tests in development mode, mock responses from `requests` before enabling network tools.

## Troubleshooting
- **FAQ not found**: Verify `FAQ_PATH` points to the correct JSON file.
- **Slack/Ticket tools disabled**: Ensure `ENABLE_NETWORK_TOOLS=true` and required URLs are set.
- **Bedrock errors**: Confirm your AWS credentials allow Bedrock invoke access for the chosen model.
- **SEC inquiry workflow incomplete**: Ensure the Google Form field IDs and base URL are configured and the attachments directory is writable.

