from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .app import invoke


async def healthcheck(request: Request) -> JSONResponse:
    """Simple liveness probe endpoint."""
    return JSONResponse({"status": "ok"})


async def invoke_agent(request: Request) -> JSONResponse:
    """Invoke the agent with the provided payload and optional headers from the request."""
    context = _build_context_from_request(request)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    result = invoke(payload, context=context)
    return JSONResponse(result)


def _build_context_from_request(request: Request):
    headers = dict(request.headers)
    session_id = headers.get("x-agentcore-session-id") or headers.get(
        "x-session-id"
    )
    actor_id = headers.get("x-agentcore-actor-id")
    identity = {
        "actor_id": actor_id,
        "external_id": headers.get("x-agentcore-external-id"),
        "tenant": headers.get("x-agentcore-tenant"),
    }

    class GatewayContext:
        def __init__(self):
            self.session_id = session_id
            self.actor_id = actor_id
            self.identity = identity
    return GatewayContext()


routes = [
    Route("/healthz", healthcheck, methods=["GET"]),
    Route("/invoke", invoke_agent, methods=["POST"]),
]

api = Starlette(debug=False, routes=routes)

