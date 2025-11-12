from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .app import (
    INTRO_MESSAGE,
    _discover_vectorstore_summaries,
    invoke,
)
from .config import load_settings
from .memory import SessionMemory
from .rag_pipeline import VECTORSTORE_DIR, query_vectorstore


SessionId = str


# In-memory context cache reused across requests.
_session_contexts: Dict[SessionId, SimpleNamespace] = {}

# Resolve frontend directory relative to repository root.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _ensure_context(session_id: SessionId) -> SimpleNamespace:
    """Return an AgentCore context object for the given session ID."""
    context = _session_contexts.get(session_id)
    if context is None:
        settings = load_settings()
        context = SimpleNamespace(
            session_id=session_id,
            actor_id=settings.actor_id,
            identity={"actor_id": settings.actor_id},
            memory=SessionMemory(),
        )
        _session_contexts[session_id] = context
    return context


async def healthcheck(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def create_session(_: Request) -> JSONResponse:
    session_id = str(uuid.uuid4())
    context = _ensure_context(session_id)
    intro = invoke({}, context=context)
    return JSONResponse(
        {
            "session_id": session_id,
            "message": intro.get("response", INTRO_MESSAGE),
        }
    )


async def chat(request: Request) -> JSONResponse:
    payload = await request.json()
    session_id: Optional[str] = payload.get("session_id")
    prompt: Optional[str] = payload.get("prompt")
    symbol: Optional[str] = payload.get("symbol")

    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    context = _ensure_context(session_id)

    invoke_payload: Dict[str, Any] = {"prompt": prompt}
    if symbol:
        invoke_payload["symbol"] = symbol

    result = invoke(invoke_payload, context=context)
    return JSONResponse({"response": result.get("response")})


async def list_vectorstores(_: Request) -> JSONResponse:
    summaries = _discover_vectorstore_summaries()
    response_data = [
        {
            "path": str(summary.path),
            "label": summary.label,
            "form": summary.form,
            "filing_date": summary.filing_date,
            "cik": summary.cik,
            "source_url": summary.source_url,
            "description": summary.describe(),
        }
        for summary in summaries
    ]
    return JSONResponse({"items": response_data})


async def query_vectorstore_route(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON payload."}, status_code=400)

    path_value = payload.get("path")
    question = payload.get("question")

    if not path_value:
        return JSONResponse({"error": "path is required"}, status_code=400)
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    vector_path = Path(path_value).resolve()
    if not vector_path.is_file():
        return JSONResponse({"error": "Vectorstore path not found."}, status_code=404)
    if VECTORSTORE_DIR.resolve() not in vector_path.parents:
        return JSONResponse({"error": "Vectorstore path is not authorized."}, status_code=403)

    settings = load_settings()
    try:
        answer = query_vectorstore(vector_path, question, settings=settings)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"answer": answer})


async def serve_index(_: Request) -> Response:
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            {"error": "Frontend assets not found. Build the frontend first."},
            status_code=500,
        )
    return FileResponse(index_path)


def create_app() -> Starlette:
    routes = [
        Route("/health", healthcheck, methods=["GET"]),
        Route("/api/session", create_session, methods=["POST"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/vectorstores", list_vectorstores, methods=["GET"]),
        Route("/api/vectorstores/query", query_vectorstore_route, methods=["POST"]),
    ]

    if _FRONTEND_DIR.exists():
        routes.append(
            Mount(
                "/",
                StaticFiles(directory=str(_FRONTEND_DIR), html=True),
                name="frontend",
            )
        )
    else:
        routes.append(Route("/", serve_index, methods=["GET"]))

    app = Starlette(routes=routes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.web:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

