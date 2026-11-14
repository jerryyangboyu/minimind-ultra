"""FastAPI HTTP server for MiniMind Copilot Studio.

Can be launched standalone (`uvicorn minimind_copilot_studio.server:app`) or in a background
daemon thread inside a notebook (`start_background_server(engine)`).
"""

from __future__ import annotations

from pathlib import Path
import socket
import threading
from typing import Any

from .engine import CopilotInferenceEngine

WEB_DIST_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"


def create_app(engine: CopilotInferenceEngine) -> Any:
    """Create a FastAPI application serving `/api/complete`, `/api/health`, and the React SPA."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    class CompleteRequest(BaseModel):
        prefix: str
        mode: str = "sft"
        suffix: str = ""
        max_new_tokens: int = 64

    app = FastAPI(title="MiniMind Copilot Studio API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoints": [
                {"key": c.key, "label": c.label, "stage": c.stage}
                for c in engine.checkpoints
            ],
        }

    @app.post("/api/complete")
    def complete(req: CompleteRequest) -> JSONResponse:
        res = engine.complete_all(
            prefix=req.prefix,
            mode=req.mode,
            suffix=req.suffix,
            max_new_tokens=req.max_new_tokens,
        )
        return JSONResponse(res)

    if (WEB_DIST_DIR / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(WEB_DIST_DIR / "assets")),
            name="assets",
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIST_DIR / "index.html"))

    return app


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_background_server(
    engine: CopilotInferenceEngine, host: str = "127.0.0.1", port: int | None = None
) -> str:
    """Start the FastAPI server in a background daemon thread and return its base URL."""
    import uvicorn

    chosen_port = port or _find_free_port()
    app = create_app(engine)
    config = uvicorn.Config(
        app, host=host, port=chosen_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return f"http://{host}:{chosen_port}"
