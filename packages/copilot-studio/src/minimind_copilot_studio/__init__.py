"""MiniMind Copilot Studio — Full-stack Python Backend + React/Monaco IDE Frontend.

Architecture:
- `engine.py`: PyTorch inference & AST validation service managing multi-stage checkpoints
  (General Base, Code Base, Code SFT, LoRA).
- `server.py`: FastAPI HTTP server exposing `/api/complete`, `/api/health`, and serving the
  compiled React + TypeScript + Monaco IDE frontend (`web/dist`).
- `notebook.py`: Google Colab & Jupyter integration that mounts the compiled React bundle
  and bridges it to the live Python kernel (or local FastAPI server).
"""

from .engine import CheckpointSpec, CopilotInferenceEngine
from .notebook import mount_copilot_studio
from .server import create_app, start_background_server

__all__ = [
    "CheckpointSpec",
    "CopilotInferenceEngine",
    "create_app",
    "mount_copilot_studio",
    "start_background_server",
]
