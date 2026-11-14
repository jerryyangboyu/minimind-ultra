"""Mount the compiled React + Monaco Copilot Studio app inside Google Colab or Jupyter."""

from __future__ import annotations

import json
from pathlib import Path
import uuid
from typing import Any

from .engine import CopilotInferenceEngine

WEB_DIST_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"


def mount_copilot_studio(
    engine: CopilotInferenceEngine,
    presets: list[dict[str, str]] | None = None,
    api_base_url: str | None = None,
) -> None:
    """Render the compiled React + TypeScript + Monaco IDE bundle in a notebook cell.

    In Google Colab, it registers a kernel RPC bridge (`google.colab.output.register_callback`)
    so the React frontend can invoke `engine.complete_all(...)` directly on the GPU kernel,
    and also supports talking to the FastAPI HTTP server (`api_base_url`).
    """
    from IPython.display import HTML, JSON, display

    cb_name = f"minimind_studio_rpc_{uuid.uuid4().hex[:8]}"

    def _rpc_handler(prefix: str, mode: str = "sft", suffix: str = "") -> Any:
        return JSON(engine.complete_all(prefix=prefix, mode=mode, suffix=suffix))

    try:
        from google.colab import output as colab_output  # type: ignore[import-not-found]
        colab_output.register_callback(cb_name, _rpc_handler)
    except Exception:
        pass

    css_text = (WEB_DIST_DIR / "assets" / "style.css").read_text(encoding="utf-8")
    js_text = (WEB_DIST_DIR / "assets" / "bundle.js").read_text(encoding="utf-8")

    studio_cfg: dict[str, Any] = {"colabCallbackName": cb_name}
    if api_base_url:
        studio_cfg["apiBaseUrl"] = api_base_url
    if presets:
        studio_cfg["presets"] = presets

    html_payload = f"""
    <style>{css_text}</style>
    <div id="minimind-copilot-root"></div>
    <script>
      window.__MINIMIND_STUDIO_CONFIG__ = {json.dumps(studio_cfg)};
    </script>
    <script type="module">
    {js_text}
    </script>
    """
    display(HTML(html_payload))
