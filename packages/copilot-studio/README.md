# `@minimind/copilot-studio` (`minimind-copilot-studio`)

Full-stack interactive IDE playground for MiniMind code completion models:

- **`web/` (Frontend)**: React 18 + TypeScript + Vite + VS Code Monaco Editor (`src/App.tsx`, `src/MonacoEditor.tsx`). Compiled production bundle lives in `web/dist/`.
- **`src/minimind_copilot_studio/` (Python Backend)**:
  - `engine.py`: `CopilotInferenceEngine` managing multiple PyTorch checkpoints (`pretrain_768`, `code_pretrain_768`, `code_sft_768`, `lora_code_768`) with AST syntax verification (`ast.parse`) and latency profiling.
  - `server.py`: FastAPI REST backend (`/api/complete`, `/api/health`) + static SPA hosting.
  - `notebook.py`: `mount_copilot_studio(engine)` for mounting the compiled React + Monaco application directly inside Google Colab / Jupyter.
