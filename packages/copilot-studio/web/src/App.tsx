import React, { useCallback, useEffect, useState } from "react";
import { MonacoEditor } from "./MonacoEditor";
import {
  CompleteResponse,
  CompletionMode,
  ModelCompletionResult,
  PresetItem,
} from "./types";

const DEFAULT_PRESETS: PresetItem[] = [
  {
    id: "factorial",
    title: "1. Factorial (Docstring → Body)",
    mode: "sft",
    prefix:
      'def factorial(n):\n    """Calculate the factorial of a non-negative integer n."""\n    ',
    suffix: "",
  },
  {
    id: "reverse_string",
    title: "2. Reverse String (Docstring → Body)",
    mode: "sft",
    prefix: 'def reverse_string(s):\n    """Reverse a given string."""\n    ',
    suffix: "",
  },
  {
    id: "remove_duplicates",
    title: "3. Remove Duplicates Preserving Order",
    mode: "sft",
    prefix:
      'def remove_duplicates(items):\n    """Remove duplicate elements from a list while preserving order."""\n    ',
    suffix: "",
  },
  {
    id: "matmul_bare",
    title: "4. Matrix Multiply (Bare Signature, OOD)",
    mode: "sft",
    prefix: "def matmul(mat1, mat2):\n    ",
    suffix: "",
  },
  {
    id: "fim_is_even",
    title: "5. Fill-in-the-Middle (Cursor Inside Function)",
    mode: "fim",
    prefix:
      'def is_even(n):\n    """Return True if n is even, else False."""\n    return ',
    suffix: "\n\nprint(is_even(4))\n",
  },
];

async function requestCompletion(
  prefix: string,
  mode: CompletionMode,
  suffix: string
): Promise<CompleteResponse> {
  const cfg = window.__MINIMIND_STUDIO_CONFIG__;
  const cbName = cfg?.colabCallbackName;

  if (
    cbName &&
    window.google?.colab?.kernel?.invokeFunction
  ) {
    const res = await window.google.colab.kernel.invokeFunction(
      cbName,
      [prefix, mode, suffix],
      {}
    );
    const payload =
      (res.data["application/json"] as CompleteResponse) ||
      JSON.parse(String(res.data["text/plain"] || "{}"));
    return payload;
  }

  const baseUrl = cfg?.apiBaseUrl || "";
  const resp = await fetch(`${baseUrl}/api/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prefix, mode, suffix }),
  });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return (await resp.json()) as CompleteResponse;
}

export const App: React.FC = () => {
  const presets =
    window.__MINIMIND_STUDIO_CONFIG__?.presets || DEFAULT_PRESETS;
  const [selectedPresetId, setSelectedPresetId] = useState<string>(
    presets[0].id
  );
  const [mode, setMode] = useState<CompletionMode>(presets[0].mode);
  const [code, setCode] = useState<string>(presets[0].prefix);
  const [suffix, setSuffix] = useState<string>(presets[0].suffix);
  const [models, setModels] = useState<ModelCompletionResult[]>([]);
  const [activeModelKey, setActiveModelKey] = useState<string>("sft");
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("Ready");

  const runInference = useCallback(
    async (curCode: string, curMode: CompletionMode, curSuffix: string) => {
      setLoading(true);
      setStatus("Running GPU inference across 4 checkpoints...");
      try {
        const res = await requestCompletion(curCode, curMode, curSuffix);
        setModels(res.models || []);
        setStatus("Live GPU completions updated");
      } catch (err: any) {
        setStatus(`Inference error: ${err?.message || String(err)}`);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    const timer = setTimeout(() => {
      runInference(code, mode, suffix);
    }, 350);
    return () => clearTimeout(timer);
  }, []);

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setSelectedPresetId(id);
    const found = presets.find((p) => p.id === id);
    if (found) {
      setCode(found.prefix);
      setMode(found.mode);
      setSuffix(found.suffix);
      runInference(found.prefix, found.mode, found.suffix);
    }
  };

  const handleAcceptIntoEditor = (m: ModelCompletionResult) => {
    if (!m.text) return;
    if (mode === "fim") {
      setCode(code + m.text + suffix);
    } else {
      const cleanPrefix = code.replace(/[ \t]+$/, "");
      const sep =
        cleanPrefix.endsWith("\n") || m.text.startsWith("\n") ? "" : "\n";
      setCode(cleanPrefix + sep + m.text + "\n");
    }
  };

  const selectedModel =
    models.find((m) => m.key === activeModelKey) || models[0];

  return (
    <div className="mm-studio-shell">
      <header className="mm-studio-header">
        <div>
          <div className="mm-studio-title">
            MiniMind Copilot Studio{" "}
            <span className="mm-chip mm-chip-accent">React + Monaco IDE</span>
          </div>
          <div className="mm-studio-subtitle">
            Compare General Base, Code Base, Full-FT SFT, and LoRA checkpoints live on your GPU (Press <code>Alt+Enter</code> in editor)
          </div>
        </div>
        <div className="mm-studio-controls">
          <select
            className="mm-select"
            value={selectedPresetId}
            onChange={handlePresetChange}
          >
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.title}
              </option>
            ))}
          </select>

          <select
            className="mm-select"
            value={mode}
            onChange={(e) => setMode(e.target.value as CompletionMode)}
          >
            <option value="sft">Mode: Docstring → Body (SFT / LoRA)</option>
            <option value="l2r">Mode: Left-to-Right Continuation</option>
            <option value="fim">Mode: Fill-in-the-Middle (FIM)</option>
          </select>

          <button
            className="mm-btn mm-btn-primary"
            disabled={loading}
            onClick={() => runInference(code, mode, suffix)}
          >
            {loading ? "⏳ Running..." : "✨ Complete Code"}
          </button>
        </div>
      </header>

      <div className="mm-studio-body">
        <div className="mm-editor-pane">
          <div className="mm-pane-bar">
            <span>VS Code Monaco Editor (Python 3)</span>
            <span className="mm-status-text">{status}</span>
          </div>
          <MonacoEditor
            value={code}
            onChange={setCode}
            onTriggerComplete={() => runInference(code, mode, suffix)}
            height={mode === "fim" ? 215 : 285}
          />
          {mode === "fim" && (
            <div className="mm-fim-suffix-box">
              <div className="mm-pane-bar">
                <span>FIM Suffix (Code Below Cursor)</span>
              </div>
              <textarea
                className="mm-suffix-textarea"
                value={suffix}
                onChange={(e) => setSuffix(e.target.value)}
              />
            </div>
          )}
        </div>

        <div className="mm-results-pane">
          <div className="mm-pane-bar">
            <span>Model Checkpoints (Click a tab or card to inspect / accept)</span>
            {selectedModel && selectedModel.text && (
              <button
                className="mm-btn mm-btn-accept"
                onClick={() => handleAcceptIntoEditor(selectedModel)}
              >
                ⇥ Accept {selectedModel.label}
              </button>
            )}
          </div>

          <div className="mm-cards-stack">
            {models.map((m) => {
              const isSelected = m.key === activeModelKey;
              return (
                <div
                  key={m.key}
                  className={`mm-model-card ${isSelected ? "mm-model-card-active" : ""}`}
                  onClick={() => setActiveModelKey(m.key)}
                >
                  <div className="mm-model-card-header">
                    <span className="mm-model-name">{m.label}</span>
                    <div className="mm-badges">
                      <span className="mm-chip">{m.stage}</span>
                      <span
                        className={`mm-chip ${
                          m.valid_syntax ? "mm-chip-ok" : "mm-chip-err"
                        }`}
                      >
                        {m.valid_syntax ? "✓ AST Valid" : "✗ Syntax Error"}
                      </span>
                      <span className="mm-chip">{m.latency_ms} ms</span>
                    </div>
                  </div>
                  <pre className="mm-model-code">
                    {m.text || "(empty completion)"}
                  </pre>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
