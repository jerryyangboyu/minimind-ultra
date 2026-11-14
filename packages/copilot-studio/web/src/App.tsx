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
    title: "1. Factorial (Cursor at End → Auto Completion)",
    prefix:
      'def factorial(n):\n    """Calculate the factorial of a non-negative integer n."""\n    ',
    suffix: "",
  },
  {
    id: "reverse_string",
    title: "2. Reverse String (Cursor at End → Auto Completion)",
    prefix: 'def reverse_string(s):\n    """Reverse a given string."""\n    ',
    suffix: "",
  },
  {
    id: "remove_duplicates",
    title: "3. Remove Duplicates (Cursor at End → Auto Completion)",
    prefix:
      'def remove_duplicates(items):\n    """Remove duplicate elements from a list while preserving order."""\n    ',
    suffix: "",
  },
  {
    id: "matmul_bare",
    title: "4. Matrix Multiply (Bare Signature → Auto Completion)",
    prefix: "def matmul(mat1, mat2):\n    ",
    suffix: "",
  },
  {
    id: "fim_is_even",
    title: "5. Fill-in-the-Middle: is_even (Cursor Inside Function → Auto FIM)",
    prefix:
      'def is_even(n):\n    """Return True if n is even, else False."""\n    return ',
    suffix: "\n\nprint(is_even(4))\n",
  },
  {
    id: "fim_factorial",
    title: "6. Fill-in-the-Middle: factorial Base Case (Cursor in Middle → Auto FIM)",
    prefix: "def factorial(n):\n    if n == 0:\n        ",
    suffix: "\n    return n * factorial(n - 1)\n",
  },
];

async function requestCompletion(
  prefix: string,
  mode: CompletionMode,
  suffix: string
): Promise<CompleteResponse> {
  const cfg = window.__MINIMIND_STUDIO_CONFIG__;
  const cbName = cfg?.colabCallbackName;

  if (cbName && window.google?.colab?.kernel?.invokeFunction) {
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
  const initialDoc = presets[0].prefix + (presets[0].suffix || "");
  const initialOffset = presets[0].prefix.length;

  const [selectedPresetId, setSelectedPresetId] = useState<string>(
    presets[0].id
  );
  const [code, setCode] = useState<string>(initialDoc);
  const [cursorOffset, setCursorOffset] = useState<number>(initialOffset);
  const [cursorLineCol, setCursorLineCol] = useState<{ line: number; col: number }>({
    line: presets[0].prefix.split("\n").length,
    col: (presets[0].prefix.split("\n").pop()?.length || 0) + 1,
  });
  const [models, setModels] = useState<ModelCompletionResult[]>([]);
  const [activeModelKey, setActiveModelKey] = useState<string>("sft");
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("Ready");

  const prefix = code.slice(0, cursorOffset);
  const suffix = code.slice(cursorOffset);
  const detectedMode: "completion" | "fim" =
    suffix.trim().length > 0 ? "fim" : "completion";

  const runInference = useCallback(
    async (curDoc: string, curOffset: number) => {
      const curPrefix = curDoc.slice(0, curOffset);
      const curSuffix = curDoc.slice(curOffset);
      const autoMode: CompletionMode =
        curSuffix.trim().length > 0 ? "fim" : "completion";
      setLoading(true);
      setStatus(
        `Running GPU inference (${
          autoMode === "fim" ? "Fill-in-the-Middle" : "Completion"
        } at cursor)...`
      );
      try {
        const res = await requestCompletion(curPrefix, autoMode, curSuffix);
        setModels(res.models || []);
        setStatus(
          `Updated • Auto Mode: ${
            autoMode === "fim" ? "Fill-in-the-Middle (FIM)" : "Completion"
          }`
        );
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
      runInference(code, cursorOffset);
    }, 350);
    return () => clearTimeout(timer);
  }, []);

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setSelectedPresetId(id);
    const found = presets.find((p) => p.id === id);
    if (found) {
      const fullDoc = found.prefix + (found.suffix || "");
      const targetOffset = found.prefix.length;
      const lines = found.prefix.split("\n");
      setCode(fullDoc);
      setCursorOffset(targetOffset);
      setCursorLineCol({
        line: lines.length,
        col: (lines[lines.length - 1]?.length || 0) + 1,
      });
      runInference(fullDoc, targetOffset);
    }
  };

  const handleAcceptIntoEditor = (m: ModelCompletionResult) => {
    if (!m.text) return;
    if (detectedMode === "fim") {
      const nextDoc = prefix + m.text + suffix;
      const nextOffset = prefix.length + m.text.length;
      setCode(nextDoc);
      setCursorOffset(nextOffset);
    } else {
      const cleanPrefix = prefix.replace(/[ \t]+$/, "");
      const sep =
        cleanPrefix.endsWith("\n") || m.text.startsWith("\n") ? "" : "\n";
      const inserted = cleanPrefix + sep + m.text + "\n";
      setCode(inserted + suffix);
      setCursorOffset(inserted.length);
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
            Move your cursor anywhere and press <code>Alt+Enter</code> (or click Complete) — mode is auto-judged by cursor position
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

          <span
            className={`mm-mode-indicator ${
              detectedMode === "fim"
                ? "mm-mode-indicator-fim"
                : "mm-mode-indicator-comp"
            }`}
            title="Automatically detected from cursor position: if code exists below the cursor, Fill-in-the-Middle (FIM) is used; otherwise End-of-File Completion is used."
          >
            {detectedMode === "fim"
              ? "⚡ Auto Mode: Fill-in-the-Middle (FIM)"
              : "⚡ Auto Mode: Completion"}
          </span>

          <button
            className="mm-btn mm-btn-primary"
            disabled={loading}
            onClick={() => runInference(code, cursorOffset)}
          >
            {loading ? "⏳ Running..." : "✨ Complete at Cursor"}
          </button>
        </div>
      </header>

      <div className="mm-studio-body">
        <div className="mm-editor-pane">
          <div className="mm-pane-bar">
            <span>
              VS Code Monaco Editor •{" "}
              <code>
                Ln {cursorLineCol.line}, Col {cursorLineCol.col}
              </code>{" "}
              ({suffix.trim().length > 0 ? "code below cursor → FIM" : "cursor at end → Completion"})
            </span>
            <span className="mm-status-text">{status}</span>
          </div>
          <MonacoEditor
            value={code}
            cursorOffset={cursorOffset}
            onChange={(nextVal, nextOff) => {
              setCode(nextVal);
              setCursorOffset(nextOff);
            }}
            onCursorChange={(off, line, col) => {
              setCursorOffset(off);
              setCursorLineCol({ line, col });
            }}
            onTriggerComplete={() => runInference(code, cursorOffset)}
            height={300}
          />
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
