import React, { useEffect, useRef } from "react";

interface MonacoEditorProps {
  value: string;
  onChange: (nextValue: string) => void;
  onTriggerComplete: () => void;
  height?: number;
}

const MONACO_LOADER_URL =
  "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js";
const MONACO_VS_PATH =
  "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs";

function ensureMonacoLoaded(): Promise<any> {
  if (window.monaco && window.monaco.editor) {
    return Promise.resolve(window.monaco);
  }
  return new Promise((resolve, reject) => {
    const initRequire = () => {
      window.require.config({ paths: { vs: MONACO_VS_PATH } });
      window.require(["vs/editor/editor.main"], () => {
        resolve(window.monaco);
      });
    };
    if (window.require) {
      initRequire();
      return;
    }
    const script = document.createElement("script");
    script.src = MONACO_LOADER_URL;
    script.async = true;
    script.onload = initRequire;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export const MonacoEditor: React.FC<MonacoEditorProps> = ({
  value,
  onChange,
  onTriggerComplete,
  height = 270,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const editorRef = useRef<any>(null);
  const onChangeRef = useRef(onChange);
  const onTriggerRef = useRef(onTriggerComplete);
  onChangeRef.current = onChange;
  onTriggerRef.current = onTriggerComplete;

  useEffect(() => {
    let disposed = false;
    ensureMonacoLoaded().then((monaco) => {
      if (disposed || !containerRef.current) return;
      const editor = monaco.editor.create(containerRef.current, {
        value,
        language: "python",
        theme: "vs-dark",
        minimap: { enabled: false },
        fontSize: 13,
        lineNumbers: "on",
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 4,
        insertSpaces: true,
        padding: { top: 10, bottom: 10 },
      });
      editorRef.current = editor;

      editor.onDidChangeModelContent(() => {
        onChangeRef.current(editor.getValue());
      });

      editor.addCommand(
        monaco.KeyMod.Alt | monaco.KeyCode.Enter,
        () => onTriggerRef.current()
      );
    });

    return () => {
      disposed = true;
      if (editorRef.current) {
        editorRef.current.dispose();
        editorRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const editor = editorRef.current;
    if (editor && editor.getValue() !== value) {
      editor.setValue(value);
    }
  }, [value]);

  return (
    <div
      ref={containerRef}
      className="mm-monaco-box"
      style={{ height: `${height}px` }}
    />
  );
};
