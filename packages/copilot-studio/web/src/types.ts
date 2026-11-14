export type CompletionMode = "sft" | "l2r" | "fim";

export interface ModelCompletionResult {
  key: string;
  label: string;
  stage: string;
  text: string;
  valid_syntax: boolean;
  latency_ms: number;
  tokens_generated: number;
}

export interface CompleteResponse {
  mode: CompletionMode;
  models: ModelCompletionResult[];
}

export interface PresetItem {
  id: string;
  title: string;
  mode: CompletionMode;
  prefix: string;
  suffix: string;
}

export interface StudioConfig {
  apiBaseUrl?: string;
  colabCallbackName?: string;
  presets: PresetItem[];
}

declare global {
  interface Window {
    __MINIMIND_STUDIO_CONFIG__?: StudioConfig;
    google?: {
      colab?: {
        kernel?: {
          invokeFunction: (
            name: string,
            args: unknown[],
            kwargs: Record<string, unknown>
          ) => Promise<{ data: Record<string, unknown> }>;
        };
      };
    };
    monaco?: any;
    require?: any;
  }
}
