"""Rich visual & interactive teaching toolkit for the MiniMind course (Lesson 2: Domain Code Model).

Provides:
- Styled HTML cards & comparison tables for IPython / Google Colab
- Tokenization color-chip visualizer (preserving whitespace & indentation)
- Interactive Charts (Data Cleaning Funnel, Specialization vs. Forgetting Perplexity, Radar / Bar charts)
- Interactive Monaco Editor Copilot Playground connected to the Colab Python kernel via google.colab.kernel
"""

from __future__ import annotations

import html
import json
import uuid
from typing import Any, Callable

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover
    def display(obj: Any) -> None:
        print(obj)

    class HTML(str):  # type: ignore[no-redef]
        pass


_BASE_CSS = """
<style>
.mm-card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
  background: #0f172a; color: #e2e8f0; border: 1px solid #1e293b;
  border-radius: 12px; padding: 16px 20px; margin: 10px 0;
  box-shadow: 0 4px 16px rgba(0,0,0,0.28);
}
.mm-title {
  font-size: 15px; font-weight: 700; color: #38bdf8; letter-spacing: 0.3px;
  margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}
.mm-sub { font-size: 12.5px; color: #94a3b8; margin-bottom: 12px; }
.mm-grid { display: grid; gap: 12px; }
.mm-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 600; background: #1e293b; color: #7dd3fc;
  border: 1px solid #334155;
}
.mm-badge-ok { background: rgba(16,185,129,0.15); color: #34d399; border-color: rgba(16,185,129,0.35); }
.mm-badge-warn { background: rgba(245,158,11,0.15); color: #fbbf24; border-color: rgba(245,158,11,0.35); }
.mm-badge-err { background: rgba(239,68,68,0.15); color: #f87171; border-color: rgba(239,68,68,0.35); }
.mm-code {
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 12.5px; line-height: 1.55; background: #020617;
  border: 1px solid #1e293b; border-radius: 8px; padding: 10px 12px;
  white-space: pre-wrap; overflow-x: auto; color: #cbd5e1;
}
.mm-gen { color: #4ade80; background: rgba(74, 222, 128, 0.10); border-radius: 3px; padding: 0 2px; font-weight: 600; }
.mm-fim-tag { color: #f472b6; font-weight: 700; }
.mm-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
.mm-table th {
  text-align: left; padding: 8px 10px; border-bottom: 1px solid #334155;
  color: #94a3b8; font-weight: 600; font-size: 12px; text-transform: uppercase;
}
.mm-table td { padding: 9px 10px; border-bottom: 1px solid #1e293b; vertical-align: middle; }
.mm-bar-bg { width: 100%; height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }
.mm-bar-fill { height: 100%; border-radius: 4px; }
</style>
"""


def render_env_banner(python_ver: str, torch_ver: str, gpu_name: str, dtype: str, ckpts: list[str], datasets: list[str]) -> None:
    """Display a clean hardware & assets status banner."""
    ckpt_chips = " ".join(f'<span class="mm-badge mm-badge-ok">📦 {html.escape(c)}</span>' for c in ckpts)
    ds_chips = " ".join(f'<span class="mm-badge">📄 {html.escape(d)}</span>' for d in datasets)
    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card">
      <div class="mm-title">⚡ MiniMind Lesson 2 — Runtime & Assets Ready</div>
      <div class="mm-grid" style="grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 12px;">
        <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
          <div style="font-size:11px;color:#64748b;">GPU ACCELERATOR</div>
          <div style="font-size:14px;font-weight:700;color:#38bdf8;margin-top:2px;">{html.escape(gpu_name)}</div>
        </div>
        <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
          <div style="font-size:11px;color:#64748b;">MIXED PRECISION</div>
          <div style="font-size:14px;font-weight:700;color:#a78bfa;margin-top:2px;">{html.escape(dtype)}</div>
        </div>
        <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
          <div style="font-size:11px;color:#64748b;">PYTORCH / PYTHON</div>
          <div style="font-size:14px;font-weight:700;color:#e2e8f0;margin-top:2px;">{html.escape(torch_ver)} / {html.escape(python_ver)}</div>
        </div>
      </div>
      <div style="font-size:12px;color:#94a3b8;margin-bottom:6px;">Checkpoints in <code>/content/minimind/out</code>: {ckpt_chips}</div>
      <div style="font-size:12px;color:#94a3b8;">Datasets in <code>/content/minimind/dataset</code>: {ds_chips}</div>
    </div>
    """
    display(HTML(html_str))


def render_token_chips(snippet: str, token_ids: list[int], token_pieces: list[str], fim_tokens: dict[str, int] | None = None) -> None:
    """Visualize how Byte-Level BPE tokenizes Python code (highlighting whitespace & indentation tokens)."""
    palette = ["#1e3a8a", "#065f46", "#581c87", "#7c2d12", "#1e293b", "#312e81"]
    chips = []
    for idx, (tid, piece) in enumerate(zip(token_ids, token_pieces)):
        vis = piece.replace("\n", "↵\n").replace(" ", "·")
        is_ws = piece.strip() == "" or piece.startswith("\n")
        bg = "#0f766e" if is_ws else palette[idx % len(palette)]
        border = "#2dd4bf" if is_ws else "#475569"
        chips.append(
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
            f'background:{bg};border:1px solid {border};border-radius:6px;padding:3px 7px;margin:2px;">'
            f'<span style="font-family:monospace;font-size:12.5px;color:#f8fafc;white-space:pre;">{html.escape(vis)}</span>'
            f'<span style="font-size:9.5px;color:#cbd5e1;margin-top:1px;">id={tid}</span>'
            f'</span>'
        )
    fim_html = ""
    if fim_tokens:
        fim_badges = " ".join(
            f'<span class="mm-badge" style="color:#f472b6;border-color:#f472b655;">{html.escape(k)} → id={v}</span>'
            for k, v in fim_tokens.items()
        )
        fim_html = f'<div style="margin-top:10px;font-size:12px;color:#94a3b8;">Repurposed Single-Token FIM Sentinels: {fim_badges}</div>'

    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card">
      <div class="mm-title">🔬 Byte-Level BPE Tokenization Inspector (Lossless Whitespace & Indentation)</div>
      <div class="mm-sub">Notice how newline + 4-space indentation (<code>↵····</code>, teal) is encoded as a single dedicated token—critical for Python syntax!</div>
      <div style="display:flex;flex-wrap:wrap;align-items:center;background:#020617;padding:10px;border-radius:8px;border:1px solid #1e293b;">
        {''.join(chips)}
      </div>
      {fim_html}
    </div>
    """
    display(HTML(html_str))


def render_completions_grid(title: str, subtitle: str, items: list[dict[str, str]]) -> None:
    """Render code completions in a clean 2-column grid with highlighted generated code."""
    cards = []
    for it in items:
        mode = it.get("mode", "Left-to-Right")
        badge_cls = "mm-badge-warn" if "FIM" in mode else "mm-badge"
        pre = html.escape(it["prefix"])
        gen = html.escape(it["completion"])
        suf = html.escape(it.get("suffix", ""))
        body = f'{pre}<span class="mm-gen">{gen}</span>{suf}'
        cards.append(
            f'<div style="background:#020617;border:1px solid #1e293b;border-radius:8px;padding:10px 12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span style="font-size:12px;font-weight:600;color:#cbd5e1;">{html.escape(it.get("label", "Probe"))}</span>'
            f'<span class="mm-badge {badge_cls}">{html.escape(mode)}</span>'
            f'</div>'
            f'<div class="mm-code" style="border:none;padding:4px 0;background:transparent;">{body}</div>'
            f'</div>'
        )
    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card">
      <div class="mm-title">{html.escape(title)}</div>
      <div class="mm-sub">{html.escape(subtitle)} (<span class="mm-gen">green</span> = model-generated tokens)</div>
      <div class="mm-grid" style="grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));">
        {''.join(cards)}
      </div>
    </div>
    """
    display(HTML(html_str))


def render_funnel_chart(stages: list[tuple[str, int]], sample_code: str, fim_sample: str) -> None:
    """Render an interactive horizontal funnel bar chart for the data cleaning pipeline plus a FIM preview."""
    max_val = max(v for _, v in stages) if stages else 1
    rows = []
    for name, count in stages:
        pct = 100.0 * count / max_val
        rows.append(
            f'<tr>'
            f'<td style="width:36%;font-weight:500;color:#e2e8f0;">{html.escape(name)}</td>'
            f'<td style="width:48%;">'
            f'  <div class="mm-bar-bg"><div class="mm-bar-fill" style="width:{pct:.1f}%;background:linear-gradient(90deg,#38bdf8,#818cf8);"></div></div>'
            f'</td>'
            f'<td style="width:16%;text-align:right;font-family:monospace;color:#38bdf8;">{count:,} ({pct:.0f}%)</td>'
            f'</tr>'
        )
    fim_pretty = (
        html.escape(fim_sample)
        .replace("&lt;|buffer1|&gt;", '<span class="mm-fim-tag">&lt;FIM_PREFIX&gt;</span>')
        .replace("&lt;|buffer2|&gt;", '<span class="mm-fim-tag">&lt;FIM_SUFFIX&gt;</span>')
        .replace("&lt;|buffer3|&gt;", '<span class="mm-fim-tag">&lt;FIM_MIDDLE&gt;</span>')
    )
    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card">
      <div class="mm-title">🧹 Domain Corpus Curation Funnel & FIM Transformation</div>
      <table class="mm-table">{''.join(rows)}</table>
      <div class="mm-grid" style="grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); margin-top: 14px;">
        <div>
          <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">✨ Curated Left-to-Right Sample (AST-Normalized PEP-8):</div>
          <div class="mm-code">{html.escape(sample_code.strip())}</div>
        </div>
        <div>
          <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">🧩 Transformed FIM Sample (Prefix–Suffix–Middle):</div>
          <div class="mm-code">{fim_pretty.strip()}</div>
        </div>
      </div>
    </div>
    """
    display(HTML(html_str))


def render_three_stage_comparison(
    bench_rows: list[dict[str, Any]],
    side_by_side: list[dict[str, str]],
) -> None:
    """Render the quantitative benchmark table + 3-column side-by-side qualitative code cards."""
    tr_list = []
    for r in bench_rows:
        rate = float(r["valid_pct"])
        color = "#34d399" if rate >= 80 else ("#fbbf24" if rate >= 50 else "#f87171")
        tr_list.append(
            f'<tr>'
            f'<td style="font-weight:600;color:#f8fafc;">{html.escape(r["stage"])}</td>'
            f'<td style="font-family:monospace;color:#38bdf8;">{html.escape(str(r.get("code_ppl", "—")))}</td>'
            f'<td>'
            f'  <div style="display:flex;align-items:center;gap:8px;">'
            f'    <div class="mm-bar-bg" style="width:110px;"><div class="mm-bar-fill" style="width:{rate:.0f}%;background:{color};"></div></div>'
            f'    <span style="font-family:monospace;color:{color};font-weight:700;">{r["valid_str"]}</span>'
            f'  </div>'
            f'</td>'
            f'<td style="font-family:monospace;color:#cbd5e1;">{r["avg_lines"]:.1f} lines</td>'
            f'</tr>'
        )

    demo_blocks = []
    for d in side_by_side:
        demo_blocks.append(
            f'<div style="margin-top:14px;border-top:1px solid #1e293b;padding-top:12px;">'
            f'<div style="font-size:12px;color:#94a3b8;margin-bottom:6px;">Prompt Signature & Docstring:</div>'
            f'<div class="mm-code" style="margin-bottom:8px;color:#f8fafc;">{html.escape(d["prompt"].strip())}</div>'
            f'<div class="mm-grid" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));">'
            f'  <div><div style="font-size:11.5px;color:#f87171;margin-bottom:4px;">① General Base (pretrain_768)</div>'
            f'  <div class="mm-code">{html.escape(d["gen"].strip())}</div></div>'
            f'  <div><div style="font-size:11.5px;color:#fbbf24;margin-bottom:4px;">② Code Base (code_pretrain_768)</div>'
            f'  <div class="mm-code">{html.escape(d["base"].strip())}</div></div>'
            f'  <div><div style="font-size:11.5px;color:#34d399;margin-bottom:4px;">③ Code SFT (code_sft_768) ✅</div>'
            f'  <div class="mm-code"><span class="mm-gen">{html.escape(d["sft"].rstrip())}</span></div></div>'
            f'</div></div>'
        )

    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card">
      <div class="mm-title">🏆 Three-Stage Code Completion Benchmark (50 Held-Out Functions)</div>
      <table class="mm-table">
        <thead><tr><th>Model Stage</th><th>Python Code PPL (↓)</th><th>Valid Python 3 Syntax (↑)</th><th>Avg Output Length</th></tr></thead>
        <tbody>{''.join(tr_list)}</tbody>
      </table>
      {''.join(demo_blocks)}
    </div>
    """
    display(HTML(html_str))


def render_forgetting_pareto(rows: list[dict[str, Any]], general_probes: list[dict[str, str]]) -> None:
    """Visualize the Specialization vs. Catastrophic Forgetting trade-off across all 4 strategies."""
    tr_list = []
    for r in rows:
        u_ppl = float(r["unseen_ppl"])
        u_badge = "mm-badge-ok" if u_ppl < 30 else ("mm-badge-warn" if u_ppl < 200 else "mm-badge-err")
        tr_list.append(
            f'<tr>'
            f'<td style="font-weight:600;color:#f8fafc;">{html.escape(r["name"])}</td>'
            f'<td style="font-family:monospace;color:#94a3b8;">{html.escape(r["params"])}</td>'
            f'<td style="font-family:monospace;color:#38bdf8;font-weight:700;">{html.escape(r["valid_str"])}</td>'
            f'<td style="font-family:monospace;color:#a78bfa;">{r["code_ppl"]:.2f}</td>'
            f'<td><span class="mm-badge {u_badge}">PPL = {u_ppl:.2f}</span></td>'
            f'<td style="font-size:12px;color:#cbd5e1;">{html.escape(r["verdict"])}</td>'
            f'</tr>'
        )

    probe_cards = []
    for p in general_probes:
        probe_cards.append(
            f'<div style="background:#020617;border:1px solid #1e293b;border-radius:8px;padding:10px 12px;margin-top:8px;">'
            f'<div style="font-size:12px;color:#38bdf8;font-weight:600;margin-bottom:6px;">🌍 General Prompt: "{html.escape(p["prompt"])} ..."</div>'
            f'<div style="font-size:12px;line-height:1.6;">'
            f'  <div><span class="mm-badge">General Base</span> <span style="color:#cbd5e1;">{html.escape(p["base"])}</span></div>'
            f'  <div style="margin-top:4px;"><span class="mm-badge mm-badge-err">Full FT + Narrow Replay</span> <span style="color:#fca5a5;">{html.escape(p["full_sft"])}</span></div>'
            f'  <div style="margin-top:4px;"><span class="mm-badge mm-badge-ok">LoRA (Base Frozen)</span> <span style="color:#86efac;">{html.escape(p["lora"])}</span></div>'
            f'</div></div>'
        )

    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card">
      <div class="mm-title">⚖️ Specialization vs. Catastrophic Forgetting — Strategy Comparison</div>
      <div class="mm-sub">Comparing Naive Data Mixing against Weight-Space Merging (WiSE-FT) and Parameter-Efficient LoRA</div>
      <table class="mm-table">
        <thead>
          <tr>
            <th>Strategy</th><th>Trainable Params</th><th>Valid Python (↑)</th>
            <th>Code PPL (↓)</th><th>Unseen English PPL (↓)</th><th>Engineering Takeaway</th>
          </tr>
        </thead>
        <tbody>{''.join(tr_list)}</tbody>
      </table>
      <div style="margin-top:12px;font-size:12.5px;color:#94a3b8;font-weight:600;">Qualitative Audit on Non-Coding General English Prompts:</div>
      {''.join(probe_cards)}
    </div>
    """
    display(HTML(html_str))


def launch_monaco_copilot_playground(
    callbacks: dict[str, Callable[..., dict[str, Any]]],
    presets: list[dict[str, str]] | None = None,
) -> None:
    """Render an interactive VS Code Monaco Editor playground in Colab connected to live PyTorch models.

    `callbacks` should map action names ('complete_all') to a Python callable registered with
    `google.colab.output.register_callback`. When the student clicks 'Run Copilot Completion' or presses
    Alt+\\ inside the Monaco Editor, JavaScript invokes the Python kernel, runs all model stages, displays
    inline ghost-text in Monaco, and updates the side-by-side comparison cards + AST syntax check badge.
    """
    cb_name = f"minimind_copilot_cb_{uuid.uuid4().hex[:8]}"
    try:
        from google.colab import output as colab_output  # type: ignore[import-not-found]
        colab_output.register_callback(cb_name, callbacks["complete_all"])
    except Exception:
        pass

    if presets is None:
        presets = [
            {
                "title": "1. Factorial (Docstring → Body)",
                "mode": "sft",
                "code": 'def factorial(n):\n    """Calculate the factorial of a non-negative integer n."""\n    ',
                "suffix": "",
            },
            {
                "title": "2. Reverse String (Docstring → Body)",
                "mode": "sft",
                "code": 'def reverse_string(s):\n    """Reverse a given string."""\n    ',
                "suffix": "",
            },
            {
                "title": "3. Remove Duplicates Preserving Order",
                "mode": "sft",
                "code": 'def remove_duplicates(items):\n    """Remove duplicate elements from a list while preserving order."""\n    ',
                "suffix": "",
            },
            {
                "title": "4. Fill-in-the-Middle (Cursor Inside Function)",
                "mode": "fim",
                "code": 'def is_even(n):\n    """Return True if n is even, else False."""\n    return ',
                "suffix": "\n\nprint(is_even(4))\n",
            },
        ]

    presets_json = json.dumps(presets)
    uid = uuid.uuid4().hex[:8]

    html_str = f"""
    {_BASE_CSS}
    <div class="mm-card" style="padding:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
        <div>
          <div class="mm-title" style="margin-bottom:2px;">🚀 MiniMind Copilot Studio — Interactive Monaco Editor Playground</div>
          <div class="mm-sub" style="margin-bottom:0;">Edit Python code in the real VS Code Monaco Editor below, choose a preset or write your own function, and click <b>✨ Complete Code</b> (or press <code>Alt+Enter</code>) to query all trained checkpoints live on the GPU!</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <select id="mm-preset-{uid}" style="background:#020617;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:6px 10px;font-size:12.5px;"></select>
          <select id="mm-mode-{uid}" style="background:#020617;color:#38bdf8;border:1px solid #334155;border-radius:6px;padding:6px 10px;font-size:12.5px;">
            <option value="sft">Mode: Docstring → Body (SFT / LoRA)</option>
            <option value="l2r">Mode: Left-to-Right Continuation</option>
            <option value="fim">Mode: Fill-in-the-Middle (Prefix + Suffix)</option>
          </select>
          <button id="mm-run-{uid}" style="background:linear-gradient(135deg,#0284c7,#4f46e5);color:#fff;border:none;border-radius:6px;padding:7px 14px;font-size:12.5px;font-weight:600;cursor:pointer;">
            ✨ Complete Code (GPU)
          </button>
          <button id="mm-accept-{uid}" style="background:#065f46;color:#a7f3d0;border:1px solid #059669;border-radius:6px;padding:7px 12px;font-size:12px;font-weight:600;cursor:pointer;display:none;">
            ⇥ Accept Code SFT into Editor
          </button>
        </div>
      </div>

      <div class="mm-grid" style="grid-template-columns: 1.15fr 1fr; align-items: stretch;">
        <div>
          <div style="font-size:11.5px;color:#94a3b8;margin-bottom:4px;display:flex;justify-content:space-between;">
            <span>📝 <b>VS Code Monaco Editor</b> (Prefix / Cursor Context)</span>
            <span id="mm-status-{uid}" style="color:#38bdf8;">Ready</span>
          </div>
          <div id="mm-monaco-{uid}" style="height:260px;border:1px solid #334155;border-radius:8px;overflow:hidden;"></div>
          <div id="mm-suffix-wrap-{uid}" style="display:none;margin-top:8px;">
            <div style="font-size:11.5px;color:#f472b6;margin-bottom:4px;">🔻 <b>FIM Suffix</b> (Code Below Cursor):</div>
            <textarea id="mm-suffix-{uid}" style="width:100%;height:56px;background:#020617;color:#cbd5e1;border:1px solid #334155;border-radius:6px;padding:6px 8px;font-family:monospace;font-size:12px;"></textarea>
          </div>
        </div>

        <div style="display:flex;flex-direction:column;gap:8px;">
          <div style="font-size:11.5px;color:#94a3b8;">🤖 <b>Live Model Outputs</b> (Click a card to preview in editor):</div>
          <div id="mm-out-base-{uid}" class="mm-code" style="flex:1;min-height:54px;">Click "✨ Complete Code" to run inference...</div>
          <div id="mm-out-code-{uid}" class="mm-code" style="flex:1;min-height:54px;"></div>
          <div id="mm-out-sft-{uid}" class="mm-code" style="flex:1;min-height:54px;"></div>
          <div id="mm-out-lora-{uid}" class="mm-code" style="flex:1;min-height:54px;"></div>
        </div>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>
    <script>
    (function() {{
      const presets = {presets_json};
      const presetSel = document.getElementById("mm-preset-{uid}");
      const modeSel   = document.getElementById("mm-mode-{uid}");
      const runBtn    = document.getElementById("mm-run-{uid}");
      const acceptBtn = document.getElementById("mm-accept-{uid}");
      const statusEl  = document.getElementById("mm-status-{uid}");
      const sufWrap   = document.getElementById("mm-suffix-wrap-{uid}");
      const sufInput  = document.getElementById("mm-suffix-{uid}");
      let editor = null;
      let lastSftCompletion = "";

      presets.forEach((p, idx) => {{
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = p.title;
        presetSel.appendChild(opt);
      }});

      function updateModeUI() {{
        sufWrap.style.display = (modeSel.value === "fim") ? "block" : "none";
      }}
      modeSel.addEventListener("change", updateModeUI);

      require.config({{ paths: {{ 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }} }});
      require(['vs/editor/editor.main'], function() {{
        editor = monaco.editor.create(document.getElementById("mm-monaco-{uid}"), {{
          value: presets[0].code,
          language: 'python',
          theme: 'vs-dark',
          minimap: {{ enabled: false }},
          fontSize: 13,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          automaticLayout: true,
          padding: {{ top: 10, bottom: 10 }}
        }});

        presetSel.addEventListener("change", () => {{
          const p = presets[parseInt(presetSel.value)];
          editor.setValue(p.code);
          modeSel.value = p.mode;
          sufInput.value = p.suffix || "";
          updateModeUI();
          runCompletion();
        }});

        editor.addCommand(monaco.KeyMod.Alt | monaco.KeyCode.Enter, runCompletion);
        // Trigger initial completion automatically once loaded
        setTimeout(runCompletion, 300);
      }});

      function renderCard(elId, title, badgeHtml, codeText, highlight) {{
        const el = document.getElementById(elId);
        const safe = codeText.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
        const body = highlight ? `<span class="mm-gen">${{safe}}</span>` : safe;
        el.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-size:11px;font-weight:700;color:#94a3b8;">${{title}}</span>${{badgeHtml}}
        </div><div>${{body || '<i style="color:#64748b;">(empty completion — stopped immediately)</i>'}}</div>`;
      }}

      async function runCompletion() {{
        if (!editor) return;
        const code = editor.getValue();
        const mode = modeSel.value;
        const suffix = sufInput.value || "";
        statusEl.textContent = "⏳ Running GPU inference across 4 checkpoints...";
        runBtn.disabled = true;

        try {{
          if (typeof google !== "undefined" && google.colab && google.colab.kernel) {{
            const res = await google.colab.kernel.invokeFunction("{cb_name}", [code, mode, suffix], {{}});
            const data = res.data["application/json"] || JSON.parse(res.data["text/plain"]);
            const badge = (ok, ms) => `<span class="mm-badge ${{ok ? 'mm-badge-ok' : 'mm-badge-err'}}">${{ok ? '✓ AST Valid' : '✗ Syntax Err'}} · ${{ms}}ms</span>`;
            renderCard("mm-out-base-{uid}", "① General Base (pretrain_768)", badge(data.base.valid, data.base.ms), data.base.text, false);
            renderCard("mm-out-code-{uid}", "② Code Base (code_pretrain_768)", badge(data.code.valid, data.code.ms), data.code.text, false);
            renderCard("mm-out-sft-{uid}",  "③ Code SFT (code_sft_768)", badge(data.sft.valid, data.sft.ms), data.sft.text, true);
            renderCard("mm-out-lora-{uid}", "④ LoRA Adapter (lora_code_768, 0.61% params)", badge(data.lora.valid, data.lora.ms), data.lora.text, true);
            lastSftCompletion = data.sft.text;
            acceptBtn.style.display = lastSftCompletion ? "inline-block" : "none";
            statusEl.textContent = "✅ Live GPU completions updated";
          }} else {{
            statusEl.textContent = "⚠️ Colab kernel bridge unavailable outside Colab";
          }}
        }} catch (err) {{
          statusEl.textContent = "Error: " + err;
        }} finally {{
          runBtn.disabled = false;
        }}
      }}

      runBtn.addEventListener("click", runCompletion);
      acceptBtn.addEventListener("click", () => {{
        if (!editor || !lastSftCompletion) return;
        const cur = editor.getValue();
        const sep = (cur.endsWith("\\n") || lastSftCompletion.startsWith("\\n")) ? "" : "\\n";
        editor.setValue(cur + sep + lastSftCompletion + (modeSel.value === "fim" ? sufInput.value : ""));
        acceptBtn.style.display = "none";
      }});
    }})();
    </script>
    """
    display(HTML(html_str))
