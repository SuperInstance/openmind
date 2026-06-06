"""Rich HTML dashboard renderer for Jupyter notebooks."""

from typing import Optional


def _esc(text: str) -> str:
    """Escape HTML entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_analysis(result, decisions) -> str:
    """Render an analysis dashboard as HTML."""
    total = len(result.functions)
    tested = sum(1 for f in result.functions if f.has_tests)
    test_pct = (tested / total * 100) if total > 0 else 0

    # Decision breakdown
    decision_counts = {}
    for d in decisions.values():
        val = d.value if hasattr(d, "value") else str(d)
        decision_counts[val] = decision_counts.get(val, 0) + 1

    # Function table rows
    func_rows = ""
    sorted_funcs = sorted(result.functions, key=lambda f: len(f.calls) + len(f.called_by), reverse=True)
    for func in sorted_funcs[:20]:
        qualified = f"{func.module}.{func.name}"
        d = decisions.get(qualified)
        d_str = d.value if d else "—"
        color = {"hardcode": "#4CAF50", "model": "#2196F3", "hybrid": "#FF9800", "cached": "#9C27B0"}.get(d_str, "#999")
        func_rows += f"""
        <tr>
            <td>{_esc(func.name)}</td>
            <td style="color:#888">{_esc(func.module)}</td>
            <td>{len(func.calls)}</td>
            <td>{len(func.called_by)}</td>
            <td>{'✓' if func.has_tests else '✗'}</td>
            <td style="color:{color}; font-weight:bold">{_esc(d_str).upper()}</td>
        </tr>"""

    # Decision bar
    bar_parts = ""
    bar_colors = {"hardcode": "#4CAF50", "model": "#2196F3", "hybrid": "#FF9800", "cached": "#9C27B0"}
    for name, count in decision_counts.items():
        pct = count / max(len(decisions), 1) * 100
        bar_parts += f'<div style="background:{bar_colors.get(name, "#999")}; width:{pct}%; display:inline-block; text-align:center; color:white; padding:4px">{name.upper()} {pct:.0f}%</div>'

    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 900px;">
        <h2>🧠 OpenMind Analysis: {_esc(result.repo_url)}</h2>

        <div style="display:flex; gap:20px; margin-bottom:20px;">
            <div style="background:#f0f0f0; padding:15px; border-radius:8px; flex:1; text-align:center;">
                <div style="font-size:2em; font-weight:bold; color:#333">{total}</div>
                <div style="color:#666">Functions</div>
            </div>
            <div style="background:#f0f0f0; padding:15px; border-radius:8px; flex:1; text-align:center;">
                <div style="font-size:2em; font-weight:bold; color:#333">{len(result.classes)}</div>
                <div style="color:#666">Classes</div>
            </div>
            <div style="background:#f0f0f0; padding:15px; border-radius:8px; flex:1; text-align:center;">
                <div style="font-size:2em; font-weight:bold; color:#333">{test_pct:.0f}%</div>
                <div style="color:#666">Test Coverage</div>
            </div>
            <div style="background:#f0f0f0; padding:15px; border-radius:8px; flex:1; text-align:center;">
                <div style="font-size:2em; font-weight:bold; color:#333">{len(result.call_graph)}</div>
                <div style="color:#666">Call Graph Nodes</div>
            </div>
        </div>

        <h3>Tripartite Decisions</h3>
        <div style="display:flex; border-radius:4px; overflow:hidden; margin-bottom:20px;">
            {bar_parts}
        </div>

        <h3>Top Functions</h3>
        <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
            <tr style="background:#f5f5f5; text-align:left;">
                <th style="padding:8px;">Function</th>
                <th style="padding:8px;">Module</th>
                <th style="padding:8px;">Calls</th>
                <th style="padding:8px;">Called By</th>
                <th style="padding:8px;">Tested</th>
                <th style="padding:8px;">Decision</th>
            </tr>
            {func_rows}
        </table>
    </div>
    """
    return html


def render_search(query, results) -> str:
    """Render search results as HTML."""
    rows = ""
    for dv, score in results:
        pct = score * 100
        bar_color = "#4CAF50" if pct > 70 else "#FF9800" if pct > 40 else "#f44336"
        rows += f"""
        <tr>
            <td style="padding:8px;"><strong>{_esc(dv.function_name)}</strong></td>
            <td style="padding:8px; color:#888;">{_esc(dv.module)}</td>
            <td style="padding:8px;">
                <div style="background:#e0e0e0; border-radius:4px; overflow:hidden; width:100%;">
                    <div style="background:{bar_color}; width:{pct}%; padding:4px; color:white; font-size:0.85em;">{pct:.1f}%</div>
                </div>
            </td>
        </tr>"""

    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 700px;">
        <h3>🔍 Search: "{_esc(query)}"</h3>
        <table style="width:100%; border-collapse:collapse;">
            <tr style="background:#f5f5f5; text-align:left;">
                <th style="padding:8px;">Function</th>
                <th style="padding:8px;">Module</th>
                <th style="padding:8px;">Similarity</th>
            </tr>
            {rows}
        </table>
    </div>
    """
    return html


def render_decision(func_name, decision) -> str:
    """Render a single tripartite decision."""
    d_val = decision.value if hasattr(decision, "value") else str(decision)
    colors = {"hardcode": "#4CAF50", "model": "#2196F3", "hybrid": "#FF9800", "cached": "#9C27B0"}
    emojis = {"hardcode": "⚙️", "model": "🧠", "hybrid": "🔄", "cached": "📦"}
    descriptions = {
        "hardcode": "Compiled/fast path — deterministic, low-latency",
        "model": "LLM inference — creative, flexible",
        "hybrid": "Cache + model fallback — balanced",
        "cached": "Pre-computed — read-only, zero-latency",
    }

    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 500px; border: 2px solid {colors.get(d_val, '#999')}; border-radius: 12px; padding: 20px; text-align: center;">
        <h3>{emojis.get(d_val, '❓')} {_esc(func_name)}</h3>
        <div style="font-size: 2em; font-weight: bold; color: {colors.get(d_val, '#999')}; margin: 10px 0;">
            {d_val.upper()}
        </div>
        <p style="color: #666;">{descriptions.get(d_val, '')}</p>
    </div>
    """
    return html


def render_hardware(cap) -> str:
    """Render hardware probe results."""
    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 500px;">
        <h3>🖥️ Hardware Capabilities</h3>
        <table style="width:100%; border-collapse:collapse;">
            <tr><td style="padding:6px; color:#888;">GPU</td><td style="padding:6px;"><strong>{_esc(str(cap.gpu_name or 'None'))}</strong> ({'✓' if cap.gpu else '✗'})</td></tr>
            <tr><td style="padding:6px; color:#888;">RAM</td><td style="padding:6px;">{cap.ram_gb} GB</td></tr>
            <tr><td style="padding:6px; color:#888;">CPU Cores</td><td style="padding:6px;">{cap.cpu_cores}</td></tr>
            <tr><td style="padding:6px; color:#888;">Architecture</td><td style="padding:6px;">{_esc(cap.arch)}</td></tr>
            <tr><td style="padding:6px; color:#888;">Device Type</td><td style="padding:6px;">{_esc(cap.device_type)}</td></tr>
        </table>
    </div>
    """
    return html
