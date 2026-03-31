"""
Modern LLM Builder & Visualizer (2026 Architectures)
Supports: GPT-style, LLaMA-style (MoE), GQA, RoPE, SwiGLU, KV Cache
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Tuple, List, Dict, Any, Optional


# Color schemes for different components
COLORS = {
    'embedding': '#60a5fa',      # blue
    'attention': '#f472b6',       # pink
    'ffn': '#a78bfa',             # purple
    'moe': '#fbbf24',             # amber
    'router': '#34d399',          # emerald
    'norm': '#2dd4bf',            # teal
    'residual': '#fb923c',        # orange
    'rope': '#e879f9',            # fuchsia
    'kv_cache': '#f87171',        # red
    'input': '#a3e635',           # lime
    'output': '#fb7185',          # rose
    'q_head': '#38bdf8',         # sky blue
    'kv_head': '#a855f7',        # purple
}


def calculate_model_stats(
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    use_moe: bool = False,
    n_experts: int = 8,
    top_k: int = 2,
    use_swiglu: bool = True,
    context_length: int = 4096
) -> Dict[str, Any]:
    """Calculate model statistics."""

    # Standard attention params
    head_dim = d_model // n_heads
    q_params = n_heads * head_dim * d_model
    k_params = n_kv_heads * head_dim * d_model
    v_params = n_kv_heads * head_dim * d_model
    o_params = d_model * n_heads * head_dim
    attention_params_per_layer = q_params + k_params + v_params + o_params

    # Layer norm
    norm_params = 2 * d_model

    # FFN / MoE
    if use_moe:
        # MoE: router + experts
        router_params = n_experts * d_model  # linear router
        expert_params = n_experts * (4 * d_model * d_ff + 4 * d_ff)  # SwiGLU: W1, W2, W3, W4
        ffn_params_per_layer = expert_params + router_params
    elif use_swiglu:
        # SwiGLU: FFN with gate
        ffn_params_per_layer = 3 * d_model * d_ff  # W1, W2 (output), W3 (gate)
    else:
        # Standard FFN
        ffn_params_per_layer = 2 * d_model * d_ff

    # Embeddings
    embedding_params = vocab_size * d_model

    # Total per layer
    params_per_layer = attention_params_per_layer + ffn_params_per_layer + norm_params

    # Total model params
    total_params = embedding_params + n_layers * params_per_layer + norm_params

    # Token/chunk (for kv cache)
    tokens_per_chunk = 32  # typical chunk size
    n_chunks = (context_length + tokens_per_chunk - 1) // tokens_per_chunk

    # KV cache size (in millions)
    kv_cache_params = 2 * n_kv_heads * head_dim * context_length
    kv_cache_mb = kv_cache_params * 4 / (1024 ** 2)  # float32

    # FLOPs estimate (forward pass per token)
    attention_flops = 4 * n_heads * head_dim * d_model  # Q, K, V, O
    ffn_flops = 2 * (3 if use_swiglu else 2) * d_model * d_ff
    if use_moe:
        ffn_flops = 2 * top_k * n_experts * d_model * d_ff

    flops_per_token = attention_flops + ffn_flops

    return {
        'total_params': total_params,
        'params_formatted': f"{total_params/1e9:.2f}B" if total_params > 1e9 else f"{total_params/1e6:.1f}M",
        'embedding_params': embedding_params,
        'attention_params': attention_params_per_layer,
        'ffn_params': ffn_params_per_layer,
        'kv_cache_mb': kv_cache_mb,
        'flops_per_token': flops_per_token,
        'head_dim': head_dim,
        'n_chunks': n_chunks,
        'expert_active': top_k if use_moe else 1,
    }


def create_llm_architecture(
    n_layers: int = 12,
    n_heads: int = 12,
    n_kv_heads: int = 12,
    d_model: int = 768,
    d_ff: int = 3072,
    vocab_size: int = 50000,
    context_length: int = 4096,
    architecture: str = "llama",
    use_moe: bool = False,
    n_experts: int = 8,
    top_k: int = 2,
    use_gqa: bool = False,
    use_rope: bool = True,
    use_swiglu: bool = True,
    use_sliding_window: bool = False,
    window_size: int = 4096,
    show_kv_cache: bool = False,
    title: str = "Modern LLM Architecture"
) -> go.Figure:
    """Create a visualization of modern LLM architecture."""

    fig = go.Figure()

    # Colors for components
    colors = COLORS

    # Calculate stats
    stats = calculate_model_stats(
        n_layers, n_heads, n_kv_heads, d_model, d_ff, vocab_size,
        use_moe, n_experts, top_k, use_swiglu, context_length
    )

    # Limit displayed layers to avoid clutter
    max_display_layers = 8
    if n_layers <= max_display_layers:
        display_layers = list(range(1, n_layers + 1))
    else:
        # Show first 3, last 3, and indicate skipped layers
        display_layers = list(range(1, 4)) + [None] + list(range(n_layers - 2, n_layers + 1))

    # Layout parameters
    layer_height = 2.5
    layer_spacing = 3.0
    n_display = len(display_layers)
    start_y = (n_display - 1) * layer_spacing / 2 + 4

    # Calculate max x extent for proper axis range
    attn_width = 1.6 if use_gqa else 2.0
    ffn_x_start = 1.2 + attn_width + 1.2
    if use_moe:
        max_x = ffn_x_start + 0.8 + min(n_experts, 4) * 0.35 + 0.5
    else:
        max_x = ffn_x_start + 1.4

    # Architecture-specific settings
    is_bert = "BERT" in architecture
    is_gpt = "GPT" in architecture
    is_llama = "LLaMA" in architecture
    norm_label = "RMSNorm" if is_llama else "LayerNorm"
    norm_color = colors['norm'] if is_llama else '#93c5fd'
    norm_border = '#0f766e' if is_llama else '#1e40af'

    # Component positions for a single layer
    def draw_layer(layer_idx: int, y_pos: float):

        # Layer label
        fig.add_annotation(
            x=-1.5, y=y_pos,
            text=f"<b>Layer {layer_idx}</b>",
            showarrow=False,
            font=dict(size=12, color='#94a3b8'),
            textangle=-90
        )

        # Pre-norm (LLaMA uses pre-norm RMSNorm, GPT uses post-norm LayerNorm, BERT uses post-norm)
        if is_llama or is_gpt:
            fig.add_shape(
                type="rect", x0=-0.8, x1=-0.2, y0=y_pos-0.3, y1=y_pos+0.3,
                fillcolor=norm_color, line=dict(color=norm_border, width=1),
                opacity=0.8
            )
            fig.add_annotation(x=-0.5, y=y_pos, text=norm_label, showarrow=False, font=dict(size=7))

        # Attention block
        attn_x = 0.0
        attn_w = 1.6 if use_gqa else 2.0

        # Attention type label varies by architecture
        if is_bert:
            attn_type_label = "Bi-Directional"
        elif is_gpt:
            attn_type_label = "Causal Masked"
        else:
            attn_type_label = "Causal"

        if use_gqa:
            # GQA: Show Q heads and shared KV
            fig.add_shape(
                type="rect", x0=attn_x, x1=attn_x+attn_w, y0=y_pos+0.15, y1=y_pos+0.45,
                fillcolor=colors['q_head'], line=dict(color='#0369a1', width=1),
                opacity=0.8
            )
            fig.add_annotation(x=attn_x+attn_w/2, y=y_pos+0.3, text=f"Q Heads ({n_heads})", showarrow=False, font=dict(size=7))

            fig.add_shape(
                type="rect", x0=attn_x, x1=attn_x+attn_w, y0=y_pos-0.15, y1=y_pos+0.1,
                fillcolor=colors['kv_head'], line=dict(color='#7c3aed', width=1),
                opacity=0.8
            )
            fig.add_annotation(x=attn_x+attn_w/2, y=y_pos-0.02, text=f"KV Heads ({n_kv_heads})", showarrow=False, font=dict(size=7))

            # Attention type indicator
            fig.add_annotation(x=attn_x+attn_w/2, y=y_pos-0.4, text=attn_type_label, showarrow=False, font=dict(size=6, color='#64748b'))
        else:
            fig.add_shape(
                type="rect", x0=attn_x, x1=attn_x+attn_w, y0=y_pos-0.3, y1=y_pos+0.3,
                fillcolor=colors['attention'], line=dict(color='#be185d', width=1),
                opacity=0.8
            )
            fig.add_annotation(x=attn_x+attn_w/2, y=y_pos+0.05, text=f"{attn_type_label}<br>Attention ({n_heads}h)", showarrow=False, font=dict(size=7))

        # RoPE indicator (not used in BERT or classic GPT)
        if use_rope and not is_bert:
            fig.add_annotation(
                x=attn_x+attn_w+0.15, y=y_pos,
                text="RoPE", showarrow=False,
                font=dict(size=8, color=colors['rope']),
                bordercolor=colors['rope'], borderwidth=1,
                borderpad=3
            )

        # Post-norm (BERT/GPT use post-norm)
        norm2_x = attn_x + attn_w + 0.4
        fig.add_shape(
            type="rect", x0=norm2_x, x1=norm2_x+0.6, y0=y_pos-0.3, y1=y_pos+0.3,
            fillcolor=norm_color, line=dict(color=norm_border, width=1),
            opacity=0.8
        )
        norm2_label = norm_label if is_llama else ("Post-LN" if (is_bert or is_gpt) else "Norm")
        fig.add_annotation(x=norm2_x+0.3, y=y_pos, text=norm2_label, showarrow=False, font=dict(size=7))

        # FFN / MoE
        ffn_x = norm2_x + 0.8
        if use_moe:
            # MoE layer
            fig.add_shape(
                type="rect", x0=ffn_x, x1=ffn_x+0.6, y0=y_pos-0.35, y1=y_pos+0.35,
                fillcolor=colors['router'], line=dict(color='#047857', width=1),
                opacity=0.8
            )
            fig.add_annotation(x=ffn_x+0.3, y=y_pos, text="Router", showarrow=False, font=dict(size=7))

            # Experts grid
            expert_colors = px.colors.qualitative.Set3
            for i in range(min(n_experts, 4)):  # Show up to 4 experts visually
                ex = ffn_x + 0.8 + i * 0.35
                fig.add_shape(
                    type="rect", x0=ex, x1=ex+0.25, y0=y_pos-0.3, y1=y_pos+0.3,
                    fillcolor=expert_colors[i % len(expert_colors)],
                    line=dict(color='#92400e', width=1),
                    opacity=0.9
                )
            if n_experts > 4:
                fig.add_annotation(x=ffn_x+0.8+4*0.35+0.1, y=y_pos, text=f"+{n_experts-4}", showarrow=False, font=dict(size=7, color='#92400e'))

            fig.add_annotation(x=ffn_x+1.5, y=y_pos+0.55, text=f"MoE (top-{top_k})", showarrow=False, font=dict(size=8, color=colors['moe']))

        elif use_swiglu:
            fig.add_shape(
                type="rect", x0=ffn_x, x1=ffn_x+1.2, y0=y_pos-0.3, y1=y_pos+0.3,
                fillcolor=colors['ffn'], line=dict(color='#6b21a8', width=1),
                opacity=0.8
            )
            fig.add_annotation(x=ffn_x+0.6, y=y_pos, text="SwiGLU<br>FFN", showarrow=False, font=dict(size=8))
        else:
            ffn_label = "GELU FFN" if is_bert else ("GELU FFN" if is_gpt else "ReLU FFN")
            ffn_color = '#c4b5fd' if is_bert else colors['ffn']
            fig.add_shape(
                type="rect", x0=ffn_x, x1=ffn_x+1.2, y0=y_pos-0.3, y1=y_pos+0.3,
                fillcolor=ffn_color, line=dict(color='#6b21a8', width=1),
                opacity=0.8
            )
            fig.add_annotation(x=ffn_x+0.6, y=y_pos, text=ffn_label, showarrow=False, font=dict(size=8))

        # Residual connection (curved arrow bypassing the layer)
        fig.add_trace(go.Scatter(
            x=[-1.0, -1.0, max_x + 0.2, max_x + 0.2],
            y=[y_pos + 0.5, y_pos + 0.7, y_pos + 0.7, y_pos + 0.5],
            mode='lines',
            line=dict(color=colors['residual'], width=1.5, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_annotation(
            x=max_x * 0.5, y=y_pos + 0.85,
            text="residual", showarrow=False,
            font=dict(size=7, color=colors['residual'])
        )

        return y_pos - layer_spacing

    # Draw layers
    y = start_y
    for layer_num in display_layers:
        if layer_num is None:
            # Draw "..." indicator for skipped layers
            fig.add_annotation(
                x=max_x / 2, y=y,
                text=f"... ({n_layers - 6} more layers) ...",
                showarrow=False,
                font=dict(size=14, color='#94a3b8')
            )
            y -= layer_spacing
        else:
            y = draw_layer(layer_num, y)

    # Input/Output embeddings - vary by architecture
    embed_text = f"Token + {'RoPE' if use_rope and is_llama else 'Pos'} Embed ({d_model}d)"
    if is_bert:
        embed_text = f"Token + Position + Segment ({d_model}d)"

    fig.add_shape(
        type="rect", x0=-1.0, x1=1.0, y0=start_y+1.2, y1=start_y+1.8,
        fillcolor=colors['embedding'], line=dict(color='#1d4ed8', width=2),
        opacity=0.9
    )
    fig.add_annotation(x=0, y=start_y+1.5, text=embed_text, showarrow=False, font=dict(size=9))

    # Arrow from embedding to first layer
    fig.add_annotation(
        x=0, y=start_y + 0.5,
        ax=0, ay=start_y + 1.2,
        text="", showarrow=True,
        axref='x', ayref='y', xref='x', yref='y',
        arrowhead=2, arrowcolor='#64748b', arrowsize=1.5
    )

    # Output head varies by architecture
    if is_bert:
        output_text = "MLM / Classification Head"
        output_color = '#86efac'
        output_border = '#166534'
    elif is_gpt:
        output_text = "Autoregressive LM Head"
        output_color = colors['output']
        output_border = '#be123c'
    else:
        output_text = "Output LM Head"
        output_color = colors['output']
        output_border = '#be123c'

    fig.add_shape(
        type="rect", x0=-1.0, x1=1.0, y0=y-0.6, y1=y+0.0,
        fillcolor=output_color, line=dict(color=output_border, width=2),
        opacity=0.9
    )
    fig.add_annotation(x=0, y=y-0.3, text=output_text, showarrow=False, font=dict(size=9))

    # Title with stats
    param_text = f"<b>{stats['params_formatted']}</b> parameters | "
    param_text += f"d_model={d_model} | "
    param_text += f"heads={n_heads}"
    if use_gqa:
        param_text += f"/{n_kv_heads}"
    if use_moe:
        param_text += f" | MoE: {n_experts} experts"
    param_text += f" | context={context_length:,}"

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#64748b;'>{param_text}</span>",
            x=0.5
        ),
        showlegend=False,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, max_x + 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=max(400, n_display * 100 + 150),
        width=900,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def create_attention_pattern(
    n_heads: int = 12,
    n_kv_heads: int = 12,
    seq_len: int = 32,
    use_gqa: bool = False,
    use_sliding_window: bool = False,
    window_size: int = 16,
    architecture: str = "LLaMA-style (Dense)",
    title: str = "Attention Pattern"
) -> go.Figure:
    """Visualize attention patterns (full, GQA, sliding window, causal/bidirectional)."""

    is_causal = "BERT" not in architecture  # BERT is bidirectional, others are causal
    sz = min(seq_len, 16)

    def make_attn_matrix(sz, is_causal, use_sliding_window, window_size):
        """Generate attention matrix based on architecture type."""
        attn = np.zeros((sz, sz))
        for i in range(sz):
            for j in range(sz):
                # Causal mask: can only attend to previous positions
                if is_causal and j > i:
                    attn[i, j] = 0
                    continue
                # Sliding window
                if use_sliding_window and abs(i - j) > window_size:
                    attn[i, j] = 0
                    continue
                # Attention weight (distance-based decay for realism)
                dist = abs(i - j)
                attn[i, j] = np.exp(-dist * 0.15)
            # Normalize row
            row_sum = attn[i].sum()
            if row_sum > 0:
                attn[i] /= row_sum
        return attn

    if use_gqa:
        # GQA: Each query head group shares one KV head
        n_groups = min(n_heads, 4)  # Show max 4 groups
        cols = min(n_groups, 4)
        rows = (n_groups + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Q Head {i*n_heads//n_groups}-{(i+1)*n_heads//n_groups-1} → KV Head {i * n_kv_heads // n_groups}" for i in range(n_groups)]
        )

        for h in range(n_groups):
            row = h // cols + 1
            col = h % cols + 1

            attn = make_attn_matrix(sz, is_causal, use_sliding_window, window_size)

            heatmap = go.Heatmap(
                z=attn,
                colorscale='Blues',
                showscale=(h == n_groups - 1),
                zmin=0, zmax=1
            )
            fig.add_trace(heatmap, row=row, col=col)
            fig.update_xaxes(title_text="Key", row=row, col=col)
            fig.update_yaxes(title_text="Query", row=row, col=col)

        attn_desc = "Causal" if is_causal else "Bidirectional"
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><span style='font-size:11px'>{attn_desc} GQA: {n_heads} Q heads → {n_kv_heads} KV heads</span>", x=0.5),
            height=300, width=700
        )

    else:
        attn = make_attn_matrix(sz, is_causal, use_sliding_window, window_size)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=attn,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Attention", x=1.02),
            zmin=0, zmax=attn.max() * 1.2
        ))

        attn_desc = "Causal" if is_causal else "Bidirectional"
        window_desc = ", sliding window" if use_sliding_window else ""
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><span style='font-size:11px'>{attn_desc} attention{window_desc} | {n_heads} heads, seq_len={seq_len}</span>", x=0.5),
            xaxis=dict(title="Key position"),
            yaxis=dict(title="Query position"),
            height=400, width=500
        )

    return fig


def create_moe_visualization(
    n_experts: int = 8,
    top_k: int = 2,
    seq_len: int = 16,
    d_model: int = 512,
    title: str = "Mixture of Experts Routing"
) -> go.Figure:
    """Visualize MoE expert selection and load balancing."""

    fig = go.Figure()

    # Token positions
    tokens = [f"Tok {i}" for i in range(seq_len)]

    # Simulate expert routing (seed based on params so it changes with input)
    np.random.seed(n_experts * 100 + top_k * 10 + seq_len + d_model)
    routing = np.random.randint(0, n_experts, size=(seq_len, top_k))

    # Expert load (for bar chart)
    expert_load = np.zeros(n_experts)
    for i in range(seq_len):
        for k in range(top_k):
            expert_load[routing[i, k]] += 1

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Expert Grid", "Token Routing", "Expert Load", "Expert Selection Matrix"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "bar"}, {"type": "heatmap"}]
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )

    # Expert grid
    grid_size = int(np.ceil(np.sqrt(n_experts)))
    for e in range(n_experts):
        row = e // grid_size
        col = e % grid_size
        fig.add_trace(go.Scatter(
            x=[col * 0.8 + 0.4], y=[2 - row * 0.8],
            mode='markers+text',
            marker=dict(size=40, color=px.colors.qualitative.Set3[e % 12]),
            text=f"E{e}",
            textposition='middle center',
            showlegend=False
        ), row=1, col=1)

    fig.update_xaxes(range=[-0.5, grid_size * 0.8], showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[-0.5, 2.5], showgrid=False, showticklabels=False, row=1, col=1)

    # Token routing visualization
    for i in range(min(seq_len, 8)):
        for k in range(top_k):
            expert = routing[i, k]
            fig.add_trace(go.Scatter(
                x=[i * 0.8, expert * 0.8 + 0.4],
                y=[1.5, 0.5],
                mode='lines',
                line=dict(
                    color=px.colors.qualitative.Set3[expert % 12],
                    width=2
                ),
                showlegend=False,
                hoverinfo='text',
                text=f"Token {i} → Expert {expert}"
            ), row=1, col=2)

    fig.update_xaxes(range=[-0.5, max(7, n_experts) * 0.8], showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(range=[0, 2], showgrid=False, showticklabels=False, row=1, col=2)

    # Expert load bar chart
    fig.add_trace(go.Bar(
        x=[f"E{i}" for i in range(n_experts)],
        y=expert_load,
        marker_color=px.colors.qualitative.Set3[:n_experts],
        showlegend=False
    ), row=2, col=1)

    fig.update_xaxes(title="Expert", row=2, col=1)
    fig.update_yaxes(title="Tokens Assigned", row=2, col=1)

    # Expert selection matrix
    selection_matrix = np.zeros((min(seq_len, 8), n_experts))
    for i in range(min(seq_len, 8)):
        for k in range(top_k):
            selection_matrix[i, routing[i, k]] = 1

    fig.add_trace(go.Heatmap(
        z=selection_matrix,
        x=[f"E{i}" for i in range(n_experts)],
        y=[f"T{i}" for i in range(min(seq_len, 8))],
        colorscale=[[0, '#f3f4f6'], [1, px.colors.qualitative.Set3[0]]],
        showscale=False
    ), row=2, col=2)

    fig.update_xaxes(title="Expert", row=2, col=2)
    fig.update_yaxes(title="Token", row=2, col=2)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:11px'>{n_experts} experts, top-{top_k} routing | Load balance: {expert_load.std()/expert_load.mean():.2f} std/mean</span>", x=0.5),
        height=600, width=900,
        showlegend=False
    )

    return fig


def create_rope_visualization(
    d_model: int = 512,
    n_heads: int = 8,
    head_dim: int = 64,
    context_length: int = 128,
    rope_theta: float = 500000,
    title: str = "Rotary Position Embeddings (RoPE)"
) -> go.Figure:
    """Visualize RoPE rotation encoding."""

    fig = go.Figure()

    # Show rotation for first few positions and dimensions
    positions_to_show = [0, 1, 2, 63, 64, 127]  # Example positions
    dims_to_show = 4  # Show first 4 dimensions

    # Calculate rotations
    freqs = 1.0 / (rope_theta ** (2 * np.arange(0, head_dim // 2, 1) / head_dim))

    # Create 3D surface of rotations
    theta_vals = []
    for pos in range(min(positions_to_show) if positions_to_show else 0, min(max(positions_to_show) + 1, context_length)):
        for dim_pair in range(head_dim // 2):
            theta = pos * freqs[dim_pair]
            theta_vals.append([pos, dim_pair, np.cos(theta), np.sin(theta)])

    theta_data = np.array(theta_vals)

    # Subplot 1: Rotation angles by position
    fig.add_trace(go.Scatter(
        x=theta_data[:, 0],
        y=theta_data[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=np.arctan2(theta_data[:, 3], theta_data[:, 2]),
            colorscale='HSV',
            showscale=True,
            colorbar=dict(title="Phase (rad)", x=1.02)
        ),
        text=[f"Pos: {int(p)}, Dim: {int(d*2)}, {int(d*2+1)}" for p, d in zip(theta_data[:, 0], theta_data[:, 1])],
        hovertemplate="Pos: %{text}<br>Phase: %{marker.color:.3f}<extra></extra>"
    ))

    # Add annotation explaining RoPE
    fig.add_annotation(
        x=0.5, y=-0.15,
        xref='paper', yref='paper',
        text="<b>RoPE</b>: Each position gets a rotation matrix. "
             "Q and K are rotated and inner product gives position-aware attention.",
        showarrow=False,
        font=dict(size=11),
        bgcolor='rgba(251, 191, 36, 0.1)',
        bordercolor='#fbbf24',
        borderwidth=1,
        borderpad=10
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:11px'>θ={rope_theta:,} | d_model={d_model} | context={context_length}</span>", x=0.5),
        xaxis=dict(title="Position"),
        yaxis=dict(title="Dimension pair (2i, 2i+1)"),
        height=400, width=700,
        plot_bgcolor='#fafafa'
    )

    return fig


def create_kv_cache_viz(
    n_kv_heads: int = 8,
    head_dim: int = 128,
    context_length: int = 128,
    max_context: int = 4096,
    title: str = "KV Cache Behavior"
) -> go.Figure:
    """Visualize KV cache during autoregressive generation."""

    fig = go.Figure()

    # Show how KV cache grows
    current_tokens = context_length
    cached_tokens = min(current_tokens, max_context)

    # Create visualization
    cache_utilization = cached_tokens / max_context

    # Memory calculation (float16 = 2 bytes)
    memory_mb = 2 * 2 * n_kv_heads * head_dim * cached_tokens / (1024 ** 2)

    # Timeline bar
    fig.add_trace(go.Bar(
        x=[cached_tokens, max_context - cached_tokens],
        y=["KV Cache"],
        orientation='h',
        marker=dict(
            color=['#34d399', '#e5e7eb'],
            line=dict(color=['#047857', '#9ca3af'], width=2)
        ),
        text=[f"{cached_tokens:,} tokens", f"{max_context - cached_tokens:,} available"],
        textposition='inside',
        insidetextanchor='middle',
        showlegend=False,
        hovertemplate="Token: %{x}<br>%{text}<extra></extra>"
    ))

    # Add stats annotations
    fig.add_annotation(
        x=0.5, y=-0.2,
        xref='paper', yref='paper',
        text=f"<b>Cache Stats:</b> {cached_tokens:,}/{max_context:,} tokens | "
             f"Memory: {memory_mb:.1f} MB (float16) | "
             f"Utilization: {cache_utilization*100:.1f}%",
        showarrow=False,
        font=dict(size=11),
        bgcolor='rgba(52, 211, 153, 0.1)',
        bordercolor='#34d399',
        borderwidth=1,
        borderpad=8
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:11px'>KV heads={n_kv_heads}, head_dim={head_dim}</span>", x=0.5),
        xaxis=dict(title="Token Position", range=[0, max_context]),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=250, width=700,
        plot_bgcolor='white',
        showlegend=False
    )

    return fig


def show_llm_builder_ui():
    """Streamlit UI for Modern LLM Builder."""

    st.markdown("## 🚀 Modern LLM Builder (2026 Architectures)")

    st.markdown("""
    <div class="info-box">
    Build and visualize modern LLM architectures with support for:
    <b>Mixture of Experts (MoE)</b>, <b>Grouped Query Attention (GQA)</b>,
    <b>RoPE</b>, <b>SwiGLU</b>, <b>Sliding Window Attention</b>, and <b>KV Cache</b>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 📐 Architecture")

        architecture = st.selectbox(
            "Architecture Type",
            ["LLaMA-style (Dense)", "LLaMA-style (MoE)", "GPT-style", "BERT-style"],
            key="llm_architecture_type"
        )

        st.markdown("### ⚙️ Core Parameters")

        n_layers = st.slider("Layers", 1, 96, 32, 1, key="llm_n_layers")
        n_heads = st.slider("Attention Heads", 1, 64, 32, 1, key="llm_n_heads")
        d_model = st.slider("Model Dimension (d_model)", 128, 2048, 512, 64, key="llm_d_model")
        d_ff = st.slider("FFN Dimension", 256, 8192, 1360, 64, key="llm_d_ff")

        vocab_size = st.selectbox(
            "Vocabulary Size",
            [10000, 30000, 50000, 64000, 100000, 128256],
            index=2,
            key="llm_vocab_size"
        )

        context_length = st.selectbox(
            "Context Length",
            [512, 1024, 2048, 4096, 8192, 16384, 32768, 100000, 1000000],
            index=3,
            format_func=lambda x: f"{x:,}" if x < 100000 else f"{x/1000:.0f}K",
            key="llm_context_length"
        )

        st.markdown("### 🔧 Modern Features")

        use_gqa = st.checkbox("Grouped Query Attention (GQA)", True, key="llm_use_gqa")
        if use_gqa:
            n_kv_heads = st.slider("KV Heads", 1, n_heads, max(1, n_heads // 4), 1,
                                  help="Fewer KV heads = more efficient", key="llm_n_kv_heads")
        else:
            n_kv_heads = n_heads

        use_swiglu = st.checkbox("SwiGLU Activation", True,
                                  help="Used in LLaMA, PaLM (better than ReLU)", key="llm_use_swiglu")

        use_rope = st.checkbox("Rotary Position Embeddings (RoPE)", True,
                                help="Most modern LLMs use RoPE", key="llm_use_rope")

        if use_rope:
            rope_theta = st.selectbox(
                "RoPE θ",
                [10000, 50000, 100000, 500000, 1000000],
                index=3,
                format_func=lambda x: f"{x:,}",
                key="llm_rope_theta"
            )

        use_sliding_window = st.checkbox("Sliding Window Attention", False,
                                         help="Efficient for long contexts (Mistral)", key="llm_use_sliding_window")

        if use_sliding_window:
            window_size = st.slider("Window Size", 32, 8192, 4096, 32, key="llm_window_size")

        use_moe = st.checkbox("Mixture of Experts (MoE)", False,
                               help="Only activate top-K experts per token", key="llm_use_moe")

        if use_moe:
            n_experts = st.slider("Number of Experts", 2, 64, 8, 1, key="llm_n_experts")
            top_k = st.slider("Top-K Routing", 1, n_experts, 2, 1, key="llm_top_k")

        show_kv_cache = st.checkbox("Show KV Cache", True,
                                     help="Visualize cache behavior", key="llm_show_kv_cache")

    with col1:
        tab_names = [
            "🏗️ Architecture",
            "👁️ Attention Pattern",
            "⚡ MoE Routing" if use_moe else "📊 Expert Config",
            "🔄 RoPE & Cache"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_names)

        with tab1:
            # Calculate stats
            stats = calculate_model_stats(
                n_layers, n_heads, n_kv_heads, d_model, d_ff, vocab_size,
                use_moe, n_experts if use_moe else 8,
                top_k if use_moe else 2, use_swiglu, context_length
            )

            # Architecture title based on selection
            arch_names = {
                "LLaMA-style (Dense)": "LLaMA-style Dense LLM",
                "LLaMA-style (MoE)": "LLaMA-style MoE LLM",
                "GPT-style": "GPT-style Autoregressive LLM",
                "BERT-style": "BERT-style Encoder LLM"
            }

            fig = create_llm_architecture(
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_model=d_model,
                d_ff=d_ff,
                vocab_size=vocab_size,
                context_length=context_length,
                architecture=architecture,
                use_moe=use_moe,
                n_experts=n_experts if use_moe else 8,
                top_k=top_k if use_moe else 2,
                use_gqa=use_gqa,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
                use_sliding_window=use_sliding_window,
                window_size=window_size if use_sliding_window else 4096,
                show_kv_cache=show_kv_cache,
                title=arch_names.get(architecture, "Modern LLM")
            )
            st.plotly_chart(fig, use_container_width=True, key=f"arch_{architecture}_{n_layers}_{n_heads}_{d_model}_{d_ff}_{use_moe}_{use_gqa}_{use_rope}_{use_swiglu}")

            # Stats summary
            st.markdown("### 📊 Model Statistics")

            stats_cols = st.columns(3)
            with stats_cols[0]:
                st.metric("Total Parameters", stats['params_formatted'])
            with stats_cols[1]:
                st.metric("KV Cache", f"{stats['kv_cache_mb']:.1f} MB")
            with stats_cols[2]:
                st.metric("Active Experts", stats['expert_active'])

            expander = st.expander("Detailed Breakdown")
            with expander:
                st.write(f"**Per Layer:**")
                st.write(f"- Attention: {stats['attention_params']/1e6:.1f}M params")
                st.write(f"- FFN/MoE: {stats['ffn_params']/1e6:.1f}M params")
                st.write(f"- Head dimension: {stats['head_dim']}")
                st.write(f"**Context:** {context_length:,} tokens")
                if use_moe:
                    st.write(f"**MoE:** {n_experts} experts, top-{top_k} routing")
                if use_gqa:
                    st.write(f"**GQA:** {n_heads} Q heads → {n_kv_heads} KV heads (compression: {n_heads/n_kv_heads:.1f}x)")

        with tab2:
            st.markdown("### 👁️ Attention Pattern Visualization")

            seq_len_vis = st.slider("Sequence Length (visualization)", 8, 64, 32, 8)

            fig_attn = create_attention_pattern(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                seq_len=seq_len_vis,
                use_gqa=use_gqa,
                use_sliding_window=use_sliding_window,
                window_size=window_size if use_sliding_window else 16,
                architecture=architecture,
                title="Attention Pattern"
            )
            st.plotly_chart(fig_attn, use_container_width=True, key=f"attn_{n_heads}_{n_kv_heads}_{seq_len_vis}_{use_gqa}_{use_sliding_window}")

            st.markdown("""
            **Attention Types:**
            - **Full Attention**: Every token attends to all others (O(n²) memory)
            - **GQA**: Query heads share KV projections (reduces KV cache)
            - **Sliding Window**: Only attend to nearby tokens (efficient for long contexts)
            """)

        with tab3:
            if use_moe:
                fig_moe = create_moe_visualization(
                    n_experts=n_experts,
                    top_k=top_k,
                    seq_len=16,
                    d_model=d_model,
                    title="Mixture of Experts Routing"
                )
                st.plotly_chart(fig_moe, use_container_width=True, key=f"moe_{n_experts}_{top_k}_{d_model}")

                st.markdown(f"""
                **MoE Analysis:**
                - **{n_experts} experts** available per layer
                - **{top_k} experts** activated per token (sparsity)
                - Expected FLOPs reduction: {100 * (1 - top_k/n_experts):.0f}% vs dense
                - Load balancing important to avoid expert collapse
                """)
            else:
                # Show FFN config instead
                st.markdown("### FFN Configuration (SwiGLU)")

                ffn_fig = go.Figure()
                layers_ffn = list(range(n_layers))
                ffn_params = [stats['ffn_params'] / 1e6] * len(layers_ffn)

                ffn_fig.add_trace(go.Bar(
                    x=layers_ffn,
                    y=ffn_params,
                    marker_color='#a78bfa',
                    name='FFN Params (M)'
                ))

                ffn_fig.update_layout(
                    title=f"<b>FFN Parameters per Layer</b><br><span style='font-size:11px'>SwiGLU: 3×d_model×d_ff matrix multiplies</span>",
                    xaxis_title="Layer",
                    yaxis_title="FFN Parameters (M)",
                    height=350, width=600,
                    plot_bgcolor='white'
                )
                st.plotly_chart(ffn_fig, use_container_width=True, key=f"ffn_{n_layers}_{d_model}_{d_ff}_{use_swiglu}")

                st.markdown(f"""
                **SwiGLU (LLaMA, PaLM):**
                - FFN(x) = (Swish(xW₁) ⊙ xW₃)W₂
                - Uses **3 matrix multiplications** instead of 2
                - Parameters: 3 × d_model × d_ff = {stats['ffn_params']/1e6:.0f}M per layer
                - Better expressiveness than standard FFN
                """)

        with tab4:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🔄 RoPE Visualization")
                if use_rope:
                    fig_rope = create_rope_visualization(
                        d_model=d_model,
                        n_heads=n_heads,
                        head_dim=d_model // n_heads,
                        context_length=min(context_length, 128),
                        rope_theta=rope_theta if use_rope else 500000,
                        title="Rotary Position Embeddings"
                    )
                    st.plotly_chart(fig_rope, use_container_width=True, key=f"rope_{d_model}_{n_heads}_{context_length}_{rope_theta if use_rope else 0}")
                else:
                    st.info("Enable RoPE to see visualization")

            with col_b:
                st.markdown("### 💾 KV Cache")
                if show_kv_cache:
                    fig_kv = create_kv_cache_viz(
                        n_kv_heads=n_kv_heads,
                        head_dim=d_model // n_heads,
                        context_length=context_length,
                        max_context=context_length,
                        title="KV Cache Behavior"
                    )
                    st.plotly_chart(fig_kv, use_container_width=True, key=f"kv_{n_kv_heads}_{d_model}_{context_length}")

                # Memory comparison
                st.markdown("**Memory Comparison:**")

                dense_kv = 2 * n_heads * (d_model // n_heads) * context_length * 4 / (1024 ** 2)
                gqa_kv = 2 * n_kv_heads * (d_model // n_heads) * context_length * 4 / (1024 ** 2)

                memory_data = {
                    'Attention Type': ['Dense (All Heads)', 'GQA (Shared KV)'],
                    'KV Cache (MB)': [dense_kv, gqa_kv]
                }

                fig_mem = go.Figure()
                fig_mem.add_trace(go.Bar(
                    x=memory_data['Attention Type'],
                    y=memory_data['KV Cache (MB)'],
                    marker_color=['#f87171', '#34d399']
                ))
                fig_mem.update_layout(
                    title=f"<b>KV Cache Memory</b><br><span style='font-size:10px'>float32 | context={context_length:,}</span>",
                    height=250, width=350,
                    yaxis_title="Memory (MB)",
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig_mem, use_container_width=True, key=f"mem_{n_heads}_{n_kv_heads}_{d_model}_{context_length}")

                st.markdown(f"""
                **GQA Savings:** {100*(1-gqa_kv/dense_kv):.1f}% less KV cache memory
                - Dense: {dense_kv:.1f} MB
                - GQA: {gqa_kv:.1f} MB
                """)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 2026 Modern LLM Architecture Guide")

    cols = st.columns(3)

    with cols[0]:
        st.markdown("""
        ### 🔑 Key Innovations

        **Grouped Query Attention (GQA)**
        - Llama 2/3, Mistral, Gemma
        - Fewer KV heads than Q heads
        - Reduces KV cache by ~50-75%

        **Mixture of Experts (MoE)**
        - Mixtral, GPT-4, DBRX
        - 8-128+ experts per layer
        - Only top-2 or top-8 active
        - Enables huge models efficiently
        """)

    with cols[1]:
        st.markdown("""
        **Rotary Position Embeddings (RoPE)**
        - LLaMA, Falcon, GPT-NeoX
        - Rotation-based, not additive
        - Better long-context generalization
        - θ (theta) controls frequency

        **Sliding Window Attention**
        - Mistral, Longformer
        - Local attention + global tokens
        - Handles 32K+ context efficiently
        """)

    with cols[2]:
        st.markdown("""
        **SwiGLU Activation**
        - LLaMA, PaLM, Chinchilla
        - Swish(x) ⊙ Gate(x) × FFN
        - Better than ReLU/GELU
        - ~10% better perplexity

        **KV Cache Optimization**
        - Critical for autoregressive inference
        - GQA reduces memory 4-8x
        - Flash Attention for speed
        """)

    # Additional learning content
    st.markdown("---")
    st.markdown("## 📈 Scaling Laws & Training Insights")

    cols2 = st.columns(2)

    with cols2[0]:
        st.markdown("""
        ### ⚖️ Scaling Laws (Chinchilla Hoffs)

        **The Problem:** GPT-3 (175B) was undertrained relative to compute.

        **The Rule:** For optimal loss, token count should scale ~20x faster than model size.

        | Model Size | Tokens to Train |
        |------------|----------------|
        | 1B         | 20B            |
        | 7B         | 140B           |
        | 70B        | 1.4T           |
        | 405B       | 8.1T           |

        **Practical Implications:**
        - LLaMA 3 8B trained on 15T tokens (vs 1T rule)
        - DeepMind's Chinchilla optimal: 1 token per param per second
        - More tokens = better sample efficiency
        """)

    with cols2[1]:
        st.markdown("""
        ### 🏋️ Training Dynamics

        **Mixed Precision Training:**
        - FP16/BF16 for forward/backward
        - FP32 for optimizer states (Adam)
        - Gradients ~2x model size

        **Memory Breakdown (7B model):**
        - Model weights: 14 GB
        - Optimizer states: 56 GB (Adam 2nd moment)
        - Activations: ~8-16 GB (sequence length)
        - KV Cache: 2-4 GB (per layer)

        **Gradient Checkpointing:** Trade 30% speed for 50% memory
        """)

    # Architecture comparison
    st.markdown("### 🔍 Architecture Comparison")

    arch_cols = st.columns(3)

    with arch_cols[0]:
        st.markdown("""
        **LLaMA (Meta)**
        - RMSNorm (instead of LayerNorm)
        - SwiGLU activation
        - RoPE position embeddings
        - Pre-normalization
        - Used in: Llama 2/3, Code Llama
        """)

    with arch_cols[1]:
        st.markdown("""
        **Mistral (Mistral AI)**
        - Sliding Window Attention
        - GQA (Grouped Query Attention)
        - RoPE with 32K context
        - Expert Choice Routing (MoE)
        - Used in: Mixtral 8x7B
        """)

    with arch_cols[2]:
        st.markdown("""
        **GPT-4 (OpenAI)**
        - Likely MoE architecture
        - Specialized experts per domain
        - Very long context (128K+)
        - Proprietary training data
        - Instruction tuning + RLHF
        """)

    # Inference optimization
    st.markdown("### ⚡ Inference Optimization Techniques")

    inf_cols = st.columns(3)

    with inf_cols[0]:
        st.markdown("""
        **Quantization:**
        - **INT8:** ~50% memory, minimal quality loss
        - **INT4:** ~75% memory, 2-5% perplexity increase
        - **GPTQ:** Post-training quantization
        - **AWQ:** Activation-aware weight quantization
        - **GGUF:** quantization format (llama.cpp)
        """)

    with inf_cols[1]:
        st.markdown("""
        **KV Cache Optimization:**
        - Paged Attention (vLLM)
        - Chunked prefill + decoding
        - Memory efficiency: 2x more sequences
        - Flash Attention 2/3: 2-4x speedup

        **Batching:**
        - Continuous batching (iteration-level)
        - Dynamic padding (variable seq len)
        - Increases throughput 10-20x
        """)

    with inf_cols[2]:
        st.markdown("""
        **Speculative Decoding:**
        - Draft model (small) proposes tokens
        - Target model verifies in parallel
        - 2-3x speedup for auto-regressive decode
        - Maintains exact same outputs

        **Distillation:**
        - DistilBERT: 40% smaller, 60% faster
        - TinyLlama: 1.1B, 3x faster
        - MedicalLLM: domain-specific distilled
        """)
