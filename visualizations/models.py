"""
Advanced Models Visualization (LLM, Diffusion, Video, Audio)
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import networkx as nx
from plotly.subplots import make_subplots


def create_transformer_architecture(
    n_layers: int = 6,
    n_heads: int = 8,
    d_model: int = 512,
    d_ff: int = 2048,
    show_decoder: bool = True,
    title: str = "Transformer Architecture"
) -> go.Figure:
    """
    Create a visualization of Transformer architecture

    Args:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        d_model: Model dimension
        d_ff: Feed-forward dimension
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Colors for different components
    colors = {
        'input': 'lightblue',
        'output': 'lightcoral',
        'encoder': 'lightgreen',
        'decoder': 'lightyellow',
        'attention': 'lightpink',
        'ffn': 'lightgray',
        'norm': 'lightseagreen',
        'residual': 'lightsalmon'
    }

    # Create a simplified diagram
    # We'll create a flow diagram from left to right

    # Component positions
    components = [
        ("Input Embedding", 0, 5, colors['input']),
        ("Positional Encoding", 0, 4, colors['input']),
    ]

    # Encoder layers
    for i in range(n_layers):
        y_pos = 3 - i * 0.5
        components.append((f"Encoder {i+1}", 1, y_pos, colors['encoder']))
        components.append((f"Multi-Head Attention", 2, y_pos + 0.1, colors['attention']))
        components.append((f"Feed Forward", 2, y_pos - 0.1, colors['ffn']))
        components.append((f"Layer Norm", 1.5, y_pos, colors['norm']))
        components.append((f"Residual", 1.2, y_pos, colors['residual']))

    # Decoder layers (if showing full transformer)
    if show_decoder:
        components.append(("Output Embedding", 0, -2, colors['input']))
        components.append(("Output Positional Encoding", 0, -3, colors['input']))

        for i in range(n_layers):
            y_pos = -4 - i * 0.5
            components.append((f"Decoder {i+1}", 1, y_pos, colors['decoder']))
            components.append((f"Masked Attention", 2, y_pos + 0.2, colors['attention']))
            components.append((f"Encoder-Decoder Attention", 2, y_pos, colors['attention']))
            components.append((f"Feed Forward", 2, y_pos - 0.2, colors['ffn']))
            components.append((f"Layer Norm", 1.5, y_pos, colors['norm']))
        components.append(("Linear", 3, -4 - (n_layers-1)*0.5, colors['output']))
        components.append(("Softmax", 4, -4 - (n_layers-1)*0.5, colors['output']))

    # Add components as rectangles
    for name, x, y, color in components:
        fig.add_shape(
            type="rect",
            x0=x - 0.4,
            x1=x + 0.4,
            y0=y - 0.2,
            y1=y + 0.2,
            fillcolor=color,
            line=dict(color="black", width=1),
            opacity=0.8
        )

        # Add text
        fig.add_annotation(
            x=x,
            y=y,
            text=name,
            showarrow=False,
            font=dict(size=8 if len(name) > 15 else 10),
            textangle=0
        )

    # Add connections
    # Input to encoder
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[4.5, 3.5],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ))

    # Encoder layers connections
    for i in range(n_layers - 1):
        y_from = 3 - i * 0.5
        y_to = 3 - (i + 1) * 0.5
        fig.add_trace(go.Scatter(
            x=[1, 1], y=[y_from, y_to],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-10, 6]),
        showlegend=False,
        plot_bgcolor='white',
        width=800,
        height=600
    )

    return fig


def create_diffusion_process(
    n_steps: int = 10,
    noise_schedule: str = 'linear',
    title: str = "Diffusion Process"
) -> go.Figure:
    """
    Visualize the diffusion (forward and reverse) process

    Args:
        n_steps: Number of diffusion steps
        noise_schedule: Type of noise schedule ('linear', 'cosine', 'sigmoid')
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    # Generate noise schedule
    if noise_schedule == 'linear':
        betas = np.linspace(0.0001, 0.02, n_steps)
    elif noise_schedule == 'cosine':
        t = np.linspace(0, np.pi, n_steps)
        betas = 0.5 * (1 - np.cos(t)) * 0.02
    else:  # sigmoid
        t = np.linspace(-6, 6, n_steps)
        betas = 0.02 / (1 + np.exp(-t))

    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Noise Schedule (β)",
            "Signal Retention (ᾱ)",
            "Forward Process",
            "Reverse Process"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )

    # 1. Noise schedule
    fig.add_trace(
        go.Scatter(x=list(range(n_steps)), y=betas,
                  mode='lines+markers', name='β (noise)',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )

    # 2. Signal retention
    fig.add_trace(
        go.Scatter(x=list(range(n_steps)), y=alpha_bars,
                  mode='lines+markers', name='ᾱ (signal)',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )

    # 3. Forward process (adding noise)
    # Generate sample image evolution
    np.random.seed(42)
    sample_signal = np.sin(np.linspace(0, 4*np.pi, 100))

    forward_samples = []
    current = sample_signal.copy()
    forward_samples.append(current)

    for step in range(1, n_steps):
        noise = np.random.randn(*current.shape) * np.sqrt(betas[step-1])
        current = np.sqrt(alphas[step-1]) * current + noise
        forward_samples.append(current)

    # Plot forward process
    for step in range(0, n_steps, max(1, n_steps//5)):
        fig.add_trace(
            go.Scatter(x=np.arange(100), y=forward_samples[step],
                      mode='lines',
                      name=f'Step {step}',
                      line=dict(width=1),
                      opacity=0.7),
            row=2, col=1
        )

    # 4. Reverse process (denoising)
    reverse_samples = []
    current = forward_samples[-1].copy()  # Start from noisy

    # Simplified reverse process
    for step in reversed(range(n_steps)):
        if step > 0:
            # Simplified denoising
            denoised = current / np.sqrt(alphas[step-1]) if step > 0 else current
            noise_estimate = (current - np.sqrt(alpha_bars[step-1]) * sample_signal
                            if step > 0 else np.zeros_like(current))
            current = denoised - np.sqrt(betas[step-1]) * noise_estimate
        reverse_samples.append(current)

    for step in range(0, n_steps, max(1, n_steps//5)):
        fig.add_trace(
            go.Scatter(x=np.arange(100), y=reverse_samples[step],
                      mode='lines',
                      name=f'Step {step}',
                      line=dict(width=1, dash='dash'),
                      opacity=0.7),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=700,
        showlegend=True
    )

    # Update axes
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_yaxes(title_text="β value", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="ᾱ value", row=1, col=2)
    fig.update_xaxes(title_text="Pixel", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Pixel", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=2)

    return fig


def create_video_model_architecture(
    temporal_layers: int = 3,
    spatial_layers: int = 4,
    title: str = "Video Model Architecture"
) -> go.Figure:
    """
    Visualize a video processing model architecture

    Args:
        temporal_layers: Number of temporal processing layers
        spatial_layers: Number of spatial processing layers
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Create a 3D visualization of video processing
    # Frames × Height × Width × Channels

    # Generate sample video data positions
    n_frames = 5
    height = 4
    width = 4

    # Initial frame positions
    frame_positions = []
    colors = []

    for f in range(n_frames):
        for h in range(height):
            for w in range(width):
                # Position in 3D space
                x = w + f * (width + 2)  # Separate frames in x direction
                y = h
                z = 0  # Initial input layer

                frame_positions.append((x, y, z))
                colors.append(f / n_frames)  # Color by frame

    # Convert to arrays
    positions = np.array(frame_positions)
    color_vals = np.array(colors)

    # Input layer
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_vals,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Frame")
        ),
        name='Input Frames'
    ))

    # Processing layers
    for layer in range(1, spatial_layers + 1):
        # Processed positions (compressed in spatial dimensions)
        compressed_height = max(1, height // (2 ** layer))
        compressed_width = max(1, width // (2 ** layer))

        layer_positions = []
        layer_colors = []

        for f in range(n_frames):
            for h in range(compressed_height):
                for w in range(compressed_width):
                    x = w + f * (compressed_width + 2)
                    y = h
                    z = layer * 2  # Higher z for later layers

                    layer_positions.append((x, y, z))
                    layer_colors.append(f / n_frames)

        layer_positions = np.array(layer_positions)

        fig.add_trace(go.Scatter3d(
            x=layer_positions[:, 0],
            y=layer_positions[:, 1],
            z=layer_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8 - layer * 1.5,
                color=layer_colors,
                colorscale='Viridis',
                opacity=0.7
            ),
            name=f'Spatial Layer {layer}'
        ))

        # Add connections from previous layer (simplified)
        if layer > 1:
            # Add some representative connections
            for f in [0, n_frames//2, n_frames-1]:
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, 0], z=[(layer-1)*2, layer*2],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    opacity=0.3
                ))

    # Temporal processing layers (on top)
    for t_layer in range(1, temporal_layers + 1):
        # Temporal aggregation positions
        temp_positions = []
        for f in range(n_frames):
            x = f * 3
            y = 0
            z = (spatial_layers + t_layer) * 2

            temp_positions.append((x, y, z))

        temp_positions = np.array(temp_positions)

        fig.add_trace(go.Scatter3d(
            x=temp_positions[:, 0],
            y=temp_positions[:, 1],
            z=temp_positions[:, 2],
            mode='markers+lines',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond'
            ),
            line=dict(color='red', width=2),
            name=f'Temporal Layer {t_layer}'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Width / Frame",
            yaxis_title="Height",
            zaxis_title="Processing Depth",
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        width=800,
        height=600
    )

    return fig


def create_audio_model_architecture(
    sample_rate: int = 16000,
    n_mels: int = 128,
    time_steps: int = 100,
    title: str = "Audio Model Architecture"
) -> go.Figure:
    """
    Visualize audio processing model architecture

    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel-frequency bands
        time_steps: Number of time steps
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Raw Audio Waveform",
            "Spectrogram",
            "Mel-spectrogram",
            "Feature Extraction",
            "Temporal Processing",
            "Classification"
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
        ]
    )

    # 1. Raw audio waveform
    np.random.seed(42)
    time = np.linspace(0, 1, sample_rate)
    frequency = 440  # A4 note
    audio_wave = 0.5 * np.sin(2 * np.pi * frequency * time)
    audio_wave += 0.2 * np.sin(2 * np.pi * 2 * frequency * time)  # Harmonic
    audio_wave += 0.1 * np.random.randn(len(time))  # Noise

    # Show only a segment
    segment = 1000
    fig.add_trace(
        go.Scatter(x=time[:segment], y=audio_wave[:segment],
                  mode='lines', line=dict(color='blue', width=1)),
        row=1, col=1
    )

    # 2. Spectrogram
    from scipy import signal
    frequencies, times, Sxx = signal.spectrogram(audio_wave, fs=sample_rate)

    fig.add_trace(
        go.Heatmap(z=10 * np.log10(Sxx + 1e-10),
                  x=times, y=frequencies,
                  colorscale='Viridis', showscale=False),
        row=1, col=2
    )

    # 3. Mel-spectrogram
    # Simplified mel filter banks
    mel_freqs = np.linspace(0, sample_rate/2, n_mels)
    mel_spectrogram = np.random.randn(n_mels, len(times))
    mel_spectrogram = np.abs(mel_spectrogram)  # Make positive

    fig.add_trace(
        go.Heatmap(z=mel_spectrogram,
                  x=times, y=mel_freqs,
                  colorscale='Hot', showscale=False),
        row=1, col=3
    )

    # 4. Feature extraction layers
    n_features = 256
    features = np.random.randn(time_steps, n_features)
    features = np.cumsum(features, axis=0)  # Make it look like learned features

    # Show first few features
    n_to_show = 10
    for i in range(n_to_show):
        fig.add_trace(
            go.Scatter(x=np.arange(time_steps), y=features[:, i],
                      mode='lines', line=dict(width=1),
                      opacity=0.7, showlegend=False),
            row=2, col=1
        )

    # 5. Temporal processing
    # Simulate RNN/CNN processing
    processed = np.zeros((time_steps, n_features))
    for t in range(1, time_steps):
        processed[t] = 0.9 * processed[t-1] + 0.1 * features[t]

    # Show mean over features
    fig.add_trace(
        go.Scatter(x=np.arange(time_steps), y=processed.mean(axis=1),
                  mode='lines', line=dict(color='green', width=2),
                  name='Temporal features'),
        row=2, col=2
    )

    # 6. Classification output
    n_classes = 5
    class_names = ["Speech", "Music", "Noise", "Silence", "Other"]
    class_probs = np.random.rand(n_classes)
    class_probs = np.exp(class_probs) / np.sum(np.exp(class_probs))

    fig.add_trace(
        go.Bar(x=class_names, y=class_probs,
              marker_color='lightcoral'),
        row=2, col=3
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=700,
        showlegend=False
    )

    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=1, col=3)
    fig.update_yaxes(title_text="Mel Frequency", row=1, col=3)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Feature Value", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=2)
    fig.update_yaxes(title_text="Feature Value", row=2, col=2)
    fig.update_xaxes(title_text="Class", row=2, col=3)
    fig.update_yaxes(title_text="Probability", row=2, col=3)

    return fig


def show_advanced_models_ui():
    """Streamlit UI for advanced models visualization"""
    st.markdown("## 🤖 Advanced Models")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")

        model_type = st.selectbox(
            "Model Type",
            ["Transformer (LLM)", "Diffusion Model", "Video Model", "Audio Model"]
        )

        if model_type == "Transformer (LLM)":
            st.markdown("#### Transformer Parameters")
            n_layers = st.slider("Number of layers", 2, 24, 6, 1)
            n_heads = st.slider("Number of heads", 1, 16, 8, 1)
            d_model = st.slider("Model dimension", 128, 1024, 512, 64)
            d_ff = st.slider("Feed-forward dimension", 256, 4096, 2048, 128)
            show_decoder = st.checkbox("Show Decoder", True)

        elif model_type == "Diffusion Model":
            st.markdown("#### Diffusion Parameters")
            n_steps = st.slider("Number of steps", 5, 100, 20, 5)
            noise_schedule = st.selectbox(
                "Noise schedule",
                ["linear", "cosine", "sigmoid"]
            )
            show_forward = st.checkbox("Show forward process", True)
            show_reverse = st.checkbox("Show reverse process", True)

        elif model_type == "Video Model":
            st.markdown("#### Video Parameters")
            temporal_layers = st.slider("Temporal layers", 1, 10, 3, 1)
            spatial_layers = st.slider("Spatial layers", 1, 8, 4, 1)
            n_frames = st.slider("Number of frames", 2, 30, 5, 1)

        elif model_type == "Audio Model":
            st.markdown("#### Audio Parameters")
            sample_rate = st.selectbox(
                "Sample rate",
                [8000, 16000, 22050, 44100, 48000],
                index=1
            )
            n_mels = st.slider("Mel bands", 32, 256, 128, 16)
            time_steps = st.slider("Time steps", 50, 500, 100, 10)

    with col1:
        st.markdown("### Visualization")

        if model_type == "Transformer (LLM)":
            fig = create_transformer_architecture(
                n_layers=n_layers,
                n_heads=n_heads,
                d_model=d_model,
                d_ff=d_ff,
                show_decoder=show_decoder,
                title=f"Transformer Architecture ({n_layers} layers, {n_heads} heads)"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif model_type == "Diffusion Model":
            fig = create_diffusion_process(
                n_steps=n_steps,
                noise_schedule=noise_schedule,
                title=f"Diffusion Process ({n_steps} steps, {noise_schedule} schedule)"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif model_type == "Video Model":
            fig = create_video_model_architecture(
                temporal_layers=temporal_layers,
                spatial_layers=spatial_layers,
                title=f"Video Model Architecture"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif model_type == "Audio Model":
            fig = create_audio_model_architecture(
                sample_rate=sample_rate,
                n_mels=n_mels,
                time_steps=time_steps,
                title=f"Audio Model Architecture"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Educational Content")

    col1, col2 = st.columns(2)

    with col1:
        if model_type == "Transformer (LLM)":
            st.markdown("### Transformer Components")
            st.markdown("""
            - **Self-Attention**: Compute attention between all positions
            - **Multi-Head Attention**: Multiple attention heads in parallel
            - **Positional Encoding**: Inject position information
            - **Feed-Forward Networks**: Position-wise MLPs
            - **Layer Normalization**: Stabilize training
            - **Residual Connections**: Gradient flow
            - **Encoder-Decoder Architecture**: For sequence-to-sequence
            """)

        elif model_type == "Diffusion Model":
            st.markdown("### Diffusion Process")
            st.markdown("""
            - **Forward Process**: Gradually add noise to data
            - **Reverse Process**: Learn to denoise (generation)
            - **Noise Schedule**: Controls noise addition rate
            - **Score Matching**: Learn gradient of data distribution
            - **Conditional Generation**: Control generation with prompts
            - **Latent Diffusion**: Operate in compressed latent space
            """)

    with col2:
        if model_type == "Video Model":
            st.markdown("### Video Processing")
            st.markdown("""
            - **Spatial Processing**: Extract features from frames (CNNs)
            - **Temporal Processing**: Model frame relationships (3D CNNs, RNNs)
            - **Spatio-temporal**: Joint modeling of space and time
            - **Optical Flow**: Motion estimation between frames
            - **Frame Sampling**: Efficient processing of long videos
            - **Temporal Attention**: Focus on important time steps
            """)

        elif model_type == "Audio Model":
            st.markdown("### Audio Processing")
            st.markdown("""
            - **Time Domain**: Raw waveform processing (WaveNet)
            - **Frequency Domain**: Spectrogram analysis
            - **Mel-spectrograms**: Perceptually relevant features
            - **MFCCs**: Compact speech representations
            - **Temporal Modeling**: RNNs, CNNs, Transformers
            - **Multi-task Learning**: Speech, music, environmental sounds
            """)

    # Additional interactive demos
    st.markdown("---")
    if st.checkbox("Show Model Comparison"):
        st.markdown("### Model Comparison")

        models_to_compare = st.multiselect(
            "Select models to compare",
            ["Transformer", "CNN", "RNN", "Diffusion", "GAN"],
            default=["Transformer", "CNN", "RNN"]
        )

        comparison_data = []
        for model in models_to_compare:
            # Generate some comparison metrics (simulated)
            np.random.seed(hash(model) % 1000)

            if model == "Transformer":
                params = np.random.randint(100, 1000) * 1e6
                speed = np.random.uniform(0.5, 2.0)
                accuracy = np.random.uniform(0.7, 0.95)
                memory = np.random.uniform(2, 16)

            elif model == "CNN":
                params = np.random.randint(10, 100) * 1e6
                speed = np.random.uniform(2.0, 10.0)
                accuracy = np.random.uniform(0.6, 0.9)
                memory = np.random.uniform(1, 4)

            elif model == "RNN":
                params = np.random.randint(1, 50) * 1e6
                speed = np.random.uniform(0.1, 1.0)
                accuracy = np.random.uniform(0.5, 0.85)
                memory = np.random.uniform(0.5, 2)

            elif model == "Diffusion":
                params = np.random.randint(500, 2000) * 1e6
                speed = np.random.uniform(0.01, 0.1)
                accuracy = np.random.uniform(0.8, 0.99)
                memory = np.random.uniform(8, 32)

            else:  # GAN
                params = np.random.randint(50, 500) * 1e6
                speed = np.random.uniform(0.5, 5.0)
                accuracy = np.random.uniform(0.6, 0.9)
                memory = np.random.uniform(2, 8)

            comparison_data.append({
                'Model': model,
                'Parameters (M)': params / 1e6,
                'Speed (relative)': speed,
                'Accuracy': accuracy,
                'Memory (GB)': memory
            })

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison)

            # Radar chart for comparison
            fig = go.Figure()

            for _, row in df_comparison.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        row['Parameters (M)'] / df_comparison['Parameters (M)'].max(),
                        row['Speed (relative)'] / df_comparison['Speed (relative)'].max(),
                        row['Accuracy'],
                        row['Memory (GB)'] / df_comparison['Memory (GB)'].max()
                    ],
                    theta=['Parameters', 'Speed', 'Accuracy', 'Memory'],
                    fill='toself',
                    name=row['Model']
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Comparison Radar Chart",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)