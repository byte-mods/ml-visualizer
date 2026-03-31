"""
Neural Network Architecture Visualizations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from typing import Tuple, List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import networkx as nx
from scipy import spatial


def create_neural_network_diagram(
    layers: List[Tuple[str, int, Optional[str]]],
    layer_spacing: float = 2.0,
    node_spacing: float = 0.5,
    node_radius: float = 0.3,
    show_weights: bool = False,
    title: str = "Neural Network Architecture"
) -> go.Figure:
    """
    Create a diagram of a neural network

    Args:
        layers: List of (layer_type, n_neurons, activation) tuples
        layer_spacing: Horizontal spacing between layers
        node_spacing: Vertical spacing between nodes
        node_radius: Radius of node circles
        show_weights: Whether to show weight connections
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Colors for different layer types
    layer_colors = {
        'input': 'lightblue',
        'dense': 'lightgreen',
        'convolutional': 'lightcoral',
        'pooling': 'lightyellow',
        'recurrent': 'lightpink',
        'attention': 'lightgray',
        'output': 'lightsalmon'
    }

    # Store node positions
    node_positions = []
    node_trace_x = []
    node_trace_y = []
    node_trace_text = []
    node_trace_color = []

    # Create nodes for each layer
    for layer_idx, (layer_type, n_neurons, activation) in enumerate(layers):
        x_pos = layer_idx * layer_spacing

        # Special handling for convolutional/pooling layers (show as rectangles)
        if layer_type in ['convolutional', 'pooling']:
            # Create rectangle instead of circles
            y_center = 0
            height = n_neurons * node_spacing * 0.5
            width = node_radius * 3

            fig.add_shape(
                type="rect",
                x0=x_pos - width/2,
                x1=x_pos + width/2,
                y0=y_center - height/2,
                y1=y_center + height/2,
                fillcolor=layer_colors.get(layer_type, 'lightgray'),
                line=dict(color="black", width=1),
                opacity=0.7
            )

            # Add label
            fig.add_annotation(
                x=x_pos,
                y=y_center + height/2 + 0.2,
                text=f"{layer_type}<br>{n_neurons} filters",
                showarrow=False,
                font=dict(size=10)
            )

            # Store position for connections
            for i in range(3):  # Show 3 representative nodes
                y_pos = y_center - height/2 + (i + 0.5) * height/3
                node_positions.append((x_pos, y_pos))
                node_trace_x.append(x_pos)
                node_trace_y.append(y_pos)
                node_trace_text.append(f"{layer_type}")
                node_trace_color.append(layer_colors.get(layer_type, 'lightgray'))

        else:
            # Regular layers with circles
            y_start = -(n_neurons - 1) * node_spacing / 2

            for neuron_idx in range(n_neurons):
                y_pos = y_start + neuron_idx * node_spacing

                node_positions.append((x_pos, y_pos))
                node_trace_x.append(x_pos)
                node_trace_y.append(y_pos)

                # Node text
                if layer_idx == 0:
                    text = f"Input {neuron_idx}"
                elif layer_idx == len(layers) - 1:
                    text = f"Output {neuron_idx}"
                else:
                    text = f"Hidden {layer_idx}.{neuron_idx}"

                if activation:
                    text += f"<br>{activation}"

                node_trace_text.append(text)
                node_trace_color.append(layer_colors.get(layer_type, 'lightgray'))

    # Add nodes as scatter plot
    fig.add_trace(go.Scatter(
        x=node_trace_x,
        y=node_trace_y,
        mode='markers+text',
        marker=dict(
            size=node_radius * 30,
            color=node_trace_color,
            line=dict(color='black', width=1)
        ),
        text=[t.split('<br>')[0] for t in node_trace_text],  # First line only
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_trace_text,
        hoverinfo='text',
        name='Neurons'
    ))

    # Add connections between layers
    if show_weights and len(node_positions) > 0:
        # Group nodes by layer
        layer_nodes = []
        current_idx = 0
        for layer_idx, (layer_type, n_neurons, activation) in enumerate(layers):
            if layer_type in ['convolutional', 'pooling']:
                layer_nodes.append(node_positions[current_idx:current_idx + 3])
                current_idx += 3
            else:
                layer_nodes.append(node_positions[current_idx:current_idx + n_neurons])
                current_idx += n_neurons

        # Create connections between consecutive layers
        for i in range(len(layer_nodes) - 1):
            layer1_nodes = layer_nodes[i]
            layer2_nodes = layer_nodes[i + 1]

            # For convolutional layers, show fewer connections
            max_connections = 50  # Limit for performance
            n_connections = len(layer1_nodes) * len(layer2_nodes)
            if n_connections > max_connections:
                # Sample connections
                connections = []
                step1 = max(1, len(layer1_nodes) // 10)
                step2 = max(1, len(layer2_nodes) // 10)
                for j in range(0, len(layer1_nodes), step1):
                    for k in range(0, len(layer2_nodes), step2):
                        connections.append((j, k))
            else:
                connections = [(j, k) for j in range(len(layer1_nodes))
                              for k in range(len(layer2_nodes))]

            # Add connection lines
            for j, k in connections:
                x_vals = [layer1_nodes[j][0], layer2_nodes[k][0]]
                y_vals = [layer1_nodes[j][1], layer2_nodes[k][1]]

                # Random weight for visualization
                weight = np.random.randn()

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(
                        width=abs(weight) * 2,
                        color='red' if weight < 0 else 'blue'
                    ),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Add layer labels at the top
    for layer_idx, (layer_type, n_neurons, activation) in enumerate(layers):
        x_pos = layer_idx * layer_spacing
        y_pos = max(node_trace_y) + 1.5 if node_trace_y else 5

        layer_label = f"{layer_type.capitalize()}"
        if layer_type not in ['convolutional', 'pooling']:
            layer_label += f" ({n_neurons})"
        if activation:
            layer_label += f"<br>{activation}"

        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=layer_label,
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            opacity=0.9
        )

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        plot_bgcolor='white',
        width=800,
        height=600
    )

    return fig


def create_attention_visualization(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    n_heads: int = 1,
    title: str = "Attention Mechanism"
) -> go.Figure:
    """
    Visualize attention mechanism

    Args:
        query: Query matrix of shape (seq_len, d_k)
        key: Key matrix of shape (seq_len, d_k)
        value: Value matrix of shape (seq_len, d_v)
        n_heads: Number of attention heads
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    seq_len = query.shape[0]

    # Compute attention scores
    attention_scores = np.dot(query, key.T) / np.sqrt(query.shape[1])
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    attention_output = np.dot(attention_weights, value)

    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Query Matrix", "Key Matrix", "Value Matrix",
            "Attention Scores", "Attention Weights", "Output"
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]
        ]
    )

    # Row 1: Input matrices
    fig.add_trace(
        go.Heatmap(z=query, colorscale='Blues', showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=key, colorscale='Greens', showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=value, colorscale='Reds', showscale=False),
        row=1, col=3
    )

    # Row 2: Attention computation
    fig.add_trace(
        go.Heatmap(z=attention_scores, colorscale='Viridis', showscale=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(z=attention_weights, colorscale='Plasma',
                  zmin=0, zmax=1, showscale=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Heatmap(z=attention_output, colorscale='Rainbow', showscale=False),
        row=2, col=3
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )

    # Update axes
    for i in range(1, 7):
        fig.update_xaxes(title_text="Dimension", row=(i-1)//3 + 1, col=(i-1)%3 + 1)
        fig.update_yaxes(title_text="Sequence", row=(i-1)//3 + 1, col=(i-1)%3 + 1)

    return fig


def create_cnn_feature_maps(
    input_image: np.ndarray,
    kernels: List[np.ndarray],
    title: str = "CNN Feature Maps"
) -> go.Figure:
    """
    Visualize CNN feature maps

    Args:
        input_image: Input image of shape (height, width, channels)
        kernels: List of convolutional kernels
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    n_kernels = len(kernels)
    n_cols = min(4, n_kernels)
    n_rows = (n_kernels + n_cols - 1) // n_cols

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Kernel {i+1}" for i in range(n_kernels)]
    )

    # Apply each kernel to the input image
    for i, kernel in enumerate(kernels):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Simple convolution for visualization
        from scipy import ndimage
        if len(input_image.shape) == 2:
            input_2d = input_image
        else:
            input_2d = np.mean(input_image, axis=-1)

        # Ensure kernel is 2D
        if len(kernel.shape) > 2:
            kernel_2d = np.mean(kernel, axis=-1)
        else:
            kernel_2d = kernel

        # Convolve
        feature_map = ndimage.convolve(input_2d, kernel_2d, mode='constant', cval=0)

        fig.add_trace(
            go.Heatmap(z=feature_map, colorscale='Gray', showscale=False),
            row=row, col=col
        )

        # Add kernel visualization as annotation
        kernel_text = f"{kernel_2d.shape[0]}x{kernel_2d.shape[1]}"
        fig.add_annotation(
            x=0.5, y=1.05,
            xref=f"x{col if n_rows > 1 else ''}",
            yref=f"y{row if n_cols > 1 else ''}",
            text=kernel_text,
            showarrow=False,
            font=dict(size=10)
        )

    fig.update_layout(
        title=title,
        height=200 * n_rows,
        showlegend=False
    )

    return fig


def create_training_progress_plot(
    history: Dict[str, List[float]],
    title: str = "Training Progress"
) -> go.Figure:
    """
    Plot training and validation metrics over epochs

    Args:
        history: Dictionary with metrics lists
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (metric_name, values) in enumerate(history.items()):
        epochs = list(range(1, len(values) + 1))

        # Determine line style
        if 'val_' in metric_name:
            line_style = 'dash'
            color = colors[(i - 1) % len(colors)] if i > 0 else colors[0]
        else:
            line_style = 'solid'
            color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=epochs,
            y=values,
            mode='lines',
            name=metric_name,
            line=dict(color=color, dash=line_style, width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
        hovermode='x unified',
        showlegend=True
    )

    return fig


def create_weight_distribution_plot(
    weights: List[np.ndarray],
    layer_names: List[str],
    title: str = "Weight Distributions"
) -> go.Figure:
    """
    Plot weight distributions for each layer

    Args:
        weights: List of weight matrices
        layer_names: Names of layers
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for i, (weight_matrix, layer_name) in enumerate(zip(weights, layer_names)):
        # Flatten weights
        flat_weights = weight_matrix.flatten()

        # Create histogram
        fig.add_trace(go.Histogram(
            x=flat_weights,
            name=layer_name,
            opacity=0.7,
            nbinsx=50
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Weight Value",
        yaxis_title="Count",
        barmode='overlay',
        bargap=0.1
    )

    # Add kernel density estimate
    fig.update_traces(
        histnorm='probability density',
        hoverinfo='x+y'
    )

    return fig


# ============================================================================
# Streamlit UI Functions
# ============================================================================

def show_neural_network_ui():
    """Streamlit UI for neural network visualizations"""
    st.markdown("## 🧠 Neural Network Architectures")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")

        visualization_type = st.selectbox(
            "Visualization Type",
            [
                "Network Diagram",
                "Attention Mechanism",
                "CNN Feature Maps",
                "Training Progress",
                "Weight Distributions"
            ]
        )

        if visualization_type == "Network Diagram":
            st.markdown("#### Layer Configuration")
            n_layers = st.slider("Number of layers", 2, 10, 4, 1)

            layers = []
            for i in range(n_layers):
                st.markdown(f"##### Layer {i}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if i == 0:
                        layer_type = "input"
                    elif i == n_layers - 1:
                        layer_type = "output"
                    else:
                        layer_type = st.selectbox(
                            "Type",
                            ["dense", "convolutional", "pooling", "recurrent"],
                            key=f"layer_type_{i}"
                        )

                with col_b:
                    if layer_type in ["convolutional", "pooling"]:
                        n_units = st.slider(
                            "Filters/kernel size",
                            1, 64, 16, 1,
                            key=f"n_units_{i}"
                        )
                    else:
                        n_units = st.slider(
                            "Neurons",
                            1, 256, 32 if i == 0 else 64,
                            1, key=f"n_units_{i}"
                        )

                with col_c:
                    if layer_type in ["dense", "output"]:
                        activation = st.selectbox(
                            "Activation",
                            ["relu", "sigmoid", "tanh", "softmax", "linear", None],
                            key=f"activation_{i}"
                        )
                    else:
                        activation = None

                layers.append((layer_type, n_units, activation))

            show_weights = st.checkbox("Show weight connections", True)
            layer_spacing = st.slider("Layer spacing", 1.0, 5.0, 2.0, 0.1)
            node_spacing = st.slider("Node spacing", 0.1, 2.0, 0.5, 0.1)

        elif visualization_type == "Attention Mechanism":
            seq_len = st.slider("Sequence length", 2, 20, 5, 1)
            d_model = st.slider("Model dimension", 4, 64, 16, 4)
            n_heads = st.slider("Number of heads", 1, 8, 1, 1)

            # Generate random queries, keys, values
            np.random.seed(42)
            query = np.random.randn(seq_len, d_model)
            key = np.random.randn(seq_len, d_model)
            value = np.random.randn(seq_len, d_model)

        elif visualization_type == "CNN Feature Maps":
            image_size = st.slider("Image size", 16, 128, 32, 16)
            n_kernels = st.slider("Number of kernels", 1, 16, 4, 1)
            kernel_size = st.slider("Kernel size", 3, 7, 3, 2)

            # Generate random image
            np.random.seed(42)
            input_image = np.random.randn(image_size, image_size, 1)

            # Generate random kernels
            kernels = []
            for i in range(n_kernels):
                kernel = np.random.randn(kernel_size, kernel_size, 1)
                kernels.append(kernel)

        elif visualization_type == "Training Progress":
            n_epochs = st.slider("Number of epochs", 10, 200, 50, 10)

            # Generate fake training history
            np.random.seed(42)
            epochs = list(range(1, n_epochs + 1))

            # Training loss
            train_loss = 1.0 + np.exp(-np.array(epochs) / 10) + 0.1 * np.random.randn(n_epochs)
            train_loss = np.maximum(train_loss, 0.1)

            # Validation loss
            val_loss = 1.2 + 0.8 * np.exp(-np.array(epochs) / 15) + 0.15 * np.random.randn(n_epochs)
            val_loss = np.maximum(val_loss, 0.1)

            # Accuracy
            train_acc = 0.3 + 0.7 * (1 - np.exp(-np.array(epochs) / 20)) + 0.05 * np.random.randn(n_epochs)
            train_acc = np.clip(train_acc, 0, 1)

            val_acc = 0.25 + 0.65 * (1 - np.exp(-np.array(epochs) / 25)) + 0.06 * np.random.randn(n_epochs)
            val_acc = np.clip(val_acc, 0, 1)

            history = {
                'loss': train_loss.tolist(),
                'val_loss': val_loss.tolist(),
                'accuracy': train_acc.tolist(),
                'val_accuracy': val_acc.tolist()
            }

        elif visualization_type == "Weight Distributions":
            n_layers = st.slider("Number of layers", 1, 5, 3, 1)

            weights = []
            layer_names = []

            for i in range(n_layers):
                # Random weights for visualization
                if i == 0:
                    # Input layer
                    n_input = st.slider(f"Layer {i} inputs", 10, 100, 32, 1, key=f"in_{i}")
                    n_output = st.slider(f"Layer {i} outputs", 10, 100, 64, 1, key=f"out_{i}")
                else:
                    n_input = n_output
                    n_output = st.slider(f"Layer {i} outputs", 10, 100, 32 if i == n_layers-1 else 64, 1, key=f"out_{i}")

                # Generate weights with different distributions
                dist_type = st.selectbox(
                    f"Distribution for layer {i}",
                    ["Normal", "Uniform", "Xavier", "He"],
                    key=f"dist_{i}"
                )

                if dist_type == "Normal":
                    layer_weights = np.random.randn(n_input, n_output) * 0.1
                elif dist_type == "Uniform":
                    layer_weights = np.random.uniform(-0.1, 0.1, (n_input, n_output))
                elif dist_type == "Xavier":
                    scale = np.sqrt(2.0 / (n_input + n_output))
                    layer_weights = np.random.randn(n_input, n_output) * scale
                else:  # He initialization
                    scale = np.sqrt(2.0 / n_input)
                    layer_weights = np.random.randn(n_input, n_output) * scale

                weights.append(layer_weights)
                layer_names.append(f"Layer {i} ({n_input}×{n_output})")

    with col1:
        st.markdown("### Visualization")

        if visualization_type == "Network Diagram":
            fig = create_neural_network_diagram(
                layers,
                layer_spacing=layer_spacing,
                node_spacing=node_spacing,
                show_weights=show_weights,
                title="Neural Network Architecture"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Attention Mechanism":
            fig = create_attention_visualization(
                query, key, value, n_heads=n_heads,
                title=f"Attention Mechanism ({n_heads} head{'s' if n_heads > 1 else ''})"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "CNN Feature Maps":
            fig = create_cnn_feature_maps(
                input_image, kernels,
                title=f"CNN Feature Maps ({n_kernels} kernels)"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Training Progress":
            fig = create_training_progress_plot(
                history,
                title="Training Progress"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Weight Distributions":
            fig = create_weight_distribution_plot(
                weights, layer_names,
                title="Weight Distributions by Layer"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Educational Content")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Architecture Types")
        st.markdown("""
        - **Feedforward Networks**: Sequential layers, no cycles
        - **Convolutional Networks**: Spatial feature extraction
        - **Recurrent Networks**: Sequential data processing
        - **Attention Networks**: Focus on relevant parts
        - **Transformers**: Self-attention based
        - **Autoencoders**: Dimensionality reduction
        - **GANs**: Generative adversarial networks
        """)

    with col2:
        st.markdown("### Design Considerations")
        st.markdown("""
        - **Depth vs Width**: More layers vs more neurons per layer
        - **Activation functions**: ReLU, sigmoid, tanh, etc.
        - **Initialization**: Xavier, He, random
        - **Regularization**: Dropout, batch norm, weight decay
        - **Optimization**: SGD, Adam, learning rate schedules
        - **Hyperparameter tuning**: Grid search, random search, Bayesian
        """)

    # Additional interactive demos
    if visualization_type == "Network Diagram" and st.checkbox("Show layer connectivity matrix"):
        st.markdown("### Layer Connectivity Matrix")

        # Create adjacency matrix
        n_total_neurons = sum([n for _, n, _ in layers])
        adj_matrix = np.zeros((n_total_neurons, n_total_neurons))

        # Simple connectivity: each neuron connects to all in next layer
        neuron_idx = 0
        for i in range(len(layers) - 1):
            n_current = layers[i][1]
            n_next = layers[i + 1][1]

            for j in range(n_current):
                for k in range(n_next):
                    adj_matrix[neuron_idx + j, neuron_idx + n_current + k] = 1

            neuron_idx += n_current

        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix,
            colorscale='Blues',
            showscale=False
        ))

        fig.update_layout(
            title="Layer Connectivity Matrix",
            xaxis_title="To Neuron",
            yaxis_title="From Neuron",
            width=600,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)