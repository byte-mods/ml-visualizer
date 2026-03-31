"""
Loss Functions and Activation Functions Visualizations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict, Any, Callable
import streamlit as st
import pandas as pd


# ============================================================================
# Loss Functions
# ============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Squared Error: MSE = (1/n) * Σ(y_true - y_pred)²"""
    return np.mean((y_true - y_pred)**2, axis=-1)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Absolute Error: MAE = (1/n) * Σ|y_true - y_pred|"""
    return np.mean(np.abs(y_true - y_pred), axis=-1)

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Huber loss: quadratic for small errors, linear for large errors"""
    error = np.abs(y_true - y_pred)
    mask = error <= delta
    loss = np.where(mask, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss, axis=-1)

def log_cosh_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Log-cosh loss: log(cosh(y_pred - y_true))"""
    error = y_pred - y_true
    return np.mean(np.log(np.cosh(error)), axis=-1)

def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Binary cross-entropy: -[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1)

def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Categorical cross-entropy: -Σ y_true * log(y_pred)"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Hinge loss: max(0, 1 - y_true * y_pred)"""
    return np.mean(np.maximum(0, 1 - y_true * y_pred), axis=-1)

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Kullback-Leibler divergence: Σ p * log(p/q)"""
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q), axis=-1)


def create_loss_function_surface(
    loss_func: Callable,
    y_true_range: Tuple[float, float] = (0, 1),
    y_pred_range: Tuple[float, float] = (0, 1),
    resolution: int = 50,
    title: str = "Loss Function Surface"
) -> go.Figure:
    """
    Create a 3D surface plot of a loss function

    Args:
        loss_func: Loss function that takes (y_true, y_pred) and returns loss
        y_true_range: Range of true values
        y_pred_range: Range of predicted values
        resolution: Number of points in each dimension
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    y_true = np.linspace(y_true_range[0], y_true_range[1], resolution)
    y_pred = np.linspace(y_pred_range[0], y_pred_range[1], resolution)
    Y_true, Y_pred = np.meshgrid(y_true, y_pred)

    # For scalar loss functions, need to handle broadcasting
    try:
        Z = loss_func(Y_true, Y_pred)
    except:
        # Handle functions that expect specific shapes
        Z = np.zeros_like(Y_true)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = loss_func(np.array([Y_true[i, j]]), np.array([Y_pred[i, j]]))

    fig = go.Figure(data=[go.Surface(z=Z, x=Y_true, y=Y_pred, colorscale='Viridis')])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="y_true",
            yaxis_title="y_pred",
            zaxis_title="Loss",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        )
    )

    return fig


def create_loss_function_contour(
    loss_func: Callable,
    y_true_range: Tuple[float, float] = (0, 1),
    y_pred_range: Tuple[float, float] = (0, 1),
    resolution: int = 100,
    title: str = "Loss Function Contour"
) -> go.Figure:
    """
    Create a contour plot of a loss function

    Args:
        loss_func: Loss function
        y_true_range: Range of true values
        y_pred_range: Range of predicted values
        resolution: Number of points in each dimension
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    y_true = np.linspace(y_true_range[0], y_true_range[1], resolution)
    y_pred = np.linspace(y_pred_range[0], y_pred_range[1], resolution)
    Y_true, Y_pred = np.meshgrid(y_true, y_pred)

    try:
        Z = loss_func(Y_true, Y_pred)
    except:
        Z = np.zeros_like(Y_true)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = loss_func(np.array([Y_true[i, j]]), np.array([Y_pred[i, j]]))

    fig = go.Figure(data=go.Contour(
        z=Z,
        x=y_true,
        y=y_pred,
        colorscale='Viridis',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title="y_true",
        yaxis_title="y_pred",
        width=600,
        height=500
    )

    return fig


def create_loss_comparison_plot(
    loss_funcs: List[Tuple[Callable, str, str]],
    error_range: Tuple[float, float] = (-3, 3),
    n_points: int = 1000,
    title: str = "Loss Functions Comparison"
) -> go.Figure:
    """
    Compare multiple loss functions on the same plot

    Args:
        loss_funcs: List of (loss_function, label, color) tuples
        error_range: Range of errors (y_pred - y_true) to plot
        n_points: Number of points
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    errors = np.linspace(error_range[0], error_range[1], n_points)
    y_true = np.zeros_like(errors)  # Assume true value is 0
    y_pred = errors  # y_pred = y_true + error

    fig = go.Figure()

    for loss_func, label, color in loss_funcs:
        losses = np.zeros_like(errors)
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            losses[i] = loss_func(np.array([yt]), np.array([yp]))

        fig.add_trace(go.Scatter(
            x=errors,
            y=losses,
            mode='lines',
            name=label,
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Error (y_pred - y_true)",
        yaxis_title="Loss",
        hovermode='x unified'
    )

    return fig


# ============================================================================
# Activation Functions
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: x if x > 0 else alpha * x"""
    return np.where(x > 0, x, alpha * x)

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU: x if x > 0 else alpha * (exp(x) - 1)"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return np.tanh(x)

def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus: log(1 + exp(x))"""
    return np.log(1 + np.exp(x))

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish: x * sigmoid(beta * x)"""
    return x * sigmoid(beta * x)

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU: x * Φ(x) where Φ is CDF of normal distribution"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax: exp(x) / Σ exp(x)"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_activation_function_plot(
    activation_funcs: List[Tuple[Callable, str, str, Dict[str, float]]],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 1000,
    show_derivative: bool = False,
    title: str = "Activation Functions"
) -> go.Figure:
    """
    Plot activation functions and optionally their derivatives

    Args:
        activation_funcs: List of (func, label, color, params) tuples
        x_range: Range of x values
        n_points: Number of points
        show_derivative: Whether to show derivative
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    fig = go.Figure()

    for func, label, color, params in activation_funcs:
        # Apply function with parameters
        y = func(x, **params) if params else func(x)

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=label,
            line=dict(color=color, width=2)
        ))

        if show_derivative:
            # Numerical derivative
            h = 1e-5
            y_deriv = func(x + h, **params) if params else func(x + h)
            y_deriv = (y_deriv - y) / h

            fig.add_trace(go.Scatter(
                x=x,
                y=y_deriv,
                mode='lines',
                name=f"{label} derivative",
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.7
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Input (x)",
        yaxis_title="Output / Derivative",
        hovermode='x unified'
    )

    return fig


def create_activation_3d_plot(
    activation_func: Callable,
    params: Dict[str, float] = None,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 50,
    title: str = "Activation Function 3D"
) -> go.Figure:
    """
    Create a 3D plot showing activation function applied to 2D inputs

    Args:
        activation_func: Activation function
        params: Function parameters
        x_range: Range of first input dimension
        y_range: Range of second input dimension
        resolution: Number of points in each dimension
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Flatten, apply activation, reshape
    inputs = np.stack([X.flatten(), Y.flatten()], axis=-1)
    if params:
        outputs = activation_func(inputs, **params)
    else:
        outputs = activation_func(inputs)

    # For scalar outputs (like sigmoid on each dimension separately)
    if outputs.ndim == 1:
        Z = outputs.reshape(resolution, resolution)
    else:
        # For vector outputs, take norm or first dimension
        Z = np.linalg.norm(outputs, axis=-1).reshape(resolution, resolution)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Input 1",
            yaxis_title="Input 2",
            zaxis_title="Activation Output",
            aspectmode='cube'
        )
    )

    return fig


# ============================================================================
# Streamlit UI Functions
# ============================================================================

def show_loss_functions_ui():
    """Streamlit UI for loss functions visualization"""
    st.markdown("## 📉 Loss Functions")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")

        visualization_type = st.selectbox(
            "Visualization Type",
            ["Function Comparison", "3D Surface", "Contour Plot"],
            key="loss_viz_type"
        )

        selected_losses = st.multiselect(
            "Select Loss Functions",
            [
                "Mean Squared Error",
                "Mean Absolute Error",
                "Huber Loss",
                "Log-cosh Loss",
                "Binary Cross-entropy",
                "Hinge Loss"
            ],
            default=["Mean Squared Error", "Mean Absolute Error", "Huber Loss"]
        )

        # Loss function mapping
        loss_func_map = {
            "Mean Squared Error": (mean_squared_error, "blue"),
            "Mean Absolute Error": (mean_absolute_error, "red"),
            "Huber Loss": (lambda yt, yp: huber_loss(yt, yp, delta=huber_delta), "green"),
            "Log-cosh Loss": (log_cosh_loss, "purple"),
            "Binary Cross-entropy": (binary_crossentropy, "orange"),
            "Hinge Loss": (hinge_loss, "brown")
        }

        # Parameters
        if "Huber Loss" in selected_losses:
            huber_delta = st.slider("Huber delta", 0.1, 5.0, 1.0, 0.1)

        if visualization_type == "3D Surface":
            selected_surface_loss = st.selectbox(
                "Loss function for 3D surface",
                selected_losses if selected_losses else ["Mean Squared Error"]
            )
            y_true_range = st.slider("y_true range", 0.0, 5.0, (0.0, 2.0), 0.1)
            y_pred_range = st.slider("y_pred range", 0.0, 5.0, (0.0, 2.0), 0.1)
            resolution = st.slider("Resolution", 20, 100, 50, 5)

        elif visualization_type == "Function Comparison":
            error_range = st.slider("Error range", -5.0, 5.0, (-3.0, 3.0), 0.1)

        elif visualization_type == "Contour Plot":
            selected_contour_loss = st.selectbox(
                "Loss function for contour plot",
                selected_losses if selected_losses else ["Mean Squared Error"]
            )
            y_true_range = st.slider("y_true range", 0.0, 5.0, (0.0, 2.0), 0.1, key="contour_true")
            y_pred_range = st.slider("y_pred range", 0.0, 5.0, (0.0, 2.0), 0.1, key="contour_pred")
            resolution = st.slider("Resolution", 20, 100, 50, 5, key="contour_res")

    with col1:
        st.markdown("### Visualization")

        if visualization_type == "Function Comparison":
            # Prepare loss functions for comparison
            loss_funcs = []
            for loss_name in selected_losses:
                func, color = loss_func_map[loss_name]
                loss_funcs.append((func, loss_name, color))

            fig = create_loss_comparison_plot(
                loss_funcs,
                error_range=error_range,
                title="Loss Functions Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "3D Surface":
            func, _ = loss_func_map[selected_surface_loss]
            fig = create_loss_function_surface(
                func,
                y_true_range=y_true_range,
                y_pred_range=y_pred_range,
                resolution=resolution,
                title=f"{selected_surface_loss} Surface"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Contour Plot":
            func, _ = loss_func_map[selected_contour_loss]
            fig = create_loss_function_contour(
                func,
                y_true_range=y_true_range,
                y_pred_range=y_pred_range,
                resolution=resolution,
                title=f"{selected_contour_loss} Contour Plot"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Chapter 4: Loss Functions - Tutorial")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Learning Objectives:</b> Understand how loss functions measure model performance and guide optimization.
    Master the mathematical formulation of key losses and their trade-offs.
    </div>
    """, unsafe_allow_html=True)

    # Mathematical Formulas
    st.markdown("### 📖 Mathematical Formulas")

    with st.expander("📐 Regression Losses"):
        st.markdown(r"""
        **Mean Squared Error (MSE / L2 Loss):**
        ```
        L_{MSE} = (1/n) Σ_{i=1}^{n} (ŷ_i - y_i)²

        Gradient: ∂L/∂ŷ = (2/n)(ŷ - y)

        Properties:
        - Penalizes large errors heavily (squared)
        - Smooth gradients everywhere
        - Sensitive to outliers
        - Units: squared error units
        ```

        **Mean Absolute Error (MAE / L1 Loss):**
        ```
        L_{MAE} = (1/n) Σ_{i=1}^{n} |ŷ_i - y_i|

        Subgradient: ∂L/∂ŷ = sign(ŷ - y)

        Properties:
        - Linear penalty for errors
        - Robust to outliers
        - Non-smooth at zero (subgradient)
        - Units: same as y units
        ```

        **Huber Loss (Smooth L1):**
        ```
        L_{Huber}(y, ŷ) = { 0.5(y - ŷ)²     if |y - ŷ| ≤ δ
                           { δ|y - ŷ| - 0.5δ²  otherwise }

        Derivative: ∂L/∂ŷ = { (ŷ - y)        if |ŷ - y| ≤ δ
                            { δ · sign(ŷ - y)  otherwise }

        Used in: Fast R-CNN, CornerNet, RetinaNet
        ```
        """)

    with st.expander("📐 Classification Losses"):
        st.markdown(r"""
        **Binary Cross-Entropy (Log Loss):**
        ```
        L_{BCE} = -(1/n) Σ [y_i · log(ŷ_i) + (1-y_i) · log(1-ŷ_i)]

        where ŷ_i = σ(z_i) = 1/(1 + e^{-z_i})

        Gradient (with sigmoid): ∂L/∂z = ŷ - y

        Properties:
        - Natural for probabilistic outputs
        - Beautiful gradient when combined with sigmoid
        - Penalizes confident wrong predictions heavily
        ```

        **Categorical Cross-Entropy:**
        ```
        L_{CCE} = -Σ_c y_c · log(ŷ_c)

        where y_c = true probability (one-hot)
              ŷ_c = predicted probability

        For softmax outputs:
        ∂L/∂z_k = ŷ_k - y_k (beautiful!)

        Used in: Image classification, language modeling
        ```

        **Hinge Loss (SVM):**
        ```
        L_{Hinge} = max(0, 1 - y · ŷ)

        where y ∈ {-1, +1} is true label
              ŷ is predicted score

        Properties:
        - Margin-based loss
        - Sparse gradients (dead zone)
        - Used in SVMs and max-margin methods
        ```
        """)

    with st.expander("📐 Regularization Losses"):
        st.markdown(r"""
        **L2 Regularization (Weight Decay):**
        ```
        L_{total} = L_{task} + λ · ||W||²_2
                   = L_{task} + λ · Σ_{i,j} W_{ij}²

        Gradient: ∂L/∂W = ∂L_task/∂W + 2λW

        Effect: Shrinks weights toward zero
        Used in: Most neural networks (AdamW)
        ```

        **L1 Regularization (Lasso):**
        ```
        L_{total} = L_{task} + λ · ||W||_1
                   = L_{task} + λ · Σ_{i,j} |W_{ij}|

        Gradient: ∂L/∂W = ∂L_task/∂W + λ · sign(W)

        Effect: Induces sparsity (feature selection)
        Used in: Compressed sensing, sparse models
        ```

        **Elastic Net:**
        ```
        L_{total} = L_{task} + λ_1||W||_1 + λ_2||W||²_2
        ```
        """)

    # ML Applications
    st.markdown("### 🛠️ ML Applications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **When to Use What:**

        | Task | Loss Function | Why |
        |------|---------------|-----|
        | Regression | MSE | Smooth, penalizes large errors |
        | Robust regression | MAE/Huber | Outlier resistant |
        | Binary classification | BCE | Probabilistic outputs |
        | Multi-class | Cross-entropy | Softmax + one-hot |
        | Ranking | Hinge/Margin | Pairwise comparisons |
        | Object detection | Focal Loss | Class imbalance |

        **Focal Loss (for class imbalance):**
        ```
        L_{focal} = -α(1-ŷ)^γ · log(ŷ)

        γ > 0: Focuses on hard examples
        γ = 2, α = 0.25: Default (RetinaNet)
        ```
        """)

    with col2:
        st.markdown("""
        **Custom Loss Functions:**

        Sometimes you need domain-specific losses:

        ```python
        # Dice loss for segmentation
        def dice_loss(y_pred, y_true):
            smooth = 1.0
            intersection = np.sum(y_pred * y_true)
            return 1 - (2. * intersection + smooth) / \
                   (np.sum(y_pred) + np.sum(y_true) + smooth)

        # Contrastive loss for embeddings
        def contrastive_loss(emb1, emb2, label, margin=1.0):
            dist = np.linalg.norm(emb1 - emb2)
            return label * dist² + (1-label) * max(0, margin-dist)²
        ```

        **KL Divergence (Distribution Matching):**
        ```
        D_{KL}(P||Q) = Σ P(x) · log(P(x)/Q(x))

        Used in:
        - Variational autoencoders
        - Knowledge distillation
        - Language model smoothing
        ```
        """)

    # Key Takeaways
    st.markdown("### 🎯 Key Takeaways")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
    1. <b>Loss functions define what you're optimizing</b> - Choose based on your task and data characteristics

    2. <b>MSE + sigmoid has vanishing gradients</b> - BCE + sigmoid has nice gradient ŷ - y

    3. <b>MAE/Huber are robust to outliers</b> - MSE is not (squared errors amplify large mistakes)

    4. <b>Regularization is added to the loss</b> - L2 shrinks weights, L1 induces sparsity

    5. <b>Custom losses enable domain adaptation</b> - Segmentation uses Dice, detection uses Focal
    </div>
    """, unsafe_allow_html=True)

def show_activation_functions_ui():
    """Streamlit UI for activation functions visualization"""
    st.markdown("## ⚡ Activation Functions")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")

        visualization_type = st.selectbox(
            "Visualization Type",
            ["Function Comparison", "3D Surface", "With Derivatives"],
            key="activation_viz_type"
        )

        selected_activations = st.multiselect(
            "Select Activation Functions",
            [
                "Sigmoid",
                "ReLU",
                "Leaky ReLU",
                "ELU",
                "Tanh",
                "Softplus",
                "Swish",
                "GELU"
            ],
            default=["Sigmoid", "ReLU", "Tanh"]
        )

        # Parameters
        leaky_alpha = 0.01
        elu_alpha = 1.0
        swish_beta = 1.0

        if "Leaky ReLU" in selected_activations:
            leaky_alpha = st.slider("Leaky ReLU alpha", 0.01, 0.5, 0.01, 0.01)

        if "ELU" in selected_activations:
            elu_alpha = st.slider("ELU alpha", 0.1, 2.0, 1.0, 0.1)

        if "Swish" in selected_activations:
            swish_beta = st.slider("Swish beta", 0.1, 5.0, 1.0, 0.1)

        # Activation function mapping
        activation_map = {
            "Sigmoid": (sigmoid, "blue", {}),
            "ReLU": (relu, "red", {}),
            "Leaky ReLU": (leaky_relu, "green", {"alpha": leaky_alpha}),
            "ELU": (elu, "purple", {"alpha": elu_alpha}),
            "Tanh": (tanh, "orange", {}),
            "Softplus": (softplus, "brown", {}),
            "Swish": (swish, "pink", {"beta": swish_beta}),
            "GELU": (gelu, "gray", {})
        }

        x_range = st.slider("Input range", -10.0, 10.0, (-5.0, 5.0), 0.1)

        if visualization_type == "3D Surface":
            selected_3d_activation = st.selectbox(
                "Activation for 3D surface",
                selected_activations if selected_activations else ["Sigmoid"]
            )
            x_range_3d = st.slider("X range", -5.0, 5.0, (-3.0, 3.0), 0.1, key="x_3d")
            y_range_3d = st.slider("Y range", -5.0, 5.0, (-3.0, 3.0), 0.1, key="y_3d")
            resolution = st.slider("Resolution", 20, 100, 50, 5)

        show_derivative = st.checkbox("Show derivatives", visualization_type == "With Derivatives")

    with col1:
        st.markdown("### Visualization")

        if visualization_type in ["Function Comparison", "With Derivatives"]:
            # Prepare activation functions
            activation_funcs = []
            for act_name in selected_activations:
                func, color, params_dict = activation_map[act_name]
                # Get current parameter values
                params = {}
                for param_name, param_value in params_dict.items():
                    if callable(param_value):
                        # Parameter is a function that returns the value
                        params[param_name] = param_value
                    else:
                        params[param_name] = param_value

                activation_funcs.append((func, act_name, color, params))

            fig = create_activation_function_plot(
                activation_funcs,
                x_range=x_range,
                show_derivative=show_derivative,
                title="Activation Functions"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "3D Surface":
            func, color, params_dict = activation_map[selected_3d_activation]
            params = {}
            for param_name, param_value in params_dict.items():
                if callable(param_value):
                    params[param_name] = param_value
                else:
                    params[param_name] = param_value

            fig = create_activation_3d_plot(
                func,
                params=params if params else None,
                x_range=x_range_3d,
                y_range=y_range_3d,
                resolution=resolution,
                title=f"{selected_3d_activation} 3D Surface"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Chapter 5: Activation Functions - Tutorial")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Learning Objectives:</b> Understand why activations are non-linear and how they enable neural networks
    to approximate complex functions. Master the mathematical properties of each activation.
    </div>
    """, unsafe_allow_html=True)

    # Mathematical Formulas
    st.markdown("### 📖 Mathematical Formulas")

    with st.expander("📐 Classic Activations"):
        st.markdown(r"""
        **Sigmoid (Logistic):**
        ```
        σ(x) = 1 / (1 + e^{-x})

        Derivative: σ'(x) = σ(x) · (1 - σ(x))

        Range: (0, 1)
        Properties:
        - Squashes to (0, 1) - good for probabilities
        - Gradient = σ(1-σ) - saturates at 0 and 1
        - Historically popular (logistic regression)
        - Problem: Vanishing gradients, not zero-centered
        ```

        **Tanh (Hyperbolic Tangent):**
        ```
        tanh(x) = (e^{x} - e^{-x}) / (e^{x} + e^{-x})

        Derivative: tanh'(x) = 1 - tanh²(x)

        Range: (-1, 1)
        Properties:
        - Zero-centered (better than sigmoid)
        - Stronger gradients near zero
        - Still saturates at extremes
        - Used in: LSTM, GRU (often better than sigmoid)
        ```
        """)

    with st.expander("📐 Modern Activations"):
        st.markdown(r"""
        **ReLU (Rectified Linear Unit):**
        ```
        ReLU(x) = max(0, x) = { x if x > 0
                               { 0 otherwise

        Derivative: ReLU'(x) = { 1 if x > 0
                                { 0 otherwise

        Range: [0, ∞)
        Properties:
        - Sparse activations (some neurons are zero)
        - No saturation for x > 0
        - Efficient computation
        - Dying ReLU problem: neurons can "die" if always negative
        - Default choice for hidden layers
        ```

        **Leaky ReLU:**
        ```
        LeakyReLU(x) = max(αx, x) where α is small (e.g., 0.01)

        Derivative: LeakyReLU'(x) = { 1 if x > 0
                                     { α otherwise

        Properties:
        - Fixes dying ReLU problem
        - Small gradient for negative values
        - α is a hyperparameter to tune
        ```

        **ELU (Exponential Linear Unit):**
        ```
        ELU(x) = { x                    if x > 0
                 { α(e^{x} - 1)          if x ≤ 0

        Derivative: ELU'(x) = { 1                    if x > 0
                               { ELU(x) + α           if x ≤ 0

        Properties:
        - Smooth approximation to ReLU
        - Negative values allow mean to be closer to zero
        - α typically = 1
        ```
        """)

    with st.expander("📐 Transformer Activations"):
        st.markdown(r"""
        **GELU (Gaussian Error Linear Unit):**
        ```
        GELU(x) = x · Φ(x)
                 = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

        where Φ is CDF of standard normal

        Approximation: GELU(x) ≈ x · σ(1.702x)

        Used in: BERT, GPT, RoBERTa, and most transformers

        Properties:
        - Probabilistic interpretation (weighted combination of ReLU and dropout)
        - Smooth everywhere
        - Better performance than ReLU in transformers
        ```

        **Swish:**
        ```
        Swish(x) = x · σ(βx) = x / (1 + e^{-βx})

        where β is a learnable parameter (or fixed = 1)

        Derivative: Swish'(x) = σ(βx) + βx · σ(βx)(1 - σ(βx))

        Properties:
        - Self-gated activation (multiplies input by sigmoid of itself)
        - Unbounded in positive region
        - Google's search paper: better than ReLU in some tasks
        ```
        """)

    with st.expander("📐 Softmax (Multi-class Output):"):
        st.markdown(r"""
        **Softmax Function:**
        ```
        softmax(x_i) = e^{x_i} / Σ_{j=1}^{K} e^{x_j}

        Properties:
        - Outputs sum to 1 (probability distribution)
        - Exponentiation amplifies differences
        - Can overflow with large x values (use log-softmax in practice)

        Log-Softmax (numerically stable):
        ```
        log_softmax(x_i) = x_i - log Σ e^{x_j}
                          = x_i - x_max - log Σ e^{x_j - x_max}
        ```

        Cross-entropy with softmax:
        ```
        L = -Σ y_i · log(softmax(x)_i)
          = -Σ y_i · (x_i - log Σ e^{x_j})  [when y is one-hot]
        ```
        """)

    # ML Applications
    st.markdown("### 🛠️ ML Applications & Best Practices")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Activation Selection Guide:**

        | Layer Type | Recommended Activations |
        |-----------|------------------------|
        | Input | None (linear) |
        | Hidden (CNN) | ReLU, Leaky ReLU, GELU |
        | Hidden (RNN) | Tanh, ReLU |
        | Hidden (Transformer) | GELU |
        | Binary Output | Sigmoid |
        | Multi-class Output | Softmax |
        | Bounds required | Sigmoid, Tanh |

        **Common Mistakes:**
        1. Using sigmoid in hidden layers (vanishing gradients)
        2. Using ReLU with unnormalized data (dying ReLU)
        3. Using softmax for multi-label (use sigmoid instead)
        """)

    with col2:
        st.markdown("""
        **Why Non-Linearity Matters:**

        Without non-linear activations:
        ```
        y = W₂(W₁x + b₁) + b₂
          = (W₂W₁)x + (W₂b₁ + b₂)
          = W'x + b'        # Still linear!

        Multiple layers = single linear transformation
        ```

        With non-linearity:
        ```
        y = σ(W₂σ(W₁x + b₁) + b₂)
        Non-linear σ allows composition of non-linear functions
        Universal Approximation: 1 hidden layer can approximate any function
        ```

        **Self-Normalizing: SELU:**
        ```
        SELU(x) = λ · { x                    if x > 0
                       { α(e^{x} - 1)        if x ≤ 0

        With proper initialization: α ≈ 1.673, λ ≈ 1.051
        Property: Output tends to zero-mean, unit-variance
        ```
        """)

    # Key Takeaways
    st.markdown("### 🎯 Key Takeaways")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
    1. <b>Non-linearity is essential</b> - Without it, deep networks = single linear layer

    2. <b>ReLU is the default for hidden layers</b> - Simple, efficient, sparse activations

    3. <b>GELU is standard in Transformers</b> - Used in BERT, GPT, and most modern LLMs

    4. <b>Sigmoid/Softmax for outputs</b> - Sigmoid for binary, Softmax for multi-class

    5. <b>Activation affects gradient flow</b> - Dying ReLU, vanishing gradients are real problems

    6. <b>Zero-centered helps</b> - Tanh is zero-centered, ReLU is not (but sparse)
    </div>
    """, unsafe_allow_html=True)

    # Softmax visualization
    st.markdown("---")
    if st.checkbox("Show Softmax Visualization"):
        st.markdown("### Softmax Function")

        n_classes = st.slider("Number of classes", 2, 10, 3, 1)
        logits = []
        for i in range(n_classes):
            logit = st.slider(f"Logit for class {i}", -5.0, 5.0, float(i), 0.1, key=f"logit_{i}")
            logits.append(logit)

        logits_array = np.array(logits).reshape(1, -1)
        probabilities = softmax(logits_array)[0]

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Class {i}" for i in range(n_classes)],
                y=probabilities,
                marker_color='lightblue',
                text=[f"{p:.3f}" for p in probabilities],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title=f"Softmax Probabilities (sum = {probabilities.sum():.3f})",
            xaxis_title="Class",
            yaxis_title="Probability",
            yaxis_range=[0, 1]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show temperature scaling
        st.markdown("#### Temperature Scaling")
        temperature = st.slider("Temperature", 0.1, 5.0, 1.0, 0.1)
        scaled_probs = softmax(logits_array / temperature)[0]

        fig2 = go.Figure(data=[
            go.Bar(
                x=[f"Class {i}" for i in range(n_classes)],
                y=scaled_probs,
                marker_color='lightgreen',
                text=[f"{p:.3f}" for p in scaled_probs],
                textposition='auto'
            )
        ])

        fig2.update_layout(
            title=f"Softmax with Temperature {temperature} (sum = {scaled_probs.sum():.3f})",
            xaxis_title="Class",
            yaxis_title="Probability",
            yaxis_range=[0, 1]
        )

        st.plotly_chart(fig2, use_container_width=True)