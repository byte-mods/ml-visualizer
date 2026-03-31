"""
Mathematical Foundations for Deep Learning
Interactive visualizations of key formulas with adjustable parameters
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Tuple, Dict, Any


def create_gradient_descent_visualization(
    func_type: str = "parabola",
    learning_rate: float = 0.1,
    n_steps: int = 50,
    start_x: float = -5.0
) -> go.Figure:
    """Visualize gradient descent on various loss landscapes."""

    fig = go.Figure()

    # Define the function and its derivative
    if func_type == "parabola":
        def f(x): return (x - 2)**2
        def df(x): return 2 * (x - 2)
        x_range = (-6, 6)
        title = "Minimizing f(x) = (x - 2)²"
    elif func_type == "cubic":
        def f(x): return (x - 2)**3 + x**2
        def df(x): return 3*(x - 2)**2 + 2*x
        x_range = (-4, 4)
        title = "Minimizing f(x) = (x - 2)³ + x²"
    elif func_type == "saddle":
        def f(x): return x[0]**2 - x[1]**2
        def df(x): return np.array([2*x[0], -2*x[1]])
        x_range = (-4, 4)
        title = "Saddle Point: f(x,y) = x² - y²"
    else:  # rosenbrock-like
        def f(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        def df(x): return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])
        x_range = (-3, 3)
        title = "Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²"

    # Plot the function
    x = np.linspace(x_range[0], x_range[1], 500)
    if func_type in ["saddle", "rosenbrock"]:
        y = np.linspace(x_range[0], x_range[1], 500)
        X, Y = np.meshgrid(x, y)
        Z = f([X, Y])
        fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Viridis', opacity=0.7))
    else:
        Y = f(x)
        fig.add_trace(go.Scatter(x=x, y=Y, mode='lines', name='f(x)', line=dict(color='blue')))

    # Gradient descent
    if func_type in ["saddle", "rosenbrock"]:
        path = [np.array([start_x, start_x])]
        current = path[-1].copy()
        for _ in range(n_steps):
            grad = df(current)
            current = current - learning_rate * grad
            path.append(current.copy())
        path = np.array(path)

        fig.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1],
            mode='markers+lines',
            marker=dict(size=8, color=range(len(path)), colorscale='Reds'),
            line=dict(color='red', width=2),
            name='Gradient Descent'
        ))
    else:
        path = [start_x]
        current = start_x
        for _ in range(n_steps):
            grad = df(current)
            current = current - learning_rate * grad
            path.append(current)

        fig.add_trace(go.Scatter(
            x=path, y=[f(x) for x in path],
            mode='markers+lines',
            marker=dict(size=10, color=range(len(path)), colorscale='Reds'),
            line=dict(color='red', width=2),
            name='Gradient Descent'
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br>Learning Rate = {learning_rate}, Steps = {n_steps}", x=0.5),
        height=450,
        showlegend=True
    )

    return fig


def create_backprop_visualization(
    input_val: float = 1.0,
    weight1: float = 0.5,
    weight2: float = 0.3,
    activation: str = "sigmoid"
) -> go.Figure:
    """Visualize backpropagation through a simple 2-layer network."""

    fig = go.Figure()

    # Forward pass
    z1 = input_val * weight1
    if activation == "sigmoid":
        a1 = 1 / (1 + np.exp(-z1))
        da1_dz1 = a1 * (1 - a1)
    elif activation == "relu":
        a1 = max(0, z1)
        da1_dz1 = 1 if z1 > 0 else 0
    else:  # tanh
        a1 = np.tanh(z1)
        da1_dz1 = 1 - a1**2

    z2 = a1 * weight2
    output = z2

    # Backward pass (gradient computation)
    dL_dz2 = 1  # Assuming MSE loss derivative at output
    dL_dw2 = dL_dz2 * a1
    dL_da1 = dL_dz2 * weight2
    dL_dz1 = dL_da1 * da1_dz1
    dL_dw1 = dL_dz1 * input_val

    # Create visualization
    # Forward pass values
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3],
        y=[1, 1, 1, 1],
        mode='markers+text',
        marker=dict(size=[30, 30, 30, 30], color=['#3498db', '#3498db', '#e74c3c', '#e74c3c']),
        text=['Input<br>x='+str(input_val), f'z₁={z1:.2f}<br>a₁={a1:.2f}', f'z₂={z2:.2f}', f'Output={output:.2f}'],
        textposition='middle center',
        textfont=dict(size=10),
        name='Forward Pass'
    ))

    # Annotations for values
    fig.add_annotation(x=0, y=1.3, text=f"Input: {input_val}", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=1, y=1.3, text=f"z₁={z1:.2f}", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=1, y=0.7, text=f"a₁={a1:.2f}", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=2, y=1.3, text=f"z₂={z2:.2f}", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=3, y=1.3, text=f"Output: {output:.2f}", showarrow=False, font=dict(size=12))

    # Connections
    for i in range(3):
        fig.add_shape(type="line", x0=i, y0=1, x1=i+1, y1=1,
                     line=dict(width=2, color='gray'))

    fig.update_layout(
        title=dict(text=f"<b>Forward Pass</b><br>2-Layer Network with {activation}", x=0.5),
        xaxis=dict(range=[-0.5, 3.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
        height=250,
        showlegend=False
    )

    # Gradient visualization
    fig2 = go.Figure()

    # Gradient values
    grad_labels = [
        f'∂L/∂w₂={dL_dw2:.2f}',
        f'∂L/∂a₁={dL_da1:.2f}',
        f'∂L/∂z₁={dL_dz1:.2f}',
        f'∂L/∂w₁={dL_dw1:.2f}'
    ]

    fig2.add_trace(go.Scatter(
        x=[1, 2, 1, 2],
        y=[1, 1, 2, 2],
        mode='markers+text',
        marker=dict(size=[30, 30, 30, 30], color=['#9b59b6', '#9b59b6', '#2ecc71', '#2ecc71']),
        text=grad_labels,
        textposition='middle center',
        textfont=dict(size=9),
        name='Gradients'
    ))

    # Arrows showing gradient flow (reverse direction)
    fig2.add_annotation(x=1.5, y=1.1, ax=1.5, ay=1.4, xref='x', yref='y', axref='x', ayref='y',
                       text='', showarrow=True, arrowhead=2, arrowcolor='red', arrowsize=2)
    fig2.add_annotation(x=2, y=1.5, ax=1, ay=1.5, xref='x', yref='y', axref='x', ayref='y',
                       text='', showarrow=True, arrowhead=2, arrowcolor='red', arrowsize=2)

    fig2.update_layout(
        title=dict(text=f"<b>Backward Pass (Gradients)</b><br>Red arrows show gradient flow", x=0.5),
        xaxis=dict(range=[0.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
        height=250,
        showlegend=False
    )

    # Combine
    combined = make_subplots(rows=2, cols=1, subplot_titles=(
        f"Forward Pass: Input={input_val}, w₁={weight1}, w₂={weight2}, act={activation}",
        f"Backward Pass: ∂L/∂w₁={dL_dw1:.3f}, ∂L/∂w₂={dL_dw2:.3f}"
    ))

    # Forward trace
    for trace in fig.data:
        combined.add_trace(trace, row=1, col=1)
    # Gradient trace
    for trace in fig2.data:
        combined.add_trace(trace, row=2, col=1)

    combined.update_layout(height=550, showlegend=False)
    return combined


def create_activation_function_visualization(
    func_type: str = "sigmoid"
) -> go.Figure:
    """Visualize activation functions and their derivatives."""

    x = np.linspace(-10, 10, 500)

    if func_type == "sigmoid":
        y = 1 / (1 + np.exp(-x))
        dy = y * (1 - y)
        title = "Sigmoid: σ(x) = 1/(1 + e⁻ˣ)"
    elif func_type == "tanh":
        y = np.tanh(x)
        dy = 1 - y**2
        title = "Tanh: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)"
    elif func_type == "relu":
        y = np.maximum(0, x)
        dy = (x > 0).astype(float)
        title = "ReLU: max(0, x)"
    elif func_type == "leaky_relu":
        alpha = 0.01
        y = np.where(x > 0, x, alpha * x)
        dy = np.where(x > 0, 1, alpha)
        title = f"Leaky ReLU: max(0.01x, x)"
    elif func_type == "elu":
        alpha = 1.0
        y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        dy = np.where(x > 0, 1, y + alpha)
        title = "ELU: x > 0 ? x : α(eˣ - 1)"
    elif func_type == "gelu":
        y = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        dy = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) + \
             0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))**2) * \
             np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
        title = "GELU: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))"
    else:  # softmax (shown for multiple inputs)
        x_soft = np.linspace(-5, 5, 100)
        y = np.exp(x_soft) / np.sum(np.exp(x_soft))
        dy = y * (1 - y)
        title = "Softmax: softmax(xᵢ) = eˣⁱ/Σⱼeˣʲ"

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Function", "Derivative"))

    fig.add_trace(go.Scatter(x=x if func_type != "softmax" else x_soft, y=y,
                             mode='lines', name='f(x)',
                             line=dict(color='blue', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=x if func_type != "softmax" else x_soft, y=dy,
                             mode='lines', name="f'(x)",
                             line=dict(color='red', width=2)), row=1, col=2)

    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5), height=400, showlegend=True)
    fig.update_xaxes(range=[-10, 10])
    fig.update_yaxes(range=[-0.5, 1.5])

    return fig


def create_loss_landscape_visualization(
    loss_type: str = "mse"
) -> go.Figure:
    """Visualize common loss functions and their gradients."""

    x = np.linspace(-5, 5, 500)
    y_true = np.zeros_like(x)

    if loss_type == "mse":
        loss = (x - y_true)**2
        grad = 2 * (x - y_true)
        title = "MSE Loss: (ŷ - y)²"
        grad_title = "Gradient: 2(ŷ - y)"
    elif loss_type == "mae":
        loss = np.abs(x - y_true)
        grad = np.sign(x - y_true)
        title = "MAE Loss: |ŷ - y|"
        grad_title = "Gradient: sign(ŷ - y)"
    elif loss_type == "huber":
        delta = 1.0
        abs_error = np.abs(x - y_true)
        loss = np.where(abs_error <= delta,
                        0.5 * abs_error**2,
                        delta * (abs_error - 0.5 * delta))
        grad = np.where(abs_error <= delta,
                        x - y_true,
                        delta * np.sign(x - y_true))
        title = f"Huber Loss (δ={delta})"
        grad_title = "Gradient: piecewise"
    elif loss_type == "cross_entropy":
        eps = 1e-10
        p = 1 / (1 + np.exp(-x))
        loss = -(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))
        grad = p - y_true
        title = "Binary Cross-Entropy: -[y log(ŷ) + (1-y)log(1-ŷ)]"
        grad_title = "Gradient: ŷ - y"

    fig = make_subplots(rows=1, cols=2, subplot_titles=(title, grad_title))

    fig.add_trace(go.Scatter(x=x, y=loss, mode='lines', name='Loss',
                             line=dict(color='blue', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=grad, mode='lines', name='Gradient',
                             line=dict(color='red', width=2)), row=1, col=2)

    # Add zero line for gradient
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)
    fig.update_xaxes(range=[-5, 5])

    return fig


def create_optimizer_comparison_visualization(
    optimizer: str = "sgd"
) -> go.Figure:
    """Visualize different optimizers on loss landscape."""

    # Create a loss landscape (stylized)
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Rosenbrock-like function
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    Z = np.log(Z + 1)  # Log for visualization

    fig = go.Figure()

    fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Viridis',
                              contours=dict(showlabels=True),
                              name='Loss Landscape'))

    # Simulate optimizer path
    if optimizer == "sgd":
        path_x, path_y = [2], [2]
        lr = 0.001
        for _ in range(100):
            gx = -2*(1 - path_x[-1]) - 400*path_x[-1]*(path_y[-1] - path_x[-1]**2)
            gy = 200*(path_y[-1] - path_x[-1]**2)
            path_x.append(path_x[-1] - lr * gx)
            path_y.append(path_y[-1] - lr * gy)
        name = "SGD (slow convergence)"

    elif optimizer == "sgd_momentum":
        path_x, path_y = [2], [2]
        vx, vy = 0, 0
        lr, momentum = 0.001, 0.9
        for _ in range(100):
            gx = -2*(1 - path_x[-1]) - 400*path_x[-1]*(path_y[-1] - path_x[-1]**2)
            gy = 200*(path_y[-1] - path_x[-1]**2)
            vx = momentum * vx - lr * gx
            vy = momentum * vy - lr * gy
            path_x.append(path_x[-1] + vx)
            path_y.append(path_y[-1] + vy)
        name = "SGD + Momentum"

    elif optimizer == "adam":
        path_x, path_y = [2], [2]
        m_x, m_y, v_x, v_y = 0, 0, 0, 0
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
        for t in range(1, 101):
            gx = -2*(1 - path_x[-1]) - 400*path_x[-1]*(path_y[-1] - path_x[-1]**2)
            gy = 200*(path_y[-1] - path_x[-1]**2)
            m_x = beta1 * m_x + (1 - beta1) * gx
            m_y = beta1 * m_y + (1 - beta1) * gy
            v_x = beta2 * v_x + (1 - beta2) * gx**2
            v_y = beta2 * v_y + (1 - beta2) * gy**2
            m_hat = m_x / (1 - beta1**t)
            v_hat = v_x / (1 - beta2**t)
            path_x.append(path_x[-1] - lr * m_hat / (np.sqrt(v_hat) + eps))
            path_y.append(path_y[-1] - lr * m_y / (np.sqrt(v_y) + eps))
        name = "Adam (adaptive)"

    fig.add_trace(go.Scatter(x=path_x, y=path_y,
                             mode='markers+lines',
                             marker=dict(size=4, color=range(len(path_x)), colorscale='Reds'),
                             line=dict(color='red', width=2),
                             name=name))

    fig.update_layout(title=f"<b>{name} on Rosenbrock Loss</b>", height=500)
    fig.update_xaxes(range=[-10, 10])
    fig.update_yaxes(range=[-10, 10])

    return fig


def show_math_fundamentals_ui():
    """Streamlit UI for mathematical foundations."""

    st.markdown("## 🔣 Mathematical Foundations of Deep Learning")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Interactive Mathematical Formulas:</b> Adjust parameters and visualize how equations change.
    Understand gradients, activations, loss functions, and optimizers.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Gradient Descent",
        "2️⃣ Backpropagation",
        "3️⃣ Activation Functions",
        "4️⃣ Loss Functions",
        "5️⃣ Optimizers",
        "6️⃣ Key Formulas"
    ])

    with tab1:
        st.markdown("### Gradient Descent: Optimizing Loss Functions")

        col1, col2 = st.columns([1, 2])

        with col1:
            func_type = st.selectbox(
                "Loss Landscape",
                ["parabola", "cubic", "saddle", "rosenbrock"],
                format_func=lambda x: {
                    "parabola": "Parabola (simple)",
                    "cubic": "Cubic (non-convex)",
                    "saddle": "Saddle Point",
                    "rosenbrock": "Rosenbrock (banana)"
                }[x]
            )
            learning_rate = st.slider("Learning Rate (α)", 0.001, 0.5, 0.1, 0.001)
            n_steps = st.slider("Number of Steps", 10, 200, 50, 5)
            start_x = st.slider("Starting Position", -5.0, 5.0, -4.0, 0.1)

        with col2:
            fig = create_gradient_descent_visualization(func_type, learning_rate, n_steps, start_x)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📐 Gradient Descent Formula"):
            st.markdown(r"""
            **Algorithm:**
            ```
            θ_{t+1} = θ_t - α ∇L(θ_t)

            where:
            - θ_t = parameters at step t
            - α = learning rate (step size)
            - ∇L(θ_t) = gradient of loss w.r.t. parameters
            ```

            **Key Insights:**
            - Learning rate too small → slow convergence
            - Learning rate too large → oscillation/divergence
            - Can get stuck in local minima (non-convex)
            - Saddle points are common in high dimensions

            **Variants:**
            | Method | Update Rule | Pros |
            |--------|-------------|------|
            | Vanilla GD | θ - α∇L | Simple |
            | Momentum | θ - α∇L + βv | Accelerates |
            | Nesterov | Look-ahead gradient | Better |
            | Adagrad | Per-param adaptive | Sparse data |
            | Adam | Momentum + RMSProp | Default choice |
            """)

    with tab2:
        st.markdown("### Backpropagation: Computing Gradients")

        col1, col2 = st.columns([1, 2])

        with col1:
            input_val = st.slider("Input (x)", -3.0, 3.0, 1.0, 0.1)
            weight1 = st.slider("Weight 1 (w₁)", -2.0, 2.0, 0.5, 0.1)
            weight2 = st.slider("Weight 2 (w₂)", -2.0, 2.0, 0.3, 0.1)
            activation = st.selectbox("Activation", ["sigmoid", "relu", "tanh"])

        with col2:
            fig = create_backprop_visualization(input_val, weight1, weight2, activation)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📐 Backpropagation Formulas"):
            st.markdown(r"""
            **Chain Rule:**
            ```
            ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

            For a simple network: Input → z₁ = xw₁ → a₁ = σ(z₁) → z₂ = a₁w₂ → Output
            ```

            **Forward Pass:**
            ```
            z₁ = x · w₁
            a₁ = σ(z₁)          (activation)
            z₂ = a₁ · w₂
            ŷ = z₂              (output)
            L = (ŷ - y)²        (loss)
            ```

            **Backward Pass (Gradient Flow):**
            ```
            ∂L/∂z₂ = 2(z₂ - y) = 2(ŷ - y)
            ∂L/∂w₂ = ∂L/∂z₂ · ∂z₂/∂w₂ = ∂L/∂z₂ · a₁
            ∂L/∂a₁ = ∂L/∂z₂ · ∂z₂/∂a₁ = ∂L/∂z₂ · w₂
            ∂L/∂z₁ = ∂L/∂a₁ · ∂a₁/∂z₁ = ∂L/∂a₁ · σ'(z₁)
            ∂L/∂w₁ = ∂L/∂z₁ · ∂z₁/∂w₁ = ∂L/∂z₁ · x
            ```

            **Matrix Form (for layer l):**
            ```
            Forward:  z^{l} = W^{l} · a^{l-1} + b^{l}
                     a^{l} = σ(z^{l})

            Backward: ∂L/∂W^{l} = (∂L/∂z^{l}) · (a^{l-1})ᵀ
                     ∂L/∂b^{l} = ∂L/∂z^{l}
                     ∂L/∂a^{l-1} = (W^{l})ᵀ · ∂L/∂z^{l}
            ```
            """)

    with tab3:
        st.markdown("### Activation Functions: Introducing Non-Linearity")

        func_type = st.selectbox(
            "Function",
            ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu"],
            format_func=lambda x: {
                "sigmoid": "Sigmoid (logistic)",
                "tanh": "Tanh (hyperbolic)",
                "relu": "ReLU (rectified)",
                "leaky_relu": "Leaky ReLU",
                "elu": "ELU (exponential)",
                "gelu": "GELU (Gaussian)"
            }[x]
        )

        fig = create_activation_function_visualization(func_type)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📐 Activation Function Formulas"):
            st.markdown(r"""
            **Sigmoid:**
            ```
            σ(x) = 1 / (1 + e^{-x})
            σ'(x) = σ(x)(1 - σ(x))
            Range: (0, 1)
            ```

            **Tanh:**
            ```
            tanh(x) = (e^{x} - e^{-x}) / (e^{x} + e^{-x})
            tanh'(x) = 1 - tanh²(x)
            Range: (-1, 1)
            ```

            **ReLU:**
            ```
            ReLU(x) = max(0, x)
            ReLU'(x) = 1 if x > 0, else 0
            ```

            **Leaky ReLU:**
            ```
            LeakyReLU(x) = max(αx, x) where α = 0.01
            LeakyReLU'(x) = 1 if x > 0, else α
            ```

            **GELU (used in GPT, BERT):**
            ```
            GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            Approximation: GELU(x) ≈ x · σ(1.702x)
            ```
            """)

    with tab4:
        st.markdown("### Loss Functions: Measuring Model Performance")

        loss_type = st.selectbox(
            "Loss Type",
            ["mse", "mae", "huber", "cross_entropy"],
            format_func=lambda x: {
                "mse": "MSE (Mean Squared Error)",
                "mae": "MAE (Mean Absolute Error)",
                "huber": "Huber Loss",
                "cross_entropy": "Cross-Entropy"
            }[x]
        )

        fig = create_loss_landscape_visualization(loss_type)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📐 Loss Function Formulas"):
            st.markdown(r"""
            **MSE (L2 Loss):**
            ```
            L_{MSE} = (1/n) Σ (ŷᵢ - yᵢ)²
            L'_{MSE} = 2(ŷ - y)        (gradient)

            Pros: Smooth gradients, penalizes large errors more
            Cons: Sensitive to outliers
            ```

            **MAE (L1 Loss):**
            ```
            L_{MAE} = (1/n) Σ |ŷᵢ - yᵢ|
            L'_{MAE} = sign(ŷ - y)

            Pros: Robust to outliers
            Cons: Non-smooth at zero, slower convergence
            ```

            **Huber Loss:**
            ```
            L_{Huber}(x) = { 0.5x²           if |x| ≤ δ
                           { δ|x| - 0.5δ²     otherwise }

            Combines MSE (near zero) and MAE (far from zero)
            ```

            **Cross-Entropy (Binary):**
            ```
            L_{CE} = -[y log(ŷ) + (1-y)log(1-ŷ)]
            L'_{CE} = ŷ - y                    (beautiful gradient!)

            Used in: Logistic regression, neural networks
            ```
            """)

    with tab5:
        st.markdown("### Optimizers: Algorithms for Finding Minima")

        optimizer = st.selectbox(
            "Optimizer",
            ["sgd", "sgd_momentum", "adam"],
            format_func=lambda x: {
                "sgd": "SGD (Stochastic Gradient Descent)",
                "sgd_momentum": "SGD + Momentum",
                "adam": "Adam (Adaptive Moment Estimation)"
            }[x]
        )

        fig = create_optimizer_comparison_visualization(optimizer)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📐 Optimizer Formulas"):
            st.markdown(r"""
            **SGD with Momentum:**
            ```
            v_t = βv_{t-1} + (1-β)∇L(θ)
            θ_t = θ_{t-1} - αv_t

            Common: β = 0.9, α = 0.01
            ```

            **Adam (default choice for modern DL):**
            ```
            m_t = β₁m_{t-1} + (1-β₁)∇L(θ)      (first moment)
            v_t = β₂v_{t-1} + (1-β₂)(∇L(θ))²  (second moment)

            m_hat = m_t / (1 - β₁^t)            (bias correction)
            v_hat = v_t / (1 - β₂^t)

            θ_t = θ_{t-1} - α · m_hat / (√v_hat + ε)

            Default: α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸
            ```
            """)

    with tab6:
        st.markdown("### 📜 Reference: Key Formulas in Deep Learning")

        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
        <b>Essential Equations for ML/DL:</b>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **1. Linear Layer (Fully Connected)**
            ```
            Forward:  y = Wx + b
                     where W ∈ ℝ^{d_out × d_in}

            Backward: ∂L/∂W = (∂L/∂y)ᵀ ⊗ x
                     ∂L/∂x = Wᵀ · (∂L/∂y)
            ```

            **2. 2D Convolution**
            ```
            y[i,j] = Σ_k Σ_l W[k,l] · x[i+k, j+l]

            Backward: gradient w.r.t. W, x computed via correlation
            ```

            **3. Batch Normalization**
            ```
            μ_B = (1/m)Σx_i           (batch mean)
            σ²_B = (1/m)Σ(x_i-μ)²    (batch variance)
            x̂ = (x - μ) / √(σ² + ε)  (normalize)
            y = γx̂ + β               (scale & shift)

            During training: uses batch stats
            During inference: uses running stats
            ```
            """)

        with col2:
            st.markdown("""
            **4. Dropout (regularization)**
            ```
            During training:
               mask ~ Bernoulli(p)      (keep probability)
               y = (mask * x) / p       (inverted dropout)

            During inference:
               y = x                     (no dropout)
            ```

            **5. Attention (Scaled Dot-Product)**
            ```
            Attention(Q,K,V) = softmax(QKᵀ / √d_k) V

            where:
               Q ∈ ℝ^{n×d_k}  (queries)
               K ∈ ℝ^{m×d_k}  (keys)
               V ∈ ℝ^{m×d_v}  (values)

            Multi-head: concat heads, project
            ```
            """)

        st.markdown("---")
        st.markdown("### 🎯 Key Takeaways")

        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
        1. <b>Gradient descent</b> is the core optimization algorithm - all variants aim to find minima faster/stable

        2. <b>Backpropagation</b> is just the chain rule applied recursively - gradients flow backward through the network

        3. <b>Activation functions</b> introduce non-linearity - without them, deep networks = single linear layer

        4. <b>Loss functions</b> define what we're optimizing - choose based on task (regression vs classification)

        5. <b>Adam</b> is the default optimizer - combines momentum and adaptive learning rates

        6. <b>BatchNorm</b> normalizes activations - reduces internal covariate shift, enables faster training
        </div>
        """, unsafe_allow_html=True)
