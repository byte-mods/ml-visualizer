"""
ML Visualization Lab - Educational Tool for Machine Learning Concepts
Main Streamlit application with interactive visualizations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Tuple, Callable, Dict, Any
import config

# Set page configuration
st.set_page_config(
    page_title="ML Visualization Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .info-box p, .info-box li {
        color: #1f1f1f !important;
        margin-bottom: 0;
    }
    .info-box strong {
        color: #1f1f1f;
    }
    .stApp {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1f1f1f !important;
    }
    .stApp p {
        color: #1f1f1f !important;
    }
    body {
        background-color: #ffffff;
        color: #1f1f1f;
    }
    .stMarkdownContainer {
        color: #1f1f1f;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    .stSidebar .stMarkdown, .stSidebar p, .stSidebar span, .stSidebar label {
        color: #1f1f1f !important;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        color: #1f1f1f !important;
    }
    .equation {
        font-family: "Times New Roman", serif;
        font-size: 1.2rem;
        background-color: #f8f9fa;
        color: #1f1f1f;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧠 ML Visualization Lab")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to",
    [
        "🏠 Home",
        "📖 Chapter 1: 3D Graphs & Vector Spaces",
        "📖 Chapter 2: Probability Distributions",
        "📖 Chapter 3: Math Foundations (Gradients & Backprop)",
        "📖 Chapter 4: Loss Functions",
        "📖 Chapter 5: Activation Functions",
        "📖 Chapter 6: Neural Network Architectures",
        "📖 Chapter 7: PyTorch Layer Builder",
        "📖 Chapter 8: Modern LLM Builder",
        "📖 Chapter 9: Advanced Models (LLM, Diffusion, etc.)",
        "📖 Chapter 10: Model Training Simulation"
    ]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This interactive tool helps visualize machine learning concepts for "
    "educational and research purposes. Adjust parameters using sliders and "
    "dropdowns to see real-time updates."
)

st.sidebar.markdown("### Settings")
plot_width = st.sidebar.slider("Plot Width", 600, 1200, config.PLOT_WIDTH, 50)
plot_height = st.sidebar.slider("Plot Height", 400, 800, config.PLOT_HEIGHT, 50)

# Import visualization modules
from visualizations import three_d_graphs, distributions, math_fundamentals, functions, architectures, models, torch_layers, llm_builder

def show_home():
    """Home page with overview and instructions"""
    st.markdown('<div class="main-header">🧠 ML Visualization Lab</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Welcome to ML Visualization Lab!</b> This interactive tutorial takes you through machine learning concepts
    from foundational mathematics to modern LLM architectures. Each chapter builds on the previous one.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📚 Tutorial Curriculum")

    st.markdown("""
    Follow this sequential path to master ML fundamentals:
    """)

    chapters = [
        ("Chapter 1", "3D Graphs & Vector Spaces", "📊", "Understand how machines see data in multiple dimensions - the foundation of all ML"),
        ("Chapter 2", "Probability Distributions", "🔔", "Learn how probability describes data and uncertainty in ML models"),
        ("Chapter 3", "Math Foundations (Gradients & Backprop)", "🔣", "Master the mathematics of gradient descent and backpropagation"),
        ("Chapter 4", "Loss Functions", "📉", "Discover how models learn by measuring their mistakes"),
        ("Chapter 5", "Activation Functions", "⚡", "Explore the nonlinearities that enable deep learning"),
        ("Chapter 6", "Neural Network Architectures", "🧠", "Build intuition for how neurons learn patterns"),
        ("Chapter 7", "PyTorch Layer Builder", "🔥", "Get hands-on with deep learning building blocks"),
        ("Chapter 8", "Modern LLM Builder", "🚀", "Master 2026 architectures: MoE, GQA, RoPE, SwiGLU"),
        ("Chapter 9", "Advanced Models", "🤖", "Explore diffusion, video, and audio models"),
        ("Chapter 10", "Model Training Simulation", "⚙️", "Put it all together in a training loop"),
    ]

    for i, (num, title, icon, desc) in enumerate(chapters):
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.markdown(f"### {icon} {num}")
        with col2:
            st.markdown(f"**{title}**")
        with col3:
            st.markdown(f"_{desc}_")

        if i < len(chapters) - 1:
            st.markdown("---")

    st.markdown("## 🎯 How to Use This Tutorial")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📖 Learning Path
        1. **Start from Chapter 1** - Each chapter builds on concepts from previous ones
        2. **Read the theory** - Each section has detailed explanations
        3. **Interact with visualizations** - Adjust sliders to build intuition
        4. **Complete the exercises** - Test your understanding
        """)

    with col2:
        st.markdown("""
        ### 🔧 Quick Navigation
        - Use the **sidebar** to jump to any chapter
        - **Sliders** adjust parameters in real-time
        - **Tabs** show different views of the same concept
        - **Expanders** reveal deeper mathematical details
        """)

    st.markdown("## 🚀 Prerequisites")

    st.markdown("""
    <div class="info-box">
    This tutorial assumes basic Python knowledge and high school mathematics (calculus, linear algebra, probability).
    Don't worry if you're rusty - we'll refresh concepts as needed!
    </div>
    """, unsafe_allow_html=True)

    # Quick examples
    st.markdown("## 📸 Preview: What You'll Build")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Understanding Data: Bell Curve")
        # Simple bell curve preview
        x = np.linspace(-4, 4, 100)
        y = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2)
        df = pd.DataFrame({'x': x, 'Probability Density': y})
        fig = px.line(df, x='x', y='Probability Density', title="Standard Normal Distribution")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*The normal distribution appears everywhere in ML - from measurement noise to activation patterns*")

    with col2:
        st.markdown("### Complex Patterns: 3D Surfaces")
        # Simple 3D surface preview
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(title="sin(√(x² + y²))", height=300, scene=dict(aspectmode='cube'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*Neural networks learn to fit complex surfaces like this to your data*")

    st.markdown("---")
    st.markdown("*Ready to begin? Select **Chapter 1: 3D Graphs & Vector Spaces** from the sidebar to start learning!*")

def show_training_simulation():
    """Model Training Simulation"""
    st.markdown('<div class="main-header">⚙️ Model Training Simulation</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Training simulation** shows how models learn from data.
    Adjust hyperparameters and observe training dynamics in real-time.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Simulation Controls")

        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            ["Simple Classifier", "Multi-layer Network", "Convolutional Network", "Recurrent Network"]
        )

        # Training hyperparameters
        st.markdown("#### Hyperparameters")

        learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.01, 0.001, format="%.4f")
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128, 256], index=2)
        n_epochs = st.slider("Number of Epochs", 10, 500, 100, 10)

        optimizer = st.selectbox(
            "Optimizer",
            ["SGD", "SGD + Momentum", "Adam", "RMSprop", "AdaGrad"]
        )

        momentum = 0.0
        if optimizer in ["SGD + Momentum", "RMSprop"]:
            momentum = st.slider("Momentum", 0.0, 0.99, 0.9, 0.01)

        # Learning rate scheduler
        use_scheduler = st.checkbox("Use Learning Rate Scheduler")
        scheduler_type = "Step Decay"
        step_size = 30
        decay_rate = 0.5
        if use_scheduler:
            scheduler_type = st.selectbox(
                "Scheduler Type",
                ["Step Decay", "Exponential Decay", "Cosine Annealing", "Reduce on Plateau"]
            )

            if scheduler_type == "Step Decay":
                step_size = st.slider("Step Size (epochs)", 10, 100, 30, 5)
                decay_rate = st.slider("Decay Rate", 0.1, 0.9, 0.5, 0.1)

        # Loss function
        loss_function = st.selectbox(
            "Loss Function",
            ["Cross-Entropy", "Mean Squared Error", "Hinge Loss", "Smooth L1"]
        )

        # Regularization
        st.markdown("#### Regularization")
        use_dropout = st.checkbox("Use Dropout")
        if use_dropout:
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)

        use_early_stopping = st.checkbox("Use Early Stopping")
        if use_early_stopping:
            patience = st.slider("Patience", 5, 50, 10, 1)

        # Generate training data
        st.markdown("#### Data")
        n_samples = st.slider("Training Samples", 100, 10000, 1000, 100)
        n_features = st.slider("Input Features", 2, 100, 10, 1)
        n_classes = st.slider("Number of Classes", 2, 10, 3, 1)

        noise_level = st.slider("Label Noise (%)", 0, 30, 0, 1) / 100.0

        # Run simulation button
        run_simulation = st.button("▶️ Run Training Simulation", type="primary")

    with col1:
        st.markdown("### Training Visualization")

        if run_simulation or 'history' in locals():
            # Generate or retrieve training history
            np.random.seed(42)

            # Simulate training with realistic curves
            epochs = np.arange(1, n_epochs + 1)

            # Base learning rate effect
            if optimizer == "Adam":
                lr_factor = 0.5  # Adam converges faster
            elif optimizer == "RMSprop":
                lr_factor = 0.6
            elif optimizer == "AdaGrad":
                lr_factor = 0.7
            else:
                lr_factor = 1.0

            # Apply scheduler if enabled
            lr_schedule = np.ones(n_epochs) * learning_rate
            if use_scheduler:
                if scheduler_type == "Step Decay":
                    lr_schedule = learning_rate * np.power(decay_rate, np.floor(epochs / step_size))
                elif scheduler_type == "Exponential Decay":
                    lr_schedule = learning_rate * np.exp(-0.01 * epochs)
                elif scheduler_type == "Cosine Annealing":
                    lr_schedule = learning_rate * 0.5 * (1 + np.cos(np.pi * epochs / n_epochs))
                elif scheduler_type == "Reduce on Plateau":
                    lr_schedule = learning_rate * np.where(
                        epochs > n_epochs * 0.5,
                        0.1,
                        1.0
                    )

            # Training loss (decreases with noise)
            if loss_function == "Cross-Entropy":
                base_loss = 1.5
            elif loss_function == "Mean Squared Error":
                base_loss = 2.0
            elif loss_function == "Hinge Loss":
                base_loss = 1.2
            else:
                base_loss = 1.0

            train_loss = base_loss * np.exp(-epochs * lr_factor * 0.05 / learning_rate)
            train_loss += noise_level * np.random.randn(n_epochs) * epochs * 0.001
            train_loss += 0.05 * np.random.randn(n_epochs)
            train_loss = np.maximum(train_loss, 0.1)

            # Validation loss (slightly higher, with overfitting at later epochs)
            val_loss = train_loss * (1.1 + 0.1 * np.random.rand(n_epochs))
            val_loss += 0.02 * np.exp(epochs * 0.005)  # Overfitting at end
            val_loss = np.minimum(val_loss, base_loss * 1.5)

            # Apply early stopping if enabled
            if use_early_stopping:
                best_epoch = np.argmin(val_loss) + 1
                if best_epoch < n_epochs - patience:
                    # Early stopping would have triggered
                    val_loss[best_epoch + patience:] = val_loss[best_epoch]

            # Training accuracy
            if loss_function == "Cross-Entropy":
                base_acc = 0.3
            else:
                base_acc = 0.35

            train_acc = base_acc + (1 - base_acc) * (1 - np.exp(-epochs * lr_factor * 0.03 / learning_rate))
            train_acc += 0.02 * np.random.randn(n_epochs)
            train_acc = np.clip(train_acc, 0, 0.99)

            val_acc = train_acc * (0.95 - 0.05 * np.exp(epochs * 0.003))
            val_acc += 0.02 * np.random.randn(n_epochs)
            val_acc = np.clip(val_acc, 0, 0.98)

            # Apply dropout effect
            if use_dropout:
                train_acc *= (1 - dropout_rate * 0.5)
                val_acc *= (1 - dropout_rate * 0.3)

            # Create the plot
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Training & Validation Loss",
                    "Training & Validation Accuracy",
                    "Learning Rate Schedule",
                    "Loss Landscape (2D Projection)"
                ),
                specs=[
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ]
            )

            # Loss plot
            fig.add_trace(go.Scatter(
                x=epochs, y=train_loss,
                mode='lines', name='Train Loss',
                line=dict(color='blue', width=2)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=epochs, y=val_loss,
                mode='lines', name='Val Loss',
                line=dict(color='red', width=2, dash='dash')
            ), row=1, col=1)

            # Accuracy plot
            fig.add_trace(go.Scatter(
                x=epochs, y=train_acc,
                mode='lines', name='Train Acc',
                line=dict(color='blue', width=2)
            ), row=1, col=2)

            fig.add_trace(go.Scatter(
                x=epochs, y=val_acc,
                mode='lines', name='Val Acc',
                line=dict(color='red', width=2, dash='dash')
            ), row=1, col=2)

            # Learning rate schedule
            fig.add_trace(go.Scatter(
                x=epochs, y=lr_schedule,
                mode='lines', name='Learning Rate',
                line=dict(color='green', width=2)
            ), row=2, col=1)

            # Loss landscape (simplified 2D projection)
            x_landscape = np.linspace(-2, 2, 50)
            y_landscape = np.linspace(-2, 2, 50)
            X, Y = np.meshgrid(x_landscape, y_landscape)

            # Simplified loss landscape
            Z = X**2 + Y**2 + 0.5 * np.sin(3 * X) * np.cos(3 * Y)

            fig.add_trace(go.Contour(
                x=x_landscape, y=y_landscape, z=Z,
                colorscale='Viridis',
                showscale=False,
                contours=dict(showlabels=True, labelfont=dict(size=10))
            ), row=2, col=2)

            # Mark current "position" in loss landscape
            final_epoch = min(n_epochs, 100)
            if use_early_stopping and 'best_epoch' in locals():
                final_epoch = best_epoch

            progress = 1 - np.exp(-final_epoch * 0.02)
            pos_x = -1.5 + progress * 2.5
            pos_y = -1.5 + progress * 2.5

            fig.add_trace(go.Scatter(
                x=[pos_x], y=[pos_y],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Current Position'
            ), row=2, col=2)

            fig.update_layout(
                height=700,
                showlegend=True,
                legend=dict(x=1.02, y=1)
            )

            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Learning Rate", row=2, col=1, type='log')
            fig.update_xaxes(title_text="Parameter 1", row=2, col=2)
            fig.update_yaxes(title_text="Parameter 2", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

            # Training statistics
            st.markdown("### Training Statistics")
            col_a, col_b, col_c, col_d = st.columns(4)

            final_train_loss = train_loss[-1]
            final_val_loss = val_loss[-1]
            final_train_acc = train_acc[-1]
            final_val_acc = val_acc[-1]

            if use_early_stopping and 'best_epoch' in locals():
                best_val_loss = val_loss[best_epoch - 1]
                best_val_acc = val_acc[best_epoch - 1]
            else:
                best_val_loss = np.min(val_loss)
                best_val_acc = np.max(val_acc)

            col_a.metric("Final Train Loss", f"{final_train_loss:.4f}")
            col_b.metric("Best Val Loss", f"{best_val_loss:.4f}")
            col_c.metric("Final Train Acc", f"{final_train_acc:.2%}")
            col_d.metric("Best Val Acc", f"{best_val_acc:.2%}")

            # Gradient flow visualization
            st.markdown("### Gradient Flow Visualization")

            layers = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
            gradient_magnitudes = np.random.rand(len(layers))
            # Simulate vanishing gradients
            gradient_magnitudes = gradient_magnitudes * np.exp(-np.arange(len(layers)) * 0.3)

            fig_grad = go.Figure(data=[
                go.Bar(
                    x=layers,
                    y=gradient_magnitudes,
                    marker_color=['blue' if g > 0.01 else 'red' for g in gradient_magnitudes],
                    text=[f"{g:.6f}" for g in gradient_magnitudes],
                    textposition='auto'
                )
            ])

            fig_grad.update_layout(
                title="Gradient Magnitude by Layer",
                xaxis_title="Layer",
                yaxis_title="Gradient Magnitude (log scale)",
                yaxis_type="log"
            )

            st.plotly_chart(fig_grad, width='stretch')

            if any(g < 0.001 for g in gradient_magnitudes):
                st.warning("⚠️ Potential vanishing gradient detected in deeper layers!")

        else:
            # Initial state - show explanation
            st.info("👆 Configure hyperparameters and click 'Run Training Simulation' to begin")

            # Show a sample training curve
            st.markdown("#### Sample Training Curve")

            sample_epochs = np.arange(1, 101)
            sample_loss = 1.5 * np.exp(-sample_epochs * 0.03) + 0.1

            fig_sample = go.Figure()
            fig_sample.add_trace(go.Scatter(
                x=sample_epochs, y=sample_loss,
                mode='lines', name='Training Loss',
                line=dict(color='blue', width=2)
            ))

            fig_sample.update_layout(
                title="Example: Typical Training Loss Curve",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )

            st.plotly_chart(fig_sample, width='stretch')

            st.markdown("""
            **How to interpret training curves:**

            1. **Rapid initial drop**: Model learns basic patterns quickly
            2. **Gradual refinement**: Model fine-tunes on data
            3. **Plateau**: Model converges to optimal performance
            4. **Divergence/Overfitting**: Loss increases or validation loss diverges

            **Common issues to watch for:**
            - **Vanishing gradients**: Loss barely decreases
            - **Exploding gradients**: Loss becomes NaN/inf
            - **Overfitting**: Val loss increases while train loss decreases
            - **Underfitting**: Both losses remain high
            """)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Training Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Optimizers")
        st.markdown("""
        - **SGD**: Vanilla stochastic gradient descent
        - **SGD + Momentum**: Adds momentum to accelerate convergence
        - **Adam**: Adaptive learning rates per parameter
        - **RMSprop**: Root mean square propagation
        - **AdaGrad**: Adaptive gradient for sparse features
        """)

        st.markdown("### Learning Rate Schedules")
        st.markdown("""
        - **Step Decay**: Reduce LR by factor at fixed intervals
        - **Exponential Decay**: Continuously decay LR exponentially
        - **Cosine Annealing**: Smooth cosine curve decay
        - **Reduce on Plateau**: Reduce when metric stops improving
        """)

    with col2:
        st.markdown("### Regularization Techniques")
        st.markdown("""
        - **Dropout**: Randomly zero out activations during training
        - **Early Stopping**: Stop when validation loss stops improving
        - **Weight Decay**: L2 regularization on weights
        - **Batch Normalization**: Normalize layer inputs
        """)

        st.markdown("### Troubleshooting")
        st.markdown("""
        - **Loss not decreasing**: Try lower learning rate or different optimizer
        - **NaN loss**: Reduce learning rate, check data normalization
        - **Overfitting**: Add dropout, use early stopping, get more data
        - **Underfitting**: Increase model capacity, train longer
        """)

# Main routing
if page == "🏠 Home":
    show_home()
elif page == "📖 Chapter 1: 3D Graphs & Vector Spaces":
    three_d_graphs.show_3d_graphs_ui()
elif page == "📖 Chapter 2: Probability Distributions":
    distributions.show_distributions_ui()
elif page == "📖 Chapter 3: Math Foundations (Gradients & Backprop)":
    math_fundamentals.show_math_fundamentals_ui()
elif page == "📖 Chapter 4: Loss Functions":
    functions.show_loss_functions_ui()
elif page == "📖 Chapter 5: Activation Functions":
    functions.show_activation_functions_ui()
elif page == "📖 Chapter 6: Neural Network Architectures":
    architectures.show_neural_network_ui()
elif page == "📖 Chapter 7: PyTorch Layer Builder":
    torch_layers.show_torch_layers_ui()
elif page == "📖 Chapter 8: Modern LLM Builder":
    llm_builder.show_llm_builder_ui()
elif page == "📖 Chapter 9: Advanced Models (LLM, Diffusion, etc.)":
    models.show_advanced_models_ui()
elif page == "📖 Chapter 10: Model Training Simulation":
    show_training_simulation()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown(
        "<div style='text-align: center; color: #1f1f1f;'>"
        "🧠 ML Visualization Lab • Educational Tool • "
        "<a href='#' style='color: #1f77b4;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )