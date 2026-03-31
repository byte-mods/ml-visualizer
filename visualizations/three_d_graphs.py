"""
3D Graphs and Vector Spaces Visualizations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict, Any
import streamlit as st
import pandas as pd


def create_surface_plot(
    func: callable,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 50,
    title: str = "3D Surface Plot",
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Create a 3D surface plot for a given function z = f(x, y)

    Args:
        func: Function that takes (x, y) and returns z
        x_range: Range of x values
        y_range: Range of y values
        resolution: Number of points in each dimension
        title: Plot title
        colorscale: Colorscale for the surface

    Returns:
        plotly.graph_objects.Figure
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colorscale)])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def create_3d_scatter(
    points: np.ndarray,
    labels: List[str] = None,
    colors: List[str] = None,
    title: str = "3D Scatter Plot",
    size: int = 5
) -> go.Figure:
    """
    Create a 3D scatter plot

    Args:
        points: Array of shape (n_points, 3)
        labels: Optional labels for each point
        colors: Optional colors for each point
        title: Plot title
        size: Marker size

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    if labels is None:
        # Single trace
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=size, color=colors if colors else 'blue'),
            name='Points'
        ))
    else:
        # Multiple traces by label
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode='markers',
                marker=dict(size=size),
                name=str(label)
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        )
    )

    return fig


def create_vector_field(
    func: callable,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    z_range: Tuple[float, float] = (-5, 5),
    resolution: int = 10,
    title: str = "3D Vector Field",
    scale: float = 0.1
) -> go.Figure:
    """
    Create a 3D vector field visualization

    Args:
        func: Function that takes (x, y, z) and returns (u, v, w)
        x_range, y_range, z_range: Coordinate ranges
        resolution: Number of vectors in each dimension
        title: Plot title
        scale: Scaling factor for vector lengths

    Returns:
        plotly.graph_objects.Figure
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)

    X, Y, Z = np.meshgrid(x, y, z)
    U, V, W = func(X, Y, Z)

    fig = go.Figure(data=go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=U.flatten(),
        v=V.flatten(),
        w=W.flatten(),
        sizemode="absolute",
        sizeref=scale,
        colorscale='Blues',
        showscale=False
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        )
    )

    return fig


def create_matrix_visualization(
    matrix: np.ndarray,
    title: str = "Matrix Visualization",
    show_values: bool = True,
    colorscale: str = "RdBu"
) -> go.Figure:
    """
    Create a 3D bar plot or heatmap for matrix visualization

    Args:
        matrix: 2D numpy array
        title: Plot title
        show_values: Whether to show values on bars
        colorscale: Colorscale for the visualization

    Returns:
        plotly.graph_objects.Figure
    """
    n_rows, n_cols = matrix.shape

    # Create meshgrid for positions
    x_pos = np.arange(n_cols)
    y_pos = np.arange(n_rows)
    X, Y = np.meshgrid(x_pos, y_pos)

    # Flatten for bar plot
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.zeros_like(x_flat)
    dz_flat = matrix.flatten()

    fig = go.Figure(data=[go.Bar3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        dx=0.8 * np.ones_like(x_flat),
        dy=0.8 * np.ones_like(y_flat),
        dz=dz_flat,
        colorscale=colorscale,
        colorbar=dict(title="Value")
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Column",
            yaxis_title="Row",
            zaxis_title="Value",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        )
    )

    return fig


def create_vector_operations_visualization(
    vectors: List[Tuple[np.ndarray, str, str]],
    origin: np.ndarray = np.array([0, 0, 0]),
    title: str = "Vector Operations",
    show_plane: bool = False
) -> go.Figure:
    """
    Visualize vector operations (addition, dot product, cross product)

    Args:
        vectors: List of (vector, label, color) tuples
        origin: Origin point for vectors
        title: Plot title
        show_plane: Whether to show the plane spanned by vectors

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Plot each vector
    for vec, label, color in vectors:
        end_point = origin + vec

        # Vector line
        fig.add_trace(go.Scatter3d(
            x=[origin[0], end_point[0]],
            y=[origin[1], end_point[1]],
            z=[origin[2], end_point[2]],
            mode='lines+markers',
            line=dict(color=color, width=5),
            marker=dict(size=4, color=color),
            name=label
        ))

        # Arrow head (cone)
        fig.add_trace(go.Cone(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            u=[vec[0] * 0.2],
            v=[vec[1] * 0.2],
            w=[vec[2] * 0.2],
            showscale=False,
            colorscale=[[0, color], [1, color]],
            sizemode="absolute",
            sizeref=0.5,
            name=f"{label} head"
        ))

    # Show plane spanned by first two vectors if requested
    if show_plane and len(vectors) >= 2:
        v1 = vectors[0][0]
        v2 = vectors[1][0]

        # Create plane grid
        s = np.linspace(-1, 1, 10)
        t = np.linspace(-1, 1, 10)
        S, T = np.meshgrid(s, t)

        X_plane = origin[0] + S * v1[0] + T * v2[0]
        Y_plane = origin[1] + S * v1[1] + T * v2[1]
        Z_plane = origin[2] + S * v1[2] + T * v2[2]

        fig.add_trace(go.Surface(
            x=X_plane,
            y=Y_plane,
            z=Z_plane,
            opacity=0.3,
            colorscale='Greys',
            showscale=False,
            name="Span plane"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        )
    )

    return fig


# Predefined functions for common 3D surfaces
def sine_wave(x: np.ndarray, y: np.ndarray, frequency: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """Sine wave surface: z = amplitude * sin(frequency * sqrt(x^2 + y^2))"""
    return amplitude * np.sin(frequency * np.sqrt(x**2 + y**2))

def paraboloid(x: np.ndarray, y: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Paraboloid: z = a*x^2 + b*y^2"""
    return a * x**2 + b * y**2

def ripple(x: np.ndarray, y: np.ndarray, frequency: float = 1.0) -> np.ndarray:
    """Ripple pattern: z = sin(x) * cos(y)"""
    return np.sin(frequency * x) * np.cos(frequency * y)

def saddle(x: np.ndarray, y: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Saddle surface: z = a*x^2 - b*y^2"""
    return a * x**2 - b * y**2

def gaussian(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """2D Gaussian: z = exp(-(x^2 + y^2) / (2*sigma^2))"""
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


# Example vector field functions
def radial_vector_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial vector field pointing away from origin"""
    r = np.sqrt(x**2 + y**2 + z**2)
    u = x / (r + 1e-10)
    v = y / (r + 1e-10)
    w = z / (r + 1e-10)
    return u, v, w

def rotational_vector_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotational vector field around z-axis"""
    u = -y
    v = x
    w = np.zeros_like(z)
    return u, v, w

def gradient_vector_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of f(x,y,z) = x^2 + y^2 - z^2"""
    u = 2 * x
    v = 2 * y
    w = -2 * z
    return u, v, w


def show_3d_graphs_ui():
    """Streamlit UI for 3D graphs visualization"""
    st.markdown("## 📊 3D Graphs & Vector Spaces")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")

        visualization_type = st.selectbox(
            "Visualization Type",
            ["Surface Plot", "3D Scatter", "Vector Field", "Matrix", "Vector Operations"]
        )

        if visualization_type == "Surface Plot":
            surface_type = st.selectbox(
                "Surface Function",
                ["Sine Wave", "Paraboloid", "Ripple", "Saddle", "Gaussian", "Custom"]
            )

            if surface_type == "Sine Wave":
                freq = st.slider("Frequency", 0.1, 5.0, 1.0, 0.1)
                amp = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)

                def custom_func(x, y):
                    return sine_wave(x, y, freq, amp)

            elif surface_type == "Paraboloid":
                a = st.slider("X coefficient (a)", 0.1, 5.0, 1.0, 0.1)
                b = st.slider("Y coefficient (b)", 0.1, 5.0, 1.0, 0.1)

                def custom_func(x, y):
                    return paraboloid(x, y, a, b)

            elif surface_type == "Ripple":
                freq = st.slider("Frequency", 0.1, 5.0, 1.0, 0.1)

                def custom_func(x, y):
                    return ripple(x, y, freq)

            elif surface_type == "Saddle":
                a = st.slider("X² coefficient", 0.1, 5.0, 1.0, 0.1)
                b = st.slider("Y² coefficient", 0.1, 5.0, 1.0, 0.1)

                def custom_func(x, y):
                    return saddle(x, y, a, b)

            elif surface_type == "Gaussian":
                sigma = st.slider("Sigma (width)", 0.1, 5.0, 1.0, 0.1)

                def custom_func(x, y):
                    return gaussian(x, y, sigma)

            else:  # Custom
                st.text_area("Custom function (use x, y):", "np.sin(x) * np.cos(y)")
                # In a real implementation, you would parse and evaluate this safely

            resolution = st.slider("Resolution", 20, 100, 50, 5)
            x_range = st.slider("X range", -10.0, 10.0, (-5.0, 5.0), 0.5)
            y_range = st.slider("Y range", -10.0, 10.0, (-5.0, 5.0), 0.5)

            # Generate the plot
            fig = create_surface_plot(
                custom_func,
                x_range,
                y_range,
                resolution,
                title=f"{surface_type} Surface"
            )

        elif visualization_type == "3D Scatter":
            # Generate random points for demonstration
            n_points = st.slider("Number of points", 10, 1000, 100, 10)
            cluster = st.checkbox("Create clusters", True)

            if cluster:
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                points = []
                labels = []
                for i in range(n_clusters):
                    center = np.random.randn(3) * 3
                    cluster_points = center + np.random.randn(n_points // n_clusters, 3) * 0.5
                    points.append(cluster_points)
                    labels.extend([f"Cluster {i+1}"] * (n_points // n_clusters))

                points = np.vstack(points)
                fig = create_3d_scatter(points, labels, title="3D Clustered Data")
            else:
                points = np.random.randn(n_points, 3)
                fig = create_3d_scatter(points, title="3D Random Points")

        elif visualization_type == "Vector Field":
            field_type = st.selectbox(
                "Field Type",
                ["Radial", "Rotational", "Gradient", "Custom"]
            )

            resolution = st.slider("Resolution", 5, 20, 8, 1)
            scale = st.slider("Vector scale", 0.01, 1.0, 0.1, 0.01)

            if field_type == "Radial":
                fig = create_vector_field(
                    radial_vector_field,
                    resolution=resolution,
                    scale=scale,
                    title="Radial Vector Field"
                )
            elif field_type == "Rotational":
                fig = create_vector_field(
                    rotational_vector_field,
                    resolution=resolution,
                    scale=scale,
                    title="Rotational Vector Field"
                )
            elif field_type == "Gradient":
                fig = create_vector_field(
                    gradient_vector_field,
                    resolution=resolution,
                    scale=scale,
                    title="Gradient Vector Field"
                )

        elif visualization_type == "Matrix":
            matrix_type = st.selectbox(
                "Matrix Type",
                ["Random", "Identity", "Diagonal", "Symmetric", "Custom"]
            )
            size = st.slider("Matrix size", 3, 20, 5, 1)

            if matrix_type == "Random":
                matrix = np.random.randn(size, size)
            elif matrix_type == "Identity":
                matrix = np.eye(size)
            elif matrix_type == "Diagonal":
                diag = np.random.randn(size)
                matrix = np.diag(diag)
            elif matrix_type == "Symmetric":
                matrix = np.random.randn(size, size)
                matrix = (matrix + matrix.T) / 2

            fig = create_matrix_visualization(
                matrix,
                title=f"{matrix_type} Matrix ({size}x{size})"
            )

        elif visualization_type == "Vector Operations":
            # Create example vectors
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, 1, 0])
            v3 = v1 + v2  # Vector addition
            v4 = np.cross(v1, v2)  # Cross product

            vectors = [
                (v1, "v₁ (1,0,0)", "red"),
                (v2, "v₂ (0,1,0)", "blue"),
                (v3, "v₁ + v₂ (addition)", "green"),
                (v4, "v₁ × v₂ (cross product)", "purple")
            ]

            show_plane = st.checkbox("Show plane spanned by v₁ and v₂", True)

            fig = create_vector_operations_visualization(
                vectors,
                title="Vector Operations",
                show_plane=show_plane
            )

    with col1:
        st.markdown("### Visualization")
        if 'fig' in locals():
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select visualization type and parameters to generate plot")

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Chapter 1: 3D Graphs & Vector Spaces - Tutorial")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Learning Objectives:</b> By the end of this chapter, you will understand how data and models exist in multi-dimensional spaces,
    why visualizations help build intuition, and how vector operations underpin all of machine learning.
    </div>
    """, unsafe_allow_html=True)

    # Theory Section
    st.markdown("### 📖 Theory: Why 3D Visualization Matters in ML")

    with st.expander("Understanding Dimensions in Machine Learning"):
        st.markdown("""
        **The Dimension Problem:**

        In ML, we work with data in many dimensions:
        - A grayscale image (32x32) is a **1,024-dimensional vector**
        - A sentence with 50 words, each with 768 features = **38,400 dimensions**
        - A neural network with 1B parameters operates in a **1-billion-dimensional space**

        **Why Visualize?**

        - Human brains evolved for 3D space
        - Patterns invisible in high-D become clear in 3D projections
        - Develops intuition for abstract mathematical operations

        **The Curse of Dimensionality:**
        As dimensions increase:
        - Data becomes sparse
        - Distances between points become similar
        - Overfitting becomes easier (more dimensions than samples)

        **Dimensionality Reduction:**
        - **PCA**: Project onto principal components (maximize variance)
        - **t-SNE**: Preserve local neighborhood structure
        - **UMAP**: Fast approximation of t-SNE with better global structure
        """)

    with st.expander("Vectors and Vector Spaces"):
        st.markdown("""
        **What is a Vector?**

        A vector is a direction and magnitude in space. In ML:
        - `v = [v₁, v₂, ..., vₙ]` represents a point or direction in n-dimensional space
        - **Magnitude**: `||v|| = √(v₁² + v₂² + ... + vₙ²)`
        - **Direction**: The ratio between components

        **Vector Operations in ML:**

        | Operation | Formula | ML Use Case |
        |-----------|---------|-------------|
        | Dot Product | `a · b = Σ aᵢbᵢ` | Similarity, attention |
        | Cross Product | `a × b` | Geometry, rotations |
        | Magnitude | `||a|| = √(a·a)` | Normalization |
        | Cosine Similarity | `(a·b)/||a||·||b||` | Semantic similarity |

        **Example - Cosine Similarity in NLP:**
        ```
        word embeddings: king = [0.8, 0.3, ...]
                       queen = [0.7, 0.4, ...]
        similarity = cos(θ) = 0.95  (nearly parallel!)
        ```
        """)

    with st.expander("Matrices as Linear Transformations"):
        st.markdown("""
        **Matrix = Collection of Vectors**

        A matrix `W` with shape (m × n) contains:
        - m row vectors (each of dimension n)
        - n column vectors (each of dimension m)

        **Common Matrix Types:**

        | Type | Property | ML Example |
        |------|----------|------------|
        | Identity | `AI = IA = A` | Skip connections |
        | Diagonal | Only main diagonal | Layer norms |
        | Symmetric | `A = Aᵀ` | Covariance matrices |
        | Orthogonal | `AAᵀ = I` | Rotation matrices |

        **Matrix Multiplication as Transformation:**
        ```
        y = Wx + b

        where:
        - W = (m × n) weight matrix
        - x = (n × 1) input vector
        - b = (m × 1) bias vector
        - y = (m × 1) output vector
        ```

        **Eigenvalues and Singular Values:**
        - Eigenvalue λ: `Wv = λv` (v is unchanged direction)
        - SVD: `W = UΣVᵀ` (any matrix decomposed)
        - Largest singular value = spectral norm (measures max stretch)
        """)

    # Applications Section
    st.markdown("### 🛠️ Applications in Machine Learning")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. Weight Spaces**
        - Neural network weights live in high-dimensional space
        - Loss landscape visualization shows optimization terrain
        - Modern optimizers (Adam, AdamW) navigate this landscape

        **2. Feature Spaces**
        - Each data point is a vector in feature space
        - Similar inputs cluster together
        - Distance metrics (Euclidean, Cosine) measure similarity

        **3. Attention Mechanism**
        - Attention scores: `softmax(QKᵀ/√d)V`
        - Q, K, V are matrices of query, key, value vectors
        - Visualize as a heatmap showing which tokens attend to which
        """)

    with col2:
        st.markdown("""
        **4. Gradient Fields**
        - Gradients point in direction of steepest ascent
        - Optimization = descending gradient (finding minima)
        - Visualize as vector field showing flow toward minimum

        **5. Embedding Spaces**
        - Word2Vec, BERT create dense vector representations
        - Analogies: king - man + woman ≈ queen
        - Visualize with dimensionality reduction (PCA, t-SNE)

        **6. Parameter Manifolds**
        - Pre-trained model weights form a manifold
        - Fine-tuning moves along manifold to new task
        - LoRA: Low-rank adaptation on this manifold
        """)

    # Key Takeaways
    st.markdown("### 🎯 Key Takeaways")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
    1. <b>Everything is a vector</b>: Data, weights, activations all live in vector spaces

    2. <b>Matrices transform</b>: Linear transformations (rotation, scaling, projection) are fundamental

    3. <b>Distance measures similarity</b>: How we measure distance determines what patterns we find

    4. <b>Higher dimensions = more expressiveness</b>: But also more computational cost and overfitting risk

    5. <b>Visualization builds intuition</b>: 3D visualizations help understand high-dimensional concepts
    </div>
    """, unsafe_allow_html=True)

    # Quiz
    with st.expander("📝 Quick Quiz: Test Your Understanding"):
        st.markdown("""
        **Q1:** What does the dot product of two normalized vectors give you?
        - (a) Euclidean distance
        - (b) Cosine similarity
        - (c) Manhattan distance
        - (d) Hamming distance

        **Q2:** Why do we use transpose in matrix multiplication for attention?
        - (a) To make dimensions match (QKᵀ)
        - (b) To increase values
        - (c) To decrease values
        - (d) No particular reason

        **Q3:** In a weight matrix W (m×n), what do the columns represent?
        - (a) Output features
        - (b) Input features
        - (c) Batch samples
        - (d) Time steps
        """)