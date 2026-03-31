# ML Visualization Lab

An interactive educational tool for visualizing machine learning concepts, from fundamentals to modern LLM architectures.

## Features

- **10 Progressive Chapters** covering ML fundamentals to advanced topics
- **Interactive Visualizations** powered by Plotly and Streamlit
- **Real-time Parameter Adjustment** with sliders and controls
- **Modern LLM Architecture Builder** with support for:
  - Mixture of Experts (MoE)
  - Grouped Query Attention (GQA)
  - Rotary Position Embeddings (RoPE)
  - SwiGLU Activation
  - KV Cache optimization

## Prerequisites

### System Requirements
- **Python 3.9+** (Python 3.10+ recommended)
- **pip** or **conda** package manager

### Required Dependencies
```
streamlit>=1.28.0
plotly>=5.18.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
altair>=5.0.0
networkx>=3.0
pillow>=10.0.0
```

### Optional Dependencies (for model demonstrations)
```
# torch>=2.0.0
# tensorflow>=2.13.0
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ml_formula.git
cd ml_formula
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n ml_viz python=3.10
conda activate ml_viz
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Project Structure

```
ml_formula/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # Apache 2.0 License
├── assets/                   # Static assets
├── utils/                    # Utility functions
└── visualizations/           # Visualization modules
    ├── architectures.py     # Neural network diagrams
    ├── distributions.py      # Probability distributions
    ├── functions.py         # Loss & activation functions
    ├── llm_builder.py        # Modern LLM architecture builder
    ├── math_fundamentals.py # Gradient descent, backprop
    ├── models.py            # Advanced models (Transformers, Diffusion)
    ├── three_d_graphs.py    # 3D surface plots & vector fields
    └── torch_layers.py      # PyTorch layer builder
```

## Chapter Curriculum

| Chapter | Topic | Description |
|---------|-------|-------------|
| 1 | 3D Graphs & Vector Spaces | Visualize functions, vector operations |
| 2 | Probability Distributions | Interactive distribution explorer |
| 3 | Math Foundations | Gradient descent & backpropagation |
| 4 | Loss Functions | MSE, MAE, Cross-Entropy, Huber |
| 5 | Activation Functions | ReLU, Sigmoid, Tanh, GELU |
| 6 | Neural Network Architectures | CNN, RNN, Attention |
| 7 | PyTorch Layer Builder | Build custom models layer-by-layer |
| 8 | Modern LLM Builder | MoE, GQA, RoPE, SwiGLU, KV Cache |
| 9 | Advanced Models | Transformers, Diffusion, Video, Audio |
| 10 | Model Training Simulation | Training dynamics & optimization |

## How to Use

1. **Navigate** using the sidebar to select chapters
2. **Adjust parameters** with sliders and dropdowns
3. **Explore tabs** for different visualization views
4. **Expand sections** for mathematical details and formulas
5. **Resize plots** using the sidebar width/height controls

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
