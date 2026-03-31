"""
Configuration constants for ML Visualization Tool
"""

# Plotting defaults
PLOT_WIDTH = 800
PLOT_HEIGHT = 600
PLOT_TEMPLATE = "plotly_white"

# Color schemes
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'quinary': '#9467bd'
}

# Default ranges for visualizations
DEFAULT_RANGES = {
    'x_range': [-10, 10],
    'y_range': [-10, 10],
    'z_range': [-10, 10],
    'probability_range': [0, 1],
    'learning_rate_range': [0.001, 1.0],
    'epoch_range': [1, 100]
}

# Model defaults
DEFAULT_MODEL_PARAMS = {
    'hidden_layers': [64, 32],
    'activation': 'relu',
    'learning_rate': 0.01,
    'batch_size': 32
}