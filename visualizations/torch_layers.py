"""
PyTorch Layer Builder - Interactive Sequential Model Builder
Allows users to create PyTorch models by stacking layers and adding custom code
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Dict, Any, Optional, Callable
import streamlit as st
import pandas as pd
import re


# ============================================================================
# PyTorch Layer Definitions
# ============================================================================

PYTORCH_LAYERS = {
    # --- Core Linear Layers ---
    "Input": {
        "category": "Input",
        "color": "#94a3b8",
        "icon": "IN",
        "class": None,  # Special handling for input
        "params": [
            {"k": "shape", "label": "Input Shape", "type": "text", "default": "[None, 784]", "help": "e.g., [None, 784] or [1, 28, 28]"}
        ],
        "code_template": "# Input shape: {shape}"
    },

    "Linear": {
        "category": "Linear",
        "color": "#38bdf8",
        "icon": "FC",
        "class": "nn.Linear",
        "params": [
            {"k": "in_features", "label": "In Features", "type": "number", "default": 784},
            {"k": "out_features", "label": "Out Features", "type": "number", "default": 128},
            {"k": "bias", "label": "Bias", "type": "bool", "default": True}
        ],
        "code_template": "nn.Linear({in_features}, {out_features}, bias={bias})"
    },

    # --- Convolution Layers ---
    "Conv1d": {
        "category": "Convolution",
        "color": "#f87171",
        "icon": "C1D",
        "class": "nn.Conv1d",
        "params": [
            {"k": "in_channels", "label": "In Channels", "type": "number", "default": 1},
            {"k": "out_channels", "label": "Out Channels", "type": "number", "default": 32},
            {"k": "kernel_size", "label": "Kernel Size", "type": "number", "default": 3},
            {"k": "stride", "label": "Stride", "type": "number", "default": 1},
            {"k": "padding", "label": "Padding", "type": "number", "default": 1},
            {"k": "dilation", "label": "Dilation", "type": "number", "default": 1},
            {"k": "bias", "label": "Bias", "type": "bool", "default": False}
        ],
        "code_template": "nn.Conv1d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, bias={bias})"
    },

    "Conv2d": {
        "category": "Convolution",
        "color": "#f87171",
        "icon": "C2D",
        "class": "nn.Conv2d",
        "params": [
            {"k": "in_channels", "label": "In Channels", "type": "number", "default": 1},
            {"k": "out_channels", "label": "Out Channels", "type": "number", "default": 32},
            {"k": "kernel_size", "label": "Kernel Size", "type": "text", "default": "3"},
            {"k": "stride", "label": "Stride", "type": "text", "default": "1"},
            {"k": "padding", "label": "Padding", "type": "text", "default": "1"},
            {"k": "dilation", "label": "Dilation", "type": "text", "default": "1"},
            {"k": "bias", "label": "Bias", "type": "bool", "default": False}
        ],
        "code_template": "nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, bias={bias})"
    },

    "Conv3d": {
        "category": "Convolution",
        "color": "#f87171",
        "icon": "C3D",
        "class": "nn.Conv3d",
        "params": [
            {"k": "in_channels", "label": "In Channels", "type": "number", "default": 1},
            {"k": "out_channels", "label": "Out Channels", "type": "number", "default": 32},
            {"k": "kernel_size", "label": "Kernel Size", "type": "text", "default": "3"},
            {"k": "stride", "label": "Stride", "type": "text", "default": "1"},
            {"k": "padding", "label": "Padding", "type": "text", "default": "1"},
            {"k": "bias", "label": "Bias", "type": "bool", "default": False}
        ],
        "code_template": "nn.Conv3d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, bias={bias})"
    },

    "TransposedConv2d": {
        "category": "Convolution",
        "color": "#ef4444",
        "icon": "TC2",
        "class": "nn.ConvTranspose2d",
        "params": [
            {"k": "in_channels", "label": "In Channels", "type": "number", "default": 512},
            {"k": "out_channels", "label": "Out Channels", "type": "number", "default": 256},
            {"k": "kernel_size", "label": "Kernel Size", "type": "text", "default": "4"},
            {"k": "stride", "label": "Stride", "type": "text", "default": "2"},
            {"k": "padding", "label": "Padding", "type": "text", "default": "1"},
            {"k": "output_padding", "label": "Output Padding", "type": "text", "default": "1"},
            {"k": "bias", "label": "Bias", "type": "bool", "default": False}
        ],
        "code_template": "nn.ConvTranspose2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding}, bias={bias})"
    },

    # --- Pooling Layers ---
    "MaxPool1d": {
        "category": "Pooling",
        "color": "#fb923c",
        "icon": "MP1",
        "class": "nn.MaxPool1d",
        "params": [
            {"k": "kernel_size", "label": "Kernel Size", "type": "number", "default": 2},
            {"k": "stride", "label": "Stride", "type": "number", "default": 2},
            {"k": "padding", "label": "Padding", "type": "number", "default": 0}
        ],
        "code_template": "nn.MaxPool1d(kernel_size={kernel_size}, stride={stride}, padding={padding})"
    },

    "MaxPool2d": {
        "category": "Pooling",
        "color": "#fb923c",
        "icon": "MP2",
        "class": "nn.MaxPool2d",
        "params": [
            {"k": "kernel_size", "label": "Kernel Size", "type": "text", "default": "2"},
            {"k": "stride", "label": "Stride", "type": "text", "default": "2"},
            {"k": "padding", "label": "Padding", "type": "text", "default": "0"}
        ],
        "code_template": "nn.MaxPool2d(kernel_size={kernel_size}, stride={stride}, padding={padding})"
    },

    "MaxPool3d": {
        "category": "Pooling",
        "color": "#fb923c",
        "icon": "MP3",
        "class": "nn.MaxPool3d",
        "params": [
            {"k": "kernel_size", "label": "Kernel Size", "type": "text", "default": "2"},
            {"k": "stride", "label": "Stride", "type": "text", "default": "2"},
            {"k": "padding", "label": "Padding", "type": "text", "default": "0"}
        ],
        "code_template": "nn.MaxPool3d(kernel_size={kernel_size}, stride={stride}, padding={padding})"
    },

    "AvgPool1d": {
        "category": "Pooling",
        "color": "#fb923c",
        "icon": "AP1",
        "class": "nn.AvgPool1d",
        "params": [
            {"k": "kernel_size", "label": "Kernel Size", "type": "number", "default": 2},
            {"k": "stride", "label": "Stride", "type": "number", "default": 2},
            {"k": "padding", "label": "Padding", "type": "number", "default": 0}
        ],
        "code_template": "nn.AvgPool1d(kernel_size={kernel_size}, stride={stride}, padding={padding})"
    },

    "AvgPool2d": {
        "category": "Pooling",
        "color": "#fb923c",
        "icon": "AP2",
        "class": "nn.AvgPool2d",
        "params": [
            {"k": "kernel_size", "label": "Kernel Size", "type": "text", "default": "2"},
            {"k": "stride", "label": "Stride", "type": "text", "default": "2"},
            {"k": "padding", "label": "Padding", "type": "text", "default": "0"}
        ],
        "code_template": "nn.AvgPool2d(kernel_size={kernel_size}, stride={stride}, padding={padding})"
    },

    "AdaptiveAvgPool1d": {
        "category": "Pooling",
        "color": "#fdba74",
        "icon": "AA1",
        "class": "nn.AdaptiveAvgPool1d",
        "params": [
            {"k": "output_size", "label": "Output Size", "type": "number", "default": 1}
        ],
        "code_template": "nn.AdaptiveAvgPool1d({output_size})"
    },

    "AdaptiveAvgPool2d": {
        "category": "Pooling",
        "color": "#fdba74",
        "icon": "AA2",
        "class": "nn.AdaptiveAvgPool2d",
        "params": [
            {"k": "output_size", "label": "Output Size", "type": "text", "default": "[1, 1]"}
        ],
        "code_template": "nn.AdaptiveAvgPool2d({output_size})"
    },

    "AdaptiveMaxPool2d": {
        "category": "Pooling",
        "color": "#fdba74",
        "icon": "AM2",
        "class": "nn.AdaptiveMaxPool2d",
        "params": [
            {"k": "output_size", "label": "Output Size", "type": "text", "default": "[1, 1]"}
        ],
        "code_template": "nn.AdaptiveMaxPool2d({output_size})"
    },

    # --- Normalization Layers ---
    "BatchNorm1d": {
        "category": "Normalization",
        "color": "#34d399",
        "icon": "BN1",
        "class": "nn.BatchNorm1d",
        "params": [
            {"k": "num_features", "label": "Num Features", "type": "number", "default": 128},
            {"k": "eps", "label": "Epsilon", "type": "float", "default": 1e-05},
            {"k": "momentum", "label": "Momentum", "type": "float", "default": 0.1},
            {"k": "affine", "label": "Affine", "type": "bool", "default": True},
            {"k": "track_running_stats", "label": "Track Stats", "type": "bool", "default": True}
        ],
        "code_template": "nn.BatchNorm1d({num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats})"
    },

    "BatchNorm2d": {
        "category": "Normalization",
        "color": "#34d399",
        "icon": "BN2",
        "class": "nn.BatchNorm2d",
        "params": [
            {"k": "num_features", "label": "Num Features", "type": "number", "default": 64},
            {"k": "eps", "label": "Epsilon", "type": "float", "default": 1e-05},
            {"k": "momentum", "label": "Momentum", "type": "float", "default": 0.1},
            {"k": "affine", "label": "Affine", "type": "bool", "default": True}
        ],
        "code_template": "nn.BatchNorm2d({num_features}, eps={eps}, momentum={momentum}, affine={affine})"
    },

    "LayerNorm": {
        "category": "Normalization",
        "color": "#34d399",
        "icon": "LN",
        "class": "nn.LayerNorm",
        "params": [
            {"k": "normalized_shape", "label": "Normalized Shape", "type": "text", "default": "[64]"},
            {"k": "eps", "label": "Epsilon", "type": "float", "default": 1e-05},
            {"k": "elementwise_affine", "label": "Elementwise Affine", "type": "bool", "default": True}
        ],
        "code_template": "nn.LayerNorm({normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine})"
    },

    "InstanceNorm1d": {
        "category": "Normalization",
        "color": "#059669",
        "icon": "IN1",
        "class": "nn.InstanceNorm1d",
        "params": [
            {"k": "num_features", "label": "Num Features", "type": "number", "default": 64},
            {"k": "eps", "label": "Epsilon", "type": "float", "default": 1e-05},
            {"k": "momentum", "label": "Momentum", "type": "float", "default": 0.1},
            {"k": "affine", "label": "Affine", "type": "bool", "default": False}
        ],
        "code_template": "nn.InstanceNorm1d({num_features}, eps={eps}, momentum={momentum}, affine={affine})"
    },

    "InstanceNorm2d": {
        "category": "Normalization",
        "color": "#059669",
        "icon": "IN2",
        "class": "nn.InstanceNorm2d",
        "params": [
            {"k": "num_features", "label": "Num Features", "type": "number", "default": 64},
            {"k": "eps", "label": "Epsilon", "type": "float", "default": 1e-05},
            {"k": "momentum", "label": "Momentum", "type": "float", "default": 0.1},
            {"k": "affine", "label": "Affine", "type": "bool", "default": False}
        ],
        "code_template": "nn.InstanceNorm2d({num_features}, eps={eps}, momentum={momentum}, affine={affine})"
    },

    "GroupNorm": {
        "category": "Normalization",
        "color": "#10b981",
        "icon": "GN",
        "class": "nn.GroupNorm",
        "params": [
            {"k": "num_groups", "label": "Num Groups", "type": "number", "default": 32},
            {"k": "num_channels", "label": "Num Channels", "type": "number", "default": 64},
            {"k": "eps", "label": "Epsilon", "type": "float", "default": 1e-05}
        ],
        "code_template": "nn.GroupNorm({num_groups}, {num_channels}, eps={eps})"
    },

    # --- Dropout Layers ---
    "Dropout": {
        "category": "Dropout",
        "color": "#6b7280",
        "icon": "DO",
        "class": "nn.Dropout",
        "params": [
            {"k": "p", "label": "Probability", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.Dropout(p={p}, inplace={inplace})"
    },

    "Dropout2d": {
        "category": "Dropout",
        "color": "#6b7280",
        "icon": "DO2",
        "class": "nn.Dropout2d",
        "params": [
            {"k": "p", "label": "Probability", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.Dropout2d(p={p}, inplace={inplace})"
    },

    "AlphaDropout": {
        "category": "Dropout",
        "color": "#4b5563",
        "icon": "AD",
        "class": "nn.AlphaDropout",
        "params": [
            {"k": "p", "label": "Probability", "type": "float", "default": 0.5},
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.AlphaDropout(p={p}, inplace={inplace})"
    },

    # --- Recurrent Layers ---
    "LSTM": {
        "category": "Recurrent",
        "color": "#fbbf24",
        "icon": "LSTM",
        "class": "nn.LSTM",
        "params": [
            {"k": "input_size", "label": "Input Size", "type": "number", "default": 256},
            {"k": "hidden_size", "label": "Hidden Size", "type": "number", "default": 128},
            {"k": "num_layers", "label": "Num Layers", "type": "number", "default": 1},
            {"k": "bias", "label": "Bias", "type": "bool", "default": True},
            {"k": "batch_first", "label": "Batch First", "type": "bool", "default": True},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.0},
            {"k": "bidirectional", "label": "Bidirectional", "type": "bool", "default": False}
        ],
        "code_template": "nn.LSTM({input_size}, {hidden_size}, num_layers={num_layers}, bias={bias}, batch_first={batch_first}, dropout={dropout}, bidirectional={bidirectional})"
    },

    "GRU": {
        "category": "Recurrent",
        "color": "#c084fc",
        "icon": "GRU",
        "class": "nn.GRU",
        "params": [
            {"k": "input_size", "label": "Input Size", "type": "number", "default": 256},
            {"k": "hidden_size", "label": "Hidden Size", "type": "number", "default": 128},
            {"k": "num_layers", "label": "Num Layers", "type": "number", "default": 1},
            {"k": "bias", "label": "Bias", "type": "bool", "default": True},
            {"k": "batch_first", "label": "Batch First", "type": "bool", "default": True},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.0},
            {"k": "bidirectional", "label": "Bidirectional", "type": "bool", "default": False}
        ],
        "code_template": "nn.GRU({input_size}, {hidden_size}, num_layers={num_layers}, bias={bias}, batch_first={batch_first}, dropout={dropout}, bidirectional={bidirectional})"
    },

    "RNN": {
        "category": "Recurrent",
        "color": "#a855f7",
        "icon": "RNN",
        "class": "nn.RNN",
        "params": [
            {"k": "input_size", "label": "Input Size", "type": "number", "default": 256},
            {"k": "hidden_size", "label": "Hidden Size", "type": "number", "default": 128},
            {"k": "num_layers", "label": "Num Layers", "type": "number", "default": 1},
            {"k": "nonlinearity", "label": "Nonlinearity", "type": "select", "options": ["relu", "tanh"], "default": "relu"},
            {"k": "bias", "label": "Bias", "type": "bool", "default": True},
            {"k": "batch_first", "label": "Batch First", "type": "bool", "default": True},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.0},
            {"k": "bidirectional", "label": "Bidirectional", "type": "bool", "default": False}
        ],
        "code_template": "nn.RNN({input_size}, {hidden_size}, num_layers={num_layers}, nonlinearity='{nonlinearity}', bias={bias}, batch_first={batch_first}, dropout={dropout}, bidirectional={bidirectional})"
    },

    # --- Embedding Layers ---
    "Embedding": {
        "category": "Embedding",
        "color": "#f472b6",
        "icon": "EM",
        "class": "nn.Embedding",
        "params": [
            {"k": "num_embeddings", "label": "Vocab Size", "type": "number", "default": 10000},
            {"k": "embedding_dim", "label": "Embedding Dim", "type": "number", "default": 256}
        ],
        "code_template": "nn.Embedding({num_embeddings}, {embedding_dim})"
    },

    "EmbeddingBag": {
        "category": "Embedding",
        "color": "#ec4899",
        "icon": "EB",
        "class": "nn.EmbeddingBag",
        "params": [
            {"k": "num_embeddings", "label": "Vocab Size", "type": "number", "default": 10000},
            {"k": "embedding_dim", "label": "Embedding Dim", "type": "number", "default": 256},
            {"k": "mode", "label": "Mode", "type": "select", "options": ["mean", "sum", "max"], "default": "mean"}
        ],
        "code_template": "nn.EmbeddingBag({num_embeddings}, {embedding_dim}, mode='{mode}')"
    },

    # --- Attention Layers ---
    "MultiheadAttention": {
        "category": "Attention",
        "color": "#a78bfa",
        "icon": "MHA",
        "class": "nn.MultiheadAttention",
        "params": [
            {"k": "embed_dim", "label": "Embed Dim", "type": "number", "default": 512},
            {"k": "num_heads", "label": "Num Heads", "type": "number", "default": 8},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.1},
            {"k": "bias", "label": "Bias", "type": "bool", "default": True},
            {"k": "add_bias_kv", "label": "Add Bias KV", "type": "bool", "default": False},
            {"k": "kdim", "label": "Key Dim", "type": "number", "default": 512},
            {"k": "vdim", "label": "Value Dim", "type": "number", "default": 512}
        ],
        "code_template": "nn.MultiheadAttention({embed_dim}, {num_heads}, dropout={dropout}, bias={bias}, add_bias_kv={add_bias_kv}, kdim={kdim}, vdim={vdim})"
    },

    # --- Transformer Layers ---
    "TransformerEncoder": {
        "category": "Transformer",
        "color": "#8b5cf6",
        "icon": "TE",
        "class": "nn.TransformerEncoder",
        "params": [
            {"k": "d_model", "label": "Model Dim", "type": "number", "default": 512},
            {"k": "nhead", "label": "Num Heads", "type": "number", "default": 8},
            {"k": "num_layers", "label": "Num Layers", "type": "number", "default": 6},
            {"k": "dim_feedforward", "label": "FF Dim", "type": "number", "default": 2048},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.1},
            {"k": "activation", "label": "Activation", "type": "select", "options": ["relu", "gelu"], "default": "relu"}
        ],
        "code_template": "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}'), num_layers={num_layers})"
    },

    "TransformerDecoder": {
        "category": "Transformer",
        "color": "#7c3aed",
        "icon": "TD",
        "class": "nn.TransformerDecoder",
        "params": [
            {"k": "d_model", "label": "Model Dim", "type": "number", "default": 512},
            {"k": "nhead", "label": "Num Heads", "type": "number", "default": 8},
            {"k": "num_layers", "label": "Num Layers", "type": "number", "default": 6},
            {"k": "dim_feedforward", "label": "FF Dim", "type": "number", "default": 2048},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.1},
            {"k": "activation", "label": "Activation", "type": "select", "options": ["relu", "gelu"], "default": "relu"}
        ],
        "code_template": "nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}'), num_layers={num_layers})"
    },

    "Transformer": {
        "category": "Transformer",
        "color": "#6d28d9",
        "icon": "TF",
        "class": "nn.Transformer",
        "params": [
            {"k": "d_model", "label": "Model Dim", "type": "number", "default": 512},
            {"k": "nhead", "label": "Num Heads", "type": "number", "default": 8},
            {"k": "num_encoder_layers", "label": "Encoder Layers", "type": "number", "default": 6},
            {"k": "num_decoder_layers", "label": "Decoder Layers", "type": "number", "default": 6},
            {"k": "dim_feedforward", "label": "FF Dim", "type": "number", "default": 2048},
            {"k": "dropout", "label": "Dropout", "type": "float", "default": 0.1},
            {"k": "activation", "label": "Activation", "type": "select", "options": ["relu", "gelu"], "default": "relu"}
        ],
        "code_template": "nn.Transformer(d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, num_decoder_layers={num_decoder_layers}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}')"
    },

    # --- Shape Manipulation ---
    "Flatten": {
        "category": "Shape",
        "color": "#fb923c",
        "icon": "FL",
        "class": "nn.Flatten",
        "params": [
            {"k": "start_dim", "label": "Start Dim", "type": "number", "default": 1},
            {"k": "end_dim", "label": "End Dim", "type": "number", "default": -1}
        ],
        "code_template": "nn.Flatten(start_dim={start_dim}, end_dim={end_dim})"
    },

    "Unflatten": {
        "category": "Shape",
        "color": "#fdba74",
        "icon": "UF",
        "class": "nn.Unflatten",
        "params": [
            {"k": "dim", "label": "Dim", "type": "number", "default": 1},
            {"k": "unflattened_size", "label": "Size", "type": "text", "default": "[1, 784]"}
        ],
        "code_template": "nn.Unflatten({dim}, {unflattened_size})"
    },

    "Reshape": {
        "category": "Shape",
        "color": "#67e8f9",
        "icon": "RS",
        "class": "nn.Reshape",
        "params": [
            {"k": "shape", "label": "Shape", "type": "text", "default": "[-1, 784]"}
        ],
        "code_template": "nn.Reshape({shape})"
    },

    "Permute": {
        "category": "Shape",
        "color": "#22d3ee",
        "icon": "PM",
        "class": "nn.Permute",
        "params": [
            {"k": "dims", "label": "Dims", "type": "text", "default": "[0, 2, 1]"}
        ],
        "code_template": "nn.Permute({dims})"
    },

    "Squeeze": {
        "category": "Shape",
        "color": "#06b6d4",
        "icon": "SQ",
        "class": "nn.Squeeze",
        "params": [
            {"k": "dim", "label": "Dim", "type": "number", "default": 1}
        ],
        "code_template": "nn.Squeeze({dim})"
    },

    "Unsqueeze": {
        "category": "Shape",
        "color": "#0891b2",
        "icon": "US",
        "class": "nn.Unsqueeze",
        "params": [
            {"k": "dim", "label": "Dim", "type": "number", "default": 1}
        ],
        "code_template": "nn.Unsqueeze({dim})"
    },

    # --- Activation Layers ---
    "ReLU": {
        "category": "Activation",
        "color": "#ef4444",
        "icon": "ReLU",
        "class": "nn.ReLU",
        "params": [
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.ReLU(inplace={inplace})"
    },

    "LeakyReLU": {
        "category": "Activation",
        "color": "#dc2626",
        "icon": "LReLU",
        "class": "nn.LeakyReLU",
        "params": [
            {"k": "negative_slope", "label": "Neg Slope", "type": "float", "default": 0.01},
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.LeakyReLU(negative_slope={negative_slope}, inplace={inplace})"
    },

    "PReLU": {
        "category": "Activation",
        "color": "#b91c1c",
        "icon": "PReLU",
        "class": "nn.PReLU",
        "params": [
            {"k": "num_parameters", "label": "Num Params", "type": "number", "default": 1},
            {"k": "init", "label": "Init", "type": "float", "default": 0.25}
        ],
        "code_template": "nn.PReLU(num_parameters={num_parameters}, init={init})"
    },

    "RReLU": {
        "category": "Activation",
        "color": "#991b1b",
        "icon": "RReLU",
        "class": "nn.RReLU",
        "params": [
            {"k": "lower", "label": "Lower", "type": "float", "default": 0.125},
            {"k": "upper", "label": "Upper", "type": "float", "default": 0.333}
        ],
        "code_template": "nn.RReLU(lower={lower}, upper={upper})"
    },

    "GELU": {
        "category": "Activation",
        "color": "#7f1d1d",
        "icon": "GELU",
        "class": "nn.GELU",
        "params": [
            {"k": "approximate", "label": "Approximate", "type": "select", "options": ["none", "tanh"], "default": "none"}
        ],
        "code_template": "nn.GELU(approximate='{approximate}')"
    },

    "SiLU": {
        "category": "Activation",
        "color": "#450a0a",
        "icon": "SiLU",
        "class": "nn.SiLU",
        "params": [
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.SiLU(inplace={inplace})"
    },

    "Mish": {
        "category": "Activation",
        "color": "#7f1d1d",
        "icon": "Mish",
        "class": "nn.Mish",
        "params": [
            {"k": "inplace", "label": "Inplace", "type": "bool", "default": False}
        ],
        "code_template": "nn.Mish(inplace={inplace})"
    },

    "Tanh": {
        "category": "Activation",
        "color": "#f97316",
        "icon": "Tanh",
        "class": "nn.Tanh",
        "params": [],
        "code_template": "nn.Tanh()"
    },

    "Sigmoid": {
        "category": "Activation",
        "color": "#ea580c",
        "icon": "Sigm",
        "class": "nn.Sigmoid",
        "params": [],
        "code_template": "nn.Sigmoid()"
    },

    "Softmax": {
        "category": "Activation",
        "color": "#c2410c",
        "icon": "SM",
        "class": "nn.Softmax",
        "params": [
            {"k": "dim", "label": "Dim", "type": "number", "default": -1}
        ],
        "code_template": "nn.Softmax(dim={dim})"
    },

    "LogSoftmax": {
        "category": "Activation",
        "color": "#9a3412",
        "icon": "LSM",
        "class": "nn.LogSoftmax",
        "params": [
            {"k": "dim", "label": "Dim", "type": "number", "default": -1}
        ],
        "code_template": "nn.LogSoftmax(dim={dim})"
    },

    # --- Output Layer ---
    "Output": {
        "category": "Output",
        "color": "#4ade80",
        "icon": "OUT",
        "class": None,
        "params": [
            {"k": "units", "label": "Output Units", "type": "number", "default": 10},
            {"k": "activation", "label": "Activation", "type": "select", "options": ["linear", "softmax", "sigmoid", "log_softmax"], "default": "linear"}
        ],
        "code_template": "# Output: {units} units with {activation}"
    },

    # --- Custom Layer ---
    "Custom": {
        "category": "Custom",
        "color": "#facc15",
        "icon": "CUST",
        "class": "Custom",
        "params": [
            {"k": "name", "label": "Layer Name", "type": "text", "default": "CustomLayer"},
            {"k": "code", "label": "PyTorch Code", "type": "textarea", "default": "# Example:\nnn.Sequential(\n    nn.Linear(128, 64),\n    nn.ReLU()\n)"}
        ],
        "code_template": "{code}"
    }
}


# Default parameter values for each layer
DEFAULT_PARAMS = {k: {p["k"]: p["default"] for p in v["params"]} for k, v in PYTORCH_LAYERS.items() if v["params"]}


# ============================================================================
# Code Generation
# ============================================================================

def generate_layer_code(layer_type: str, params: Dict[str, Any]) -> str:
    """Generate PyTorch code for a single layer"""
    if layer_type == "Custom":
        return params.get("code", "# Custom code")
    if layer_type == "Output":
        activation = params.get("activation", "linear")
        if activation == "softmax":
            return "nn.Sequential(nn.Linear(params['in'], params['out']), nn.Softmax(dim=-1))"
        elif activation == "sigmoid":
            return "nn.Sequential(nn.Linear(params['in'], params['out']), nn.Sigmoid())"
        elif activation == "log_softmax":
            return "nn.Sequential(nn.Linear(params['in'], params['out']), nn.LogSoftmax(dim=-1))"
        else:
            return "# Output layer - specify Linear in_features/out_features"
    if layer_type == "Input":
        return f"# Input shape: {params.get('shape', '[None, ...]')}"

    layer_info = PYTORCH_LAYERS.get(layer_type)
    if not layer_info or not layer_info["code_template"]:
        return f"# {layer_type}"

    template = layer_info["code_template"]
    # Replace all placeholders
    for key, value in params.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, str) and not (value.startswith('[') or value.startswith('(')):
            value = f"'{value}'"
        template = template.replace(f"{{{key}}}", str(value))

    return template


def generate_sequential_code(layers: List[Dict[str, Any]]) -> str:
    """Generate complete PyTorch Sequential model code"""
    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "class SequentialModel(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.layers = nn.Sequential("
    ]

    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        params = layer.get("params", {})

        if layer_type == "Input":
            lines.append(f"        # Layer {i+1}: {layer_type} - {params.get('shape', '')}")
        elif layer_type == "Output":
            lines.append(f"        # Layer {i+1}: {layer_type} - {params.get('units', '')} units, {params.get('activation', '')}")
        else:
            code = generate_layer_code(layer_type, params)
            lines.append(f"        # Layer {i+1}: {layer_type}")
            for line in code.split('\n'):
                lines.append(f"        {line}")

    lines.extend([
        "        )",
        "",
        "    def forward(self, x):",
        "        return self.layers(x)",
        "",
        "",
        "# Create model instance",
        "model = SequentialModel()",
        "print(model)",
        "",
        "# Example forward pass",
        "# x = torch.randn(1, 784)  # Example input",
        "# output = model(x)",
        "# print(output.shape)"
    ])

    return '\n'.join(lines)


def count_parameters(layers: List[Dict[str, Any]]) -> Dict[str, int]:
    """Estimate parameter count for the model"""
    total_params = 0
    trainable_params = 0

    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        params = layer.get("params", {})

        if layer_type == "Linear":
            in_f = params.get("in_features", 0)
            out_f = params.get("out_features", 0)
            bias = params.get("bias", True)
            count = in_f * out_f + (out_f if bias else 0)
            total_params += count
            trainable_params += count

        elif layer_type == "Conv2d":
            in_c = params.get("in_channels", 1)
            out_c = params.get("out_channels", 1)
            kernel = params.get("kernel_size", 3)
            if isinstance(kernel, str):
                kernel = int(kernel.split('x')[0]) if 'x' in kernel else int(kernel)
            bias = params.get("bias", False)
            count = in_c * out_c * kernel * kernel + (out_c if bias else 0)
            total_params += count
            trainable_params += count

        elif layer_type == "Conv1d":
            in_c = params.get("in_channels", 1)
            out_c = params.get("out_channels", 1)
            kernel = params.get("kernel_size", 3)
            bias = params.get("bias", False)
            count = in_c * out_c * kernel + (out_c if bias else 0)
            total_params += count
            trainable_params += count

        elif layer_type == "LSTM" or layer_type == "GRU":
            input_size = params.get("input_size", 0)
            hidden_size = params.get("hidden_size", 0)
            num_layers = params.get("num_layers", 1)
            bidirectional = params.get("bidirectional", False)
            directions = 2 if bidirectional else 1

            # Approximate parameter count
            if layer_type == "LSTM":
                # LSTM has 4 gates: input, forget, cell, output
                count = 4 * hidden_size * (input_size + hidden_size + 1) * directions * num_layers
            else:
                # GRU has 3 gates: update, reset, new
                count = 3 * hidden_size * (input_size + hidden_size + 1) * directions * num_layers
            total_params += count
            trainable_params += count

        elif layer_type == "Embedding":
            num_embeddings = params.get("num_embeddings", 0)
            embedding_dim = params.get("embedding_dim", 0)
            count = num_embeddings * embedding_dim
            total_params += count
            trainable_params += count

        elif layer_type == "BatchNorm1d" or layer_type == "BatchNorm2d":
            num_features = params.get("num_features", 0)
            # 2 * num_features for weight and bias
            count = 2 * num_features
            total_params += count
            trainable_params += count

        elif layer_type == "LayerNorm":
            normalized_shape = params.get("normalized_shape", [1])
            if isinstance(normalized_shape, str):
                try:
                    normalized_shape = eval(normalized_shape)
                except:
                    normalized_shape = [1]
            count = 2 * np.prod(normalized_shape)
            total_params += count
            trainable_params += count

        elif layer_type == "MultiheadAttention":
            embed_dim = params.get("embed_dim", 512)
            num_heads = params.get("num_heads", 8)
            kdim = params.get("kdim", embed_dim)
            vdim = params.get("vdim", embed_dim)
            # Approximate
            count = 3 * embed_dim * (kdim + vdim + embed_dim) + embed_dim * 2
            total_params += count
            trainable_params += count

        elif layer_type in ["Dropout", "Flatten", "ReLU", "Sigmoid", "Tanh"]:
            pass  # No parameters

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


# ============================================================================
# Architecture Visualization
# ============================================================================

def create_architecture_diagram(layers: List[Dict], width: int = 800, height: int = 600) -> go.Figure:
    """Create an SVG-style architecture diagram using Plotly"""

    node_height = 60
    node_width = 200
    gap = 20
    total_height = len(layers) * (node_height + gap) + gap + 40

    fig = go.Figure()

    # Calculate positions
    for i, layer in enumerate(layers):
        layer_type = layer["type"]
        params = layer.get("params", {})

        layer_info = PYTORCH_LAYERS.get(layer_type, {})
        color = layer_info.get("color", "#6b7280")
        icon = layer_info.get("icon", "??")

        y = gap + i * (node_height + gap)
        x = 0.5 * width

        # Get primary parameter for display
        primary = ""
        if layer_type == "Input":
            primary = params.get("shape", "")
        elif layer_type == "Linear":
            primary = f"{params.get('in_features', '?')} → {params.get('out_features', '?')}"
        elif layer_type == "Conv2d":
            primary = f"{params.get('in_channels', '?')}CH → {params.get('out_channels', '?')}CH"
        elif layer_type == "LSTM" or layer_type == "GRU":
            primary = f"Hidden: {params.get('hidden_size', '?')}"
        elif layer_type == "Dropout":
            primary = f"p={params.get('p', '?')}"
        elif layer_type == "BatchNorm1d" or layer_type == "BatchNorm2d":
            primary = f"Features: {params.get('num_features', '?')}"
        elif layer_type == "Flatten":
            primary = f"dim {params.get('start_dim', '?')} → {params.get('end_dim', '?')}"
        elif layer_type == "MaxPool2d":
            primary = f"Pool: {params.get('kernel_size', '?')}"
        elif layer_type == "Output":
            primary = f"Units: {params.get('units', '?')}"
        elif layer_type == "Custom":
            primary = params.get("name", "Custom")
        else:
            # Use first parameter as primary
            if params:
                first_key = list(params.keys())[0]
                primary = str(params[first_key])

        # Node background
        fig.add_trace(go.Scatter(
            x=[x - node_width/2, x + node_width/2, x + node_width/2, x - node_width/2, x - node_width/2],
            y=[y, y, y + node_height, y + node_height, y],
            mode='lines',
            fill='toself',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
            line=dict(color=color, width=2),
            name=f"Layer {i+1}: {layer_type}",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"<b>{layer_type}</b><br>{primary}<br><i>Layer {i+1}</i>"
        ))

        # Icon box
        icon_x = x - node_width/2 + 35
        fig.add_trace(go.Scatter(
            x=[icon_x - 20, icon_x + 20, icon_x + 20, icon_x - 20, icon_x - 20],
            y=[y + 10, y + 10, y + node_height - 10, y + node_height - 10, y + 10],
            mode='lines',
            fill='toself',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)',
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Icon text
        fig.add_annotation(
            x=icon_x,
            y=y + node_height/2,
            text=icon,
            showarrow=False,
            font=dict(size=10, color=color, family="monospace"),
            align='center',
            valign='middle'
        )

        # Layer type text
        fig.add_annotation(
            x=x - node_width/2 + 80,
            y=y + 22,
            text=layer_type,
            showarrow=False,
            font=dict(size=14, color='#e2e8f0', family="sans-serif", weight="bold"),
            align='left',
            valign='middle'
        )

        # Primary parameter text
        fig.add_annotation(
            x=x - node_width/2 + 80,
            y=y + 40,
            text=primary[:30] + ("..." if len(primary) > 30 else ""),
            showarrow=False,
            font=dict(size=11, color='#94a3b8', family="monospace"),
            align='left',
            valign='middle'
        )

        # Layer number
        fig.add_annotation(
            x=x + node_width/2 - 10,
            y=y + 15,
            text=f"#{i+1}",
            showarrow=False,
            font=dict(size=9, color='#475569', family="monospace"),
            align='right',
            valign='middle'
        )

        # Connector line to next layer
        if i < len(layers) - 1:
            next_y = gap + (i + 1) * (node_height + gap) + node_height/2
            curr_y = y + node_height/2

            fig.add_trace(go.Scatter(
                x=[x, x, x, x],
                y=[curr_y + 15, curr_y + 25, next_y - 25, next_y - 15],
                mode='lines',
                line=dict(color='rgba(56, 189, 248, 0.4)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Arrow head
            fig.add_annotation(
                x=x,
                y=next_y - 20,
                ax=x,
                ay=next_y - 5,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='rgba(56, 189, 248, 0.6)'
            )

    # Layout
    fig.update_layout(
        title=dict(
            text="PyTorch Sequential Model Architecture",
            font=dict(size=18, color='#e2e8f0', family="sans-serif")
        ),
        showlegend=False,
        plot_bgcolor='rgba(6, 9, 16, 1)',
        paper_bgcolor='rgba(6, 9, 16, 1)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, width]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, total_height],
            scaleanchor='x',
            scaleratio=1
        ),
        width=width,
        height=max(height, total_height),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# ============================================================================
# Streamlit UI
# ============================================================================

def show_torch_layers_ui():
    """Streamlit UI for PyTorch Layer Builder"""
    st.markdown("## 🔥 PyTorch Layer Builder")

    # Check for torch availability
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
        st.warning("PyTorch is not installed. The visualization will work, but code generation is limited.")

    col1, col2, col3 = st.columns([1, 2, 1])

    # Initialize session state for layers
    if 'torch_layers' not in st.session_state:
        st.session_state.torch_layers = [
            {"id": 1, "type": "Input", "params": {"shape": "[None, 784]"}},
            {"id": 2, "type": "Linear", "params": {"in_features": 784, "out_features": 128, "bias": True}},
            {"id": 3, "type": "ReLU", "params": {"inplace": False}},
            {"id": 4, "type": "Dropout", "params": {"p": 0.5, "inplace": False}},
            {"id": 5, "type": "Linear", "params": {"in_features": 128, "out_features": 64, "bias": True}},
            {"id": 6, "type": "Output", "params": {"units": 10, "activation": "softmax"}}
        ]
        st.session_state.torch_next_id = 7
        st.session_state.torch_selected = None

    with col1:
        st.markdown("### 📚 Layers")

        # Layer type selector
        layer_categories = {}
        for name, info in PYTORCH_LAYERS.items():
            cat = info.get("category", "Other")
            if cat not in layer_categories:
                layer_categories[cat] = []
            layer_categories[cat].append(name)

        selected_add_type = st.selectbox(
            "Add layer type",
            options=list(layer_categories.keys()),
            key="torch_cat_select"
        )

        layer_type_to_add = st.selectbox(
            "Layer",
            options=layer_categories[selected_add_type],
            key="torch_layer_select"
        )

        if st.button("➕ Add Layer", type="primary", width='stretch'):
            new_layer = {
                "id": st.session_state.torch_next_id,
                "type": layer_type_to_add,
                "params": dict(DEFAULT_PARAMS.get(layer_type_to_add, {}))
            }
            st.session_state.torch_layers.append(new_layer)
            st.session_state.torch_next_id += 1
            st.session_state.torch_selected = new_layer["id"]
            st.rerun()

        st.markdown("---")

        # Layer list
        for i, layer in enumerate(st.session_state.torch_layers):
            layer_info = PYTORCH_LAYERS.get(layer["type"], {})
            color = layer_info.get("color", "#6b7280")
            icon = layer_info.get("icon", "??")
            is_selected = st.session_state.torch_selected == layer["id"]

            # Layer card
            card_style = f"""
            <div style="
                background: {'rgba(56, 189, 248, 0.1)' if is_selected else 'rgba(255,255,255,0.03)'};
                border: 1px solid {'rgba(56, 189, 248, 0.4)' if is_selected else 'rgba(255,255,255,0.08)'};
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 8px;
                cursor: pointer;
                transition: all 0.15s;
            ">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="
                        width: 32px; height: 32px; border-radius: 6px;
                        background: {color}25; border: 1px solid {color}60;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 9px; font-weight: 700; font-family: monospace;
                        color: {color};
                    ">{icon}</div>
                    <div style="flex: 1;">
                        <div style="font-size: 13px; font-weight: 500; color: #e2e8f0;">{layer['type']}</div>
                        <div style="font-size: 10px; color: #64748b; font-family: monospace;">
                            #{i+1}
                        </div>
                    </div>
                </div>
            </div>
            """

            if st.markdown(card_style, unsafe_allow_html=True):
                st.session_state.torch_selected = layer["id"]

            # Clickable row buttons
            cols_row = st.columns([1, 1, 1, 1])
            with cols_row[0]:
                if st.button("⬆", key=f"up_{layer['id']}", disabled=(i == 0)):
                    if i > 0:
                        st.session_state.torch_layers[i], st.session_state.torch_layers[i-1] = \
                            st.session_state.torch_layers[i-1], st.session_state.torch_layers[i]
                        st.rerun()
            with cols_row[1]:
                if st.button("⬇", key=f"down_{layer['id']}", disabled=(i == len(st.session_state.torch_layers) - 1)):
                    if i < len(st.session_state.torch_layers) - 1:
                        st.session_state.torch_layers[i], st.session_state.torch_layers[i+1] = \
                            st.session_state.torch_layers[i+1], st.session_state.torch_layers[i]
                        st.rerun()
            with cols_row[2]:
                if st.button("✏️", key=f"edit_{layer['id']}"):
                    st.session_state.torch_selected = layer["id"]
                    st.rerun()
            with cols_row[3]:
                if st.button("🗑️", key=f"del_{layer['id']}"):
                    st.session_state.torch_layers.pop(i)
                    if st.session_state.torch_selected == layer["id"]:
                        st.session_state.torch_selected = None
                    st.rerun()

    with col2:
        st.markdown("### 🏗️ Architecture")

        # Parameter count
        param_counts = count_parameters(st.session_state.torch_layers)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Total Params", f"{param_counts['total']:,}")
        metric_cols[1].metric("Trainable", f"{param_counts['trainable']:,}")
        metric_cols[2].metric("Layers", len(st.session_state.torch_layers))

        # Architecture diagram
        fig = create_architecture_diagram(st.session_state.torch_layers, width=600, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Code preview
        with st.expander("📄 Generated PyTorch Code", expanded=True):
            code = generate_sequential_code(st.session_state.torch_layers)
            st.code(code, language="python")

    with col3:
        st.markdown("### ⚙️ Properties")

        # Find selected layer
        selected_layer = None
        selected_idx = None
        for i, l in enumerate(st.session_state.torch_layers):
            if l["id"] == st.session_state.torch_selected:
                selected_layer = l
                selected_idx = i
                break

        if selected_layer:
            layer_type = selected_layer["type"]
            layer_info = PYTORCH_LAYERS.get(layer_type, {})
            color = layer_info.get("color", "#6b7280")

            st.markdown(f"""
            <div style="
                background: {color}15;
                border: 1px solid {color}40;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 16px;
            ">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="
                        width: 36px; height: 36px; border-radius: 6px;
                        background: {color}30; border: 1px solid {color}60;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 10px; font-weight: 700; font-family: monospace;
                        color: {color};
                    ">{layer_info.get('icon', '??')}</div>
                    <div>
                        <div style="font-size: 15px; font-weight: 600; color: {color};">{layer_type}</div>
                        <div style="font-size: 11px; color: #475569; font-family: monospace;">id: {selected_layer['id']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show parameters
            params = selected_layer.get("params", {})

            for param_info in layer_info.get("params", []):
                param_key = param_info["k"]
                param_label = param_info["label"]
                param_type = param_info.get("type", "text")
                param_default = param_info.get("default", "")
                param_options = param_info.get("options", [])

                current_val = params.get(param_key, param_default)

                if param_type == "text":
                    new_val = st.text_input(
                        param_label,
                        value=str(current_val),
                        key=f"param_{selected_layer['id']}_{param_key}"
                    )
                    # Try to parse as int/float if possible
                    try:
                        if '.' in str(new_val):
                            new_val = float(new_val)
                        else:
                            new_val = int(new_val)
                    except:
                        pass
                    params[param_key] = new_val

                elif param_type == "number":
                    new_val = st.number_input(
                        param_label,
                        value=int(current_val) if isinstance(current_val, (int, float)) else 0,
                        key=f"param_{selected_layer['id']}_{param_key}"
                    )
                    params[param_key] = int(new_val)

                elif param_type == "float":
                    min_val = param_info.get("min", 0.0)
                    max_val = param_info.get("max", 1.0)
                    new_val = st.slider(
                        param_label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(current_val) if current_val else min_val,
                        step=0.01,
                        key=f"param_{selected_layer['id']}_{param_key}"
                    )
                    params[param_key] = float(new_val)

                elif param_type == "bool":
                    new_val = st.checkbox(
                        param_label,
                        value=bool(current_val),
                        key=f"param_{selected_layer['id']}_{param_key}"
                    )
                    params[param_key] = new_val

                elif param_type == "select":
                    new_val = st.selectbox(
                        param_label,
                        options=param_options,
                        index=param_options.index(str(current_val)) if str(current_val) in param_options else 0,
                        key=f"param_{selected_layer['id']}_{param_key}"
                    )
                    params[param_key] = new_val

                elif param_type == "textarea":
                    new_val = st.text_area(
                        param_label,
                        value=str(current_val),
                        height=150,
                        key=f"param_{selected_layer['id']}_{param_key}"
                    )
                    params[param_key] = new_val

            # Update layer
            st.session_state.torch_layers[selected_idx]["params"] = params

            # Delete button
            st.markdown("---")
            if st.button("🗑️ Delete Layer", type="secondary", width='stretch'):
                st.session_state.torch_layers.pop(selected_idx)
                st.session_state.torch_selected = None
                st.rerun()

        else:
            st.info("👈 Select a layer to edit its properties")

    # Model summary
    st.markdown("---")
    st.markdown("### 📊 Model Summary")

    # Create summary table
    summary_data = []
    for i, layer in enumerate(st.session_state.torch_layers):
        layer_type = layer["type"]
        params = layer.get("params", {})

        # Estimate output shape (simplified)
        if layer_type == "Input":
            shape = params.get("shape", "?")
        elif layer_type == "Linear":
            shape = f"[N, {params.get('out_features', '?')}]"
        elif layer_type == "Conv2d":
            shape = f"[N, {params.get('out_channels', '?')}, ?, ?]"
        elif layer_type == "MaxPool2d":
            shape = "[N, C, H/2, W/2]"
        elif layer_type == "Flatten":
            shape = "[N, Flat]"
        elif layer_type == "LSTM":
            bi = "×2" if params.get("bidirectional", False) else ""
            shape = f"[N, Seq, {params.get('hidden_size', '?')}{bi}]"
        elif layer_type == "Output":
            shape = f"[N, {params.get('units', '?')}]"
        elif layer_type == "Custom":
            shape = "?"
        else:
            shape = "?"

        # Estimate params
        if layer_type in ["Linear"]:
            in_f = params.get("in_features", 0)
            out_f = params.get("out_features", 0)
            p_count = in_f * out_f + (out_f if params.get("bias", True) else 0)
        elif layer_type == "Conv2d":
            in_c = params.get("in_channels", 1)
            out_c = params.get("out_channels", 1)
            k = params.get("kernel_size", 3)
            if isinstance(k, str):
                k = int(k.split('x')[0]) if 'x' in k else int(k)
            p_count = in_c * out_c * k * k + (out_c if params.get("bias", False) else 0)
        elif layer_type == "Embedding":
            p_count = params.get("num_embeddings", 0) * params.get("embedding_dim", 0)
        elif layer_type in ["Dropout", "ReLU", "Flatten", "Softmax"]:
            p_count = 0
        elif layer_type in ["BatchNorm1d", "BatchNorm2d"]:
            p_count = 2 * params.get("num_features", 0)
        else:
            p_count = "?"

        summary_data.append({
            "Layer": f"{i+1}. {layer_type}",
            "Output Shape": shape,
            "Params": f"{p_count:,}" if isinstance(p_count, int) else p_count
        })

    st.dataframe(pd.DataFrame(summary_data), width='stretch')

    # Export options
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        code = generate_sequential_code(st.session_state.torch_layers)
        st.download_button(
            "📥 Download Model Code",
            data=code,
            file_name="model.py",
            mime="text/x-python",
            width='stretch'
        )

    with col_exp2:
        # Export as JSON
        import json
        model_json = json.dumps({
            "layers": st.session_state.torch_layers,
            "metadata": {
                "total_params": param_counts['total'],
                "trainable_params": param_counts['trainable'],
                "num_layers": len(st.session_state.torch_layers)
            }
        }, indent=2)
        st.download_button(
            "📥 Export as JSON",
            data=model_json,
            file_name="model_config.json",
            mime="application/json",
            width='stretch'
        )

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 PyTorch Layer Reference")

    with st.expander("📖 Layer Categories"):
        st.markdown("""
        ### Linear Layers
        - **Linear**: Fully connected layer `y = xW^T + b`
        - Input: `(batch, in_features)`, Output: `(batch, out_features)`

        ### Convolution Layers
        - **Conv1d/2d/3d**: Convolutional layers for 1D/2D/3D signals
        - Parameters: in_channels, out_channels, kernel_size, stride, padding
        - Use Conv1d for text/audio, Conv2d for images, Conv3d for video/volumes

        ### Pooling Layers
        - **MaxPool**: Max pooling - preserves sharp features
        - **AvgPool**: Average pooling - smooths features
        - **Adaptive**: Adapts to any input size

        ### Normalization Layers
        - **BatchNorm**: Normalizes across batch dimension
        - **LayerNorm**: Normalizes across feature dimension
        - **GroupNorm**: Normalizes across channel groups (good for small batches)

        ### Recurrent Layers
        - **LSTM/GRU**: Long Short-Term Memory / Gated Recurrent Units
        - Use `batch_first=True` for `(batch, seq, features)` input

        ### Transformer Layers
        - **MultiheadAttention**: Self-attention mechanism
        - **TransformerEncoder/Decoder**: Pre-defined encoder/decoder stacks

        ### Dropout
        - **Dropout**: Randomly zeros elements during training
        - **Dropout2d**: Zeros entire channels
        - Set `inplace=True` to save memory
        """)

    with st.expander("💡 Best Practices"):
        st.markdown("""
        1. **Input Shape**: Always start with an Input layer to define your data shape
        2. **Activation Functions**: Add ReLU/GELU after Linear/Conv layers (not after Softmax)
        3. **BatchNorm**: Place after Conv/Linear and before activation
        4. **Dropout**: Use between layers during training, set rate 0.2-0.5
        5. **Pool early**: Reduce spatial dimensions early in CNNs
        6. **Flatten before Linear**: After Conv layers, flatten before Dense layers
        7. **Match dimensions**: Ensure layer output matches next layer input
        8. **For Classification**: End with Linear → Softmax (or use CrossEntropyLoss directly)
        """)
