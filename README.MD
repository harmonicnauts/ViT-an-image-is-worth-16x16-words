# Vision Transformer (ViT) – A Theoretical Exploration

This repository presents an educational implementation of the **Vision Transformer (ViT)** architecture using PyTorch, inspired by the foundational paper _"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"_ by Dosovitskiy et al. The goal is to gain a conceptual understanding of how transformers—originally developed for natural language processing—can be adapted for vision tasks by treating image patches as token sequences.

## Background

Traditional Convolutional Neural Networks (CNNs) have dominated computer vision tasks for years. However, CNNs have inherent inductive biases (e.g., locality, translation equivariance), which may limit their ability to model long-range dependencies. Transformers, originally built for language modeling, offer a powerful alternative due to their **self-attention mechanism**, which enables **global context modeling** across inputs.

The Vision Transformer (ViT) leverages this by **flattening images into sequences of patches**, allowing standard transformer encoders to process them similarly to word tokens in NLP.

## Core Components and Theory

### 1. Patch Embedding

Instead of convolutional filters, ViT divides the input image
$( x \in \mathbb{R}^{H \times W \times C})$
into a sequence of flattened image patches of size $( P \times P )$. Each patch is linearly projected into a latent embedding space:

$[
\text{patches} \rightarrow \text{flatten} \rightarrow \text{Linear Projection} \in \mathbb{R}^{N \times D}
]$

Where:

- $( N = \frac{HW}{P^2})$: Number of patches
- $( D )$: Embedding dimension

### 2. Positional Encoding

Unlike sequences in language, image patches lack an inherent order. Therefore, **positional encodings** $( \mathbf{E_{Pos}} \in \mathbb{R}^{(N + 1) \times D} )$ are added to retain spatial structure:

$[
z_0^0 = \text{PatchEmbeddings} + \text{PositionalEncoding}
]$

These encodings can be fixed (e.g., sinusoidal) or learnable.

### 3. Transformer Encoder

The transformer encoder consists of repeated blocks of the following components:

- **Layer Normalization**: Stabilizes training by normalizing inputs across features.
- **Multi-Head Self-Attention (MSA)**:
  $[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  ]$
  Multiple attention heads capture information from different representation subspaces.

- **Feedforward Neural Network (MLP)**:
  A two-layer MLP with GELU activations:
  $[
  \text{MLP}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
  ]$

- **Residual Connections**: Ensure stable gradient flow and preserve low-level features.

Each transformer encoder block updates the hidden state as:
$[
z_\ell^{'} = \text{MSA}(\text{LN}(z_{\ell - 1})) + z_{\ell - 1}
]$
$[
z_\ell = \text{MLP}(\text{LN}(z_\ell^{'})) + z_\ell^{'}
]$

### 4. Classification Token

A learnable **[CLS] token** is prepended to the sequence and intended to aggregate global information. After the transformer blocks, its final hidden state is used for classification via a linear head.

## Theoretical Advantages

- **Global Receptive Field**: Unlike CNNs, every patch attends to every other patch from the first layer.
- **Scalability**: Transformers scale well with large datasets and model sizes.
- **Inductive Bias Minimization**: With fewer hard-coded assumptions, transformers learn inductive priors from data.
