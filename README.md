GPT Model Implementation

This repository contains a basic implementation of a Generative Pre-trained Transformer (GPT) model using PyTorch. The model is designed to demonstrate the core components of a transformer architecture, including multi-headed self-attention and feed-forward neural networks.

Features
Transformer Block: Implements a standard transformer block with multi-headed self-attention and a position-wise feed-forward network.

Multi-Headed Self-Attention: Includes a SingleHeadAttention module and combines them into MultiHeadedSelfAttention.

Positional Encoding: Incorporates positional embeddings to account for the order of tokens in the input sequence.

Skip Connections and Layer Normalization: Utilizes skip connections and layer normalization within the transformer blocks for stable training.

Masked Attention: Applies a lower triangular mask in the self-attention mechanism, crucial for autoregressive language modeling.

Project Structure
gpt.py: Contains the main GPT model definition and its sub-modules.

Requirements
Python 3.x

PyTorch

You can install PyTorch by following the instructions on the official PyTorch website: https://pytorch.org/get-started/locally/


Model Architecture
The GPT model is composed of the following key components:

Token Embedding Layer: Converts input token IDs into dense vector representations.

Positional Embedding Layer: Adds positional information to the token embeddings, allowing the model to understand the order of words.

Transformer Blocks: A sequence of identical TransformerBlock modules. Each block consists of:

Multi-Headed Self-Attention (MHSA): Allows the model to weigh the importance of different parts of the input sequence when processing each token. It includes:

SingleHeadAttention: Computes attention scores and applies a mask for autoregressive generation.

Vanilla Neural Network (Feed-Forward Network): A simple two-layer feed-forward network applied to each position independently.

Layer Normalization and Skip Connections: Applied around both the MHSA and feed-forward network for improved training stability.

Final Layer Normalization: Applied after the stack of transformer blocks.

Vocabulary Projection Layer: A linear layer that projects the output of the transformer blocks back to the vocabulary size, producing logits for the next token prediction.

