# Custom Quantization Engine (PTQ & QAT)

A modular PyTorch framework built from scratch to perform Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). This project simulates cycle-accurate fixed-point DSP and NPU hardware constraints for edge AI inference, demonstrating the low-level mathematics required to compress models for memory-constrained environments.

## The Engineering Challenge

Deploying neural networks to edge devices (DSPs, NPUs, microcontrollers) requires converting 32-bit floating-point parameters into 8-bit or 4-bit integers. This repository solves two core mathematical bottlenecks in that pipeline:

1.  **The Zero-Gradient Problem:** The mathematical derivative of a `round()` function is zero, which immediately halts backpropagation. This repository implements a custom **Straight-Through Estimator (STE)** at the autograd level to pass gradients unaltered, enabling the network to learn and compensate for quantization noise.
2.  **Asymmetric vs. Symmetric Scaling:** Weights naturally center around zero (symmetric), but activations (especially post-ReLU) do not. This engine tracks running statistics and dynamically shifts the `zero_point` to fully utilize the 8-bit integer grid (-128 to 127) depending on the tensor's distribution.

## Architecture

*   `quantizers/base.py`: The `FakeQuantizeSTE` autograd engine that handles the fixed-point simulation and gradient routing.
*   `quantizers/min_max.py`: The memory bank module that tracks running statistics and computes `scale` and `zero_point` calibration boundaries.
*   `quantizers/layers.py`: The `QuantizedLinear` layer wrapper that applies quantization independently to both weights and incoming activations before the Multiply-Accumulate (MAC) operation.
*   `scripts/train.py`: The end-to-end execution pipeline demonstrating FP32 pre-training, passive PTQ calibration, and active QAT fine-tuning.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/kunalsankhe/PTQ_QAT_Pipeline.git
cd PTQ_QAT_Pipeline
