# SutraWorks Model - Local AI Development Workspace

This workspace implements efficient, local AI systems for MacBook Air (16GB RAM) using pure Rust.

## Architecture

1. **Model Compression (Quantization)** - Run SOTA models via AWQ 4-bit quantization
2. **PEFT/QLoRA** - Parameter-efficient fine-tuning with LoRA adapters
3. **Efficient Architectures** - RWKV (RNN) and Mamba (SSM) implementations
4. **Neuro-Symbolic AI** - Hybrid neural + symbolic reasoning systems

## Development Guidelines

- All implementations in pure Rust for maximum performance
- Target: 16GB unified memory constraint
- Focus: CPU/edge device optimization
- No GPU dependency required
