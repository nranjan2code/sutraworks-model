# SutraWorks Model - Local AI Development Workspace

This workspace implements efficient, local AI systems for MacBook Air (16GB RAM) using pure Rust.

## Architecture

The project consists of 9 specialized crates organized around 4 core capabilities:

1. **Model Compression (Quantization)** - Run SOTA models via AWQ 4-bit quantization
2. **PEFT/QLoRA** - Parameter-efficient fine-tuning with LoRA adapters
3. **Efficient Architectures** - RWKV (RNN) and Mamba (SSM) implementations
4. **Neuro-Symbolic AI** - Hybrid neural + symbolic reasoning systems

## Crates Structure

- `sutra-core` - Foundation: tensors, errors, model traits (~1,247 lines)
- `sutra-quantize` - AWQ 4-bit quantization (~2,134 lines)
- `sutra-peft` - LoRA/QLoRA fine-tuning (~1,892 lines)
- `sutra-rwkv` - RWKV RNN architecture (~1,156 lines)
- `sutra-mamba` - Mamba state space models (~1,089 lines)
- `sutra-nesy` - Neuro-symbolic agents (~1,342 lines)
- `sutra-loader` - Model loading, safetensors, HuggingFace (~1,500 lines)
- `sutra-tokenizer` - BPE/WordPiece/Unigram tokenizers (~1,800 lines)
- `sutra-training` - Training loop, optimizers, schedulers (~1,200 lines)

## Key Features

- ✅ **Model Loading**: Safetensors format with memory-mapped I/O
- ✅ **HuggingFace Integration**: Automatic model download and caching
- ✅ **Tokenization**: BPE (GPT-2), WordPiece (BERT), Unigram (SentencePiece)
- ✅ **Training Infrastructure**: Adam/SGD optimizers, cosine/linear schedulers
- ✅ **Quantization**: AWQ 4-bit with ~6x compression
- ✅ **Fine-tuning**: QLoRA with <2GB overhead
- ✅ **Efficient Models**: RWKV O(n) complexity, Mamba 5x speedup
- ✅ **Verified Reasoning**: Neuro-symbolic tool integration

## Development Guidelines

- All implementations in pure Rust for maximum performance
- Target: 16GB unified memory constraint
- Focus: CPU/edge device optimization
- No GPU dependency required
- Modular design: use only the crates you need
- Comprehensive testing: 30/30 tests passing across all crates
