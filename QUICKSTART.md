# Quick Start Guide

## Installation

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Clone and Build**:
   ```bash
   git clone https://github.com/sutraworks/model
   cd sutraworks-model
   cargo build --release
   ```

3. **Run Tests**:
   ```bash
   cargo test --all
   # ✅ 51 tests should pass
   ```

## Running Examples

### 1. End-to-End Pipeline (⭐ Complete Workflow)
Experience the full AI stack from tokenization to inference:
```bash
cargo run --example end_to_end --release
```

**What it does**:
- ✅ BPE tokenization (GPT-2 style)
- ✅ Embedding lookup and tensor ops
- ✅ 4-bit quantization (6x compression!)
- ✅ RWKV model inference
- ✅ Token sampling and decoding

**Output**: Complete pipeline in <0.1 seconds, <100MB memory!

### 2. Model Loader
Load models from HuggingFace and safetensors:
```bash
cargo run --example model_loader --release
```

**Output**: Demonstrates model registry, safetensors loading, HuggingFace downloads.

### 3. Quantization Demo
See 4-bit model compression in action:
```bash
cargo run --example quantization_demo --release
```

**Output**: Demonstrates ~6x memory reduction with AWQ quantization.

### 4. QLoRA Fine-Tuning
Learn about parameter-efficient fine-tuning:
```bash
cargo run --example qlora_training --release
```

**Output**: Shows how to fine-tune 3B models with <8GB RAM using training infrastructure.

### 5. RWKV Inference
Explore efficient RNN-based inference:
```bash
cargo run --example rwkv_inference --release
```

**Output**: Demonstrates constant memory, linear O(n) complexity inference.

### 6. Mamba Inference
Experience 5x faster throughput:
```bash
cargo run --example mamba_inference --release
```

**Output**: Shows linear-time state space model advantages.

### 7. Neuro-Symbolic Agent
Build hybrid AI with verified reasoning:
```bash
cargo run --example nesy_agent --release
```

**Output**: Demonstrates combining neural nets with symbolic tools for guaranteed correctness.

## Using in Your Project

Add to your `Cargo.toml`:

```toml
[dependencies]
# Core functionality
sutra-core = { path = "path/to/sutraworks-model/crates/sutra-core" }

# Model loading and tokenization
sutra-loader = { path = "path/to/sutraworks-model/crates/sutra-loader" }
sutra-tokenizer = { path = "path/to/sutraworks-model/crates/sutra-tokenizer" }

# Training and optimization
sutra-training = { path = "path/to/sutraworks-model/crates/sutra-training" }
sutra-peft = { path = "path/to/sutraworks-model/crates/sutra-peft" }

# Model architectures
sutra-rwkv = { path = "path/to/sutraworks-model/crates/sutra-rwkv" }
sutra-mamba = { path = "path/to/sutraworks-model/crates/sutra-mamba" }

# Advanced features
sutra-quantize = { path = "path/to/sutraworks-model/crates/sutra-quantize" }
sutra-nesy = { path = "path/to/sutraworks-model/crates/sutra-nesy" }
```

## Next Steps

1. **Load Models**: Use `sutra-loader` to download from HuggingFace Hub
2. **Tokenize Data**: Choose BPE, WordPiece, or Unigram from `sutra-tokenizer`
3. **Train/Fine-tune**: Use `sutra-training` for optimizers and `sutra-peft` for QLoRA
4. **Quantize**: Compress with `sutra-quantize` AWQ 4-bit for deployment
5. **Deploy**: Run locally on your MacBook Air with efficient RWKV/Mamba models

## Resources

- **Documentation**: See README.md for detailed architecture
- **API Docs**: Run `cargo doc --open` for full API reference
- **Examples**: Check `examples/` directory for more use cases

## Performance Tips

### For Maximum Speed
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### For Minimum Binary Size
```bash
cargo build --release --config profile.release.strip=true
```

### For Development
```bash
cargo build  # Fast compilation, debug symbols
```

## Troubleshooting

### Out of Memory?
- Reduce model size
- Use 4-bit quantization
- Decrease batch size

### Slow Compilation?
- Use `cargo check` instead of `cargo build` for quick validation
- Enable incremental compilation (default in dev mode)

### Need Help?
- Check examples for working code
- Run tests: `cargo test --all`
- Open an issue on GitHub

## What's Next?

Explore the complete AI development workflow:

1. **Load Models** → Download from HuggingFace with `sutra-loader`
2. **Tokenize** → Prepare data with BPE/WordPiece/Unigram
3. **Train** → Use modern optimizers (Adam, SGD) and schedulers
4. **Fine-tune** → QLoRA for parameter-efficient adaptation
5. **Quantize** → Compress to 4-bit for efficient deployment
6. **Deploy** → Run RWKV/Mamba architectures on CPU
7. **Verify** → Add NeSy tools for guaranteed correctness

Start with the examples and experiment!
