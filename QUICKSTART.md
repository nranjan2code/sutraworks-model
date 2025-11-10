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
   ```

## Running Examples

### 1. Quantization Demo
See 4-bit model compression in action:
```bash
cargo run --example quantization_demo --release
```

**Output**: Demonstrates ~4x memory reduction with AWQ quantization.

### 2. QLoRA Fine-Tuning
Learn about parameter-efficient fine-tuning:
```bash
cargo run --example qlora_training --release
```

**Output**: Shows how to fine-tune 3B models with <8GB RAM.

### 3. RWKV Inference
Explore efficient RNN-based inference:
```bash
cargo run --example rwkv_inference --release
```

**Output**: Demonstrates constant memory, linear complexity inference.

### 4. Mamba Inference
Experience 5x faster throughput:
```bash
cargo run --example mamba_inference --release
```

**Output**: Shows linear-time state space model advantages.

### 5. Neuro-Symbolic Agent
Build hybrid AI with verified reasoning:
```bash
cargo run --example nesy_agent --release
```

**Output**: Demonstrates combining neural nets with symbolic tools.

## Using in Your Project

Add to your `Cargo.toml`:

```toml
[dependencies]
sutra-core = { path = "path/to/sutraworks-model/crates/sutra-core" }
sutra-quantize = { path = "path/to/sutraworks-model/crates/sutra-quantize" }
sutra-peft = { path = "path/to/sutraworks-model/crates/sutra-peft" }
sutra-rwkv = { path = "path/to/sutraworks-model/crates/sutra-rwkv" }
sutra-mamba = { path = "path/to/sutraworks-model/crates/sutra-mamba" }
sutra-nesy = { path = "path/to/sutraworks-model/crates/sutra-nesy" }
```

## Next Steps

1. **Download Models**: Get pre-trained RWKV or Mamba models from Hugging Face
2. **Quantize**: Use `sutra-quantize` to compress to 4-bit
3. **Fine-tune**: Specialize models with `sutra-peft` on your data
4. **Deploy**: Run locally on your MacBook Air

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

Explore the four core capabilities:

1. **Quantization** → Compress models to fit in 16GB
2. **QLoRA** → Fine-tune on your personal data
3. **RWKV/Mamba** → Run efficient architectures on CPU
4. **NeSy** → Build verified reasoning systems

Start with the examples and experiment!
