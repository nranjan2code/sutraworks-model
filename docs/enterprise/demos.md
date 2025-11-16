# Enterprise Demos - Live Interactive Demonstrations

**See SutraWorks AI in Action** - Live, interactive demonstrations of real-world enterprise applications.

## 🚀 Quick Start

```bash
# Run the live trading terminal (recommended)
cargo run --example trading_terminal_demo --release

# Live-updating professional terminal, press Ctrl+C to exit
```

## 📊 Available Demonstrations

### 1. 💼 Live Trading Terminal (⭐ START HERE)

**File**: `examples/trading_terminal_demo.rs`

**What it demonstrates**:
- ✅ Professional Bloomberg-style trading interface
- ✅ Real-time market data with 2-second auto-refresh
- ✅ Complete backtest engine with P&L tracking
- ✅ Real performance metrics: Sharpe ratio, Sortino, Max Drawdown, VaR
- ✅ ASCII price charts and equity curves
- ✅ ANSI color-coded display (green profits, red losses)
- ✅ Live BUY/SELL/HOLD signals from Mamba AI model

**Run it**:
```bash
cargo run --example trading_terminal_demo --release
# Press Ctrl+C to exit the live terminal
```

**Live Terminal Features**:
```
📊 Live Market Data: Price updates every 2 seconds with ASCII charts
💰 Real P&L Tracking: Actual backtest with trade-by-trade breakdown
📈 Performance Metrics: Sharpe -2.27, Max DD 7.46%, Win Rate 66.7%
🤖 AI Signals: BUY/SELL/HOLD from real Mamba model inference (<1ms)
🎨 Professional UI: ANSI colors, Bloomberg-style layout
```

**What's Real vs Simulated**:
- ✅ **REAL**: Model inference (Mamba SSM with selective scan)
- ✅ **REAL**: Quantization (AWQ 4-bit compression 7.42x)
- ✅ **REAL**: Backtest engine (P&L, Sharpe, metrics calculated)
- ✅ **REAL**: Live screen updates (continuous refresh loop)
- ✅ **REAL**: Performance metrics (measured latency <1ms)
- ⚠️  **SIMULATED**: Market data (generated with realistic price action)

### 2. 📊 Quantization Benchmark

**What it demonstrates**:
- ✅ AWQ 4-bit quantization with real bit-packing
- ✅ 7.42x compression ratio measurement
- ✅ Performance comparison (quantized vs unquantized)
- ✅ Memory usage optimization

**Run it**:
```bash
cargo run --example quantization_benchmark --release
```

### 3. 🔄 End-to-End Pipeline

**What it demonstrates**:
- ✅ Complete workflow: tokenize → embed → infer → quantize → decode
- ✅ Real model loading with SafeTensors
- ✅ Production pipeline integration
- ✅ Error handling and type safety

**Run it**:
```bash
cargo run --example end_to_end --release
```

## 🎯 Enterprise Value Proposition

### Proven Cost Savings
- **Infrastructure**: 7.42x model compression reduces storage and bandwidth costs
- **Hardware**: Optimized for 16GB MacBook Air (vs requiring expensive GPU servers)
- **Latency**: <1ms inference enables real-time applications
- **Development**: Production-ready framework accelerates time-to-market

### Real-World Applications
- **Financial Services**: Algorithmic trading, risk management, fraud detection
- **Healthcare**: Clinical decision support, diagnostic assistance
- **Manufacturing**: Predictive maintenance, quality control
- **Retail**: Demand forecasting, personalized recommendations

### Technical Advantages
- **Pure Rust**: Memory safety and performance without garbage collection
- **No GPU Required**: Runs efficiently on CPU-only systems
- **Production Ready**: 57/57 tests passing, enterprise code quality
- **Modular Design**: Use only the components you need

## 📈 Performance Metrics

### Measured Results
- **Model Compression**: 402MB → 54MB (7.42x reduction)
- **Inference Speed**: <1ms for trading decisions
- **Memory Usage**: 6-12MB for demo configurations
- **Test Coverage**: 100% success rate (57/57 tests)

### Scalability
- **Model Size**: Supports up to 3B parameters with quantization
- **Throughput**: 73,634 tokens/second demonstrated
- **Concurrency**: Thread-safe operations for parallel processing

## 🚀 Getting Started

1. **Clone and Build**:
   ```bash
   git clone https://github.com/nranjan2code/sutraworks-model
   cd sutraworks-model
   cargo build --release
   ```

2. **Run Live Demo**:
   ```bash
   cargo run --example trading_terminal_demo --release
   ```

3. **Explore All Demos**:
   ```bash
   # See all available examples
   ls examples/
   
   # Run any example
   cargo run --example <example_name> --release
   ```

## 📞 Next Steps

- **[Production Deployment](deployment.md)** - Deploy to your infrastructure
- **[API Reference](../api/core.md)** - Integrate into your applications
- **[Architecture Overview](../architecture/overview.md)** - Understand the design
- **[Tutorials](../tutorials/quantization.md)** - Learn advanced techniques

---

**Ready to see SutraWorks in action? Start with the trading terminal demo above!**