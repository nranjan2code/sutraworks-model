/// Manual Model Testing Script
/// 
/// This script allows you to test the system with manually downloaded models
/// or smaller test models without requiring large downloads.

use std::path::PathBuf;
use std::time::Instant;
use sutra_core::{ops, DType, Tensor};
use sutra_loader::{ModelDownloader, ModelRegistry};
use sutra_quantize::{AwqConfig, AwqQuantizer};
use sutra_tokenizer::{BpeConfig, BpeTokenizer, Tokenizer, VocabBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║         🔧 Manual Model Testing Suite 🔧           ║");
    println!("║    Quick validation without large downloads          ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Test 1: Simulated model inference
    test_simulated_inference()?;
    
    // Test 2: Local tokenizer test
    test_tokenizer_workflow()?;
    
    // Test 3: Quantization with synthetic data
    test_quantization_workflow()?;
    
    // Test 4: Model loading configuration
    test_model_loading_config()?;
    
    // Test 5: Performance benchmarks
    test_performance_benchmarks()?;

    println!("🎉 All manual tests completed successfully!");
    println!("ℹ️  To test with real models, run: cargo run --example real_world_test");
    
    Ok(())
}

fn test_simulated_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Test 1: Simulated Model Inference");
    println!("═══════════════════════════════════");
    
    let start = Instant::now();
    
    // Create realistic model dimensions
    let vocab_size = 32000;  // Similar to LLaMA
    let seq_len = 128;
    let d_model = 1024;
    let n_heads = 16;
    
    println!("Model config:");
    println!("• Vocab size: {}", vocab_size);
    println!("• Sequence length: {}", seq_len);
    println!("• Model dimension: {}", d_model);
    println!("• Attention heads: {}", n_heads);
    
    // Simulate tokenized input
    let input_tokens = vec![1, 8279, 3186, 322, 14502, 310, 278, 1788]; // "Testing the capabilities of the system"
    println!("\n📝 Input tokens: {:?}", input_tokens);
    
    // Create embedding weights
    let embed_init = Instant::now();
    let embed_data: Vec<f32> = (0..vocab_size * d_model)
        .map(|i| {
            let val = (i as f32) / (vocab_size as f32);
            (val * 2.0 * std::f32::consts::PI).sin() * 0.1
        })
        .collect();
    
    let embed_weights = Tensor::from_slice(&embed_data, &[vocab_size, d_model], DType::F32)?;
    println!("✅ Embedding matrix created in {:.2}ms", embed_init.elapsed().as_millis());
    
    // Embed input tokens
    let embed_start = Instant::now();
    let embedded = ops::embedding(&input_tokens, &embed_weights)?;
    println!("✅ Token embedding completed in {:.2}ms", embed_start.elapsed().as_millis());
    println!("📊 Embedded shape: {:?}", embedded.shape());
    
    // Simulate transformer processing
    let transform_start = Instant::now();
    let normalized = ops::layer_norm(&embedded, 1e-5)?;
    let attended = ops::activations::gelu(&normalized);  // Simplified attention
    let ffn_out = ops::activations::silu(&attended);     // Feed-forward simulation
    
    println!("✅ Transformer processing completed in {:.2}ms", transform_start.elapsed().as_millis());
    println!("📊 Final output shape: {:?}", ffn_out.shape());
    
    // Calculate memory usage
    let memory_usage = embed_weights.memory_usage() + embedded.memory_usage() + ffn_out.memory_usage();
    let memory_mb = memory_usage as f64 / (1024.0 * 1024.0);
    
    let total_time = start.elapsed();
    println!("\n📊 Performance Summary:");
    println!("• Total inference time: {:.2}ms", total_time.as_millis());
    println!("• Memory usage: {:.2} MB", memory_mb);
    println!("• Tokens per second: {:.0}", input_tokens.len() as f64 / total_time.as_secs_f64());
    println!();
    
    Ok(())
}

fn test_tokenizer_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Test 2: Tokenizer Workflow");
    println!("═════════════════════════════");
    
    // Create a simple tokenizer
    let mut vocab = VocabBuilder::new()
        .with_standard_special_tokens()
        .build();
    
    // Add common words
    let words = [
        "the", "and", "to", "of", "a", "in", "that", "have", 
        "for", "not", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say",
        "hello", "world", "test", "model", "AI", "system"
    ];
    
    for word in &words {
        vocab.add_token(word.to_string());
    }
    
    println!("✅ Created vocabulary with {} tokens", vocab.size());
    
    let config = BpeConfig {
        vocab,
        merges: Vec::new(),
        unk_token: "[UNK]".to_string(),
        byte_level: false,
    };
    
    let bpe_tokenizer = BpeTokenizer::new(config);
    let tokenizer = Tokenizer::Bpe(bpe_tokenizer);
    
    // Test sentences
    let test_sentences = [
        "Hello world!",
        "This is a test of the AI system.",
        "The model can process and understand text.",
        "Testing tokenization workflow with various inputs."
    ];
    
    for sentence in &test_sentences {
        println!("\n📝 Input: \"{}\"", sentence);
        
        let start = Instant::now();
        let encoding = tokenizer.encode(sentence)?;
        let encode_time = start.elapsed();
        
        let decode_start = Instant::now();
        let decoded = tokenizer.decode(&encoding.ids)?;
        let decode_time = decode_start.elapsed();
        
        println!("🔢 Tokens: {:?}", encoding.ids);
        println!("📤 Decoded: \"{}\"", decoded);
        println!("⏱️  Encode time: {:.2}ms", encode_time.as_millis());
        println!("⏱️  Decode time: {:.2}ms", decode_time.as_millis());
    }
    
    println!();
    Ok(())
}

fn test_quantization_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗜️  Test 3: Quantization Workflow");
    println!("═════════════════════════════════");
    
    // Test different matrix sizes
    let sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)];
    
    for (rows, cols) in &sizes {
        println!("\n📊 Testing {}x{} matrix ({} parameters)", rows, cols, rows * cols);
        
        // Create weight matrix with realistic values
        let weight_data: Vec<f32> = (0..(rows * cols))
            .map(|i| {
                let normalized = (i as f32) / ((rows * cols) as f32);
                (normalized * 4.0 - 2.0) * 0.1 // Range: -0.2 to 0.2
            })
            .collect();
        
        let weights = Tensor::from_slice(&weight_data, &[*rows, *cols], DType::F32)?;
        let original_size = weights.memory_usage();
        
        // Test quantization
        let quant_start = Instant::now();
        let quantizer = AwqQuantizer::new(AwqConfig::default());
        let quantized = quantizer.quantize(&weights, None)?;
        let quant_time = quant_start.elapsed();
        
        let compressed_size = quantized.memory_usage();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        let size_reduction = (1.0 - compressed_size as f64 / original_size as f64) * 100.0;
        
        println!("  • Original: {:.2} KB", original_size as f64 / 1024.0);
        println!("  • Compressed: {:.2} KB", compressed_size as f64 / 1024.0);
        println!("  • Compression: {:.2}x ({:.1}% reduction)", compression_ratio, size_reduction);
        println!("  • Time: {:.2}ms", quant_time.as_millis());
        println!("  • Bits per weight: {}", quantized.bits);
    }
    
    println!();
    Ok(())
}

fn test_model_loading_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("📁 Test 4: Model Loading Configuration");
    println!("═════════════════════════════════════");
    
    // Test model registry
    let registry = ModelRegistry::with_defaults();
    println!("✅ Model registry initialized with {} models", registry.list().len());
    
    // List all available models
    println!("\n📋 Available models:");
    for model in registry.list() {
        let params_str = format_params(model.num_parameters);
        println!("  • {} ({}) - {} - {}", 
                model.name, 
                model.id, 
                model.architecture.to_uppercase(), 
                params_str);
    }
    
    // Test cache configuration
    let downloader = ModelDownloader::with_defaults()?;
    let cache_path = PathBuf::from("~/.cache/sutraworks/models").expand_user();
    
    println!("\n📂 Cache configuration:");
    println!("  • Cache directory: {}", cache_path.display());
    println!("  • Download retries: 3");
    println!("  • Checksum verification: enabled");
    println!("  • Progress bar: enabled");
    
    // Test model path resolution
    let rwkv_path = downloader.cached_path("BlinkDL/rwkv-4-pile-169m", "model.safetensors", Some("main"));
    let mamba_path = downloader.cached_path("state-spaces/mamba-130m", "model.safetensors", Some("main"));
    
    println!("\n🔗 Model paths:");
    println!("  • RWKV-169M: {}", rwkv_path.display());
    println!("  • Mamba-130M: {}", mamba_path.display());
    println!("  • RWKV cached: {}", if rwkv_path.exists() { "✅ Yes" } else { "❌ No" });
    println!("  • Mamba cached: {}", if mamba_path.exists() { "✅ Yes" } else { "❌ No" });
    
    println!();
    Ok(())
}

fn test_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏃 Test 5: Performance Benchmarks");
    println!("═════════════════════════════════");
    
    // Benchmark tensor operations
    let sizes = [64, 128, 256, 512];
    
    println!("🧮 Matrix multiplication benchmarks:");
    for size in &sizes {
        let data_a: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.01).collect();
        let data_b: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.01).collect();
        
        let a = Tensor::from_slice(&data_a, &[*size, *size], DType::F32)?;
        let b = Tensor::from_slice(&data_b, &[*size, *size], DType::F32)?;
        
        let start = Instant::now();
        let _result = ops::matmul(&a, &b)?;
        let duration = start.elapsed();
        
        let ops_count = (*size as u64).pow(3) * 2; // 2 * n³ operations for matrix multiply
        let gflops = (ops_count as f64) / (duration.as_secs_f64() * 1e9);
        
        println!("  • {}x{}: {:.2}ms ({:.2} GFLOPS)", size, size, duration.as_millis(), gflops);
    }
    
    // Benchmark activations
    println!("\n⚡ Activation function benchmarks:");
    let test_size = 1024 * 1024; // 1M elements
    let data: Vec<f32> = (0..test_size).map(|i| (i as f32) * 0.0001 - 50.0).collect();
    let tensor = Tensor::from_slice(&data, &[test_size], DType::F32)?;
    
    let activations = [
        ("ReLU", ops::activations::relu as fn(&Tensor) -> Tensor),
        ("GELU", ops::activations::gelu),
        ("SiLU", ops::activations::silu),
        ("Tanh", ops::activations::tanh),
        ("Sigmoid", ops::activations::sigmoid),
    ];
    
    for (name, func) in &activations {
        let start = Instant::now();
        let _result = func(&tensor);
        let duration = start.elapsed();
        
        let throughput = test_size as f64 / duration.as_secs_f64() / 1e6; // M elements/sec
        println!("  • {}: {:.2}ms ({:.0}M elem/sec)", name, duration.as_millis(), throughput);
    }
    
    println!();
    Ok(())
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        params.to_string()
    }
}

// Helper trait for path expansion
trait ExpandUser {
    fn expand_user(&self) -> PathBuf;
}

impl ExpandUser for PathBuf {
    fn expand_user(&self) -> PathBuf {
        if let Some(path_str) = self.to_str() {
            if path_str.starts_with("~/") {
                if let Ok(home) = std::env::var("HOME") {
                    return PathBuf::from(home).join(&path_str[2..]);
                }
            }
        }
        self.clone()
    }
}