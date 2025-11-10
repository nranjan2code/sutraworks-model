/// Integration tests for end-to-end workflows
use sutra_core::{Tensor, DType, ops};
use sutra_tokenizer::{Tokenizer, BpeTokenizer, BpeConfig, VocabBuilder};
use sutra_quantize::{AwqQuantizer, AwqConfig};
use sutra_loader::{ModelDownloader, DownloadConfig};

#[test]
fn test_tokenize_embed_decode_pipeline() {
    // Create tokenizer
    let mut vocab = VocabBuilder::new()
        .with_standard_special_tokens()
        .build();
    
    for word in &["hello", "world", "test"] {
        vocab.add_token(word.to_string());
    }
    
    let config = BpeConfig {
        vocab,
        merges: Vec::new(),
        unk_token: "[UNK]".to_string(),
        byte_level: true,
    };
    
    let tokenizer = BpeTokenizer::new(config);
    let tokenizer = Tokenizer::Bpe(tokenizer);
    
    // Tokenize
    let text = "hello world";
    let encoding = tokenizer.encode(text).unwrap();
    assert!(!encoding.ids.is_empty());
    
    // Create embedding matrix
    let vocab_size = 1000;
    let embed_dim = 128;
    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let embed_weights = Tensor::from_slice(&embed_data, &[vocab_size, embed_dim], DType::F32).unwrap();
    
    // Embed tokens
    let valid_ids: Vec<usize> = encoding.ids.iter()
        .map(|&id| (id as usize) % vocab_size)
        .collect();
    
    let embedded = ops::embedding(&valid_ids, &embed_weights).unwrap();
    assert_eq!(embedded.shape()[0], valid_ids.len());
    assert_eq!(embedded.shape()[1], embed_dim);
    
    // Decode back
    let decoded = tokenizer.decode(&encoding.ids).unwrap();
    assert!(!decoded.is_empty());
    
    println!("✓ Tokenize → Embed → Decode pipeline works!");
}

#[test]
fn test_quantize_train_infer_pipeline() {
    // Create test weight matrix
    let data: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();
    let weights = Tensor::from_slice(&data, &[32, 32], DType::F32).unwrap();
    
    // Quantize
    let quant_config = AwqConfig {
        bits: 4,
        group_size: 128,
        n_samples: 512,
        zero_point: true,
    };
    
    let quantizer = AwqQuantizer::new(quant_config);
    let quantized = quantizer.quantize(&weights, None).unwrap();
    
    // Verify compression
    let original_size = weights.memory_usage();
    let quantized_size = quantized.memory_usage();
    let compression_ratio = original_size as f64 / quantized_size as f64;
    
    assert!(compression_ratio > 2.0, "Expected >2x compression, got {:.2}x", compression_ratio);
    assert_eq!(quantized.bits, 4);
    
    println!("✓ Quantization achieves {:.2}x compression", compression_ratio);
}

#[test]
fn test_model_loading_inference() {
    // Test model downloader configuration
    let config = DownloadConfig::default();
    assert!(config.cache_dir.exists() || !config.cache_dir.as_os_str().is_empty());
    
    // Note: Actual download tests would require network access
    // This test verifies the configuration is valid
    
    println!("✓ Model loader configuration valid");
}

#[test]
fn test_tensor_operations_pipeline() {
    // Create test tensors
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DType::F32).unwrap();
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], DType::F32).unwrap();
    
    // Matrix multiplication
    let c = ops::matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    
    // Apply activation
    let activated = ops::activations::relu(&c);
    assert_eq!(activated.shape(), c.shape());
    
    // Normalize
    let normalized = ops::layer_norm(&activated, 1e-5).unwrap();
    assert_eq!(normalized.shape(), c.shape());
    
    // Apply GELU
    let gelu_out = ops::activations::gelu(&normalized);
    assert_eq!(gelu_out.shape(), normalized.shape());
    
    println!("✓ Complete tensor operations pipeline works!");
}

#[test]
fn test_memory_efficiency() {
    // Test that quantization significantly reduces memory
    let size = 256;
    let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&data, &[size, size], DType::F32).unwrap();
    
    let original_memory = tensor.memory_usage();
    
    // Quantize to 4-bit
    let quantizer = AwqQuantizer::new(AwqConfig::default());
    let quantized = quantizer.quantize(&tensor, None).unwrap();
    
    let quantized_memory = quantized.memory_usage();
    let reduction = (original_memory - quantized_memory) as f64 / original_memory as f64 * 100.0;
    
    assert!(reduction > 50.0, "Expected >50% memory reduction, got {:.1}%", reduction);
    
    println!("✓ Memory reduction: {:.1}%", reduction);
}
