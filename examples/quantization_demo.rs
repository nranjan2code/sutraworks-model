/// Example: Model Quantization with AWQ
/// 
/// Demonstrates 4-bit quantization for running SOTA models on 16GB RAM

use sutra_core::{Tensor, DType};
use sutra_quantize::{AwqQuantizer, AwqConfig, Dequantizer};
use ndarray::{Array, IxDyn};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AWQ Quantization Demo ===\n");
    
    // Create a sample weight matrix (simulating a model layer)
    let size = 2048;
    let data: Vec<f32> = (0..size*size)
        .map(|i| (i as f32 / (size * size) as f32) * 2.0 - 1.0)
        .collect();
    
    let arr = Array::from_shape_vec(IxDyn(&[size, size]), data)?;
    let tensor = Tensor::new(arr, DType::F32);
    
    println!("Original tensor:");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Memory: {:.2} MB", tensor.memory_usage() as f64 / 1_048_576.0);
    
    // Configure AWQ quantization
    let config = AwqConfig {
        bits: 4,
        group_size: 128,
        n_samples: 512,
        zero_point: true,
    };
    
    println!("\nQuantization config:");
    println!("  Bits: {}", config.bits);
    println!("  Group size: {}", config.group_size);
    
    // Quantize
    let quantizer = AwqQuantizer::new(config);
    let quantized = quantizer.quantize(&tensor, None)?;
    
    println!("\nQuantized tensor:");
    println!("  Memory: {:.2} MB", quantized.memory_usage() as f64 / 1_048_576.0);
    println!("  Compression ratio: {:.2}x", quantized.compression_ratio());
    
    // Dequantize for inference
    let dequantizer = Dequantizer;
    let restored = dequantizer.dequantize_awq(&quantized)?;
    
    println!("\nRestored tensor:");
    println!("  Shape: {:?}", restored.shape());
    
    println!("\n✓ Quantization complete!");
    println!("  Memory saved: {:.2} MB", 
        (tensor.memory_usage() - quantized.memory_usage()) as f64 / 1_048_576.0);
    
    Ok(())
}
