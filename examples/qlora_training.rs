/// Example: QLoRA Fine-Tuning
/// 
/// Demonstrates parameter-efficient fine-tuning with quantized base model

use sutra_peft::{QLoraConfig, QLoraLayer, LoraConfig, QLoraMemoryEstimator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QLoRA Fine-Tuning Demo ===\n");
    
    // Model dimensions
    let hidden_dim = 4096;
    
    // Configure LoRA
    let lora_config = LoraConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.1,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
    };
    
    // Configure QLoRA
    let qlora_config = QLoraConfig {
        lora: lora_config,
        quant_bits: 4,
        double_quant: true,
    };
    
    println!("QLoRA Configuration:");
    println!("  LoRA rank: {}", qlora_config.lora.rank);
    println!("  Scaling (alpha/rank): {:.2}", qlora_config.lora.scaling());
    println!("  Quantization: {}-bit", qlora_config.quant_bits);
    
    // Create QLoRA layer
    let layer = QLoraLayer::new(hidden_dim, hidden_dim, qlora_config)?;
    
    println!("\nLayer statistics:");
    println!("  Trainable parameters: {}", layer.trainable_parameters());
    println!("  Memory usage: {:.2} MB", layer.memory_usage() as f64 / 1_048_576.0);
    
    // Memory estimation for full model
    println!("\n=== Memory Estimation ===");
    let model_params = 3_000_000_000; // 3B model
    let rank = 8;
    let num_layers = 32;
    
    let estimate = QLoraMemoryEstimator::estimate(model_params, rank, num_layers);
    
    println!("Fine-tuning 3B parameter model:");
    println!("  Base model (4-bit): {:.2} GB", estimate.base_model as f64 / 1e9);
    println!("  LoRA adapters: {:.2} GB", estimate.adapters as f64 / 1e9);
    println!("  Optimizer states: {:.2} GB", estimate.optimizer_states as f64 / 1e9);
    println!("  Gradients: {:.2} GB", estimate.gradients as f64 / 1e9);
    println!("  Total: {:.2} GB", estimate.total_gb());
    
    if estimate.fits_in(16) {
        println!("\n✓ Fits in 16GB MacBook Air!");
    } else {
        println!("\n✗ Requires more than 16GB");
    }
    
    println!("\n=== Parameter Efficiency ===");
    let full_params = hidden_dim * hidden_dim;
    let lora_params = layer.trainable_parameters();
    let reduction = full_params as f32 / lora_params as f32;
    
    println!("Full fine-tuning: {} parameters", full_params);
    println!("QLoRA: {} parameters", lora_params);
    println!("Reduction: {:.1}x fewer parameters", reduction);
    
    Ok(())
}
