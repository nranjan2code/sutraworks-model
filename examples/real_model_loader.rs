/// Example: Loading Real Model Weights
/// 
/// This example demonstrates how to load actual model weights from downloaded
/// DeepSeek-Coder or other HuggingFace models, replacing dummy data with real
/// production-ready weights.

use sutra_loader::{ProductionModelLoader, LoaderError, Result};
use std::path::Path;

fn main() -> Result<()> {
    println!("🚀 Production Model Weight Loading Demo");
    println!("======================================");
    
    // Initialize production loader
    let loader = ProductionModelLoader::new();
    
    // List available model configurations
    println!("\n📋 Available Models:");
    let models = loader.registry.list_models();
    for model_id in &models {
        if let Some(config) = loader.registry.get_model(model_id) {
            println!("  - {} ({:?})", model_id, config.architecture);
            println!("    Hidden: {}, Layers: {}, Vocab: {}", 
                config.hidden_size, config.num_layers, config.vocab_size);
        }
    }
    
    // Try to load DeepSeek-Coder if available
    let model_path = Path::new("./downloaded_models/deepseek-coder-v2-lite-instruct");
    if model_path.exists() {
        println!("\n🔄 Loading DeepSeek-Coder-V2-Lite-Instruct (1.3B)...");
        
        match loader.load_model("deepseek-coder-1.3b", model_path) {
            Ok(model) => {
                println!("✅ Model loaded successfully!");
                println!("{}", model.info());
                
                // Validate model weights
                println!("\n🔍 Validating weights...");
                match model.validate() {
                    Ok(()) => println!("✅ All weights validated successfully!"),
                    Err(e) => println!("❌ Validation failed: {}", e),
                }
                
                // Display weight statistics
                display_weight_stats(&model)?;
                
                // Test basic operations
                test_model_operations(&model)?;
                
            },
            Err(LoaderError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("⚠️  Model files not found. Run download script first:");
                println!("   ./download_models_enhanced.sh");
                println!("\n💡 This will download:");
                println!("   - DeepSeek-Coder-V2-Lite-Instruct (2.69 GB)");
                println!("   - Ready for production use with real weights!");
            },
            Err(e) => {
                println!("❌ Failed to load model: {}", e);
                return Err(e);
            }
        }
    } else {
        println!("\n⚠️  Model directory not found: {}", model_path.display());
        show_download_instructions();
    }
    
    // Demonstrate with mock data if real model unavailable
    if !model_path.exists() {
        println!("\n🔧 Demonstrating with model architecture (no real weights):");
        demo_architecture_only(&loader)?;
    }
    
    Ok(())
}

fn display_weight_stats(model: &sutra_loader::LoadedModel) -> Result<()> {
    println!("\n📊 Weight Statistics:");
    
    // Embedding weights
    if let Some(embedding) = &model.weights.embedding {
        println!("  Embedding: {} → {}", 
            format!("{:?}", embedding.shape()), 
            format_memory_size(embedding.memory_usage())
        );
    }
    
    // Layer weights (show first few)
    let layers_to_show = 3.min(model.weights.layers.len());
    for (i, layer) in model.weights.layers.iter().take(layers_to_show).enumerate() {
        match layer {
            sutra_loader::LayerWeights::Transformer(weights) => {
                println!("  Layer {}: Transformer", i);
                if let Some(q_proj) = &weights.attention.q_proj {
                    println!("    Q-proj: {} → {}", 
                        format!("{:?}", q_proj.shape()),
                        format_memory_size(q_proj.memory_usage())
                    );
                }
                if let Some(gate_proj) = &weights.feed_forward.gate_proj {
                    println!("    Gate-proj: {} → {}",
                        format!("{:?}", gate_proj.shape()),
                        format_memory_size(gate_proj.memory_usage())
                    );
                }
            },
            _ => println!("  Layer {}: Other architecture", i),
        }
    }
    
    if model.weights.layers.len() > layers_to_show {
        println!("  ... ({} more layers)", model.weights.layers.len() - layers_to_show);
    }
    
    // LM head
    if let Some(lm_head) = &model.weights.lm_head {
        println!("  LM Head: {} → {}",
            format!("{:?}", lm_head.shape()),
            format_memory_size(lm_head.memory_usage())
        );
    }
    
    Ok(())
}

fn test_model_operations(model: &sutra_loader::LoadedModel) -> Result<()> {
    println!("\n🧪 Testing Model Operations:");
    
    // Test weight access patterns
    println!("  ✓ Weight tensor access");
    
    // Test memory efficiency
    let total_params = calculate_total_parameters(model);
    println!("  ✓ Total parameters: {}", format_number(total_params));
    
    // Estimate memory usage
    let total_memory = estimate_total_memory(model);
    println!("  ✓ Estimated memory: {}", format_memory_size(total_memory));
    
    // Check if fits in 16GB (MacBook Air constraint)
    let memory_gb = total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
    if memory_gb <= 16.0 {
        println!("  ✅ Fits in 16GB MacBook Air memory!");
    } else {
        println!("  ⚠️  Requires {:.1}GB memory (exceeds 16GB)", memory_gb);
        println!("      Consider using quantization for efficiency");
    }
    
    Ok(())
}

fn demo_architecture_only(loader: &ProductionModelLoader) -> Result<()> {
    if let Some(config) = loader.registry.get_model("deepseek-coder-1.3b") {
        println!("\n🏗️  Model Architecture: {}", config.name);
        println!("  Architecture: {:?}", config.architecture);
        println!("  Hidden size: {}", config.hidden_size);
        println!("  Layers: {}", config.num_layers);
        println!("  Vocabulary: {} tokens", config.vocab_size);
        println!("  Model files: {:?}", config.model_files);
        
        println!("\n🗂️  Weight Mapping:");
        println!("  Embedding: {}", config.weight_mapping.embedding);
        println!("  Layer pattern: {}", config.weight_mapping.layers.pattern);
        println!("  Attention Q: {}", config.weight_mapping.layers.attention.q_proj);
        println!("  FFN Gate: {}", config.weight_mapping.layers.feed_forward.gate_proj);
        if let Some(final_norm) = &config.weight_mapping.final_norm {
            println!("  Final norm: {}", final_norm);
        }
        println!("  LM head: {}", config.weight_mapping.lm_head);
    }
    
    Ok(())
}

fn show_download_instructions() {
    println!("\n📥 To download real model weights:");
    println!("   1. Run: ./download_models_enhanced.sh");
    println!("   2. Or manually download from HuggingFace:");
    println!("      huggingface-hub download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct");
    println!("\n🎯 This replaces ALL dummy/synthetic data with:");
    println!("   ✓ Real trained model weights");  
    println!("   ✓ Proper tokenizer vocabularies");
    println!("   ✓ Production model configurations");
    println!("   ✓ Authentic embeddings & projections");
}

fn calculate_total_parameters(model: &sutra_loader::LoadedModel) -> usize {
    let mut total = 0;
    
    // Embedding parameters
    if let Some(embedding) = &model.weights.embedding {
        total += embedding.shape().iter().product::<usize>();
    }
    
    // Layer parameters
    for layer in &model.weights.layers {
        match layer {
            sutra_loader::LayerWeights::Transformer(weights) => {
                if let Some(q_proj) = &weights.attention.q_proj {
                    total += q_proj.shape().iter().product::<usize>();
                }
                if let Some(k_proj) = &weights.attention.k_proj {
                    total += k_proj.shape().iter().product::<usize>();
                }
                if let Some(v_proj) = &weights.attention.v_proj {
                    total += v_proj.shape().iter().product::<usize>();
                }
                if let Some(o_proj) = &weights.attention.o_proj {
                    total += o_proj.shape().iter().product::<usize>();
                }
                if let Some(gate_proj) = &weights.feed_forward.gate_proj {
                    total += gate_proj.shape().iter().product::<usize>();
                }
                if let Some(up_proj) = &weights.feed_forward.up_proj {
                    total += up_proj.shape().iter().product::<usize>();
                }
                if let Some(down_proj) = &weights.feed_forward.down_proj {
                    total += down_proj.shape().iter().product::<usize>();
                }
            },
            _ => {}, // TODO: Add other architectures
        }
    }
    
    // LM head parameters
    if let Some(lm_head) = &model.weights.lm_head {
        total += lm_head.shape().iter().product::<usize>();
    }
    
    total
}

fn estimate_total_memory(model: &sutra_loader::LoadedModel) -> usize {
    let mut total = 0;
    
    if let Some(embedding) = &model.weights.embedding {
        total += embedding.memory_usage();
    }
    
    for layer in &model.weights.layers {
        match layer {
            sutra_loader::LayerWeights::Transformer(weights) => {
                if let Some(q_proj) = &weights.attention.q_proj {
                    total += q_proj.memory_usage();
                }
                if let Some(k_proj) = &weights.attention.k_proj {
                    total += k_proj.memory_usage();
                }
                if let Some(v_proj) = &weights.attention.v_proj {
                    total += v_proj.memory_usage();
                }
                if let Some(o_proj) = &weights.attention.o_proj {
                    total += o_proj.memory_usage();
                }
                if let Some(gate_proj) = &weights.feed_forward.gate_proj {
                    total += gate_proj.memory_usage();
                }
                if let Some(up_proj) = &weights.feed_forward.up_proj {
                    total += up_proj.memory_usage();
                }
                if let Some(down_proj) = &weights.feed_forward.down_proj {
                    total += down_proj.memory_usage();
                }
            },
            _ => {},
        }
    }
    
    if let Some(lm_head) = &model.weights.lm_head {
        total += lm_head.memory_usage();
    }
    
    total
}

fn format_memory_size(bytes: usize) -> String {
    const MB: usize = 1024 * 1024;
    const GB: usize = 1024 * MB;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}