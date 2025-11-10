/// Example: Loading model weights from safetensors format
/// 
/// This demonstrates:
/// - Loading pre-trained weights from safetensors files
/// - Downloading models from HuggingFace
/// - Using the model registry
/// - Inspecting tensor metadata

use sutra_loader::prelude::*;

fn main() -> Result<()> {
    println!("=== SutraWorks Model Loader Demo ===\n");
    
    // 1. Model Registry
    demo_model_registry()?;
    
    // 2. Safetensors Loader (with mock data)
    demo_safetensors_loader()?;
    
    // 3. Model Downloader
    demo_downloader()?;
    
    Ok(())
}

fn demo_model_registry() -> Result<()> {
    println!("📚 Model Registry Demo");
    println!("─────────────────────\n");
    
    let registry = ModelRegistry::with_defaults();
    
    // List all models
    println!("Available models:");
    for model in registry.list() {
        println!("  • {} - {} ({} params, {})",
                 model.id,
                 model.name,
                 format_params(model.num_parameters),
                 model.architecture);
    }
    println!();
    
    // Search for RWKV models
    println!("RWKV models:");
    for model in registry.search("rwkv") {
        println!("  • {} - {}", model.id, model.name);
        if let ModelSource::HuggingFace { repo, .. } = &model.source {
            println!("    Repo: {}", repo);
        }
    }
    println!();
    
    // Get specific model
    let model = registry.get("mamba-1.4b")?;
    println!("Mamba 1.4B details:");
    println!("  Architecture: {}", model.architecture);
    println!("  Parameters: {}", format_params(model.num_parameters));
    println!("  Weight file: {}", model.weight_file);
    if let Some(config) = &model.config_file {
        println!("  Config file: {}", config);
    }
    println!();
    
    Ok(())
}

fn demo_safetensors_loader() -> Result<()> {
    println!("📦 Safetensors Loader Demo");
    println!("──────────────────────────\n");
    
    // Note: This would work with actual safetensors files
    println!("Example usage:");
    println!("  let loader = SafetensorsLoader::new(\"model.safetensors\")?;");
    println!("  let tensors = loader.list_tensors();");
    println!("  let weight = loader.load_tensor(\"model.layers.0.weight\")?;");
    println!();
    
    println!("Features:");
    println!("  • Memory-mapped I/O for efficient loading");
    println!("  • Zero-copy deserialization");
    println!("  • Automatic dtype conversion (f32, f16, i32, u8)");
    println!("  • Batch loading of multiple tensors");
    println!();
    
    Ok(())
}

fn demo_downloader() -> Result<()> {
    println!("⬇️  Model Downloader Demo");
    println!("─────────────────────────\n");
    
    let downloader = ModelDownloader::with_defaults()?;
    
    println!("Download configuration:");
    println!("  Cache directory: {:?}", std::env::temp_dir().join("sutraworks/models"));
    println!("  Verify checksums: enabled");
    println!("  Progress display: enabled");
    println!("  Max retries: 3");
    println!();
    
    println!("Example download:");
    println!("  downloader.download_hf(");
    println!("      \"BlinkDL/rwkv-4-pile-169m\",");
    println!("      \"model.safetensors\",");
    println!("      Some(\"main\")");
    println!("  )?;");
    println!();
    
    println!("Features:");
    println!("  • HuggingFace Hub integration");
    println!("  • Automatic caching");
    println!("  • SHA256 checksum verification");
    println!("  • Progress bar with download speed");
    println!("  • Retry logic with exponential backoff");
    println!();
    
    // Show cached path example
    let cached = downloader.cached_path("test/model", "weights.safetensors", Some("main"));
    println!("Cached path example: {:?}", cached);
    println!();
    
    Ok(())
}

fn format_params(num: u64) -> String {
    if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{}M", num / 1_000_000)
    } else if num >= 1_000 {
        format!("{}K", num / 1_000)
    } else {
        num.to_string()
    }
}
