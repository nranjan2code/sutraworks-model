/// Model loading and weight management for SutraWorks
/// 
/// This crate provides functionality to:
/// - Load model weights from safetensors format
/// - Download pre-trained models from remote repositories
/// - Memory-mapped I/O for efficient large model handling
/// - Weight validation and integrity checking

pub mod safetensors_loader;
pub mod downloader;
pub mod model_registry;
pub mod error;

pub use safetensors_loader::{SafetensorsLoader, TensorInfo};
pub use downloader::{ModelDownloader, DownloadConfig};
pub use model_registry::{ModelRegistry, ModelInfo, ModelSource};
pub use error::{LoaderError, Result};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{SafetensorsLoader, ModelDownloader, ModelRegistry};
    pub use crate::{TensorInfo, DownloadConfig, ModelInfo, ModelSource};
    pub use crate::{LoaderError, Result};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_imports() {
        // Ensure all public modules are accessible
        let _ = safetensors_loader::SafetensorsLoader::new("test.safetensors");
    }
}
