//! Model loading and weight management for SutraWorks
//!
//! This crate provides functionality to:
//! - Load model weights from safetensors format
//! - Download pre-trained models from remote repositories
//! - Memory-mapped I/O for efficient large model handling
//! - Weight validation and integrity checking

pub mod downloader;
pub mod error;
pub mod model_registry;
pub mod safetensors_loader;

pub use downloader::{DownloadConfig, ModelDownloader};
pub use error::{LoaderError, Result};
pub use model_registry::{ModelInfo, ModelRegistry, ModelSource};
pub use safetensors_loader::{SafetensorsLoader, TensorInfo};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{DownloadConfig, ModelInfo, ModelSource, TensorInfo};
    pub use crate::{LoaderError, Result};
    pub use crate::{ModelDownloader, ModelRegistry, SafetensorsLoader};
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
