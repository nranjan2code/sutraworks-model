/// Core types and utilities shared across all SutraWorks crates
pub mod tensor;
pub mod model;
pub mod error;

pub use tensor::{Tensor, TensorView, DType};
pub use model::{ModelConfig, ModelWeights};
pub use error::{SutraError, Result};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{Tensor, TensorView, DType, ModelConfig, ModelWeights};
    pub use crate::{SutraError, Result};
}
