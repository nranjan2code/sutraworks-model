//! Model compression through quantization
//!
//! Implements AWQ (Activation-aware Weight Quantization) for 4-bit model compression.
//! AWQ preserves model accuracy by protecting salient weights during quantization.

pub mod awq;
pub mod dequantizer;
pub mod quantizer;

pub use awq::{AwqConfig, AwqQuantizer};
pub use dequantizer::Dequantizer;
pub use quantizer::{QuantizedTensor, Quantizer};

pub mod prelude {
    pub use crate::{AwqConfig, AwqQuantizer, Dequantizer, QuantizedTensor, Quantizer};
}
