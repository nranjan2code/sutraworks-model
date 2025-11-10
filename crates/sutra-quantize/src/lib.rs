//! Model compression through quantization
//! 
//! Implements AWQ (Activation-aware Weight Quantization) for 4-bit model compression.
//! AWQ preserves model accuracy by protecting salient weights during quantization.

pub mod awq;
pub mod quantizer;
pub mod dequantizer;

pub use awq::{AwqQuantizer, AwqConfig};
pub use quantizer::{Quantizer, QuantizedTensor};
pub use dequantizer::Dequantizer;

pub mod prelude {
    pub use crate::{AwqQuantizer, AwqConfig, Quantizer, QuantizedTensor, Dequantizer};
}
