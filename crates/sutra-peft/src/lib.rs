//! Parameter-Efficient Fine-Tuning (PEFT) with LoRA and QLoRA
//! 
//! Enables fine-tuning large models with minimal memory by:
//! - Training only small adapter matrices (LoRA)
//! - Working with quantized base models (QLoRA)

pub mod lora;
pub mod qlora;
pub mod adapter;
pub mod trainer;

pub use lora::{LoraConfig, LoraLayer};
pub use qlora::{QLoraConfig, QLoraLayer};
pub use adapter::{Adapter, AdapterManager};
pub use trainer::{PeftTrainer, TrainingConfig};

pub mod prelude {
    pub use crate::{
        LoraConfig, LoraLayer,
        QLoraConfig, QLoraLayer,
        Adapter, AdapterManager,
        PeftTrainer, TrainingConfig,
    };
}
