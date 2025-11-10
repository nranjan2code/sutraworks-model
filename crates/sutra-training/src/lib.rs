/// Training infrastructure for SutraWorks models
/// 
/// Features:
/// - Optimizers (Adam, SGD, AdamW)
/// - Learning rate schedulers
/// - Gradient computation and backpropagation
/// - Loss functions
/// - Training loops with checkpointing
/// - Mixed precision training support

pub mod optimizer;
pub mod scheduler;
pub mod loss;
pub mod trainer;
pub mod gradient;
pub mod error;

pub use optimizer::{Optimizer, Adam, AdamConfig, Sgd, SgdConfig, AdamW};
pub use scheduler::{LRScheduler, CosineScheduler, LinearScheduler};
pub use loss::{Loss, CrossEntropyLoss, MSELoss};
pub use trainer::{Trainer, TrainerConfig, TrainingState};
pub use gradient::GradientAccumulator;
pub use error::{TrainingError, Result};

pub mod prelude {
    pub use crate::{Trainer, TrainerConfig, TrainingState};
    pub use crate::{Adam, AdamConfig, Sgd, SgdConfig};
    pub use crate::{LRScheduler, CosineScheduler, LinearScheduler};
    pub use crate::{Loss, CrossEntropyLoss, MSELoss};
    pub use crate::{TrainingError, Result};
}
