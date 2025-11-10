//! Mamba: Linear-Time Sequence Modeling with Selective State Spaces
//! 
//! Mamba achieves Transformer-quality performance with linear complexity:
//! - O(n) time complexity (vs O(n²) for Transformers)
//! - 5x higher throughput than Transformers
//! - Scales linearly with sequence length
//! - Selective state space mechanism

pub mod model;
pub mod layer;
pub mod ssm;
pub mod selective;

pub use model::{MambaModel, MambaConfig};
pub use layer::MambaLayer;
pub use ssm::StateSpaceModel;

pub mod prelude {
    pub use crate::{MambaModel, MambaConfig, MambaLayer, StateSpaceModel};
}
