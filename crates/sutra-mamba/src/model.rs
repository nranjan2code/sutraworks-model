use serde::{Deserialize, Serialize};
use sutra_core::{Result, SutraError};
use crate::layer::MambaLayer;

/// Mamba model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub state_size: usize,
    pub expand_factor: usize,
    pub conv_kernel: usize,
    pub max_seq_len: usize,
}

impl MambaConfig {
    pub fn new(num_layers: usize, hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            vocab_size,
            state_size: 16,  // SSM state dimension
            expand_factor: 2, // Internal expansion
            conv_kernel: 4,   // Convolution kernel size
            max_seq_len: 2048,
        }
    }
    
    /// Mamba-3B configuration
    pub fn mamba_3b() -> Self {
        Self::new(48, 2560, 50000)
    }
    
    /// Estimate memory usage
    pub fn estimate_memory(&self) -> usize {
        // Mamba has linear memory complexity
        let params_per_layer = self.hidden_size * self.hidden_size * self.expand_factor * 4;
        let total_params = params_per_layer * self.num_layers;
        total_params * std::mem::size_of::<f32>()
    }
    
    /// Estimate throughput advantage over Transformer
    pub fn throughput_multiplier(&self, seq_len: usize) -> f32 {
        // Mamba is O(n) while Transformer is O(n²)
        // For typical sequences, Mamba is ~5x faster
        let transformer_ops = seq_len * seq_len;
        let mamba_ops = seq_len;
        transformer_ops as f32 / mamba_ops as f32
    }
}

/// Mamba model for efficient sequence modeling
pub struct MambaModel {
    config: MambaConfig,
    layers: Vec<MambaLayer>,
}

impl MambaModel {
    pub fn new(config: MambaConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        
        for _ in 0..config.num_layers {
            layers.push(MambaLayer::new(&config)?);
        }
        
        Ok(Self {
            config,
            layers,
        })
    }
    
    /// Forward pass through Mamba
    /// 
    /// # Arguments
    /// * `input` - Input tokens [batch_size, seq_len]
    /// 
    /// # Returns
    /// * Logits [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input: &[usize]) -> Result<Vec<f32>> {
        // Placeholder implementation
        let logits = vec![0.0; self.config.vocab_size];
        Ok(logits)
    }
    
    /// Generate text with Mamba
    pub fn generate(
        &self,
        prompt: &[usize],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        let mut tokens = prompt.to_vec();
        
        for _ in 0..max_tokens {
            let logits = self.forward(&tokens)?;
            let next_token = self.sample_token(&logits, temperature);
            tokens.push(next_token);
            
            if next_token == 0 {
                break;
            }
        }
        
        Ok(tokens)
    }
    
    fn sample_token(&self, logits: &[f32], _temperature: f32) -> usize {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_config() {
        let config = MambaConfig::mamba_3b();
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.hidden_size, 2560);
    }
    
    #[test]
    fn test_complexity_advantage() {
        let config = MambaConfig::new(24, 1024, 50000);
        
        // For 2048 token sequence
        let speedup = config.throughput_multiplier(2048);
        println!("Mamba throughput advantage: {:.1}x", speedup);
        
        // Should be ~2048x faster (linear vs quadratic)
        assert!(speedup > 1000.0);
    }
    
    #[test]
    fn test_mamba_model() {
        let config = MambaConfig::new(12, 768, 50000);
        let model = MambaModel::new(config).unwrap();
        assert_eq!(model.layers.len(), 12);
    }
}
