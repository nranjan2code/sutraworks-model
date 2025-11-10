use serde::{Deserialize, Serialize};
use sutra_core::{Result, SutraError};
use crate::layer::RwkvLayer;
use crate::state::RwkvState;

/// RWKV model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RwkvConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
}

impl RwkvConfig {
    pub fn new(num_layers: usize, hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            vocab_size,
            max_seq_len: 2048,
            layer_norm_eps: 1e-5,
        }
    }
    
    /// Estimate memory usage for inference
    pub fn estimate_memory(&self) -> usize {
        // RWKV has constant memory complexity
        // State: hidden_size * num_layers * 2 (for att and ffn states)
        let state_mem = self.hidden_size * self.num_layers * 2 * std::mem::size_of::<f32>();
        
        // Weights: rough estimate
        let weight_mem = self.hidden_size * self.hidden_size * self.num_layers * 4 * 4;
        
        state_mem + weight_mem
    }
}

/// RWKV model for efficient inference
pub struct RwkvModel {
    config: RwkvConfig,
    layers: Vec<RwkvLayer>,
}

impl RwkvModel {
    pub fn new(config: RwkvConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        
        for layer_idx in 0..config.num_layers {
            layers.push(RwkvLayer::new(config.hidden_size, layer_idx)?);
        }
        
        Ok(Self {
            config,
            layers,
        })
    }
    
    /// Forward pass through the model
    /// 
    /// # Arguments
    /// * `input` - Input token IDs [batch_size, seq_len]
    /// * `state` - Optional previous state for sequential generation
    /// 
    /// # Returns
    /// * Logits [batch_size, seq_len, vocab_size]
    /// * Updated state for next step
    pub fn forward(
        &self,
        input: &[usize],
        state: Option<RwkvState>,
    ) -> Result<(Vec<f32>, RwkvState)> {
        let mut state = state.unwrap_or_else(|| RwkvState::new(&self.config));
        
        // In a real implementation, this would:
        // 1. Embed tokens
        // 2. Process through RWKV layers
        // 3. Generate logits
        // 4. Update state
        
        // Placeholder implementation
        let logits = vec![0.0; self.config.vocab_size];
        
        Ok((logits, state))
    }
    
    /// Generate text autoregressively
    pub fn generate(
        &self,
        prompt: &[usize],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        let mut tokens = prompt.to_vec();
        let mut state = RwkvState::new(&self.config);
        
        for _ in 0..max_tokens {
            let (logits, new_state) = self.forward(&tokens, Some(state))?;
            state = new_state;
            
            // Sample next token (placeholder)
            let next_token = self.sample_token(&logits, temperature);
            tokens.push(next_token);
            
            // Check for EOS token
            if next_token == 0 {
                break;
            }
        }
        
        Ok(tokens)
    }
    
    fn sample_token(&self, logits: &[f32], temperature: f32) -> usize {
        // Placeholder: return token with highest logit
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    pub fn config(&self) -> &RwkvConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rwkv_model_creation() {
        let config = RwkvConfig::new(12, 768, 50000);
        let model = RwkvModel::new(config).unwrap();
        assert_eq!(model.layers.len(), 12);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let config = RwkvConfig::new(24, 1024, 50000);
        let memory = config.estimate_memory();
        let memory_gb = memory as f64 / 1_073_741_824.0;
        
        println!("RWKV-24L-1024D estimated memory: {:.2} GB", memory_gb);
        
        // RWKV should fit comfortably in 16GB
        assert!(memory_gb < 10.0);
    }
}
