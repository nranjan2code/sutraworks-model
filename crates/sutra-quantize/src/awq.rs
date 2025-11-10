use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sutra_core::{Tensor, DType, Result, SutraError};

/// Configuration for AWQ quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqConfig {
    /// Number of bits for quantization (typically 4)
    pub bits: u8,
    /// Group size for quantization (typically 128)
    pub group_size: usize,
    /// Number of calibration samples
    pub n_samples: usize,
    /// Zero-point offset
    pub zero_point: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            n_samples: 512,
            zero_point: true,
        }
    }
}

/// AWQ (Activation-aware Weight Quantization) quantizer
/// 
/// AWQ identifies and protects salient weights during quantization,
/// resulting in better model quality at 4-bit precision.
pub struct AwqQuantizer {
    config: AwqConfig,
}

impl AwqQuantizer {
    pub fn new(config: AwqConfig) -> Self {
        Self { config }
    }
    
    /// Quantize a tensor using AWQ method
    /// 
    /// # Arguments
    /// * `tensor` - Input tensor to quantize
    /// * `activations` - Optional activation statistics for computing salience
    pub fn quantize(&self, tensor: &Tensor, activations: Option<&Array1<f32>>) -> Result<QuantizedWeights> {
        let data = tensor.data();
        
        if data.ndim() != 2 {
            return Err(SutraError::InvalidShape(
                "AWQ quantization requires 2D weight matrices".to_string()
            ));
        }
        
        let shape = data.shape();
        let (out_features, in_features) = (shape[0], shape[1]);
        
        // Compute salience scores (importance of each weight)
        let salience = self.compute_salience(data, activations);
        
        // Perform grouped quantization with salience-aware scaling
        let quantized = self.quantize_with_salience(data, &salience)?;
        
        Ok(quantized)
    }
    
    /// Compute salience scores for weights
    /// Salient weights have higher impact on model output
    fn compute_salience(&self, weights: &ndarray::ArrayD<f32>, activations: Option<&Array1<f32>>) -> Array1<f32> {
        let shape = weights.shape();
        let in_features = shape[1];
        
        match activations {
            Some(acts) => {
                // Use activation magnitudes as proxy for salience
                acts.clone()
            }
            None => {
                // Fallback: use weight magnitudes
                let weights_2d = weights.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                weights_2d.map_axis(Axis(0), |col| {
                    col.iter().map(|&x| x.abs()).sum::<f32>()
                })
            }
        }
    }
    
    /// Quantize weights with salience-aware scaling
    fn quantize_with_salience(&self, weights: &ndarray::ArrayD<f32>, salience: &Array1<f32>) -> Result<QuantizedWeights> {
        let weights_2d = weights.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| SutraError::QuantizationError(format!("Shape error: {}", e)))?;
        
        let shape = weights_2d.shape();
        let (out_features, in_features) = (shape[0], shape[1]);
        let group_size = self.config.group_size;
        let n_groups = (in_features + group_size - 1) / group_size;
        
        // Storage for quantized values
        let mut qweights = Vec::with_capacity(out_features * in_features);
        let mut scales = Vec::with_capacity(out_features * n_groups);
        let mut zeros = if self.config.zero_point {
            Some(Vec::with_capacity(out_features * n_groups))
        } else {
            None
        };
        
        let qmax = (1 << self.config.bits) - 1;
        let qmax_f = qmax as f32;
        
        // Quantize each output feature
        for row in weights_2d.axis_iter(Axis(0)) {
            // Process in groups
            for g in 0..n_groups {
                let start = g * group_size;
                let end = (start + group_size).min(in_features);
                let group = row.slice(ndarray::s![start..end]);
                
                // Compute group statistics with salience weighting
                let group_salience = salience.slice(ndarray::s![start..end]);
                let weighted_values: Vec<f32> = group.iter()
                    .zip(group_salience.iter())
                    .map(|(&w, &s)| w * s.sqrt())
                    .collect();
                
                let min_val = weighted_values.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = weighted_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                
                let scale = (max_val - min_val) / qmax_f;
                let scale = if scale.abs() < 1e-8 { 1.0 } else { scale };
                
                let zero = if self.config.zero_point {
                    let z = (-min_val / scale).round().clamp(0.0, qmax_f) as u8;
                    zeros.as_mut().unwrap().push(z);
                    z as f32
                } else {
                    0.0
                };
                
                scales.push(scale);
                
                // Quantize group
                for &val in group.iter() {
                    let qval = ((val / scale) + zero).round().clamp(0.0, qmax_f) as u8;
                    qweights.push(qval);
                }
            }
        }
        
        Ok(QuantizedWeights {
            qweights,
            scales,
            zeros,
            shape: vec![out_features, in_features],
            bits: self.config.bits,
            group_size,
        })
    }
}

/// Quantized weight representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedWeights {
    /// Quantized weight values (packed)
    pub qweights: Vec<u8>,
    /// Scale factors per group
    pub scales: Vec<f32>,
    /// Zero points per group (optional)
    pub zeros: Option<Vec<u8>>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Bits per weight
    pub bits: u8,
    /// Group size for quantization
    pub group_size: usize,
}

impl QuantizedWeights {
    /// Compute memory savings from quantization
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.shape.iter().product::<usize>() * 4; // f32
        let quantized_size = self.qweights.len() + 
                            self.scales.len() * 4 + 
                            self.zeros.as_ref().map_or(0, |z| z.len());
        original_size as f32 / quantized_size as f32
    }
    
    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.qweights.len() + 
        self.scales.len() * 4 + 
        self.zeros.as_ref().map_or(0, |z| z.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD, IxDyn};

    #[test]
    fn test_awq_quantization() {
        let config = AwqConfig::default();
        let quantizer = AwqQuantizer::new(config);
        
        // Create test tensor
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();
        let arr = Array::from_shape_vec(IxDyn(&[32, 32]), data).unwrap();
        let tensor = Tensor::new(arr, DType::F32);
        
        let result = quantizer.quantize(&tensor, None).unwrap();
        
        assert_eq!(result.bits, 4);
        assert!(result.compression_ratio() > 1.0);
    }
}
