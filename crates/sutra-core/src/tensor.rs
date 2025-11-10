use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use crate::error::{Result, SutraError};

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    I32,
    I8,
    U8,
    I4,  // 4-bit integer for quantization
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::I4 => 1, // Packed, but accounting at byte level
        }
    }
}

/// Multi-dimensional tensor with dynamic shape
pub struct Tensor {
    data: ArrayD<f32>,
    dtype: DType,
    name: Option<String>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>, dtype: DType) -> Self {
        Self {
            data,
            dtype,
            name: None,
        }
    }
    
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self::new(data, dtype)
    }
    
    pub fn from_slice(data: &[f32], shape: &[usize], dtype: DType) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        if data.len() != total_size {
            return Err(SutraError::InvalidShape(
                format!("Data length {} doesn't match shape {:?}", data.len(), shape)
            ));
        }
        
        let arr = Array::from_shape_vec(IxDyn(shape), data.to_vec())
            .map_err(|e| SutraError::InvalidShape(e.to_string()))?;
        
        Ok(Self::new(arr, dtype))
    }
    
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    pub fn data(&self) -> &ArrayD<f32> {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }
    
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() * self.dtype.size_bytes()
    }
}

/// View into a tensor for zero-copy operations
pub struct TensorView<'a> {
    data: &'a ArrayD<f32>,
    dtype: DType,
}

impl<'a> TensorView<'a> {
    pub fn new(data: &'a ArrayD<f32>, dtype: DType) -> Self {
        Self { data, dtype }
    }
    
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    pub fn data(&self) -> &ArrayD<f32> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, &[2, 2], DType::F32).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::F32);
    }
    
    #[test]
    fn test_tensor_memory() {
        let tensor = Tensor::zeros(&[100, 100], DType::F32);
        assert_eq!(tensor.memory_usage(), 10000 * 4); // 40KB
    }
}
