use tfhe::{prelude::FheTrivialEncrypt, FheUint8};

use crate::utils::{rsqrt, CacheTensor, Tensor, Vec4D};

pub struct RMSNorm {
    eps: FheUint8,
    weight: Tensor,
}

impl RMSNorm {
    pub fn new(eps: FheUint8, weight: Tensor) -> Self {
        Self { eps, weight }
    }

    fn norm(&self, x: &Tensor) -> Tensor {
        let sum = x
            .values
            .iter()
            .fold(FheUint8::encrypt_trivial(0u8), |sum, x| sum + (x * x))
            + &self.eps;
        let mean = sum / FheUint8::encrypt_trivial(x.size() as u8);
        let factor = rsqrt(&mean);

        let factor_vec = vec![factor; x.size()];
        let factor_tensor = Tensor::from_cipher(factor_vec);

        x.mul(&factor_tensor).unwrap()
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let output = self.norm(x);
        output.mul(&self.weight).unwrap()
    }
}

pub struct KVCache {
    cache_k: CacheTensor,
    cache_v: CacheTensor,
}

impl KVCache {
    pub fn new(max_batch_size: usize, max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        let dims = vec![max_batch_size, max_seq_len, n_heads, head_dim];
        let cache_k = CacheTensor::zeros(&dims).unwrap();
        let cache_v = CacheTensor::zeros(&dims).unwrap();

        Self { cache_k, cache_v }
    }

    pub fn update(&self, batch_size: usize, start_pos: usize, xk: usize, xv: usize) {
        
    }

    pub fn get_keys(
        &self,
        max_batch_size: usize,
        max_seq_len: usize,
        locak_kv_heads: usize,
        head_dim: usize,
    ) {
    }

    pub fn get_values(
        &self,
        max_batch_size: usize,
        max_seq_len: usize,
        locak_kv_heads: usize,
        head_dim: usize,
    ) {
    }
}
