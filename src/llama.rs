use tfhe::{prelude::FheTrivialEncrypt, FheUint8};

use crate::{
    tensor::{CacheTensor, Tensor},
    utils::rsqrt,
};

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

pub type CacheSlice = [Vec<Vec<Vec<FheUint8>>>];

impl KVCache {
    pub fn new(max_batch_size: usize, max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        let dims = vec![max_batch_size, max_seq_len, n_heads, head_dim];
        let cache_k = CacheTensor::zeros(&dims).unwrap();
        let cache_v = CacheTensor::zeros(&dims).unwrap();

        Self { cache_k, cache_v }
    }

    pub fn update(&self, batch_size: usize, start_pos: usize, xk: &Tensor, xv: &Tensor) -> () {
        let mut change_keys =
            self.cache_k.values[0..batch_size][start_pos..start_pos + xk.size()].to_vec();
        let mut change_values =
            self.cache_v.values[0..batch_size][start_pos..start_pos + xv.size()].to_vec();

        for layer_one in change_keys.iter_mut() {
            for layer_two in layer_one.iter_mut() {
                for layer_three in layer_two.iter_mut() {
                    for i in 0..xk.values.len() {
                        if i < layer_three.len() {
                            layer_three[i] = xk.values[i].to_owned();
                        }
                    }
                }
            }
        }

        for layer_one in change_values.iter_mut() {
            for layer_two in layer_one.iter_mut() {
                for layer_three in layer_two.iter_mut() {
                    for i in 0..xk.values.len() {
                        if i < layer_three.len() {
                            layer_three[i] = xk.values[i].to_owned();
                        }
                    }
                }
            }
        }
    }

    pub fn get_keys(&self, batch_size: usize, start_pos: usize, seq_len: usize) -> &CacheSlice {
        &self.cache_k.values[0..batch_size][0..start_pos + seq_len]
    }

    pub fn get_values(&self, batch_size: usize, start_pos: usize, seq_len: usize) -> &CacheSlice {
        &self.cache_v.values[0..batch_size][0..start_pos + seq_len]
    }
}

#[cfg(test)]
mod tests {
    use tfhe::{
        generate_keys,
        prelude::{FheEq, FheTrivialEncrypt, IfThenElse},
        set_server_key, ConfigBuilder, FheUint8,
    };

    use crate::{
        llama::{KVCache, RMSNorm},
        tensor::Tensor,
    };

    // very slow
    #[test]
    fn test_norm() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);

        // Server-side
        set_server_key(server_key);
        let mut cipher = Vec::new();

        for i in 0_u8..5_u8 {
            let element = FheUint8::encrypt_trivial(i);
            cipher.push(element);
        }

        println!("Setup succesful. Deriving tensors...");
        let cipher_tensor = Tensor::from_cipher(cipher);
        let weights = Tensor::random_weights(cipher_tensor.size());
        let eps = FheUint8::encrypt_trivial(3_u8);

        println!("Tensors established. Normalizing values...");
        let normalizer = RMSNorm::new(eps, weights);
        let normalized_vals = normalizer.forward(&cipher_tensor); //slow line

        assert!(normalized_vals.size() == cipher_tensor.size());

        for (norm_val, ciph_val) in normalized_vals.values.iter().zip(cipher_tensor.values) {
            let eq: u8 = (norm_val)
                .eq(&ciph_val)
                .if_then_else(
                    &FheUint8::encrypt_trivial(1u8),
                    &FheUint8::encrypt_trivial(0u8),
                )
                .try_decrypt_trivial()
                .unwrap();

            assert_ne!(eq, 1_u8);
        }
    }

    #[test]
    fn test_new_cache() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);

        // Server-side
        set_server_key(server_key);
        let cache = KVCache::new(2, 3, 4, 2);
        assert_eq!(cache.cache_k.values.len(), 2);
        assert_eq!(cache.cache_k.values[0].len(), 3);
        assert_eq!(cache.cache_k.values[0][0].len(), 4);
        assert_eq!(cache.cache_k.values[0][0][0].len(), 2);
    }

    #[test]
    fn test_update_cache() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);

        // Server-side
        set_server_key(server_key);
        let cache = KVCache::new(2, 3, 4, 2);
        let xk = Tensor::from_cipher(vec![FheUint8::encrypt_trivial(1_u8); 2]);
        let xv = Tensor::from_cipher(vec![FheUint8::encrypt_trivial(4_u8); 2]);
        cache.update(2, 0, &xk, &xv);

        let k_eq: u8 = (cache.cache_k.values[0][0][0][0])
            .eq(&FheUint8::encrypt_trivial(0_u8))
            .if_then_else(
                &FheUint8::encrypt_trivial(1u8),
                &FheUint8::encrypt_trivial(0u8),
            )
            .try_decrypt_trivial()
            .unwrap();

        let v_eq: u8 = (cache.cache_v.values[0][0][0][0])
            .eq(&FheUint8::encrypt_trivial(0_u8))
            .if_then_else(
                &FheUint8::encrypt_trivial(1u8),
                &FheUint8::encrypt_trivial(0u8),
            )
            .try_decrypt_trivial()
            .unwrap();

        assert_eq!(k_eq, 1_u8);
        assert_eq!(v_eq, 1_u8);
    }
}
