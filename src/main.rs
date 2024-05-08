use llama::RMSNorm;
use tensor::Tensor;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint8};
use utils::encrypt_str;

use crate::llama::{CacheSlice, KVCache};

mod llama;
mod tensor;
mod utils;

fn main() {
    let config = ConfigBuilder::default().build();

    // Client-side
    println!("Starting setup...");
    let (client_key, server_key) = generate_keys(config);
    let cleartext = "Hello";
    let cipher = encrypt_str(&client_key, &cleartext).unwrap();

    // Server-side
    println!("Setup succesful. Deriving tensors...");
    set_server_key(server_key);
    let cipher_tensor = Tensor::from_cipher(cipher);
    let weights = Tensor::ones(cipher_tensor.size());
    let eps = FheUint8::encrypt_trivial(3_u8);

    println!("Tensors established. Normalizing values...");
    let _normalizer = RMSNorm::new(eps, weights);
    // slow operation
    // let normalized_vals = normalizer.forward(&cipher_tensor);

    println!("Values normalized. Establishing QKV cache...");
    let cache = KVCache::new(2, 3, 4, 2);
    let xk = Tensor::from_cipher(vec![FheUint8::encrypt_trivial(1_u8); 2]);
    let xv = Tensor::from_cipher(vec![FheUint8::encrypt_trivial(4_u8); 2]);

    cache.update(2, 0, &xk, &xv);
    let _keys: &CacheSlice = cache.get_keys(2, 0, 2);
    let _values: &CacheSlice = cache.get_values(2, 0, 2);

    println!("Complete!");
}
