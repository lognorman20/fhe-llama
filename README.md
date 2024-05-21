# LLAMA Operations in Fully Homomorphic Encryption

An implementation of the RMSNorm normalizer for Meta's
[LLAMA](https://github.com/meta-llama/llama/tree/main) model and projection
Queries, Keys, and Values using the FHE library
[tfhe-rs](https://github.com/zama-ai/tfhe-rs?tab=readme-ov-file#a-simple-example)
from Zama. This project takes in an arbitrary string of words, converts them into an encrypted vector representing a user's prompt, transforms this vector into a custom `Tensor`, and outputs an encrypted `Q`, `K`, and `V` vector.

`src/tensor.rs` contains a bare bones implementation of a Tensor. A custom implementation was developed because of the lack of support from popular existing Tensor and ndarray libraries in Rust for mathematical operations of encrypted data types such as [tfhe-rs](https://github.com/zama-ai/tfhe-rs?tab=readme-ov-file#a-simple-example)'s `FheUint`.

`src/llama.rs` contains an implementation of the [RMSNorm](https://arxiv.org/abs/1910.07467) normalizer. It loosely follows the official implementation in the [open source code of LLAMA](https://github.com/meta-llama/llama/). Further, this file implements a QKV cache to be used in a model.

`src/utils.rs` contains auxiliary functions for encryption of arbitrary strings and `RMSNorm` normalization.

Additionally, each file has its own set of brief tests.

`src/main.rs` can be run as an example usage of the project as shown below:
```rust
fn main() {
    let config = ConfigBuilder::default().build();

    // Client-side
    println!("Starting setup...");
    let (client_key, server_key) = generate_keys(config);
    let cleartext = "Hello world!";
    let cipher = encrypt_str(&client_key, &cleartext).unwrap();

    // Server-side
    println!("Setup succesful. Deriving tensors...");
    set_server_key(server_key);
    let cipher_tensor = Tensor::from_cipher(cipher);
    let weights = Tensor::random_weights(cipher_tensor.size());
    let eps = FheUint8::encrypt_trivial(3_u8);

    println!("Tensors established. Normalizing values...");
    let normalizer = RMSNorm::new(eps, weights);
    let normalized_vals = normalizer.forward(&cipher_tensor);

    println!("Values normalized. Establishing QKV cache...");
    let cache = KVCache::new(2, 4, 3, 4);
    let xk = Tensor::from_cipher(vec![FheUint8::encrypt_trivial(1_u8); 2]);
    let xv = Tensor::from_cipher(cipher_tensor.values);

    cache.update(2, 0, &xk, &xv);
    let keys: &CacheSlice = cache.get_keys(2, 0, 2);
    let values: &CacheSlice = cache.get_values(2, 0, 2);

    println!("Complete!");
}
```