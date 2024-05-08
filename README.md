# LLAMA Operations in Fully Homomorphic Encryption

An implementation of the RMSNorm normalizer for Meta's
[LLAMA](https://github.com/meta-llama/llama/tree/main) model and projection
Queries, Keys, and Values using the FHE library
[tfhe-rs](https://github.com/zama-ai/tfhe-rs?tab=readme-ov-file#a-simple-example)
from Zama. This project takes in an arbitrary string of words, converts them into an encrypted vector representing a user's prompt, transforms this vector into a custom `Tensor`, and outputs an encrypted `Q`, `K`, and `V` vector.

`src/tensor.rs` contains a bare bones implementation of a Tensor. A custom implementation was developed because of the lack of support for mathematical operations of encrypted data types such as [tfhe-rs](https://github.com/zama-ai/tfhe-rs?tab=readme-ov-file#a-simple-example)'s `FheUint`.

`src/llama.rs` contains an implementation of the [RMSNorm](https://arxiv.org/abs/1910.07467) normalizer. It loosely follows the official implementation in the [open source code of LLAMA](https://github.com/meta-llama/llama/). Further, this file implements a QKV cache to be used in a model.

`src/utils.rs` contains auxiliary functions for encryption of arbitrary strings and `RMSNorm` normalization.

Additionally, each file has its own set of brief tests.

`src/main.rs` can be run as an example usage of the project.