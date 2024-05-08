use tfhe::{prelude::FheTrivialEncrypt, FheUint8};

use crate::utils::{StringCipherText, Vec4D};

#[derive(Default)]
pub struct Tensor {
    pub values: Vec<FheUint8>,
    dim: Vec<usize>,
}

impl Tensor {
    pub fn from_cipher(values: StringCipherText) -> Self {
        let dim = vec![1; values.len()];
        Self { values, dim }
    }

    pub fn ones(sz: usize) -> Self {
        let values = (0..sz).map(|_| FheUint8::encrypt_trivial(1_u8)).collect();
        let dim = vec![1; sz];

        Self { values, dim }
    }

    pub fn random_weights(sz: usize) -> Self {
        let values = vec![FheUint8::generate_oblivious_pseudo_random(tfhe::Seed(37), 8); sz];
        let dim = vec![1; sz];

        Self { values, dim }
    }

    // Basic dot product between two 1D tensors
    pub fn dot(&self, other: &Self) -> Result<FheUint8, &'static str> {
        if self.dim() != other.dim() {
            return Err("Two tensors are not the same size..");
        }

        let mut result = FheUint8::encrypt_trivial(0_u8);
        for i in 0..self.size() {
            result += &self.values[i] * &other.values[i];
        }

        Ok(result)
    }

    // Matrix multiplication -- very slow
    pub fn mul(&self, other: &Self) -> Result<Self, &'static str> {
        if self.dim() != other.dim() {
            return Err("Tensors must have the same size for element-wise multiplication.");
        }

        let values = (0..self.size())
            .map(|i| &self.values[i] * &other.values[i]) // multiplication op is slow
            .collect();

        Ok(Self {
            values: values,
            dim: vec![self.size()],
        })
    }

    pub fn dim(&self) -> Vec<usize> {
        self.dim.clone()
    }

    pub fn size(&self) -> usize {
        self.values.len()
    }
}

pub struct CacheTensor {
    pub values: Vec4D,
}

impl CacheTensor {
    pub fn zeros(dim: &Vec<usize>) -> Result<Self, &'static str> {
        if dim.len() != 4 {
            return Err("Dimensions should have 4 parameters");
        }

        let (W, X, Y, Z) = (dim[0], dim[1], dim[2], dim[3]);

        let output = vec![vec![vec![vec![FheUint8::encrypt_trivial(0_u8); Z]; Y]; X]; W];

        Ok(Self { values: output })
    }
}

#[cfg(test)]
mod tests {
    use tfhe::{generate_keys, prelude::FheDecrypt, set_server_key, ConfigBuilder};

    use crate::utils::encrypt_str;

    use super::*;

    #[test]
    fn test_dot_product() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);
        set_server_key(server_key);

        let tensor1 = Tensor {
            values: vec![
                FheUint8::encrypt_trivial(1u8),
                FheUint8::encrypt_trivial(2u8),
                FheUint8::encrypt_trivial(3u8),
            ],
            dim: vec![3],
        };
        let tensor2 = Tensor {
            values: vec![
                FheUint8::encrypt_trivial(4u8),
                FheUint8::encrypt_trivial(5u8),
                FheUint8::encrypt_trivial(6u8),
            ],
            dim: vec![3],
        };

        let result = tensor1.dot(&tensor2).unwrap();

        let decrypted: u8 = result.decrypt(&client_key);

        assert_eq!(decrypted, 32u8)
    }

    // test is slow because multiplication itself is slow
    #[test]
    fn test_mul() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);
        set_server_key(server_key);

        let tensor1 = Tensor::from_cipher(encrypt_str(&client_key, "123").unwrap());
        let tensor2 = Tensor::from_cipher(encrypt_str(&client_key, "456").unwrap());

        let result = tensor1.mul(&tensor2).unwrap();

        assert_eq!(1, 1);

        let decrypted_values: Vec<u8> = result
            .values
            .iter()
            .map(|val| val.decrypt(&client_key))
            .collect();

        assert_eq!(decrypted_values, vec![4, 10, 18]);
    }
}
