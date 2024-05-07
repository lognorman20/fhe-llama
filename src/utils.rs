use std::ops::Mul;

use num::integer::sqrt;
use tfhe::{
    prelude::{FheDecrypt, FheEncrypt, FheEq, FheOrd, FheTrivialEncrypt, IfThenElse},
    shortint::client_key,
    ClientKey, FheBool, FheUint, FheUint8, FheUint8Id,
};

pub type StringCipherText = Vec<FheUint8>;

#[derive(Default)]
pub struct Tensor {
    pub values: Vec<FheUint8>,
    dim: Vec<usize>,
}

impl Tensor {
    pub fn new(values: StringCipherText) -> Self {
        let dim = vec![1; values.len()];
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

    // Matrix multiplication
    pub fn mul(&self, other: &Self) -> Result<Self, &'static str> {
        if self.dim() != other.dim() {
            return Err("Tensors must have the same size for element-wise multiplication.");
        }

        let mut result_values = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            // todo: enumerated for debugging, clean later
            let self_val = &self.values[i];
            let other_val = &other.values[i];
            let product = self_val * other_val;

            result_values.push(product);
        }

        Ok(Self {
            values: result_values,
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

pub fn encrypt_str(client_key: &ClientKey, s: &str) -> Result<StringCipherText, &'static str> {
    if !s.is_ascii() {
        return Err("not an ascii character");
    }
    Ok(s.as_bytes()
        .iter()
        .map(|byte| FheUint8::encrypt(*byte, client_key))
        .collect())
}

// Sqrt using binary search
pub fn rsqrt(n: &FheUint8) -> FheUint8 {
    let mut low: FheUint8 = FheUint8::encrypt_trivial(0u8);
    let mut high: FheUint8 = n.clone();
    let mut mid: FheUint8;

    // while low.lt(&high) {
    //     mid = (&high + &low) / 2;
    //     let target = &mid * &mid;
    //     if target.gt(&high).try_decrypt_trivial().unwrap() {
    //         high = mid - 1;
    //     } else if target.lt(n).try_decrypt_trivial().unwrap() {
    //         low = &mid + 1;
    //     } else {
    //         return mid;
    //     }
    // }

    loop {
        mid = (&high + &low) / 2;
        let target = &mid * &mid;
        let greater_than: u8 = target.gt(n).if_then_else(
            &FheUint8::encrypt_trivial(1u8),
            &FheUint8::encrypt_trivial(0u8),
        ).try_decrypt_trivial().unwrap();
        let less_than: u8 = target.lt(n).if_then_else(
            &FheUint8::encrypt_trivial(1u8),
            &FheUint8::encrypt_trivial(0u8),
        ).try_decrypt_trivial().unwrap();
        let equal: u8 = target.eq(n).if_then_else(
            &FheUint8::encrypt_trivial(1u8),
            &FheUint8::encrypt_trivial(0u8),
        ).try_decrypt_trivial().unwrap();
        let low_check: u8 = low.gt(&high).if_then_else(
            &FheUint8::encrypt_trivial(1u8),
            &FheUint8::encrypt_trivial(0u8),
        ).try_decrypt_trivial().unwrap();

        if greater_than != 0 {
            high = &mid - 1;
        } else if less_than != 0 {
            low = &mid + 1;
        } else if equal == 1 {
            return mid;
        }  else if low_check != 0 {
            return high;
        }
    }
}

#[cfg(test)]
mod tests {
    use tfhe::{generate_keys, prelude::FheDecrypt, set_server_key, ConfigBuilder};

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

    // this test is slow cuz multiplication itself is slow
    #[test]
    fn test_mul() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);
        set_server_key(server_key);

        let tensor1 = Tensor::new(encrypt_str(&client_key, "123").unwrap());
        let tensor2 = Tensor::new(encrypt_str(&client_key, "456").unwrap());

        let result = tensor1.mul(&tensor2).unwrap();

        assert_eq!(1, 1);

        let decrypted_values: Vec<u8> = result
            .values
            .iter()
            .map(|val| val.decrypt(&client_key))
            .collect();

        assert_eq!(decrypted_values, vec![4, 10, 18]);
    }

    #[test]
    fn test_rqsqrt() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);

        set_server_key(server_key);
        let a = FheUint8::encrypt_trivial(37u8);
        let res = rsqrt(&a);
        let decrypted: u8 = res.decrypt(&client_key);

        assert_eq!(decrypted, 6u8);
    }
}