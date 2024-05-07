use tfhe::{prelude::FheTrivialEncrypt, FheUint8};

use crate::utils::Tensor;

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

        todo!()
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }
}
