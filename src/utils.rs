use tfhe::{
    prelude::{FheEncrypt, FheOrd, FheTrivialEncrypt, IfThenElse},
    ClientKey, FheUint8,
};

pub type StringCipherText = Vec<FheUint8>;
pub type Vec4D = Vec<Vec<Vec<Vec<FheUint8>>>>;

pub fn encrypt_str(client_key: &ClientKey, s: &str) -> Result<StringCipherText, &'static str> {
    if !s.is_ascii() {
        return Err("not an ascii character");
    }
    Ok(s.as_bytes()
        .iter()
        .map(|byte| FheUint8::encrypt(*byte, client_key))
        .collect())
}

// Integer square root using Newton's method (https://math.mit.edu/~stevenj/18.335/newton-sqrt.pdf)
pub fn isqrt(n: &FheUint8) -> FheUint8 {
    let mut x = n.clone();
    let mut y = (&x + 1) / 2;
    let mut less_than: u8 = (&y)
        .lt(&x)
        .if_then_else(
            &FheUint8::encrypt_trivial(1u8),
            &FheUint8::encrypt_trivial(0u8),
        )
        .try_decrypt_trivial()
        .unwrap();

    while less_than == 1 {
        x = y.clone();
        y = (&x + (n / &x)) / 2;

        less_than = (&y)
            .lt(&x)
            .if_then_else(
                &FheUint8::encrypt_trivial(1u8),
                &FheUint8::encrypt_trivial(0u8),
            )
            .try_decrypt_trivial()
            .unwrap();
    }

    x
}

#[cfg(test)]
mod tests {
    use tfhe::{generate_keys, prelude::FheDecrypt, set_server_key, ConfigBuilder};

    use super::*;

    #[test]
    fn test_rqsqrt() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);

        set_server_key(server_key);
        let a = FheUint8::encrypt_trivial(100u8);
        let res = isqrt(&a);
        let decrypted: u8 = res.decrypt(&client_key);

        assert_eq!(decrypted, 10u8);
    }
}
