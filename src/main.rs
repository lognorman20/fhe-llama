use tfhe::{prelude::*, FheBool, FheUint, FheUint8Id};
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint8};
use utils::{encrypt_str, rsqrt};

use crate::utils::Tensor;

mod utils;
mod llama;

/// OUTLINE ////////////////////////////////////////////////////////////////////
// Take in the user's input as a string
// Encrypt the user's text into a vector of characters (unnormalized characters)
// Pass the encrypted vector into rmsnorm
// Pass the normalized vector into the projection function
// Return the encrypted projection to be decrypted
///////////////////////////////////////////////////////////////////////////////

fn int_example() {
    let config = ConfigBuilder::default().build();

    // Client-side
    let (client_key, server_key) = generate_keys(config);

    let clear_a = 3u8;
    let clear_b = 12u8;

    let a = FheUint8::encrypt(clear_a, &client_key);
    let b = FheUint8::encrypt(clear_b, &client_key);

    //Server-side
    set_server_key(server_key);
    let result = a * b;

    //Client-side
    let decrypted_result: u8 = result.decrypt(&client_key);

    let clear_result = clear_a * clear_b;

    assert_eq!(decrypted_result, clear_result);
}

fn main() {
    let config = ConfigBuilder::default().build();

    // // Client-side
    let (client_key, server_key) = generate_keys(config);
    // let cipher = encrypt_str(&client_key, "SPOTEMGOTEM").unwrap();
    // let t = Tensor::new(cipher);

    set_server_key(server_key);
    let a = FheUint8::encrypt_trivial(4u8);
    let res = rsqrt(&a);
    let decrypted: u8 = res.decrypt(&client_key);

    println!("{:?}", decrypted);
}
