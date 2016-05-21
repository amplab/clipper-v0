#![allow(dead_code, unused_variables)]
use rest::InputType;
use std::net::TcpStream;
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write, Cursor};
use std::mem;
use server::{Input, Output};
use features::FeatureReq;



/// Clipper supports 4 input formats: Strings, Ints, Floats, and Bytes.
///
///
///
///
///
/// RPC HEADER format
/// _1 byte: input format_
///
///     - 1: fixed length i32
///     - 2: fixed length f64
///     - 3: fixed length u8
///     - 4: variable length i32
///     - 5: variable length f64
///     - 6: variable length u8
///     - 7: variable length utf-8 encoded string
///
/// _4 bytes: the number of inputs as u32_
///
/// Format specific headers:
/// fixed length formats {1,2,3}:
/// _4 bytes: input length_
/// The input contents sequentially
///
/// Variable length numeric inputs:
/// Each input has the format
/// <4 bytes for the input length, followed by the input>
///
/// Strings:
/// A list of 4 byte i32s indicating the length of each string,
/// followed by all of the strings concatenated together and compressed
/// with LZ4

const FIXEDINT_CODE: u8 = 1;
const FIXEDFLOAT_CODE: u8 = 2;
const FIXEDBYTE_CODE: u8 = 3;
const VARINT_CODE: u8 = 4;
const VARFLOAT_CODE: u8 = 5;
const VARBYTE_CODE: u8 = 6;
const STRING_CODE: u8 = 7;

pub fn send_batch(stream: &mut TcpStream,
                  inputs: &Vec<FeatureReq>,
                  input_type: &InputType)
                  -> Vec<Output> {
    assert!(inputs.len() > 0);
    let message = match input_type {
        &InputType::Integer(l) => {
            if l < 0 {
                encode_var_ints(inputs)
            } else {
                encode_fixed_ints(inputs, l)
            }
        }
        &InputType::Float(l) => {
            if l < 0 {
                encode_var_floats(inputs)
            } else {
                encode_fixed_floats(inputs, l)
            }
        }
        &InputType::Byte(l) => {
            if l < 0 {
                encode_var_bytes(inputs)
            } else {
                encode_fixed_bytes(inputs, l)
            }
        }
        &InputType::Str => encode_strs(inputs),
    };
    stream.write_all(&message[..]).unwrap();
    stream.flush().unwrap();

    let num_response_bytes = inputs.len() * mem::size_of::<Output>();
    let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
    stream.read_exact(&mut response_buffer).unwrap();
    let mut cursor = Cursor::new(response_buffer);
    let mut responses: Vec<Output> = Vec::with_capacity(inputs.len());
    for i in 0..inputs.len() {
        responses.push(cursor.read_f64::<LittleEndian>().unwrap());
    }
    responses
}

fn encode_var_ints(inputs: &Vec<FeatureReq>) -> Vec<u8> {
    // let mut message_len = 5; // 1 byte type + 4 bytes num inputs
    let mut message = Vec::new();
    message.push(VARINT_CODE);
    message.write_u32::<LittleEndian>(inputs.len() as u32).unwrap();
    assert!(message.len() == 5);

    let intsize = mem::size_of::<i32>;
    for x in inputs.iter() {
        match x.input {
            // for each input: 4 bytes length + len*sizeof(int)
            Input::Ints {i: ref f, length: _} => {
                message.write_u32::<LittleEndian>(f.len() as u32).unwrap();
                for xi in f.iter() {
                    message.write_i32::<LittleEndian>(*xi).unwrap();
                }
            }
            _ => unreachable!(),
        }
    }
    message
}

fn encode_fixed_ints(inputs: &Vec<FeatureReq>, length: i32) -> Vec<u8> {
    // let mut message_len = 5; // 1 byte type + 4 bytes num inputs
    let mut message = Vec::new();
    message.push(FIXEDINT_CODE);
    message.write_u32::<LittleEndian>(inputs.len() as u32).unwrap();
    message.write_u32::<LittleEndian>(length as u32).unwrap();
    assert!(message.len() == 9);
    let intsize = mem::size_of::<i32>;
    for x in inputs.iter() {
        match x.input {
            // for each input: 4 bytes length + len*sizeof(int)
            Input::Ints {i: ref f, length: _} => {
                for xi in f.iter() {
                    message.write_i32::<LittleEndian>(*xi).unwrap();
                }
            }
            _ => unreachable!(),
        }
    }
    message
}

fn encode_fixed_floats(inputs: &Vec<FeatureReq>, length: i32) -> Vec<u8> {
    unreachable!()
}

fn encode_var_floats(inputs: &Vec<FeatureReq>) -> Vec<u8> {
    unreachable!()
}

fn encode_fixed_bytes(inputs: &Vec<FeatureReq>, length: i32) -> Vec<u8> {
    unreachable!()
}

fn encode_var_bytes(inputs: &Vec<FeatureReq>) -> Vec<u8> {
    unreachable!()
}

fn encode_strs(inputs: &Vec<FeatureReq>) -> Vec<u8> {
    unreachable!()
}
