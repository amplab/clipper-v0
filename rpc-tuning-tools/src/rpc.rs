
extern crate time;
extern crate byteorder;
use std::net::TcpListener;
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write, Cursor};
use std::mem;
use std::time::Duration;
use std::thread;

const SHUTDOWN_CODE: u8 = 0;
const FIXEDINT_CODE: u8 = 1;
const FIXEDFLOAT_CODE: u8 = 2;
const FIXEDBYTE_CODE: u8 = 3;
const VARINT_CODE: u8 = 4;
const VARFLOAT_CODE: u8 = 5;
const VARBYTE_CODE: u8 = 6;
const STRING_CODE: u8 = 7;


fn main() {
    start_listening();
}

fn start_listening() {
    let listener = TcpListener::bind("0.0.0.0:7777").unwrap();
    let (mut stream, _) = listener.accept().unwrap();
    // let mut response_message: Vec<u8> = Vec::new();
    // for i in 0..100 {
    //     response_message.write_u32::<LittleEndian>(i).unwrap();
    // }
    // assert_eq!(response_message.len(), 100 * mem::size_of::<u32>());
    loop {
        let header_bytes = 2 * mem::size_of::<u32>() + 1;
        let mut header_buffer: Vec<u8> = vec![0; header_bytes];
        stream.read_exact(&mut header_buffer).unwrap();
        let start_time = time::precise_time_ns();
        let mut header_cursor = Cursor::new(header_buffer);
        let code = header_cursor.read_u8().unwrap();
        // println!("read type code");
        assert_eq!(code, FIXEDFLOAT_CODE);
        let num_inputs = header_cursor.read_u32::<LittleEndian>().unwrap() as usize;
        let input_len = header_cursor.read_u32::<LittleEndian>().unwrap() as usize;
        assert_eq!(input_len, 784);
        // println!("read header");

        let payload_bytes = num_inputs * input_len * mem::size_of::<f64>();
        let mut payload_buffer: Vec<u8> = vec![0; payload_bytes];
        stream.read_exact(&mut payload_buffer).unwrap();
        let mut cursor = Cursor::new(payload_buffer);
        let mut inputs = Vec::with_capacity(num_inputs);
        for _ in 0..num_inputs {
            let mut cur_input = Vec::with_capacity(input_len as usize);
            for _ in 0..input_len {
                cur_input.push(cursor.read_f64::<LittleEndian>().unwrap());
            }
            inputs.push(cur_input);
        }
        let end_time = time::precise_time_ns();
        let latency = (end_time - start_time) as f64 / 1000.0;
        println!("Latency {}", latency);
        let mut response_message: Vec<u8> = Vec::new();
        for _ in 0..num_inputs {
            response_message.write_f64::<LittleEndian>(1.0).unwrap();
        }
        // thread::sleep(Duration::from_millis(15));
        stream.write_all(&response_message[..]).unwrap();
        stream.flush();
    }
}
