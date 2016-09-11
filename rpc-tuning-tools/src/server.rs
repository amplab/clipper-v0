
extern crate time;
extern crate byteorder;
use std::net::TcpListener;
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write, Cursor};
use std::mem;
use std::time::Duration;
use std::thread;

fn main() {
    start_listening();
}

fn start_listening() {
    let listener = TcpListener::bind("127.0.0.1:7777").unwrap();
    let (mut stream, _) = listener.accept().unwrap();
    let mut response_message: Vec<u8> = Vec::new();
    for i in 0..100 {
        response_message.write_u32::<LittleEndian>(i).unwrap();
    }
    assert_eq!(response_message.len(), 100 * mem::size_of::<u32>());
    loop {
        let header_bytes = mem::size_of::<u32>();
        let mut header_buffer: Vec<u8> = vec![0; header_bytes];
        stream.read_exact(&mut header_buffer).unwrap();
        let mut cursor = Cursor::new(header_buffer);
        let payload_bytes = cursor.read_u32::<LittleEndian>().unwrap() as usize;
        println!("Expecting payload length: {}", payload_bytes);
        let mut payload_buffer: Vec<u8> = vec![0; payload_bytes];
        stream.read_exact(&mut payload_buffer).unwrap();
        println!("stream reading finished");
        // thread::sleep(Duration::from_millis(15));
        stream.write_all(&response_message[..]).unwrap();
        stream.flush();
        println!("responding with {} bytes", response_message.len());
    }
}
