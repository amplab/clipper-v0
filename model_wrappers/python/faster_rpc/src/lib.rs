#![allow(dead_code)]
extern crate time;
extern crate byteorder;
extern crate libc;
use std::net::{TcpListener, TcpStream};
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write, Cursor};
use std::mem;
use std::slice;
// use std::time::Duration;
// use std::thread;
use std::str;
use std::ffi::CStr;
use libc::{uint32_t, c_char};

const SHUTDOWN_CODE: u8 = 0;
const FIXEDINT_CODE: u8 = 1;
const FIXEDFLOAT_CODE: u8 = 2;
const FIXEDBYTE_CODE: u8 = 3;
const VARINT_CODE: u8 = 4;
const VARFLOAT_CODE: u8 = 5;
const VARBYTE_CODE: u8 = 6;
const STRING_CODE: u8 = 7;


#[repr(C)]
#[derive(Clone, Debug)]
pub struct Header {
    pub code: u8,
    pub num_inputs: u32,
    pub input_len: u32,
}

pub struct ModelWrapperServer {
    listener: TcpListener,
    stream: Option<TcpStream>,
    header: Option<Header>,
}

impl ModelWrapperServer {
    pub fn new(address: &str) -> ModelWrapperServer {

        let mw = ModelWrapperServer {
            listener: TcpListener::bind(address).unwrap(),
            stream: None,
            header: None,
        };
        println!("Starting to serve");
        mw
    }

    /// Blocking call that waits for a new incoming connection
    pub fn wait_for_connection(&mut self) {
        match self.stream {
            Some(_) => println!("Already listening on a stream"),
            None => {
                self.stream = Some(self.listener.accept().unwrap().0);
                println!("Handling new connection");
            }
        };
    }

    /// We split reading header and payload into 2 methods because
    /// the header informs us what the type of the payload is
    pub fn get_next_request_header(&mut self) -> Header {
        let header_bytes = 2 * mem::size_of::<u32>() + 1;
        let mut header_buffer: Vec<u8> = vec![0; header_bytes];
        self.stream.as_ref().unwrap().read_exact(&mut header_buffer).unwrap();
        let mut header_cursor = Cursor::new(header_buffer);
        let code = header_cursor.read_u8().unwrap();
        // self.header = Some(header_buffer);
        // println!("read type code");
        // assert_eq!(code, FIXEDFLOAT_CODE);
        let num_inputs = header_cursor.read_u32::<LittleEndian>().unwrap();
        let input_len = header_cursor.read_u32::<LittleEndian>().unwrap();
        let h = Header {
            code: code,
            num_inputs: num_inputs,
            input_len: input_len,
        };
        self.header = Some(h.clone());
        h

    }

    pub fn get_fixed_floats_payload(&mut self, input_buffer: &mut [f64]) {

        let h = self.header.as_ref().unwrap();
        let total_inputs = (h.num_inputs * h.input_len) as usize;
        let payload_bytes = total_inputs * mem::size_of::<f64>();
        let mut payload_buffer: Vec<u8> = vec![0; payload_bytes];
        self.stream.as_ref().unwrap().read_exact(&mut payload_buffer).unwrap();
        let mut cursor = Cursor::new(payload_buffer);
        assert!(input_buffer.len() == total_inputs);
        for idx in 0..total_inputs {
            input_buffer[idx] = cursor.read_f64::<LittleEndian>().unwrap();
        }
    }

    pub fn send_response(&mut self, response_buffer: &[f64]) {
        let mut response_message: Vec<u8> = Vec::new();
        for i in 0..response_buffer.len() {
            response_message.write_f64::<LittleEndian>(response_buffer[i]).unwrap();
        }
        // thread::sleep(Duration::from_millis(15));
        self.stream.as_ref().unwrap().write_all(&response_message[..]).unwrap();
        self.stream.as_ref().unwrap().flush().unwrap();
        self.header = None;
    }

    pub fn send_shutdown_message(&mut self) {
        println!("Shutting down connection");
        let mut response_message: Vec<u8> = Vec::new();
        response_message.write_u32::<LittleEndian>(1234).unwrap();
        // thread::sleep(Duration::from_millis(15));
        self.stream.as_ref().unwrap().write_all(&response_message[..]).unwrap();
        self.stream.as_ref().unwrap().flush().unwrap();
        self.header = None;
        self.stream = None;
    }
}


#[no_mangle]
pub extern "C" fn init_server(address: *const c_char) -> *mut ModelWrapperServer {
    let addr_cstr = unsafe {
        assert!(!address.is_null());
        CStr::from_ptr(address)
    };
    let addr_str = str::from_utf8(addr_cstr.to_bytes()).unwrap();
    Box::into_raw(Box::new(ModelWrapperServer::new(addr_str)))
}

#[no_mangle]
pub extern "C" fn server_free(ptr: *mut ModelWrapperServer) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
pub extern "C" fn wait_for_connection(ptr: *mut ModelWrapperServer) {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.wait_for_connection();
}

#[no_mangle]
pub extern "C" fn get_next_request_header(ptr: *mut ModelWrapperServer) -> Header {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.get_next_request_header()
}

#[no_mangle]
pub extern "C" fn get_fixed_floats_payload(ptr: *mut ModelWrapperServer,
                                           input_buffer_ptr: *mut f64,
                                           len: uint32_t) {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let input_buffer: &mut [f64] = unsafe {
        assert!(!input_buffer_ptr.is_null());
        let slice: &mut [f64] = slice::from_raw_parts_mut(input_buffer_ptr, len as usize);
        // let mut v = vec![0.0_f64; len as usize];
        // v[..].clone_from_slice(slice);
        slice
    };
    server.get_fixed_floats_payload(input_buffer);
}

#[no_mangle]
pub extern "C" fn send_response(ptr: *mut ModelWrapperServer,
                                response_buffer_ptr: *const f64,
                                len: uint32_t) {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let response_buffer: &[f64] = unsafe {
        assert!(!response_buffer_ptr.is_null());
        let slice: &[f64] = slice::from_raw_parts(response_buffer_ptr, len as usize);
        // let mut v = vec![0.0_f64; len as usize];
        // v[..].clone_from_slice(slice);
        slice
    };
    println!("SENDING RESPONSE: {:?} (Rust)", response_buffer);
    server.send_response(response_buffer);
}

#[no_mangle]
pub extern "C" fn send_shutdown_message(ptr: *mut ModelWrapperServer) {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.send_shutdown_message();
}



// fn start_listening() {
//     let listener = TcpListener::bind("0.0.0.0:7777").unwrap();
//     let (mut stream, _) = listener.accept().unwrap();
//     // let mut response_message: Vec<u8> = Vec::new();
//     // for i in 0..100 {
//     //     response_message.write_u32::<LittleEndian>(i).unwrap();
//     // }
//     // assert_eq!(response_message.len(), 100 * mem::size_of::<u32>());
//     loop {
//         let header_bytes = 2 * mem::size_of::<u32>() + 1;
//         let mut header_buffer: Vec<u8> = vec![0; header_bytes];
//         stream.read_exact(&mut header_buffer).unwrap();
//         let mut header_cursor = Cursor::new(header_buffer);
//         let code = header_cursor.read_u8().unwrap();
//         // println!("read type code");
//         assert_eq!(code, FIXEDFLOAT_CODE);
//         let num_inputs = header_cursor.read_u32::<LittleEndian>().unwrap() as usize;
//         let input_len = header_cursor.read_u32::<LittleEndian>().unwrap() as usize;
//         assert_eq!(input_len, 784);
//         // println!("read header");
//
//         let payload_bytes = num_inputs * input_len * mem::size_of::<f64>();
//         let mut payload_buffer: Vec<u8> = vec![0; payload_bytes];
//         stream.read_exact(&mut payload_buffer).unwrap();
//         let mut cursor = Cursor::new(payload_buffer);
//         let mut inputs = Vec::with_capacity(num_inputs);
//         for _ in 0..num_inputs {
//             let mut cur_input = Vec::with_capacity(input_len as usize);
//             for _ in 0..input_len {
//                 cur_input.push(cursor.read_f64::<LittleEndian>().unwrap());
//             }
//             inputs.push(cur_input);
//         }
//         let mut response_message: Vec<u8> = Vec::new();
//         for _ in 0..num_inputs {
//             response_message.write_f64::<LittleEndian>(1.0).unwrap();
//         }
//         // thread::sleep(Duration::from_millis(15));
//         stream.write_all(&response_message[..]).unwrap();
//         stream.flush();
//     }
// }
