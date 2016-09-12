extern crate time;
extern crate byteorder;
extern crate rand;

use std::net::TcpStream;
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write, Cursor};
use std::mem;
use rand::{thread_rng, Rng};
use std::env;
use std::slice;

const SHUTDOWN_CODE: u8 = 0;
// const FIXEDINT_CODE: u8 = 1;
const FIXEDFLOAT_CODE: u8 = 2;
// const FIXEDBYTE_CODE: u8 = 3;
// const VARINT_CODE: u8 = 4;
// const VARFLOAT_CODE: u8 = 5;
// const VARBYTE_CODE: u8 = 6;
// const STRING_CODE: u8 = 7;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    let num_messages = args[1].parse::<usize>().unwrap();
    // let message_size_bytes = args[2].parse::<usize>().unwrap();
    // let (duration, latencies) = send_messages(num_messages, message_size_bytes);
    // let bytes_proc = (num_messages * (message_size_bytes + 4)) as f64;

    let (duration, batch_lat_tracker, ser_lat_tracker, send_lat_tracker, bp) =
        send_clipper_messages(num_messages, 3072, 128);
    let bytes_proc = bp as f64;

    let duration_sec = duration as f64 / (1000.0 * 1000.0 * 1000.0);
    let mb_proc = bytes_proc / (1000.0 * 1000.0);
    let thru = mb_proc / duration_sec;
    let mean_batch_lat = compute_mean_lat_micros(&batch_lat_tracker);
    let mean_ser_lat = compute_mean_lat_micros(&ser_lat_tracker);
    let mean_send_lat = compute_mean_lat_micros(&send_lat_tracker);
    // let mean_lat = latencies.iter().fold(0, |acc, &x| acc + x) as f64 / (latencies.len() as f64) /
    //                (1000.0);

    println!("Processed {} bytes in {} seconds. Throughput: {} MBps, recorded batches: {},\n\
             mean batch latency: {} us (end to end latency)\n\
             mean ser latency:   {} us (message serialization time)\n\
             mean send latency:  {} us (just sending and receiving bytes)\n",
             bytes_proc,
             duration_sec,
             thru,
             batch_lat_tracker.len(),
             mean_batch_lat,
             mean_ser_lat,
             mean_send_lat);

}

fn compute_mean_lat_micros(latencies: &Vec<u64>) -> f64 {
    latencies.iter().fold(0, |acc, &x| acc + x) as f64 / (latencies.len() as f64) / (1000.0)
}


fn send_clipper_messages(num_messages: usize,
                         input_len: usize,
                         batch_size: usize)
                         -> (u64, Vec<u64>, Vec<u64>, Vec<u64>, u64) {
    let mut inputs = Vec::new();
    let mut rng = thread_rng();
    for _ in 0..batch_size {
        inputs.push(rng.gen_iter::<f64>().take(input_len).collect::<Vec<f64>>());
    }

    // let message = encode_fixed_floats(&inputs);

    let mut total_bytes_sent: u64 = 0;

    let expected_response: Vec<f64> = vec![1.0; inputs.len()];
    // let expected_response = (0..100).collect::<Vec<u32>>();

    let mut stream: TcpStream = TcpStream::connect("127.0.0.1:6001").unwrap();
    stream.set_nodelay(true).unwrap();
    let mut batch_lat_tracker = Vec::new();
    let mut ser_lat_tracker = Vec::new();
    let mut send_lat_tracker = Vec::new();

    let bench_start_time = time::precise_time_ns();
    for m in 0..num_messages {
        if m % 500 == 0 {
            println!("Sent {} messages", m);
        }
        let start_time = time::precise_time_ns();
        let (message, ser_duration) = fast_encode_fixed_floats(&inputs);
        let send_start = time::precise_time_ns();
        stream.write_all(&message[..]).unwrap();
        stream.flush().unwrap();
        total_bytes_sent += message.len() as u64;
        // let num_response_bytes = 100 * mem::size_of::<u32>();
        let num_response_bytes = inputs.len() * mem::size_of::<f64>();
        let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
        stream.read_exact(&mut response_buffer).unwrap();
        let send_end = time::precise_time_ns();
        // let mut cursor = Cursor::new(response_buffer);
        let mut responses: Vec<f64> = Vec::with_capacity(inputs.len());
        let res_ptr = response_buffer.as_ptr();
        let f64_amount = inputs.len();

        let res_buffer_f64: &[f64] =
            unsafe { slice::from_raw_parts(res_ptr as *const f64, f64_amount) };
        responses.extend_from_slice(res_buffer_f64);

        // for _ in 0..inputs.len() {
        //     responses.push(cursor.read_f64::<LittleEndian>().unwrap());
        // }
        // assert_eq!(responses, expected_response);
        let end_time = time::precise_time_ns();
        batch_lat_tracker.push(end_time - start_time);
        ser_lat_tracker.push(ser_duration);
        send_lat_tracker.push(send_end - send_start);
    }
    let bench_end_time = time::precise_time_ns();

    {
        let mut message = Vec::new();
        message.push(SHUTDOWN_CODE);
        message.write_u32::<LittleEndian>(0 as u32).unwrap();
        message.write_u32::<LittleEndian>(0 as u32).unwrap();
        stream.write_all(&message[..]).unwrap();
        stream.flush().unwrap();
        let mut response_buffer: Vec<u8> = vec![0; 4];
        stream.read_exact(&mut response_buffer).unwrap();
        let mut cursor = Cursor::new(response_buffer);
        let response = cursor.read_u32::<LittleEndian>().unwrap();
        // info!("SHUTDOWN RESPONSE: {}", response);
        assert_eq!(response, 1234_u32);
    }
    (bench_end_time - bench_start_time,
     batch_lat_tracker,
     ser_lat_tracker,
     send_lat_tracker,
     total_bytes_sent)
}

fn fast_encode_fixed_floats(inputs: &Vec<Vec<f64>>) -> (Vec<u8>, u64) {
    let ser_start = time::precise_time_ns();
    let length = inputs[0].len();
    let num_bytes_total = 9 + inputs.len() * length * mem::size_of::<f64>();
    let mut message: Vec<u8> = Vec::with_capacity(num_bytes_total);
    // message.write_u32::<LittleEndian>((inputs.len() * 784 * 8) as u32 + 9).unwrap();
    message.push(FIXEDFLOAT_CODE);
    message.write_u32::<LittleEndian>(inputs.len() as u32).unwrap();
    message.write_u32::<LittleEndian>(length as u32).unwrap();
    // assert!(message.len() == 9);
    // let floatsize = mem::size_of::<f64>;
    for x in inputs.iter() {
        let ptr = x.as_ptr();
        let u8_amount = x.len() * mem::size_of::<f64>();
        let x_u8: &[u8] = unsafe { slice::from_raw_parts(ptr as *const u8, u8_amount) };
        message.extend_from_slice(x_u8);

        // for xi in x.iter() {
        //     message.write_f64::<LittleEndian>(*xi).unwrap();
        // }
    }
    let ser_end = time::precise_time_ns();
    (message, ser_end - ser_start)
}

fn baseline_encode_fixed_floats(inputs: &Vec<Vec<f64>>) -> (Vec<u8>, u64) {
    let ser_start = time::precise_time_ns();
    let length = 784;
    let mut message = Vec::new();
    // message.write_u32::<LittleEndian>((inputs.len() * 784 * 8) as u32 + 9).unwrap();
    message.push(FIXEDFLOAT_CODE);
    message.write_u32::<LittleEndian>(inputs.len() as u32).unwrap();
    message.write_u32::<LittleEndian>(length as u32).unwrap();
    // assert!(message.len() == 9);
    // let floatsize = mem::size_of::<f64>;
    for x in inputs.iter() {
        for xi in x.iter() {
            message.write_f64::<LittleEndian>(*xi).unwrap();
        }
    }
    let ser_end = time::precise_time_ns();
    (message, ser_end - ser_start)
}





#[allow(dead_code)]
fn send_messages(num_messages: usize, message_size_bytes: usize) -> (u64, Vec<u64>) {
    let mut stream: TcpStream = TcpStream::connect("127.0.0.1:7777").unwrap();
    stream.set_nodelay(true).unwrap();
    let expected_response = (0..100).collect::<Vec<u32>>();
    let mut lat_tracker = Vec::new();
    let message = gen_message(message_size_bytes);

    let bench_start_time = time::precise_time_ns();
    for m in 0..num_messages {
        if m % 200 == 0 {
            println!("Sent {} messages", m);
        }
        let start_time = time::precise_time_ns();
        // let message = gen_message(message_size_bytes);
        println!("Message length: {}", message.len());
        println!("Payload length: {}", message.len() - 4);
        stream.write_all(&message[..]).unwrap();
        stream.flush().unwrap();
        let num_response_bytes = 100 * mem::size_of::<u32>();
        let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
        stream.read_exact(&mut response_buffer).unwrap();
        let mut cursor = Cursor::new(response_buffer);
        let mut responses: Vec<u32> = Vec::with_capacity(100);
        for _ in 0..100 {
            responses.push(cursor.read_u32::<LittleEndian>().unwrap());
        }
        assert_eq!(responses, expected_response);
        let end_time = time::precise_time_ns();
        lat_tracker.push(end_time - start_time);
    }
    let bench_end_time = time::precise_time_ns();
    (bench_end_time - bench_start_time, lat_tracker)
}

#[allow(dead_code)]
fn gen_message(size: usize) -> Vec<u8> {
    let mut message = Vec::new();
    message.write_u32::<LittleEndian>(size as u32).unwrap();
    let mut rng = thread_rng();
    // let x = rng.gen_iter::<u8>().take(size).collect::<Vec<u8>>();
    message.extend(rng.gen_iter::<u8>().take(size));
    assert!(message.len() == size + 4);
    message
}
