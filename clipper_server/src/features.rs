
// use gj;
// use gj::{EventLoop, Promise};
// use gj::io;
// use capnp;
// use std::time::{Duration, PreciseTime};
use time;
// use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use std::net::{ToSocketAddrs, SocketAddr, TcpStream};
// use feature_capnp::feature;
// use capnp::{primitive_list, message};
use std::{thread, mem, slice};
use std::sync::{RwLock, Arc};
use std::sync::mpsc;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use num_cpus;
use linear_models::linalg;
use digits;
use std::hash::{Hash, SipHasher, Hasher};
use std::io::{Read, Write};
// use std::net::{self};
use net2::{TcpBuilder, TcpStreamExt};
use byteorder::{LittleEndian, WriteBytesExt};

pub type HashKey = u64;

pub struct FeatureReq {
    pub hash_key: HashKey,
    pub input: Vec<f64>,
    pub req_start_time: time::PreciseTime,
}

#[derive(Clone)]
pub struct FeatureHandle<H: FeatureHash + Send + Sync> {
    // addr: SocketAddr,
    pub name: String,
    pub queue: mpsc::Sender<FeatureReq>,
    // TODO: need a better concurrent hashmap: preferable lock free wait free
    // This should actually be reasonably simple, because we don't need to resize
    // (fixed size cache) and things never get evicted. Neither of these is strictly
    // true but it's a good approximation for now.
    pub cache: Arc<RwLock<HashMap<HashKey, f64>>>,
    pub hasher: Arc<H>,
    pub latencies: Arc<RwLock<Vec<i64>>>
    // thread_handle: ::std::thread::JoinHandle<()>,
}


pub trait FeatureHash {
    fn hash(&self, input: &Vec<f64>, salt: Option<i32>) -> HashKey;
}

#[derive(Clone)]
pub struct SimpleHasher;

impl FeatureHash for SimpleHasher {
    fn hash(&self, input: &Vec<f64>, salt: Option<i32>) -> HashKey {
        // lame way to get around rust's lack of floating point equality. Basically,
        // we need a better hash function.
        let mut int_vec: Vec<i32> = Vec::new();
        for i in input {
            let iv = (i * 10000000.0) as i32;
            int_vec.push(iv);
        }
        if salt.is_some() {
            int_vec.push(salt.unwrap());
        }
        let mut s = SipHasher::new();
        int_vec.hash(&mut s);
        s.finish()
    }
}


// pub struct LocalitySensitiveHash {
//     hash_size: i32
// }
//
// impl FeatureHash for LocalitySensitiveHash {
//     fn hash(&self, input: &Vec<f64>) -> u64 {
//         
//     }
// }


pub fn feature_batch_latency(batch_size: usize) {
    let mnist_path = "/crankshaw-local/mnist/data/test.data".to_string();
    let all_test_data = digits::load_mnist_dense(&mnist_path).unwrap();

    let addr_vec = vec!["127.0.0.1:6001".to_string()];
    let names = vec!["pyspark-svm".to_string()];
    let (mut features, mut handles): (Vec<_>, Vec<_>) = addr_vec.into_iter().map(|a| get_addr(a))
                    .zip(names.into_iter())
                    .map(|(a, n)| create_feature_worker(n, a, batch_size))
                    .unzip();

    assert!(features.len() == 1);
    let feat = features.pop().unwrap();
    let hand = handles.pop().unwrap();

    let mut rng = thread_rng();
    let num_trials = 10000;
    let num_reqs = num_trials * batch_size;
    for i in 0..num_reqs {
        let example_idx: usize = rng.gen_range(0, all_test_data.xs.len());
        let input = (*all_test_data.xs[example_idx]).clone();
        let req = FeatureReq {
            hash_key: 11,
            input: input,
            req_start_time: time::PreciseTime::now(),
        };
        feat.queue.send(req).unwrap();
    }

    thread::sleep(::std::time::Duration::new(20, 0));

    let l = feat.latencies.read().unwrap();
    let mut avg_t: f64 = 0.0;
    let mut avg_l: f64 = 0.0;
    for i in l.iter() {
        avg_t += (batch_size as f64) / (*i as f64) * 1000.0 * 1000.0;
        avg_l += *i as f64 / 1000.0;
    }
    println!("batch size: {}, trials: {}, average thruput (pred/s): {}, average lat (ms): {}",
            batch_size,
             l.len(),
             avg_t / (l.len() as f64),
             avg_l / (l.len() as f64));

    hand.join().unwrap();
}


pub fn create_feature_worker(name: String, addr: SocketAddr,
                             batch_size: usize)
    -> (FeatureHandle<SimpleHasher>, ::std::thread::JoinHandle<()>) {

    let (tx, rx) = mpsc::channel();

    let feature_cache: Arc<RwLock<HashMap<HashKey, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    let latencies: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
    let handle = {
        let thread_cache = feature_cache.clone();
        let latencies = latencies.clone();
        let name = name.clone();
        thread::spawn(move || {
            feature_worker(name, rx, thread_cache, addr, latencies, batch_size);
        })
    };
    (FeatureHandle {
        name: name.clone(),
        queue: tx,
        cache: feature_cache,
        hasher: Arc::new(SimpleHasher),
        latencies: latencies
        // thread_handle: handle,
    }, handle)
}

pub fn get_addr(a: String) -> SocketAddr {
    a.to_socket_addrs().unwrap().next().unwrap()
}


fn update_batch_size(cur_batch: usize, cur_time_micros: u64, max_time_micros: u64) -> usize {
    let batch_increment = 2;
    if cur_time_micros < (max_time_micros - 3000) {
        cur_batch + batch_increment
    } else if cur_time_micros < max_time_micros {
        cur_batch
    } else {
        cur_batch / 2
    }
}

fn feature_worker(name: String,
                  rx: mpsc::Receiver<FeatureReq>,
                  cache: Arc<RwLock<HashMap<HashKey, f64>>>,
                  address: SocketAddr,
                  latencies: Arc<RwLock<Vec<i64>>>,
                  batch_size: usize) {

    // println!("starting worker: {}", name);
    let mut stream: TcpStream = TcpStream::connect(address).unwrap();
    stream.set_nodelay(true).unwrap();
    stream.set_read_timeout(None).unwrap();
    let max_batch_size = batch_size;
    let mut cur_batch_size = 1;
    // let mut bench_latencies = Vec::new();
    // let mut loop_counter = 0;
    // let mut epoch_start = time::PreciseTime::now();
    // let mut epoch_count = 0;

    loop {
        let mut batch: Vec<FeatureReq> = Vec::new();
        // block until new request, then try to get more requests
        let first_req = rx.recv().unwrap();
        batch.push(first_req);
        let start_time = time::PreciseTime::now();
        // while batch.len() < cur_batch_size {
        while batch.len() < max_batch_size {
            if let Ok(req) = rx.try_recv() {
                // let req_latency = req.req_start_time.to(time::PreciseTime::now()).num_microseconds().unwrap();
                // println!("req->features latency {} (ms)", (req_latency as f64 / 1000.0));
                batch.push(req);
            } else {
                break;
            }
        }
        assert!(batch.len() > 0);
        // send batch
        let mut header_wtr: Vec<u8> = vec![];
        header_wtr.write_u16::<LittleEndian>(batch.len() as u16).unwrap();
        stream.write_all(&header_wtr).unwrap();
        for r in batch.iter() {
            stream.write_all(floats_to_bytes(r.input.clone())).unwrap();
        }
        stream.flush();

        // read response: assumes 1 f64 for each entry in batch
        let num_response_bytes = batch.len()*mem::size_of::<f64>();
        let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
        stream.read_exact(&mut response_buffer).unwrap();
        // make immutable
        let response_buffer = response_buffer;
        let response_floats = bytes_to_floats(&response_buffer);
        let end_time = time::PreciseTime::now();
        let latency = start_time.to(end_time).num_microseconds().unwrap();
        let max_req_latency = batch.first().unwrap().req_start_time.to(end_time).num_microseconds().unwrap();
        let min_req_latency = batch.last().unwrap().req_start_time.to(end_time).num_microseconds().unwrap();
        if latency > 20*1000 {
            println!("latency: {}, batch size: {}", (latency as f64 / 1000.0), batch.len());
        }
        // only try to increase the batch size if we actually sent a batch of maximum size
        if batch.len() == cur_batch_size {
            cur_batch_size = update_batch_size(cur_batch_size, latency as u64, 20*1000);
            println!("{} updated batch size to {}", name, cur_batch_size);
        }

        let mut l = latencies.write().unwrap();
        l.push(latency);
        let mut w = cache.write().unwrap();
        for r in 0..batch.len() {
            let hash = batch[r].hash_key;
            if !w.contains_key(&hash) {
                w.insert(hash, response_floats[r]);
            } else {
                // println!("CACHE HIT");
                // let existing_res = w.get(&hash).unwrap();
                // if result != *existing_res {
                //     // println!("{} CACHE ERR: existing: {}, new: {}",
                //     //          name, existing_res, result);
                // } else {
                //     println!("{} CACHE HIT", name);
                // }
            }
        }
        // let loop_end_time = time::PreciseTime::now();
        // let loop_latency = start_time.to(loop_end_time).num_microseconds().unwrap() as f64 / 1000.0;
        // bench_latencies.push(loop_latency);
        // epoch_count += batch.len();
        // if epoch_count >= 10000 {
        //     let epoch_end = time::PreciseTime::now();
        //     let epoch_time = epoch_start.to(epoch_end).num_microseconds().unwrap();
        //     // let xs = vec![Arc::new(bench_latencies.clone())];
        //     let xs = bench_latencies.clone().into_iter().map(|x| {
        //         Arc::new(vec![x])
        //     }).collect::<Vec<Arc<Vec<f64>>>>();
        //     let (mut mean, mut var) = linalg::mean_and_var(&xs);
        //     // println!("mean: {:?}, var: {:?}", mean, var);
        //     assert!(mean.len() == 1 && var.len() == 1);
        //     let max_lat = bench_latencies.iter().fold(0.0, |m, &x| {
        //         if x > m {
        //             x
        //         } else {
        //             m
        //         }
        //     });
        //     println!("batch_size: {}, thru: {:.3} (qps), mean_lat (ms): {}, var_lat (ms): {}, max_lat (ms): {}",
        //              batch_size,
        //              epoch_count as f64 / epoch_time as f64 * 1000.0 * 1000.0,
        //              mean.pop().unwrap(), var.pop().unwrap(), max_lat);
        //     epoch_start = time::PreciseTime::now();
        //     epoch_count = 0;
        //     bench_latencies.clear();
        // }
        // loop_counter += 1;
        
        // println!("feature: {}, batch_size: {}, latency: {}, max_req_latency: {}, min_req_latency {}, loop_latency: {}",
        //          name, batch.len(), (latency as f64 / 1000.0),
        //         (max_req_latency as f64 / 1000.0),
        //         (min_req_latency as f64 / 1000.0),
        //         (loop_latency as f64 / 1000.0)
        //         );
    }
}


// TODO: this is super broken, and for some reason the
// 
fn floats_to_bytes<'a>(v: Vec<f64>) -> &'a [u8] {
    let byte_arr: &[u8] = unsafe {
        let float_ptr: *const f64 = v[..].as_ptr();
        let num_elems = v.len()*mem::size_of::<f64>();
        slice::from_raw_parts(float_ptr as *const u8, num_elems)
    };
    byte_arr
}

// fn ints_to_bytes<'a>(v: Vec<u32>) -> &'a [u8] {
//     let byte_arr: &[u8] = unsafe {
//         let int_ptr: *const u32 = v[..].as_ptr();
//         let num_elems = v.len()*mem::size_of::<u32>();
//         slice::from_raw_parts(int_ptr as *const u8, num_elems)
//     };
//     byte_arr
// }

fn bytes_to_floats(bytes: &[u8]) -> &[f64] {
    let float_arr: &[f64] = unsafe {
        let byte_ptr: *const u8 = bytes.as_ptr();
        let num_elems = bytes.len() / mem::size_of::<f64>();
        slice::from_raw_parts(byte_ptr as *const f64, num_elems)
    };
    float_arr
}





// pub fn feature_lats_main(feature_addrs: Vec<(String, SocketAddr)>) {
//
//     // let addr_vec = vec!["127.0.0.1:6001".to_string(), "127.0.0.1:6002".to_string(), "127.0.0.1:6003".to_string()];
//     // let names = vec!["TEN_rf".to_string(), "HUNDRED_rf".to_string(), "FIVE_HUNDO_rf".to_string()];
//     // let addr_vec = (1..11).map(|i| format!("127.0.0.1:600{}", i)).collect::<Vec<String>>();
//     // let names = (1..11).map(|i| format!("Spark_predictor_{}", i)).collect::<Vec<String>>();
//     // let addr_vec = vec!["169.229.49.167:6001".to_string()];
//     // let names = vec!["CAFFE on c67 with GPU".to_string()];
//     let num_features = feature_addrs.len();
//     let handles: Vec<::std::thread::JoinHandle<()>> =
//             feature_addrs.into_iter().map(|(n, a)| create_blocking_features(n, a)).collect();
//
//     for h in handles {
//         h.join().unwrap();
//     }
//     // handle.join().unwrap();
//     println!("done");
// }
//
// fn create_blocking_features(name: String, address: SocketAddr)
//                             -> ::std::thread::JoinHandle<()> {
//     // let name = name.clone();
//     println!("{}", name);
//     thread::spawn(move || {
//         blocking_feature_requests(name, address);
//     })
// }
//
// fn blocking_feature_requests(name: String, address: SocketAddr) {
//
//     let num_events = 5000;
//     EventLoop::top_level(move |wait_scope| {
//         let (reader, writer) = try!(::gj::io::tcp::Stream::connect(address).wait(wait_scope))
//             .split();
//         let network = Box::new(twoparty::VatNetwork::new(reader,
//                                                          writer,
//                                                          rpc_twoparty_capnp::Side::Client,
//                                                          Default::default()));
//         let mut rpc_system = RpcSystem::new(network, None);
//         let feature_rpc: feature::Client = rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
//         println!("rpc connection established");
//         // feature_send_loop(name, feature_rpc, rx).lift()
//
//         for _ in 0..num_events {
//
//             let start_time = time::PreciseTime::now();
//             let feature_vec = random_features(784);
//
//             // println!("sending {} reqs", new_features.len());
//             // println!("sending request");
//
//             let mut request = feature_rpc.compute_feature_request();
//             {
//                 let mut builder = request.get();
//                 let mut inp_entries = builder.init_inp(feature_vec.len() as u32);
//                 for i in 0..feature_vec.len() {
//                     inp_entries.set(i as u32, feature_vec[i]);
//                 }
//             }
//
//             // request.get().set_inp(message.get_root::<primitive_list::Builder>().as_reader());
//             
//             let name = name.clone();
//             try!(request.send().promise.then(move |response| {
//                 let res = pry!(response.get()).get_result();
//                 // let result = response.get().unwrap().get_result();
//                 let end_time = time::PreciseTime::now();
//                 let latency = start_time.to(end_time).num_milliseconds();
//                 println!("got response: {} from {} in {} ms, putting in cache",
//                          res,
//                          name,
//                          latency);
//                 // feature_send_loop(name, feature_rpc, rx)
//                 Promise::ok(())
//             }).wait(wait_scope));
//             // thread::sleep(::std::time::Duration::new(1, 0));
//         }
//         println!("done with requests");
//         Ok(())
//     })
//     .expect("top level error");
// }

pub fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}



