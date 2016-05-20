use time;
use std::net::{ToSocketAddrs, SocketAddr, TcpStream};
use std::{thread, mem, slice};
use std::sync::{RwLock, Arc};
use std::sync::mpsc;
use std::collections::HashMap;
use std::cmp;
use rand::{thread_rng, Rng};
use std::sync::atomic::AtomicUsize;
use toml;
use server;
// use std::hash::{Hash, SipHasher, Hasher};
use std::io::{Read, Write};
// use std::net::{self};
use net2::TcpStreamExt;
use byteorder::{LittleEndian, WriteBytesExt};
use metrics;
use hashing::{FeatureHash, SimpleHasher, HashKey};


pub struct FeatureReq {
    pub hash_key: HashKey,
    pub input: server::Input,
    pub req_start_time: time::PreciseTime,
}

#[derive(Clone)]
pub struct FeatureHandle<H: FeatureHash + Send + Sync> {
    pub name: String,
    // TODO: need a better concurrent hashmap: preferable lock free wait free
    // This should actually be reasonably simple, because we don't need to resize
    // (fixed size cache) and things never get evicted. Neither of these is strictly
    // true but it's a good approximation for now.
    pub cache: Arc<RwLock<HashMap<HashKey, f64>>>,
    pub hasher: Arc<H>,
    queues: Vec<mpsc::Sender<FeatureReq>>,
    next_instance: Arc<AtomicUsize>,
}




impl<H: FeatureHash + Send + Sync> FeatureHandle<H> {
    pub fn request_feature(&self, req: FeatureReq) {
        // let inst = self.next_instance.fetch_add(1, Ordering::Relaxed) % self.queues.len();
        let inst = req.hash_key as usize % self.queues.len();
        self.queues[inst].send(req).unwrap();
    }
}

pub fn create_feature_worker(name: String,
                             addrs: Vec<SocketAddr>,
                             batch_size: usize,
                             metric_register: Arc<RwLock<metrics::Registry>>)
                             -> (FeatureHandle<SimpleHasher>,
                                 Vec<::std::thread::JoinHandle<()>>) {

    let latency_hist: Arc<metrics::Histogram> = {
        let metric_name = format!("{}_faas_latency", name);
        metric_register.write().unwrap().create_histogram(metric_name, 2056)
    };

    let thruput_meter: Arc<metrics::Meter> = {
        let metric_name = format!("{}_faas_thruput", name);
        metric_register.write().unwrap().create_meter(metric_name)
    };

    let predictions_counter: Arc<metrics::Counter> = {
        let metric_name = format!("{}_prediction_counter", name);
        metric_register.write().unwrap().create_counter(metric_name)
    };

    let feature_cache: Arc<RwLock<HashMap<HashKey, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    // let latencies: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
    let mut handles = Vec::new();
    let mut queues = Vec::new();
    for a in addrs.iter() {
        let (tx, rx) = mpsc::channel();
        queues.push(tx);
        let handle = {
            // cache is shared
            let thread_cache = feature_cache.clone();
            // latency tracking is shared
            // let latencies = latencies.clone();
            let name = name.clone();
            let addr = a.clone();
            let latency_hist = latency_hist.clone();
            let thruput_meter = thruput_meter.clone();
            let predictions_counter = predictions_counter.clone();
            thread::spawn(move || {
                feature_worker(name,
                               rx,
                               thread_cache,
                               addr,
                               latency_hist,
                               thruput_meter,
                               predictions_counter,
                               batch_size);
            })
        };
        handles.push(handle);
    }
    info!("Creating feature worker with {} replicas", queues.len());
    (FeatureHandle {
        name: name.clone(),
        queues: queues,
        cache: feature_cache,
        hasher: Arc::new(SimpleHasher),
        next_instance: Arc::new(AtomicUsize::new(0)), // thread_handle: handle,
    },
     handles)
}

pub fn get_addr(a: String) -> SocketAddr {
    a.to_socket_addrs().unwrap().next().unwrap()
}

pub fn get_addrs(addrs: Vec<toml::Value>) -> Vec<SocketAddr> {
    addrs.into_iter().map(|a| get_addr(a.as_str().unwrap().to_string())).collect::<Vec<_>>()
    // a.to_socket_addrs().unwrap().next().unwrap()
}

pub fn get_addrs_str(addrs: Vec<String>) -> Vec<SocketAddr> {
    addrs.into_iter().map(|a| get_addr(a)).collect::<Vec<_>>()
    // a.to_socket_addrs().unwrap().next().unwrap()
}


fn update_batch_size(cur_batch: usize, cur_time_micros: u64, max_time_micros: u64) -> usize {
    let batch_increment = 2;
    let backoff = 0.9;
    let epsilon = (0.1 * max_time_micros as f64).ceil() as u64;
    if cur_time_micros < (max_time_micros - epsilon) {
        let new_batch = cur_batch + batch_increment;
        debug!("increasing batch to {}", new_batch);
        new_batch as usize
    } else if cur_time_micros < max_time_micros {
        cur_batch
    } else {
        // don't try to set the batch size below 1
        let new_batch = cmp::max((cur_batch as f64 * backoff).floor() as u64, 1);
        debug!("decreasing batch to {}", new_batch);
        new_batch as usize
    }
}

fn feature_worker(name: String,
                  rx: mpsc::Receiver<FeatureReq>,
                  cache: Arc<RwLock<HashMap<HashKey, f64>>>,
                  address: SocketAddr,
                  latency_hist: Arc<metrics::Histogram>,
                  thruput_meter: Arc<metrics::Meter>,
                  predictions_counter: Arc<metrics::Counter>,
                  batch_size: usize) {

    // if the batch_size is less than 1 (these are unsigned
    // integers, so that means batch size == 0), we assume dynamic batching
    let dynamic_batching = batch_size < 1;
    if dynamic_batching {
        info!("using dynamic batch size for {}", name);
    }
    // println!("starting worker: {}", name);
    let mut stream: TcpStream = TcpStream::connect(address).unwrap();
    stream.set_nodelay(true).unwrap();
    stream.set_read_timeout(None).unwrap();
    // let max_batch_size = batch_size;
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
        let max_batch_size = if dynamic_batching {
            cur_batch_size
        } else {
            batch_size
        };
        assert!(max_batch_size >= 1);
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
            match r.input {
                server::Input::Floats { ref f, length: _ } => {
                    stream.write_all(floats_to_bytes(f.clone())).unwrap()
                }
                _ => panic!("unimplemented input type"),
            }
        }
        stream.flush().unwrap();
        // read response: assumes 1 f64 for each entry in batch
        let num_response_bytes = batch.len() * mem::size_of::<f64>();
        let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
        stream.read_exact(&mut response_buffer).unwrap();
        // make immutable
        let response_buffer = response_buffer;
        let response_floats = bytes_to_floats(&response_buffer);
        let end_time = time::PreciseTime::now();
        let latency = start_time.to(end_time).num_microseconds().unwrap();
        for _ in 0..batch.len() {
            latency_hist.insert(latency);
        }
        thruput_meter.mark(batch.len());
        predictions_counter.incr(batch.len() as isize);
        if latency > server::SLA * 1000 {
            debug!("latency: {}, batch size: {}",
                   (latency as f64 / 1000.0),
                   batch.len());
        }
        if dynamic_batching {
            // only try to increase the batch size if we actually sent a batch of maximum size
            if batch.len() == cur_batch_size {
                cur_batch_size = update_batch_size(cur_batch_size,
                                                   latency as u64,
                                                   server::SLA as u64 * 1000 as u64);
                // debug!("{} updated batch size to {}", name, cur_batch_size);
            }
        }

        // let mut l = latencies.write().unwrap();
        // l.push(latency);
        let mut w = cache.write().unwrap();
        for r in 0..batch.len() {
            let hashed_query = batch[r].hash_key;
            if !w.contains_key(&hashed_query) {
                w.insert(hashed_query, response_floats[r]);
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
    }
}


// TODO: this is super broken, and for some reason the
//
fn floats_to_bytes<'a>(v: Vec<f64>) -> &'a [u8] {
    let byte_arr: &[u8] = unsafe {
        let float_ptr: *const f64 = v[..].as_ptr();
        let num_elems = v.len() * mem::size_of::<f64>();
        slice::from_raw_parts(float_ptr as *const u8, num_elems)
    };
    byte_arr
}

fn bytes_to_floats(bytes: &[u8]) -> &[f64] {
    let float_arr: &[f64] = unsafe {
        let byte_ptr: *const u8 = bytes.as_ptr();
        let num_elems = bytes.len() / mem::size_of::<f64>();
        slice::from_raw_parts(byte_ptr as *const f64, num_elems)
    };
    float_arr
}


pub fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}
