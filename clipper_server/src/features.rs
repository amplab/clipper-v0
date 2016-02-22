
use gj;
use gj::{EventLoop, Promise};
// use gj::io;
use capnp;
// use std::time::{Duration, PreciseTime};
use time;
use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use std::net::{ToSocketAddrs, SocketAddr};
use feature_capnp::feature;
// use capnp::{primitive_list, message};
use std::thread;
use std::sync::{RwLock, Arc};
use std::sync::mpsc;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use num_cpus;
use linear_models::linalg;
use digits;
use std::hash::Hash;

type hashkey = u64;

#[derive(Clone)]
pub struct FeatureHandle {
    // addr: SocketAddr,
    pub name: String,
    pub queue: mpsc::Sender<(hashkey, Vec<f64>)>,
    // TODO: need a better concurrent hashmap: preferable lock free wait free
    // This should actually be reasonably simple, because we don't need to resize
    // (fixed size cache) and things never get evicted. Neither of these is strictly
    // true but it's a good approximation for now.
    pub cache: Arc<RwLock<HashMap<hashkey, f64>>>,
    pub hasher: Arc<FeatureHash>,
    pub latencies: Arc<RwLock<Vec<i64>>>
    // thread_handle: ::std::thread::JoinHandle<()>,
}


trait FeatureHash {
    fn hash(&self, input: &Vec<f64>) -> hashkey;
}

pub struct SimpleHasher;

impl FeatureHash for SimpleHash {
    fn hash(&self, input: &Vec<f64>) -> hashkey {
        input.hash() as hashkey
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






pub fn create_feature_worker(name: String, addr: SocketAddr)
    -> (FeatureHandle, ::std::thread::JoinHandle<()>) {

    let (tx, rx) = mpsc::channel();

    let feature_cache: Arc<RwLock<HashMap<hashkey, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    let latencies: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
    let handle = {
        let thread_cache = feature_cache.clone();
        let latencies = latencies.clone();
        let name = name.clone();
        thread::spawn(move || {
            feature_worker(name, rx, thread_cache, addr, latencies);
        })
    };
    (FeatureHandle {
        name: name.clone(),
        queue: tx,
        cache: feature_cache,
        hasher: SimpleHasher,
        latencies: latencies
        // thread_handle: handle,
    }, handle)
}

pub fn get_addr(a: String) -> SocketAddr {
    a.to_socket_addrs().unwrap().next().unwrap()
}

fn feature_worker(name: String,
                  rx: mpsc::Receiver<(hashkey, Vec<f64>)>,
                  cache: Arc<RwLock<HashMap<hashkey, f64>>>,
                  address: SocketAddr,
                  latencies: Arc<RwLock<Vec<i64>>>) {
    println!("starting worker: {}", name);

    EventLoop::top_level(move |wait_scope| {
        let (reader, writer) = try!(::gj::io::tcp::Stream::connect(address).wait(wait_scope))
                                   .split();
        let network = Box::new(twoparty::VatNetwork::new(reader,
                                                         writer,
                                                         rpc_twoparty_capnp::Side::Client,
                                                         Default::default()));
        let mut rpc_system = RpcSystem::new(network, None);
        let feature_rpc: feature::Client = rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
        println!("rpc connection established");
        feature_send_loop(name, feature_rpc, rx, cache, latencies)
            .lift().wait(wait_scope)
    }).expect("top level error");
}


fn feature_send_loop(name: String,
                     feature_rpc: feature::Client,
                     rx: mpsc::Receiver<(u64, Vec<f64>)>,
                     cache: Arc<RwLock<HashMap<hashkey, f64>>>,
                     latencies: Arc<RwLock<Vec<i64>>>)
                     -> Promise<(), ::std::io::Error> {

    // TODO batch feature requests
    // let mut new_features = Vec::new();
    // while rx.try_recv() {
    //     let data = rx.recv().unwrap();
    //     new_features.push(data);
    // }
    // println!("entering feature_send_loop");


    // try_recv() never blocks, will return immediately if pending data, else will error
    if let Ok(input) = rx.try_recv() {

        let start_time = time::PreciseTime::now();
        let hash = input.0;
        let feature_vec = input.1;


        // TODO(caching): check for cache presence before sending request
        let mut request = feature_rpc.compute_feature_request();
        {
            let mut builder = request.get();
            let mut inp_entries = builder.init_inp(feature_vec.len() as u32);
            for i in 0..feature_vec.len() {
                inp_entries.set(i as u32, feature_vec[i]);
            }
        }

        // request.get().set_inp(message.get_root::<primitive_list::Builder>().as_reader());
        request.send().promise.then_else(move |r| {
            match r {
                Ok(response) => {
                    let result = response.get().unwrap().get_result();
                    let end_time = time::PreciseTime::now();
                    let latency = start_time.to(end_time).num_microseconds().unwrap();
                    println!("got response: {} from {} in {} ms, putting in cache",
                             result,
                             name,
                             (latency as f64) / 1000.0);

                    {
                        let mut l = latencies.write().unwrap();
                        l.append(latency);
                        let mut w = cache.write().unwrap();
                        if !w.contains_key(hash) {
                            w.insert(hash, result);
                        } else {
                            let existing_res = w.get(hash).unwrap();
                            if result != existing_res {
                                println!("{} CACHE ERR: existing: {}, new: {}",
                                         name, existing_res, result);
                            } else {
                                println!("{} CACHE HIT", name);
                            }
                        }
                    }
                    feature_send_loop(name, feature_rpc, rx, cache, latencies)
                }
                Err(e) => {
                    println!("failed: {}", e);
                    feature_send_loop(name, feature_rpc, rx, cache, latencies)
                }
            }
        })
    } else {
        // if there's nothing in the queue, we don't need to spin, back off a little bit
        // println!("nothing in queue, waiting 1 s");
        // TODO change to 5ms
        gj::io::Timer.after_delay(::std::time::Duration::from_millis(1000))
                     .then(move |()| feature_send_loop(name, feature_rpc, rx, cache, latencies))
    }
}




pub fn feature_lats_main(feature_addrs: Vec<(String, SocketAddr)>) {

    // let addr_vec = vec!["127.0.0.1:6001".to_string(), "127.0.0.1:6002".to_string(), "127.0.0.1:6003".to_string()];
    // let names = vec!["TEN_rf".to_string(), "HUNDRED_rf".to_string(), "FIVE_HUNDO_rf".to_string()];
    // let addr_vec = (1..11).map(|i| format!("127.0.0.1:600{}", i)).collect::<Vec<String>>();
    // let names = (1..11).map(|i| format!("Spark_predictor_{}", i)).collect::<Vec<String>>();
    // let addr_vec = vec!["169.229.49.167:6001".to_string()];
    // let names = vec!["CAFFE on c67 with GPU".to_string()];
    let num_features = feature_addrs.len();
    let handles: Vec<::std::thread::JoinHandle<()>> =
            feature_addrs.into_iter().map(|(n, a)| create_blocking_features(n, a)).collect();

    for h in handles {
        h.join().unwrap();
    }
    // handle.join().unwrap();
    println!("done");
}

fn create_blocking_features(name: String, address: SocketAddr)
                            -> ::std::thread::JoinHandle<()> {
    // let name = name.clone();
    println!("{}", name);
    thread::spawn(move || {
        blocking_feature_requests(name, address);
    })
}

fn blocking_feature_requests(name: String, address: SocketAddr) {

    let num_events = 5000;
    EventLoop::top_level(move |wait_scope| {
        let (reader, writer) = try!(::gj::io::tcp::Stream::connect(address).wait(wait_scope))
            .split();
        let network = Box::new(twoparty::VatNetwork::new(reader,
                                                         writer,
                                                         rpc_twoparty_capnp::Side::Client,
                                                         Default::default()));
        let mut rpc_system = RpcSystem::new(network, None);
        let feature_rpc: feature::Client = rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
        println!("rpc connection established");
        // feature_send_loop(name, feature_rpc, rx).lift()

        for _ in 0..num_events {

            let start_time = time::PreciseTime::now();
            let feature_vec = random_features(784);

            // println!("sending {} reqs", new_features.len());
            // println!("sending request");

            let mut request = feature_rpc.compute_feature_request();
            {
                let mut builder = request.get();
                let mut inp_entries = builder.init_inp(feature_vec.len() as u32);
                for i in 0..feature_vec.len() {
                    inp_entries.set(i as u32, feature_vec[i]);
                }
            }

            // request.get().set_inp(message.get_root::<primitive_list::Builder>().as_reader());
            
            let name = name.clone();
            try!(request.send().promise.then(move |response| {
                let res = pry!(response.get()).get_result();
                // let result = response.get().unwrap().get_result();
                let end_time = time::PreciseTime::now();
                let latency = start_time.to(end_time).num_milliseconds();
                println!("got response: {} from {} in {} ms, putting in cache",
                         res,
                         name,
                         latency);
                // feature_send_loop(name, feature_rpc, rx)
                Promise::ok(())
            }).wait(wait_scope));
            // thread::sleep(::std::time::Duration::new(1, 0));
        }
        println!("done with requests");
        Ok(())
    })
    .expect("top level error");
}

pub fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}
