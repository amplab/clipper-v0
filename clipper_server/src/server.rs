// Copyright (c) 2013-2015 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_mut)]

// use gj::{Promise, EventLoop};

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
use linalg;

const SLA: i64 = 20;



// pub fn benchmark(num_requests: usize, features: &Vec<FeatureHandle>) {
//
//     let train_path = "/Users/crankshaw/model-serving/data/mnist_data/train-mnist-dense-with-labels\
//                       .data";
//     let test_path = "/Users/crankshaw/model-serving/data/mnist_data/test-mnist-dense-with-labels.\
//                      data";
//
//     let all_train_data = digits::load_mnist_dense(train_path).unwrap();
//     let norm_train_data = digits::normalize(&all_train_data);
//     println!("Training data loaded: {} points", norm_train_data.ys.len());
//
//     let all_test_data = digits::load_mnist_dense(test_path).unwrap();
//     let norm_test_data = digits::normalize(&all_test_data);
//
//     println!("Test data loaded: {} points", norm_test_data.ys.len());
//
//     for i in 0..200 {
//
//     }
// }



fn anytime_features(features: &Vec<FeatureHandle>, input: &Vec<f64>) -> Vec<f64> {
    // TODO check caches
    // for f in features {
    //     f.cache.read()
    // }
    vec![-3.2, 5.1]
}


struct Request {
    start_time: time::PreciseTime,
    user: u32, // TODO: remove this because each feature has it's own hash
    input: Vec<f64>,
}

impl Request {

    fn new(user: u32, input: Vec<f64>) -> Request {
        Request { start_time: time::PreciseTime::now(), user: user, input: input}
    }

}

struct Update {
    start_time: time::PreciseTime, // just for monitoring purposes
    user: u32,
    input: Vec<f64>,
    label: f32,
}

fn start_update_worker(feature_handles: Vec<FeatureHandle>) -> mpsc::Sender<Update> {
    panic!("unimplemented method");
}

// TODO make into a trait to support various kinds of models
struct TaskModel {
    w: Vec<f64>
}

// Because we don't have a good concurrent hash map, assume we know how many
// users there will be ahead of time. Then we can have a vec of RwLock.
fn make_prediction(features: &Vec<FeatureHandle>, input: &Vec<f64>,
                   task_model: &TaskModel) -> f64 {
    let fs = anytime_features(features, input);
    linalg::dot(&fs, &task_model.w)
}

fn start_prediction_worker(worker_id: i32,
                           sla_millis: i64,
                           feature_handles: Vec<FeatureHandle>,
                           user_models: Arc<Vec<RwLock<TaskModel>>>,
                           counter: Arc<AtomicUsize>) -> mpsc::Sender<Request> {

    let sla = time::Duration::milliseconds(sla_millis);
    let epsilon = time::Duration::milliseconds(3);
    let (sender, receiver) = mpsc::channel::<Request>();
    let join_guard = thread::spawn(move || {
        println!("starting response worker {} with {}ms SLA", worker_id, sla_millis);
        loop {
            let req = receiver.recv().unwrap();
            // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
            let elapsed_time = req.start_time.to(time::PreciseTime::now());
            if elapsed_time < sla - epsilon {
                let sleep_time = ::std::time::Duration::new(
                    0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
                // println!("sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            // TODO: actually compute prediction
            // return result
            debug_assert!(req.user < user_models.len() as u32);
            let lock = (&user_models).get(*(&req.user) as usize).unwrap();
            let task_model = lock.read().unwrap();
            let pred = make_prediction(&feature_handles, &req.input, &task_model);
            let end_time = time::PreciseTime::now();
            let latency = req.start_time.to(end_time).num_milliseconds();
            counter.fetch_add(1, Ordering::Relaxed);
            // if latency > 25 {
            //     println!("worked {} made prediction. latency: {} ms", worker_id, latency);
            // }
            // TODO actually respond to request
        }
    });
    // (join_guard, sender)
    sender
}



struct Dispatcher {
    workers: Vec<mpsc::Sender<Request>>,
    next_worker: usize,
    features: Vec<FeatureHandle>,
}

impl Dispatcher {

    fn new(num_workers: usize,
           sla_millis: i64,
           features: Vec<FeatureHandle>,
           user_models: Arc<Vec<RwLock<TaskModel>>>,
           counter: Arc<AtomicUsize>) -> Dispatcher {
        println!("creating dispatcher with {} workers", num_workers);
        let mut worker_threads = Vec::new();
        for i in 0..num_workers {
            let worker = start_prediction_worker(i as i32,
                                                 sla_millis,
                                                 features.clone(),
                                                 user_models.clone(),
                                                 counter.clone());
            worker_threads.push(worker);
        }
        Dispatcher {workers: worker_threads, next_worker: 0, features: features}
    }

    /// Dispatch a request.
    ///
    /// Requires self to be mutable so that we can increment `next_worker`
    fn dispatch(&mut self, req: Request) {
        get_features(&self.features, req.input.clone());
        self.workers[self.next_worker].send(req).unwrap();
        self.increment_worker();
    }

    // for now do round robin scheduling
    fn increment_worker(&mut self) {
        self.next_worker = (self.next_worker + 1) % self.workers.len();
    }
}

fn init_user_models(num_users: usize, num_features: usize) -> Arc<Vec<RwLock<TaskModel>>> {
    let mut rng = thread_rng();
    let mut models = Vec::with_capacity(num_users);
    for i in 0..num_users {
        let model = RwLock::new(TaskModel {
            w: rng.gen_iter::<f64>().take(num_features).collect::<Vec<f64>>()
        });
        models.push(model);
    }
    Arc::new(models)
}

pub fn main() {
    // let addr_vec = vec!["127.0.0.1:6001".to_string()];
    // let names = vec!["sklearn".to_string()];
    let addr_vec = vec!["127.0.0.1:6001".to_string(), "127.0.0.1:6002".to_string(), "127.0.0.1:6003".to_string()];
    let names = vec!["TEN_rf".to_string(), "HUNDRED_rf".to_string(), "FIVE_HUNDO_rf".to_string()];
    let num_features = names.len();
    let num_users = 500;
    let (features, handles): (Vec<_>, Vec<_>) = addr_vec.into_iter()
                                                        .map(|a| get_addr(a))
                                                        .zip(names.into_iter())
                                                        .map(|(a, n)| create_feature_worker(a, n))
                                                        .unzip();

    let counter = Arc::new(AtomicUsize::new(0));
    let num_events = 100;
    // let num_workers = num_cpus::get();
    let num_workers = 1;
    let mut dispatcher = Dispatcher::new(num_workers,
                                         SLA,
                                         features,
                                         init_user_models(num_users, num_features),
                                         counter.clone());

    // create monitoring thread to check incremental thruput
    thread::sleep(::std::time::Duration::new(3, 0));
    let mon_thread_join_handle = {
        let counter = counter.clone();
        thread::spawn(move || {
            let bench_start = time::PreciseTime::now();
            let mut last_count = 0;
            let mut last_time = bench_start;
            loop {
                thread::sleep(::std::time::Duration::new(3, 0));
                let cur_time = last_time.to(time::PreciseTime::now()).num_milliseconds() as f64 / 1000.0;
                let cur_count = counter.load(Ordering::Relaxed);
                println!("processed {} events in {} seconds: {} preds/sec",
                         cur_count - last_count,
                         cur_time,
                         ((cur_count - last_count) as f64 / cur_time));

                if cur_count >= num_events {
                    println!("BENCHMARK FINISHED");
                    break;
                }
                println!("sleeping...");
            }
        })
    };
    // let new_request = Request::new(11_u32, random_features(784));
    // dispatcher.dispatch(new_request);



    println!("sending batch with no delays");
    let mut rng = thread_rng();
    for _ in 0..num_events {
        dispatcher.dispatch(Request::new(rng.gen_range(0, num_users as u32), random_features(784)));
    }

    println!("waiting for features to finish");
    mon_thread_join_handle.join().unwrap();
    for h in handles {
        h.join().unwrap();
    }
    // handle.join().unwrap();
    println!("done");
}

// pub fn eval_feature_latency() {
//
// }

fn get_features(fs: &Vec<FeatureHandle>, input: Vec<f64>) {
    let hash = 11_u32;
    for f in fs {
        f.queue.send((hash, input.clone())).unwrap();
    }
}

fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}


#[derive(Clone)]
struct FeatureHandle {
    // addr: SocketAddr,
    name: String,
    queue: mpsc::Sender<(u32, Vec<f64>)>,
    // TODO: need a better concurrent hashmap: preferable lock free wait free
    // This should actually be reasonably simple, because we don't need to resize
    // (fixed size cache) and things never get evicted. Neither of these is strictly
    // true but it's a good approximation for now.
    cache: Arc<RwLock<HashMap<u32, f64>>>,
    // thread_handle: ::std::thread::JoinHandle<()>,
}

fn create_feature_worker(addr: SocketAddr, name: String)
    -> (FeatureHandle, ::std::thread::JoinHandle<()>) {

    let (tx, rx) = mpsc::channel();

    let feature_cache: Arc<RwLock<HashMap<u32, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    let handle = {
        let thread_cache = feature_cache.clone();
        let name = name.clone();
        thread::spawn(move || {
            feature_worker(name, rx, thread_cache, addr);
        })
    };
    (FeatureHandle {
        name: name.clone(),
        queue: tx,
        cache: feature_cache,
        // thread_handle: handle,
    }, handle)
}

fn get_addr(a: String) -> SocketAddr {
    a.to_socket_addrs().unwrap().next().unwrap()
}


fn feature_worker(name: String,
                  rx: mpsc::Receiver<(u32, Vec<f64>)>,
                  cache: Arc<RwLock<HashMap<u32, f64>>>,
                  address: SocketAddr) {
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
        feature_send_loop(name, feature_rpc, rx).lift().wait(wait_scope)
    })
        .expect("top level error");
}


fn feature_send_loop(name: String,
                     feature_rpc: feature::Client,
                     rx: mpsc::Receiver<(u32, Vec<f64>)>)
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
        let feature_vec = input.1;

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
        request.send().promise.then_else(move |r| {
            match r {
                Ok(response) => {
                    let result = response.get().unwrap().get_result();
                    let end_time = time::PreciseTime::now();
                    let latency = start_time.to(end_time).num_milliseconds();
                    println!("got response: {} from {} in {} ms, putting in cache",
                             result,
                             name,
                             latency);
                    feature_send_loop(name, feature_rpc, rx)
                }
                Err(e) => {
                    println!("failed: {}", e);
                    feature_send_loop(name, feature_rpc, rx)
                }
            }
        })
    } else {
        // if there's nothing in the queue, we don't need to spin, back off a little bit
        // println!("nothing in queue, waiting 1 s");
        // TODO change to 5ms
        gj::io::Timer.after_delay(::std::time::Duration::from_millis(1000))
                     .then(move |()| feature_send_loop(name, feature_rpc, rx))
    }
}

pub fn feature_lats_main() {

    let addr_vec = vec!["127.0.0.1:6001".to_string(), "127.0.0.1:6002".to_string(), "127.0.0.1:6003".to_string()];
    let names = vec!["TEN_rf".to_string(), "HUNDRED_rf".to_string(), "FIVE_HUNDO_rf".to_string()];
    // let addr_vec = vec!["127.0.0.1:6001".to_string()];
    // let names = vec!["SVM_SKLEARN".to_string()];
    let num_features = names.len();
    let handles: Vec<::std::thread::JoinHandle<()>> = addr_vec.into_iter()
                                            .map(|a| get_addr(a))
                                            .zip(names.into_iter())
                                            .map(|(a, n)| create_blocking_features(n, a))
                                            .collect();
                                            // .unzip();

    for h in handles {
        h.join().unwrap();
    }
    // handle.join().unwrap();
    println!("done");
}

fn create_blocking_features(name: String, address: SocketAddr)
                            -> ::std::thread::JoinHandle<()> {
    // let name = name.clone();
    thread::spawn(move || {
        blocking_feature_requests(name, address);
    })
}

fn blocking_feature_requests(name: String, address: SocketAddr) {

    let num_events = 150;
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
            thread::sleep(::std::time::Duration::new(1, 0));
        }
        println!("done with requests");
        Ok(())
    })
    .expect("top level error");
}

