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


// pub fn main() {
//
//     gj::EventLoop::top_level(|wait_scope| {
//         let addr1 = try!("127.0.0.1:6001".to_socket_addrs()).next().expect("couldn't parse");
//         let addr2 = try!("127.0.0.1:6002".to_socket_addrs()).next().expect("couldn't parse");
//
//         let (reader1, writer1) = try!(::gj::io::tcp::Stream::connect(addr1).wait(wait_scope))
//                                      .split();
//         let network1 = Box::new(twoparty::VatNetwork::new(reader1,
//                                                           writer1,
//                                                           rpc_twoparty_capnp::Side::Client,
//                                                           Default::default()));
//         let mut rpc_system1 = RpcSystem::new(network1, None);
//         let feature1: feature::Client = rpc_system1.bootstrap(rpc_twoparty_capnp::Side::Server);
//
//         let feature_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//
//         // let feature_list = primitive_list::
//         println!("{}", feature_vec.len());
//
//         let predict1 = {
//             let mut request = feature1.compute_feature_request();
//
//             {
//                 let mut builder = request.get();
//                 let mut inp_entries = builder.init_inp(feature_vec.len() as u32);
//                 for i in 0..feature_vec.len() {
//                     inp_entries.set(i as u32, feature_vec[i]);
//                 }
//             }
//
//             // request.get().set_inp(message.get_root::<primitive_list::Builder>().as_reader());
//             let predict_promise = request.send();
//             // let read_promise = predict_promise.pipeline.get_value().read_request().send();
//
//             let response = try!(predict_promise.promise.wait(wait_scope));
//
//             try!(response.get()).get_result()
//         };
//         println!("got a response: {}", predict1);
//         Ok(())
//     }).expect("top level error");
// }

pub fn main() {


    let names = vec!["sklearn".to_string(), "spark".to_string()];
    let features: Vec<FeatureHandle> = vec!["127.0.0.1:6001".to_string(), "127.0.0.1:6002".to_string()]
        .into_iter()
        .map(|a| get_addr(a))
        .zip(names.into_iter())
        .map(|(a, n)| create_feature_worker(a, n))
        .collect();
    // .next().expect("couldn't parse");

    


    thread::sleep(::std::time::Duration::new(3, 0));

    get_features(&features, 11_u32, random_features(784));
    get_features(&features, 12_u32, random_features(784));
    get_features(&features, 13_u32, random_features(784));
    get_features(&features, 14_u32, random_features(784));

    // {
    //     let c = (&feature_cache).read().unwrap();
    //     println!("{:?}", c.get(&11_u32));
    // }
    println!("waiting for features to finish");
    for f in features {
        f.thread_handle.join().unwrap();
    }
    // handle.join().unwrap();
    println!("done");
}

fn get_features(fs: &Vec<FeatureHandle>, hash: u32, input: Vec<f64>) {
    for f in fs {
        f.queue.send((hash, input.clone())).unwrap();
    }
}

fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}

struct FeatureHandle {
    // addr: SocketAddr,
    name: String,
    queue: mpsc::Sender<(u32, Vec<f64>)>,
    cache: Arc<RwLock<HashMap<u32, f64>>>,
    thread_handle: ::std::thread::JoinHandle<()>
}

fn create_feature_worker(addr: SocketAddr, name: String)
    -> FeatureHandle {

    let (tx, rx) = mpsc::channel();

    let feature_cache: Arc<RwLock<HashMap<u32, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    let handle = {
        let thread_cache = feature_cache.clone();
        let name = name.clone();
        thread::spawn(move || {
            feature_worker(name, rx, thread_cache, addr);
        })
    };
    FeatureHandle {name: name.clone(), queue: tx, cache: feature_cache, thread_handle: handle}
}

fn get_addr(a: String) -> SocketAddr {
    a.to_socket_addrs().unwrap().next().unwrap()
}


fn feature_worker(
    name: String,
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
    }).expect("top level error");
}


fn feature_send_loop(
    name: String,
    feature_rpc: feature::Client,
    rx: mpsc::Receiver<(u32, Vec<f64>)>) -> Promise<(), ::std::io::Error> {

    // TODO batch feature requests
    // let mut new_features = Vec::new();
    // while rx.try_recv() {
    //     let data = rx.recv().unwrap();
    //     new_features.push(data);
    // }
    println!("entering feature_send_loop");


    // try_recv() never blocks, will return immediately if pending data, else will error
    if let Ok(input) = rx.try_recv() {

        let start_time = time::PreciseTime::now();
        let feature_vec = input.1;

        // println!("sending {} reqs", new_features.len());
        println!("sending request");

        let mut request = feature_rpc.compute_feature_request();
        {
            let mut builder = request.get();
            let mut inp_entries = builder.init_inp(feature_vec.len() as u32);
            for i in 0..feature_vec.len() {
                inp_entries.set(i as u32, feature_vec[i]);
            }
        }

        // request.get().set_inp(message.get_root::<primitive_list::Builder>().as_reader());
        request.send().promise.then_else(move |r| match r {
            Ok(response) => {
                let result = response.get().unwrap().get_result();
                let end_time = time::PreciseTime::now();
                let latency = start_time.to(end_time).num_microseconds().unwrap();
                println!("got response: {} from {} in {} us, putting in cache", result, name, latency);
                feature_send_loop(name, feature_rpc, rx)
            }
            Err(e) => {
                println!("failed: {}", e);
                feature_send_loop(name, feature_rpc, rx)
            }
        })
    } else {
        // if there's nothing in the queue, we don't need to spin, back off a little bit
        println!("nothing in queue, waiting 1 s");
        // TODO change to 5ms
        gj::io::Timer.after_delay(::std::time::Duration::from_millis(1000)).then(move |()| {
            feature_send_loop(name, feature_rpc, rx)
        })
    }
}





