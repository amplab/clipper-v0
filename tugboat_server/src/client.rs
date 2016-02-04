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
use std::time::Duration;
use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use std::net::{ToSocketAddrs, SocketAddr};
use feature_capnp::feature;
use capnp::{primitive_list, message};
use std::thread;
use std::sync::{RwLock, Arc};
use std::sync::mpsc;
use std::collections::HashMap;
use rand::{thread_rng, Rng};


pub fn main() {

    gj::EventLoop::top_level(|wait_scope| {
        let addr1 = try!("127.0.0.1:6001".to_socket_addrs()).next().expect("couldn't parse");
        let addr2 = try!("127.0.0.1:6002".to_socket_addrs()).next().expect("couldn't parse");

        let (reader1, writer1) = try!(::gj::io::tcp::Stream::connect(addr1).wait(wait_scope))
                                     .split();
        let network1 = Box::new(twoparty::VatNetwork::new(reader1,
                                                          writer1,
                                                          rpc_twoparty_capnp::Side::Client,
                                                          Default::default()));
        let mut rpc_system1 = RpcSystem::new(network1, None);
        let feature1: feature::Client = rpc_system1.bootstrap(rpc_twoparty_capnp::Side::Server);

        let feature_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // let feature_list = primitive_list::
        println!("{}", feature_vec.len());

        let predict1 = {
            let mut request = feature1.compute_feature_request();

            {
                let mut builder = request.get();
                let mut inp_entries = builder.init_inp(feature_vec.len() as u32);
                for i in 0..feature_vec.len() {
                    inp_entries.set(i as u32, feature_vec[i]);
                }
            }

            // request.get().set_inp(message.get_root::<primitive_list::Builder>().as_reader());
            let predict_promise = request.send();
            // let read_promise = predict_promise.pipeline.get_value().read_request().send();

            let response = try!(predict_promise.promise.wait(wait_scope));

            try!(response.get()).get_result()
        };
        println!("got a response: {}", predict1);
        Ok(())
    }).expect("top level error");
}

fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}

pub fn other_main() {

    let (tx, rx) = mpsc::channel();

    let feature_cache: Arc<RwLock<HashMap<u32, f64>>> = Arc::new(RwLock::new(HashMap::new()));

    let addr: SocketAddr = "127.0.0.1:6001".to_socket_addrs().unwrap().next().expect("couldn't parse");
    let handle = {
        let thread_cache = feature_cache.clone();
        thread::spawn(move || {
            feature_worker(rx, thread_cache, addr);
        })
    };

    thread::sleep(Duration::new(3, 0));
    tx.send((11_u32, random_features(784))).unwrap();
    tx.send((12_u32, random_features(784))).unwrap();
    tx.send((13_u32, random_features(784))).unwrap();
    tx.send((14_u32, random_features(784))).unwrap();

    // {
    //     let c = (&feature_cache).read().unwrap();
    //     println!("{:?}", c.get(&11_u32));
    // }
    println!("waiting for feature to finish");
    handle.join().unwrap();
    println!("done");
    
}

fn feature_worker(rx: mpsc::Receiver<(u32, Vec<f64>)>, cache: Arc<RwLock<HashMap<u32, f64>>>, address: SocketAddr) {
    println!("starting worker");

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
        feature_send_loop(feature_rpc, rx).lift().wait(wait_scope)
    }).expect("top level error");

}


fn feature_send_loop(feature_rpc: feature::Client,
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
                println!("got response: {}, putting in cache", result);
                feature_send_loop(feature_rpc, rx)
            }
            Err(e) => {
                println!("failed: {}", e);
                feature_send_loop(feature_rpc, rx)
            }
        })
    } else {
        // if there's nothing in the queue, we don't need to spin, back off a little bit
        println!("nothing in queue, waiting 1 s");
        // TODO change to 5ms
        gj::io::Timer.after_delay(Duration::from_millis(1000)).then(move |()| {
            feature_send_loop(feature_rpc, rx)
        })
    }
}

