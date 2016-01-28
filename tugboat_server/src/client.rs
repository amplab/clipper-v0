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
// use gj::{EventLoop, Promise};
// use gj::io::{AsyncRead, AsyncWrite, unix};
use std::time::Duration;
use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use std::net::ToSocketAddrs;
use feature_capnp::feature;


pub fn main() {

  gj::EventLoop::top_level(|wait_scope| {
    let addr1 = try!("127.0.0.1:6001".to_socket_addrs()).next().expect("couldn't parse");
    let addr2 = try!("127.0.0.1:6002".to_socket_addrs()).next().expect("couldn't parse");

    let (reader1, writer1) = try!(::gj::io::tcp::Stream::connect(addr1).wait(wait_scope)).split();
    let network1 =
        Box::new(twoparty::VatNetwork::new(reader1, writer1,
                                           rpc_twoparty_capnp::Side::Client,
                                           Default::default()));
    let mut rpc_system1 = RpcSystem::new(network1, None);
    let feature1: feature::Client = rpc_system1.bootstrap(rpc_twoparty_capnp::Side::Server);

    let (reader2, writer2) = try!(::gj::io::tcp::Stream::connect(addr2).wait(wait_scope)).split();
    let network2 =
        Box::new(twoparty::VatNetwork::new(reader2, writer2,
                                           rpc_twoparty_capnp::Side::Client,
                                           Default::default()));
    let mut rpc_system2 = RpcSystem::new(network2, None);
    let feature2: feature::Client = rpc_system2.bootstrap(rpc_twoparty_capnp::Side::Server);
    let feature_vec = [2u8,3u8,4u8];

    let predict1 = {
        let mut request = feature1.compute_feature_request();
        request.get().set_input(&feature_vec);
        let predict_promise = request.send();
        // let read_promise = predict_promise.pipeline.get_value().read_request().send();

        let response = try!(predict_promise.promise.wait(wait_scope));

        try!(response.get()).get_result()
    };
    println!("got a response: {}", predict1);

    let predict2 = {
        let mut request = feature2.compute_feature_request();
        request.get().set_input(&[127,128,255,12,255]);
        let predict_promise = request.send();
        // let read_promise = predict_promise.pipeline.get_value().read_request().send();

        let response = try!(predict_promise.promise.wait(wait_scope));

        try!(response.get()).get_result()
    };
    println!("got a response: {}", predict2);

    // let add = {
    //     // Get the "add" function from the server.
    //     let mut request = calculator1.get_operator_request();
    //     request.get().set_op(calculator::Operator::Add);
    //     request.send().pipeline.get_func()
    // };
    //
    // let subtract = {
    //     // Get the "subtract" function from the server.
    //     let mut request = calculator.get_operator_request();
    //     request.get().set_op(calculator::Operator::Subtract);
    //     request.send().pipeline.get_func()
    // };
    //
    // // Build the request to evaluate 123 + 45 - 67.
    // let mut request = calculator.evaluate_request();
    //
    // {
    //     let mut subtract_call = request.get().init_expression().init_call();
    //     subtract_call.set_function(subtract);
    //     let mut subtract_params = subtract_call.init_params(2);
    //     subtract_params.borrow().get(1).set_literal(67.0);
    //
    //     let mut add_call = subtract_params.get(0).init_call();
    //     add_call.set_function(add);
    //     let mut add_params = add_call.init_params(2);
    //     add_params.borrow().get(0).set_literal(123.0);
    //     add_params.get(1).set_literal(45.0);
    // }
    //
    // // Send the evaluate() request, read() the result, and wait for read() to
    // // finish.
    // let eval_promise = request.send();
    // let read_promise = eval_promise.pipeline.get_value().read_request().send();
    //
    // let response = try!(read_promise.promise.wait(wait_scope));
    // assert_eq!(try!(response.get()).get_value(), 101.0);
    //
    // println!("PASS");
    //

    Ok(())
  }).expect("top level error");
}

