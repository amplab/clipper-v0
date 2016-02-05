#![crate_name="tugboat_server"]
#![crate_type="bin"]


extern crate capnp;
extern crate capnp_rpc;
extern crate rand;
extern crate time;
#[macro_use]
extern crate gj;

pub mod feature_capnp {
  include!(concat!(env!("OUT_DIR"), "/feature_capnp.rs"));
}

pub mod client;
pub mod server;

pub fn main() {
    let args : Vec<String> = ::std::env::args().collect();
    if args.len() >= 2 {
        match &args[1][..] {
            "client" => return client::main(),
            // "server" => return server::main(),
            _ => ()
        }
    }

    println!("usage: {} [client | server] ADDRESS", args[0]);
}
