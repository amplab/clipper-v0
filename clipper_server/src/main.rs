#![crate_name="clipper"]
#![crate_type="bin"]


extern crate capnp;
extern crate capnp_rpc;
extern crate rand;
extern crate time;
#[macro_use]
extern crate gj;
extern crate eventual;

pub mod feature_capnp {
  include!(concat!(env!("OUT_DIR"), "/feature_capnp.rs"));
}

pub mod client;
pub mod server;
pub mod digits;
pub mod linalg;
pub mod bench;

pub fn main() {
    let args : Vec<String> = ::std::env::args().collect();
    if args.len() >= 2 {
        match &args[1][..] {
            "client" => return client::main(),
            "gj_timers" => return bench::gj_timers(args[2].parse::<u32>().unwrap()),
            "ev_timers" => return bench::eventual_timers(args[2].parse::<u32>().unwrap()),
            _ => ()
        }
    }

    println!("usage: {} [client | server] ADDRESS", args[0]);
}
