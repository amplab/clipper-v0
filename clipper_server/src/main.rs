#![crate_name="clipper"]
#![crate_type="bin"]


extern crate capnp;
extern crate capnp_rpc;
extern crate rand;
extern crate time;
#[macro_use]
extern crate gj;
extern crate eventual;
extern crate mio;
extern crate num_cpus;

pub mod feature_capnp {
  include!(concat!(env!("OUT_DIR"), "/feature_capnp.rs"));
}

pub mod linalg;
pub mod server;
pub mod digits;
pub mod bench;

pub fn main() {
    let args : Vec<String> = ::std::env::args().collect();
    if args.len() >= 2 {
        match &args[1][..] {
            "start" => return server::main(),
            "feature_lats" => return server::feature_lats_main(),
            "gj_timers" => return bench::gj_timers(args[2].parse::<u32>().unwrap()),
            // "ev_timers" => return bench::eventual_timers(args[2].parse::<u32>().unwrap()),
            "mio_timers" => return bench::mio_timers(args[2].parse::<u32>().unwrap()),
            "clipper_timers" => return bench::clipper_timers(args[2].parse::<u32>().unwrap(), args[3].parse::<usize>().unwrap()),
            _ => ()
        }
    }

    println!("usage: {} [client | server] ADDRESS", args[0]);
}
