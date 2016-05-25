// #![crate_name="clipper"]
// #![deny(missing_docs)]
#![deny(warnings)]
extern crate rand;
extern crate time;
extern crate libc;
extern crate toml;
extern crate num_cpus;
extern crate linear_models;
extern crate rustc_serialize;
extern crate net2;
extern crate byteorder;
#[macro_use]
extern crate log;

pub mod server;
pub mod digits;
pub mod features;
pub mod metrics;
pub mod hashing;
pub mod rpc;
