#![cfg_attr(feature = "serde_derive", feature(rustc_macro))]
#![deny(warnings)]

extern crate serde;
extern crate serde_json;

#[cfg(feature = "serde_derive")]
#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate log;
extern crate env_logger;

#[cfg(feature = "serde_derive")]
include!("clipper-rest.rs.in");

#[cfg(not(feature = "serde_derive"))]
include!(concat!(env!("OUT_DIR"), "/clipper-rest.rs"));
