#![cfg_attr(feature = "serde_macros", feature(custom_derive, plugin))]
#![cfg_attr(feature = "serde_macros", plugin(serde_macros))]
#![deny(warnings)]

extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate log;
extern crate env_logger;

#[cfg(feature = "serde_macros")]
include!("clipper-rest.rs.in");

#[cfg(not(feature = "serde_macros"))]
include!(concat!(env!("OUT_DIR"), "/clipper-rest.rs"));
