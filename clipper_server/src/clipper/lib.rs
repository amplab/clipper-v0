#![cfg_attr(feature = "serde_macros", feature(custom_derive, plugin))]
#![cfg_attr(feature = "serde_macros", plugin(serde_macros))]
#![deny(warnings)]

extern crate serde;
extern crate bincode;
#[macro_use]
extern crate log;
extern crate lz4;
extern crate curl;
extern crate url;
extern crate regex;

#[cfg(feature = "serde_macros")]
include!("lib.rs.in");

#[cfg(not(feature = "serde_macros"))]
include!(concat!(env!("OUT_DIR"), "/lib.rs"));
