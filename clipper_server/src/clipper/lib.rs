#![cfg_attr(feature = "serde_derive", feature(rustc_macro))]
// #![cfg_attr(feature = "serde_derive", plugin(serde_macros))]
#![deny(warnings)]

extern crate serde;
extern crate bincode;

#[cfg(feature = "serde_derive")]
#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate log;
extern crate lz4;
extern crate curl;
extern crate url;
extern crate regex;

#[cfg(feature = "serde_derive")]
include!("lib.rs.in");

#[cfg(not(feature = "serde_derive"))]
include!(concat!(env!("OUT_DIR"), "/lib.rs"));
