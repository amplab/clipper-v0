#![crate_name="clipper"]
#![crate_type="bin"]


// extern crate capnp;
// extern crate capnp_rpc;
extern crate rand;
extern crate time;
// #[macro_use]
// extern crate gj;
// extern crate eventual;
// extern crate mio;
extern crate num_cpus;
extern crate linear_models;
extern crate toml;
// extern crate getopts;
extern crate rustc_serialize;
extern crate docopt;
extern crate net2;
extern crate byteorder;
// extern crate quickersort;

use docopt::Docopt;
use std::error::Error;
use std::result::Result;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::path::Path;
use std::io::{BufReader, BufWriter};

// use getopts::Options;
use std::env;
use std::net::{ToSocketAddrs, SocketAddr};

// pub mod feature_capnp {
//   include!(concat!(env!("OUT_DIR"), "/feature_capnp.rs"));
// }

// pub mod linalg;
pub mod server;
pub mod digits;
// pub mod bench;
pub mod features;
pub mod digits_benchmark;
// pub mod rpc;


const USAGE: &'static str = "
Clipper Server

Usage:
  clipper digits --feature-conf=</path/to/features.toml> --bench-conf=<digits.conf>
  clipper featurelats <b>
  clipper start --feature-conf=</path/to/features.toml>
  clipper -h

Options:
  -h --help                              Show this screen.
  --feature-conf=</path/to/features>             Path to features config file.
  --bench-conf=</path/to/digits.conf>    Path to mnist data file
  // --batch-size=<bs>                     Size of feature batch 
  
";

#[derive(Debug, RustcDecodable)]
struct Args {
    // flag_speed: isize,
    // flag_drifting: bool,
    flag_bench_conf: Option<String>,
    flag_feature_conf: String,
    arg_b: Option<usize>,
    cmd_digits: bool,
    cmd_featurelats: bool,
    cmd_start: bool,
}

pub fn main() {

    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.decode())
                            .unwrap_or_else(|e| e.exit());

    println!("{:?}", args);

    // let features = parse_feature_config(&"features.toml".to_string());
    if args.cmd_digits {
      let features = parse_feature_config(&args.flag_feature_conf);
      let digits_conf = parse_digits_config(&args.flag_bench_conf.unwrap());
      digits_benchmark::run(features, digits_conf);
    } else if args.cmd_featurelats {
        features::feature_batch_latency(args.arg_b.unwrap());
        // println!("unimplemented");
    } else if args.cmd_start {
        let features = parse_feature_config(&args.flag_feature_conf);
        server::main(features);
    }



    // let args : Vec<String> = ::std::env::args().collect();
    // if args.len() >= 2 {
    //     match &args[1][..] {
    //         "start" => return server::main(),
    //         "feature_lats" => return features::feature_lats_main(),
    //         "gj_timers" => return bench::gj_timers(args[2].parse::<u32>().unwrap()),
    //         // "ev_timers" => return bench::eventual_timers(args[2].parse::<u32>().unwrap()),
    //         "mio_timers" => return bench::mio_timers(args[2].parse::<u32>().unwrap()),
    //         "clipper_timers" => return bench::clipper_timers(args[2].parse::<u32>().unwrap(), args[3].parse::<usize>().unwrap()),
    //         _ => ()
    //     }
    // }
    //
    // println!("usage: {} [client | server] ADDRESS", args[0]);
}

fn parse_feature_config(fname: &String) -> Vec<(String, Vec<SocketAddr>)> {
    
    let path = Path::new(fname);
    let display = path.display();

    let mut file = match File::open(&path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => panic!(format!("couldn't open {}: REASON: {}", display,
                                                   Error::description(&why))),
        Ok(file) => BufReader::new(file),
    };

    let mut toml_string = String::new();
    match file.read_to_string(&mut toml_string) {
      Err(why) => panic!("couldn't read {}: {}", display,
                         Error::description(&why)),
                         Ok(_) => print!("{} contains:\n{}", display, toml_string),
    }

    let features = toml::Parser::new(&toml_string).parse().unwrap();
    let fs = features.get("features").unwrap()
                     .as_table().unwrap().iter()
                     .map(|(k, v)| (k.clone(), features::get_addrs(v.as_slice().unwrap().to_vec())))
                     .collect::<Vec<_>>();
    println!("{:?}", fs);
    fs
}


fn parse_digits_config(fname: &String) -> digits_benchmark::DigitsBenchConfig {
    let path = Path::new(fname);
    let display = path.display();

    let mut file = match File::open(&path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => panic!(format!("couldn't open {}: REASON: {}", display,
                                                   Error::description(&why))),
        Ok(file) => BufReader::new(file),
    };

    let mut toml_string = String::new();
    match file.read_to_string(&mut toml_string) {
      Err(why) => panic!("couldn't read {}: {}", display,
                         Error::description(&why)),
                         Ok(_) => print!("{} contains:\n{}", display, toml_string),
    }

    let conf = toml::Parser::new(&toml_string).parse().unwrap();
    let dbc = digits_benchmark::DigitsBenchConfig {
      num_users: conf.get("users")
        .unwrap_or(&toml::Value::Integer(100)).as_integer().unwrap() as usize,
      num_train_examples: conf.get("train_examples")
        .unwrap_or(&toml::Value::Integer(30)).as_integer().unwrap() as usize,
      num_test_examples: conf.get("test_examples")
        .unwrap_or(&toml::Value::Integer(50)).as_integer().unwrap() as usize,
      mnist_path: conf.get("mnist_path")
        .unwrap().as_str().unwrap().to_string(),
      num_events: conf.get("num_events")
        .unwrap_or(&toml::Value::Integer(100000)).as_integer().unwrap() as usize,
      num_workers: conf.get("worker_threads")
        .unwrap_or(&toml::Value::Integer(2)).as_integer().unwrap() as usize,
      target_qps: conf.get("target_qps")
        .unwrap_or(&toml::Value::Integer(5000)).as_integer().unwrap() as usize,
      query_batch_size: conf.get("query_batch_size")
        .unwrap_or(&toml::Value::Integer(200)).as_integer().unwrap() as usize,
      max_features: conf.get("max_features")
        .unwrap_or(&toml::Value::Integer(10)).as_integer().unwrap() as usize,
      salt_hash: conf.get("salt_hash")
        .unwrap_or(&toml::Value::Boolean(true)).as_bool().unwrap(),
      feature_batch_size: conf.get("feature_batch_size")
        .unwrap_or(&toml::Value::Integer(100)).as_integer().unwrap() as usize,

    };
    // let fs = features.get("features").unwrap()
    //                  .as_table().unwrap().iter()
    //                  .map(|(k, v)| (k.clone(), features::get_addr(v.as_str().unwrap().to_string())))
    //                  .collect::<Vec<_>>();
    // println!("{:?}", fs);
    dbc

}







