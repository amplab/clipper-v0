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
extern crate linear_models;
extern crate toml;
// extern crate getopts;
extern crate rustc_serialize;
extern crate docopt;

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

pub mod feature_capnp {
  include!(concat!(env!("OUT_DIR"), "/feature_capnp.rs"));
}

// pub mod linalg;
pub mod server;
pub mod digits;
pub mod bench;
pub mod features;
pub mod digits_benchmark;


const USAGE: &'static str = "
Clipper Server

Usage:
  clipper digits --conf=</path/to/features.toml> --mnist=<test.data>
  clipper featurelats --conf=</path/to/features.toml>
  clipper start --conf=</path/to/features.toml>

Options:
  -h --help                             Show this screen.
  --conf=</path/to/features>            Path to features config file.
  --users=<num_users>                   Number of users [default: 100]
  --traindata=<num_examples>            Number of training examples per user [default: 10]
  --testdata=<num_examples>             Number of test examples per user [default: 20]
  --mnist=</path/to/mnist/test.data>    Path to mnist data file
  
";

#[derive(Debug, RustcDecodable)]
struct Args {
    // flag_speed: isize,
    // flag_drifting: bool,
    flag_mnist: Option<String>,
    flag_conf: String,
    flag_users: usize,
    flag_traindata: usize,
    flag_testdata: usize,
    cmd_digits: bool,
    cmd_featurelats: bool,
    cmd_start: bool,
}

pub fn main() {

    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.decode())
                            .unwrap_or_else(|e| e.exit());

    println!("{:?}", args);

    let features = parse_feature_config(&args.flag_conf);
    // let features = parse_feature_config(&"features.toml".to_string());
    if args.cmd_digits {
      let mnist_path = args.flag_mnist.expect("Running digits requires path to mnist data file");
      digits_benchmark::run(features,
                            args.flag_users,
                            args.flag_traindata,
                            args.flag_testdata,
                            mnist_path);
    } else if args.cmd_featurelats {
        features::feature_lats_main(features);
    } else if args.cmd_start {
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

fn parse_feature_config(fname: &String) -> Vec<(String, SocketAddr)> {
    
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
                     .map(|(k, v)| (k.clone(), features::get_addr(v.as_str().unwrap().to_string())))
                     .collect::<Vec<_>>();
    println!("{:?}", fs);
    fs
}







