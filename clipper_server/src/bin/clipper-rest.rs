// #![deny(missing_docs)]
#![deny(warnings)]

extern crate docopt;
extern crate clipper;
extern crate toml;
extern crate rustc_serialize;
extern crate linear_models;
extern crate rand;
extern crate hyper;
extern crate time;
extern crate serde;
#[macro_use]
extern crate log;
extern crate env_logger;

use docopt::Docopt;

pub mod rest;


#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Clipper Server

Usage:
  clipper start --conf=</path/to/conf.toml>
  clipper -h

Options:
  -h --help                             Show this screen.
  --conf=</path/to/conf.toml>           Path to features config file.
";

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(dead_code)]
const OLD_USAGE: &'static str = "
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
    // flag_bench_conf: Option<String>,
    flag_feature_conf: String,
    // arg_b: Option<usize>,
    // cmd_digits: bool,
    // cmd_featurelats: bool,
    cmd_start: bool,
}


pub fn main() {

    env_logger::init().unwrap();

    let args: Args = Docopt::new(USAGE)
                         .and_then(|d| d.decode())
                         .unwrap_or_else(|e| e.exit());

    info!("{:?}", args);

    if args.cmd_start {
        // let features = parse_feature_config(&args.flag_feature_conf);
        rest::start(&args.flag_feature_conf);
    }

    // if args.cmd_digits {
    //     let features = parse_feature_config(&args.flag_feature_conf);
    //     let digits_conf = parse_digits_config(&args.flag_bench_conf.unwrap());
    //     digits_benchmark::run(features, digits_conf);
    // } else if args.cmd_featurelats {
    //     faas_benchmark::feature_batch_latency(args.arg_b.unwrap());
    // } else if args.cmd_start {
    //     let features = parse_feature_config(&args.flag_feature_conf);
    //     rest::start_listening(features, InputType::Integer(784));
    // }
}

// fn parse_feature_config(fname: &String) -> Vec<(String, Vec<SocketAddr>)> {
//
//     let path = Path::new(fname);
//     let display = path.display();
//
//     let mut file = match File::open(&path) {
//         // The `description` method of `io::Error` returns a string that
//         // describes the error
//         Err(why) => {
//             panic!(format!("couldn't open {}: REASON: {}",
//                            display,
//                            Error::description(&why)))
//         }
//         Ok(file) => BufReader::new(file),
//     };
//
//     let mut toml_string = String::new();
//     match file.read_to_string(&mut toml_string) {
//         Err(why) => panic!("couldn't read {}: {}", display, Error::description(&why)),
//         Ok(_) => print!("{} contains:\n{}", display, toml_string),
//     }
//
//     let features = toml::Parser::new(&toml_string).parse().unwrap();
//     let fs = features.get("features")
//                      .unwrap()
//                      .as_table()
//                      .unwrap()
//                      .iter()
//                      .map(|(k, v)| {
//                          (k.clone(),
//                           features::get_addrs(v.as_slice().unwrap().to_vec()))
//                      })
//                      .collect::<Vec<_>>();
//     info!("{:?}", fs);
//     fs
// }


// fn parse_digits_config(fname: &String) -> digits_benchmark::DigitsBenchConfig {
//     let path = Path::new(fname);
//     let display = path.display();
//
//     let mut file = match File::open(&path) {
//         // The `description` method of `io::Error` returns a string that
//         // describes the error
//         Err(why) => {
//             panic!(format!("couldn't open {}: REASON: {}",
//                            display,
//                            Error::description(&why)))
//         }
//         Ok(file) => BufReader::new(file),
//     };
//
//     let mut toml_string = String::new();
//     match file.read_to_string(&mut toml_string) {
//         Err(why) => panic!("couldn't read {}: {}", display, Error::description(&why)),
//         Ok(_) => print!("{} contains:\n{}", display, toml_string),
//     }
//
//     let conf = toml::Parser::new(&toml_string).parse().unwrap();
//     let dbc = digits_benchmark::DigitsBenchConfig {
//         num_users: conf.get("users")
//                        .unwrap_or(&toml::Value::Integer(100))
//                        .as_integer()
//                        .unwrap() as usize,
//         num_train_examples: conf.get("train_examples")
//                                 .unwrap_or(&toml::Value::Integer(30))
//                                 .as_integer()
//                                 .unwrap() as usize,
//         num_test_examples: conf.get("test_examples")
//                                .unwrap_or(&toml::Value::Integer(50))
//                                .as_integer()
//                                .unwrap() as usize,
//         mnist_path: conf.get("mnist_path")
//                         .unwrap()
//                         .as_str()
//                         .unwrap()
//                         .to_string(),
//         num_events: conf.get("num_events")
//                         .unwrap_or(&toml::Value::Integer(100000))
//                         .as_integer()
//                         .unwrap() as usize,
//         num_workers: conf.get("worker_threads")
//                          .unwrap_or(&toml::Value::Integer(2))
//                          .as_integer()
//                          .unwrap() as usize,
//         target_qps: conf.get("target_qps")
//                         .unwrap_or(&toml::Value::Integer(5000))
//                         .as_integer()
//                         .unwrap() as usize,
//         query_batch_size: conf.get("query_batch_size")
//                               .unwrap_or(&toml::Value::Integer(200))
//                               .as_integer()
//                               .unwrap() as usize,
//         max_features: conf.get("max_features")
//                           .unwrap_or(&toml::Value::Integer(10))
//                           .as_integer()
//                           .unwrap() as usize,
//         salt_hash: conf.get("salt_hash")
//                        .unwrap_or(&toml::Value::Boolean(true))
//                        .as_bool()
//                        .unwrap(),
//         feature_batch_size: conf.get("feature_batch_size")
//                                 .unwrap_or(&toml::Value::Integer(100))
//                                 .as_integer()
//                                 .unwrap() as usize,
//     };
//     // let fs = features.get("features").unwrap()
//     //                  .as_table().unwrap().iter()
//     //                  .map(|(k, v)| (k.clone(), features::get_addr(v.as_str().unwrap().to_string())))
//     //                  .collect::<Vec<_>>();
//     // println!("{:?}", fs);
//     dbc
//
// }
