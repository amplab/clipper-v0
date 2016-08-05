#![deny(warnings)]

extern crate clipper;
// extern crate time;
extern crate toml;
extern crate rand;
extern crate rustc_serialize;
extern crate docopt;

#[macro_use]
extern crate log;
extern crate env_logger;

use rand::{thread_rng, Rng};
use clipper::server::{ClipperServer, Input, PredictionRequest};
use clipper::configuration;
use clipper::metrics;
use clipper::correction_policy::{LinearCorrectionState, LogisticRegressionPolicy};
use std::thread;
use std::time::Duration;
use std::sync::{mpsc, Arc, RwLock};
use docopt::Docopt;
use toml::{Parser, Value};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
use std::error::Error;

mod digits;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Clipper Server

Usage:
  clipper digits --conf=</path/to/conf.toml>
  clipper -h

Options:
  -h --help                             Show this screen.
  --conf=</path/to/conf.toml>           Path to features config file.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_conf: String,
    cmd_digits: bool,
}

fn main() {
    env_logger::init().unwrap();

    let args: Args = Docopt::new(USAGE)
                         .and_then(|d| d.decode())
                         .unwrap_or_else(|e| e.exit());

    info!("{:?}", args);
    if args.cmd_digits {
        start_digits_benchmark(&args.flag_conf);
    }
}

// #[allow(dead_code)]
fn launch_monitor_thread(metrics_register: Arc<RwLock<metrics::Registry>>,
                         report_interval_secs: u64,
                         shutdown_signal_rx: mpsc::Receiver<()>)
                         -> ::std::thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            match shutdown_signal_rx.try_recv() {
                Ok(_) | Err(mpsc::TryRecvError::Empty) => {
                    thread::sleep(Duration::new(report_interval_secs, 0));
                    let m = metrics_register.read().unwrap();
                    info!("{}", m.report());
                    // m.reset();
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        info!("Shutting down metrics thread");
    })
}

#[allow(unused_variables)] // needed for metrics shutdown signal
fn start_digits_benchmark(conf_path: &String) {
    let path = Path::new(conf_path);
    let display = path.display();

    let mut file = match File::open(&path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => {
            panic!(format!("couldn't open {}: REASON: {}",
                           display,
                           Error::description(&why)))
        }
        Ok(file) => BufReader::new(file),
    };

    let mut toml_string = String::new();
    match file.read_to_string(&mut toml_string) {
        Err(why) => panic!("couldn't read {}: {}", display, Error::description(&why)),
        Ok(_) => print!("{} contains:\n{}", display, toml_string),
    }
    let pc = Parser::new(&toml_string).parse().unwrap();
    let mnist_path = pc.get("mnist_path").unwrap().as_str().unwrap().to_string();
    let num_requests = pc.get("num_benchmark_requests")
                         .unwrap_or(&Value::Integer(10000))
                         .as_integer()
                         .unwrap() as usize;
    let all_test_data = digits::load_mnist_dense(&mnist_path).unwrap();
    let norm_test_data = digits::normalize(&all_test_data);

    info!("MNIST data loaded: {} points", norm_test_data.ys.len());
    // let input_type = InputType::Integer(784);


    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    let clipper = Arc::new(ClipperServer::<LogisticRegressionPolicy,
                                           LinearCorrectionState>::new(config));

    let report_interval_secs = 5;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();
    let _ = launch_monitor_thread(clipper.get_metrics(),
                                  report_interval_secs,
                                  metrics_signal_rx);


    info!("starting benchmark");
    let (sender, receiver) = mpsc::channel::<()>();

    let mut events_fired = 0;
    let mut rng = thread_rng();
    let num_users = 1;
    while events_fired < num_requests {
        if events_fired % 20000 == 0 {
            info!("Submitted {} requests", events_fired);
        }
        let idx = rng.gen_range::<usize>(0, norm_test_data.ys.len());
        let input = Input::Floats {
            f: norm_test_data.xs[idx].clone(),
            length: 784,
        };
        let user = rng.gen_range::<u32>(0, num_users);
        {
            let sender = sender.clone();
            // let req_num = events_fired;
            let on_pred = Box::new(move |_| {
                sender.send(()).unwrap();
                // if req_num % 100 == 0 {
                //     info!("completed prediction {}", req_num);
                // }
            });
            let r = PredictionRequest::new(user, input, on_pred);
            clipper.schedule_prediction(r);
        }
        events_fired += 1;
    }

    for _ in 0..num_requests {
        receiver.recv().unwrap();
    }
    // let _ = receiver.iter().take(num_requests).collect::<Vec<_>>();
}
