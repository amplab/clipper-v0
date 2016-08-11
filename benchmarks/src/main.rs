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
#[allow(unused_imports)]
use clipper::server::{ClipperServer, Input, PredictionRequest, UpdateRequest, Update};
use clipper::configuration;
use clipper::metrics;
#[allow(unused_imports)]
use clipper::correction_policy::{LinearCorrectionState, LogisticRegressionPolicy, AveragePolicy};
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

const LABEL: f64 = 3.0;

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
                Ok(_) |
                Err(mpsc::TryRecvError::Empty) => {
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
        .unwrap_or(&Value::Integer(100000))
        .as_integer()
        .unwrap() as usize;
    let target_qps = pc.get("target_qps")
        .unwrap_or(&Value::Integer(10000))
        .as_integer()
        .unwrap() as usize;
    let batch_size = pc.get("bench_batch_size")
        .unwrap_or(&Value::Integer(100))
        .as_integer()
        .unwrap() as usize;
    let all_test_data = digits::load_mnist_dense(&mnist_path).unwrap();
    // let norm_test_data = all_test_data;
    let norm_test_data = digits::normalize(&all_test_data);
    let mut first_three_pos = -1;
    let mut last_three_pos = -1;
    for i in 0..all_test_data.ys.len() {
      if all_test_data.ys[i] == LABEL {
        if first_three_pos < 0 {
          first_three_pos = i as i32;
        }
      } else if all_test_data.ys[i] == LABEL + 1.0 {
        if last_three_pos < 0 {
          last_three_pos = (i - 1) as i32;
        }
      }
    }
    assert_eq!(all_test_data.ys[first_three_pos as usize], LABEL);
    assert_eq!(all_test_data.ys[(first_three_pos - 1) as usize], LABEL - 1.0);
    assert_eq!(all_test_data.ys[last_three_pos as usize], LABEL);
    assert_eq!(all_test_data.ys[(last_three_pos + 1) as usize], LABEL + 1.0);


    info!("MNIST data loaded: {} points", norm_test_data.ys.len());
    // let input_type = InputType::Integer(784);


    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    let clipper = Arc::new(ClipperServer::<LogisticRegressionPolicy,
                                           LinearCorrectionState>::new(config));
    // let clipper = Arc::new(ClipperServer::<AveragePolicy,
    //                                        ()>::new(config));

    let report_interval_secs = 10;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();


    info!("starting benchmark");
    let (sender, receiver) = mpsc::channel::<(f64, bool)>();

    let accuracy_counter = {
        let acc_counter_name = format!("digits accuracy ratio");
        let clipper_metrics = clipper.get_metrics();
        let mut m = clipper_metrics.write().unwrap();
        m.create_ratio_counter(acc_counter_name)
    };

    let pred_ones_counter = {
        let counter_name = format!("pred_ones_counter");
        let clipper_metrics = clipper.get_metrics();
        let mut m = clipper_metrics.write().unwrap();
        m.create_counter(counter_name)
    };

    let pred_zeros_counter = {
        let counter_name = format!("pred_zeros_counter");
        let clipper_metrics = clipper.get_metrics();
        let mut m = clipper_metrics.write().unwrap();
        m.create_counter(counter_name)
    };

    let receiver_jh = thread::spawn(move || {
        for _ in 0..num_requests {
            let (pred, correct) = receiver.recv().unwrap();
            if correct {
                accuracy_counter.incr(1,1);
            } else {
                accuracy_counter.incr(0,1);
            }
            
            if pred == 1.0 {
              pred_ones_counter.incr(1);
            } else if pred == -1.0 {
              pred_zeros_counter.incr(1);
            } else {
              warn!("unexpected label: {}", pred);
            }
        }
    });

    let mut events_fired = 0;
    let mut rng = thread_rng();
    let num_users = 1;
    // let batch_size = 200;
    let inter_batch_sleep_time_ms = 1000 / (target_qps / batch_size) as u64;

    thread::sleep(Duration::from_secs(10));

    // train correction policy
    let mut num_pos_examples = 0;
    let mut updates = Vec::new();
    while num_pos_examples < 200 {
        // let idx = rng.gen_range::<usize>(0, norm_test_data.ys.len());

        let idx = if updates.len() % 2 == 0 {
          rng.gen_range::<usize>(last_three_pos as usize + 10, norm_test_data.ys.len())
        } else {
          rng.gen_range::<usize>(first_three_pos as usize, last_three_pos as usize)
        };
        let label = norm_test_data.ys[idx];
        let input_data = norm_test_data.xs[idx].clone();
        let y = if label == LABEL {
          num_pos_examples += 1;
          1.0
        } else {
          -1.0
        };
        let input = Input::Floats {
            f: input_data,
            length: 784,
        };
        updates.push(Update { query: Arc::new(input), label: y });
    }
    // let user = rng.gen_range::<u32>(0, num_users);
    let user = 1;

    let update_req = UpdateRequest::new(user, updates);
    clipper.schedule_update(update_req);
    thread::sleep(Duration::from_secs(10));
    {
      let metrics_register = clipper.get_metrics();
      let m = metrics_register.read().unwrap();
      info!("{}", m.report());
      m.reset();
    }
    let _ = launch_monitor_thread(clipper.get_metrics(),
                                  report_interval_secs,
                                  metrics_signal_rx);
    while events_fired < num_requests {
        for _ in 0..batch_size {
            if events_fired % 20000 == 0 {
                info!("Submitted {} requests", events_fired);
            }
            let idx = if events_fired % 2 == 0 {
              rng.gen_range::<usize>(last_three_pos as usize + 10, norm_test_data.ys.len())
            } else {
              rng.gen_range::<usize>(first_three_pos as usize, last_three_pos as usize)
            };
            let input_data = norm_test_data.xs[idx].clone();
            // make each input unique so there are no cache hits
            // input_data[783] = events_fired as f64;
            let label = norm_test_data.ys[idx];
            let y = if label == LABEL {
                1.0
            } else {
                -1.0
            };
            let input = Input::Floats {
                f: input_data,
                length: 784,
            };
            // let user = rng.gen_range::<u32>(0, num_users);
            {
                let sender = sender.clone();
                // let req_num = events_fired;
                let on_pred = Box::new(move |pred_y| {
                    sender.send((pred_y, pred_y == y)).unwrap();
                    // if req_num % 100 == 0 {
                    //     info!("completed prediction {}", req_num);
                    // }
                });
                let r = PredictionRequest::new(user, input, on_pred);
                clipper.schedule_prediction(r);
            }
            events_fired += 1;
        }
        thread::sleep(Duration::from_millis(inter_batch_sleep_time_ms));
    }

    receiver_jh.join().unwrap();

    // for _ in 0..num_requests {
    //     let correct = receiver.recv().unwrap();
    //     if correct {
    //         accuracy_counter.incr(1,1);
    //     } else {
    //         accuracy_counter.incr(0,1);
    //     }
    // }
    // let _ = receiver.iter().take(num_requests).collect::<Vec<_>>();
}
