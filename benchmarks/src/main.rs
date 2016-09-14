#![deny(warnings)]

extern crate clipper;
// extern crate time;
extern crate toml;
extern crate rand;
extern crate rustc_serialize;
extern crate docopt;
extern crate time;

#[macro_use]
extern crate log;
extern crate env_logger;

use rand::{thread_rng, Rng, ThreadRng};
#[allow(unused_imports)]
use clipper::server::{ClipperServer, Input, InputType, PredictionRequest, UpdateRequest, Update};
use clipper::configuration;
use clipper::metrics;
#[allow(unused_imports)]
use clipper::correction_policy::{LinearCorrectionState, LogisticRegressionPolicy, AveragePolicy};
// use clipper::batching::BatchStrategy;
use std::thread;
use std::time::Duration;
// use time;
use std::sync::{mpsc, Arc, RwLock};
// use docopt::Docopt;
use toml::{Parser, Value};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::{BufReader, BufWriter};
use std::error::Error;
use std::collections::VecDeque;
use std::env;

mod digits;

// #[cfg_attr(rustfmt, rustfmt_skip)]
// const USAGE: &'static str = "
// Clipper Server
//
// Usage:
//   clipper digits --conf=</path/to/conf.toml>
//   clipper imagenet --conf=</path/to/conf.toml>
//   clipper -h
//
// Options:
//   -h --help                             Show this screen.
//   --conf=</path/to/conf.toml>           Path to features config file.
// ";

const LABEL: f64 = 3.0;

#[derive(Debug, RustcDecodable)]
struct Args {
    flag_conf: String,
    cmd_digits: bool,
    cmd_imagenet: bool,
}

fn main() {
    env_logger::init().unwrap();

    let conf_key = "CLIPPER_CONF_PATH";
    let command_key = "CLIPPER_BENCH_COMMAND";

    // let args: Args = Docopt::new(USAGE)
    //     .and_then(|d| d.decode())
    //     .unwrap_or_else(|e| e.exit());

    // info!("{:?}", args);
    let conf = env::var(conf_key).unwrap();
    match env::var(command_key).unwrap().as_str() {
        "digits" => start_digits_benchmark(&conf),
        "imagenet" => start_imagenet_benchmark(&conf),
        "cifar" => start_cifar_benchmark(&conf),
        "thruputjkdshfkjsdhfds" => start_thruput_benchmark(&conf),
        _ => panic!("Invalid benchmark command"),
    }

    // if args.cmd_digits {
    //     start_digits_benchmark(&args.flag_conf);
    // } else if args.cmd_imagenet {
    //     start_imagenet_benchmark(&args.flag_conf);
    // }
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


trait LoadGenerator {
    fn next_request(&mut self) -> bool;
}

#[allow(dead_code)]
struct BatchedUniformLoadGenerator {
    bench_batch_size: usize,
    total_requests: usize,
    // requested_load_qps: usize,
    inter_batch_sleep_time_ms: u64,

    // mutable
    current_batch_count: usize,
    total_count: usize,
}

impl BatchedUniformLoadGenerator {
    #[allow(dead_code)]
    pub fn new(batch_size: usize,
               total_reqs: usize,
               requested_load_qps: usize)
               -> BatchedUniformLoadGenerator {

        let sleep_time = 1000 / (requested_load_qps / batch_size) as u64;
        info!("SLEEP TIME: {}", sleep_time);
        BatchedUniformLoadGenerator {
            bench_batch_size: batch_size,
            total_requests: total_reqs,
            current_batch_count: 0,
            total_count: 0,
            // requested_load_qps: requested_load_qps,
            inter_batch_sleep_time_ms: sleep_time,
        }
    }
}

impl LoadGenerator for BatchedUniformLoadGenerator {
    fn next_request(&mut self) -> bool {
        if self.total_count < self.total_requests {
            self.total_count += 1;
            self.current_batch_count += 1;
            if self.current_batch_count >= self.bench_batch_size {
                self.current_batch_count = 0;
                // info!("INTER BATCH SLEEPING!! total count: {}", self.total_count);
                thread::sleep(Duration::from_millis(self.inter_batch_sleep_time_ms));
            }
            true
        } else {
            false
        }
    }
}



struct UniformLoadGenerator {
    total_requests: usize,
    spin_time_nanos: u64,
    total_count: usize,
}

impl UniformLoadGenerator {
    pub fn new(total_reqs: usize, requested_load_qps: usize) -> UniformLoadGenerator {

        // let sleep_time = 1000 / (requested_load_qps / batch_size) as u64;
        // info!("SLEEP TIME: {}", sleep_time);
        let nanos_per_sec: u64 = 1000 * 1000 * 1000;
        let spin_time_nanos: u64 = nanos_per_sec / (requested_load_qps as u64);
        info!("SPIN TIME: {}", spin_time_nanos);
        UniformLoadGenerator {
            total_requests: total_reqs,
            spin_time_nanos: spin_time_nanos,
            total_count: 0,
        }
    }
}

impl LoadGenerator for UniformLoadGenerator {
    fn next_request(&mut self) -> bool {
        if self.total_count < self.total_requests {
            self.total_count += 1;
            let spin_end = time::precise_time_ns() + self.spin_time_nanos;
            while time::precise_time_ns() < spin_end {
            }
            true
        } else {
            false
        }
    }
}


trait RequestGenerator {
    fn get_next_request(&mut self, request_num: usize) -> (Vec<f64>, f64);
    fn get_next_update(&mut self) -> Option<(Vec<f64>, f64)> {
        unimplemented!();
    }
}



struct CacheHitRequestGenerator {
    rng: ThreadRng,
    cached_inputs: Vec<(Vec<f64>, f64)>,
    cache_hit_rate: f64,
}

impl CacheHitRequestGenerator {
    pub fn new(cache_hit_rate: f64) -> CacheHitRequestGenerator {
        CacheHitRequestGenerator {
            rng: thread_rng(),
            cached_inputs: Vec::new(),
            cache_hit_rate: cache_hit_rate,
        }
    }

    fn random_features(&mut self, d: usize) -> Vec<f64> {
        self.rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
    }
}

impl RequestGenerator for CacheHitRequestGenerator {
    fn get_next_request(&mut self, _: usize) -> (Vec<f64>, f64) {

        if self.cached_inputs.len() < 100 {
            let input = self.random_features(784);
            let label = self.rng.gen::<bool>();
            let y = if label { 1.0 } else { -1.0 };
            self.cached_inputs.push((input.clone(), y));
            (input, y)
        } else {
            let cache_hit = self.rng.gen_range::<f64>(0.0, 1.0) <= self.cache_hit_rate;
            if cache_hit {
                let idx = self.rng.gen_range::<usize>(0, self.cached_inputs.len());
                self.cached_inputs[idx].clone()

            } else {
                let input = self.random_features(784);
                let label = self.rng.gen::<bool>();
                let y = if label { 1.0 } else { -1.0 };
                (input, y)
            }
        }
    }
}

struct CachedUpdatesRequestGenerator {
    rng: ThreadRng,
    sent_predictions: VecDeque<(Vec<f64>, f64)>,
}

impl CachedUpdatesRequestGenerator {
    pub fn new() -> CachedUpdatesRequestGenerator {
        CachedUpdatesRequestGenerator {
            rng: thread_rng(),
            sent_predictions: VecDeque::with_capacity(100000),
        }
    }

    fn random_features(&mut self, d: usize) -> Vec<f64> {
        self.rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
    }
}

impl RequestGenerator for CachedUpdatesRequestGenerator {
    fn get_next_request(&mut self, _: usize) -> (Vec<f64>, f64) {
        let input = self.random_features(784);
        // let label = self.rng.gen::<bool>();
        // let y = if label { 1.0 } else { -1.0 };
        let y = 1.0;
        self.sent_predictions.push_back((input.clone(), y));
        (input, y)
    }

    fn get_next_update(&mut self) -> Option<(Vec<f64>, f64)> {
        // let the prediction cache warm up first
        if self.sent_predictions.len() > 100000 {
            self.sent_predictions.pop_front()
        } else {
            None
        }
    }
}

/// A request generator that randomly selects positive and
/// negative examples with equal frequency. Good for accuracy
/// experiments.
struct BalancedRequestGenerator {
    min_pos_index: usize,
    max_pos_index: usize,
    min_neg_index: usize,
    max_neg_index: usize,
    rng: ThreadRng,
    data: digits::TrainingData,
}

impl BalancedRequestGenerator {
    pub fn new(min_pos_index: usize,
               max_pos_index: usize,
               min_neg_index: usize,
               max_neg_index: usize,
               data: digits::TrainingData)
               -> BalancedRequestGenerator {

        BalancedRequestGenerator {
            min_pos_index: min_pos_index,
            max_pos_index: max_pos_index,
            min_neg_index: min_neg_index,
            max_neg_index: max_neg_index,
            rng: thread_rng(),
            data: data,
        }
    }
}

/// ///////////////////////////

struct RandomRequestGenerator {
    rng: ThreadRng,
    data: Vec<Vec<f64>>,
}

impl RandomRequestGenerator {
    pub fn new(data: Vec<Vec<f64>>) -> RandomRequestGenerator {
        RandomRequestGenerator {
            rng: thread_rng(),
            data: data,
        }
    }
}

impl RequestGenerator for RandomRequestGenerator {
    fn get_next_request(&mut self, _: usize) -> (Vec<f64>, f64) {
        let idx = self.rng.gen_range::<usize>(0, self.data.len());
        let input_data = self.data[idx].clone();
        let y = 1.0;
        (input_data, y)
    }
}

struct RandomSizeRequestGenerator {
    // size: usize,
    message: Vec<f64>,
}

impl RandomSizeRequestGenerator {
    pub fn new(size: usize) -> RandomSizeRequestGenerator {
        let mut rng = thread_rng();
        let message = rng.gen_iter::<f64>().take(size).collect::<Vec<f64>>();
        RandomSizeRequestGenerator {
            // size: size,
            message: message,
        }
    }
}

impl RequestGenerator for RandomSizeRequestGenerator {
    fn get_next_request(&mut self, _: usize) -> (Vec<f64>, f64) {
        (self.message.clone(), 1.0)
        // (self.rng.gen_iter::<f64>().take(self.size).collect::<Vec<f64>>(), 1.0)
    }
}



/// /////////////

impl RequestGenerator for BalancedRequestGenerator {
    fn get_next_request(&mut self, request_num: usize) -> (Vec<f64>, f64) {
        let idx = if request_num % 2 == 0 {
            self.rng.gen_range::<usize>(self.min_neg_index, self.max_neg_index)
        } else {
            self.rng.gen_range::<usize>(self.min_pos_index, self.max_pos_index)
        };

        let input_data = self.data.xs[idx].clone();
        let label = self.data.ys[idx];
        let y = if label == LABEL { 1.0 } else { -1.0 };
        (input_data, y)
    }
}

#[allow(unused_variables)] // needed for metrics shutdown signal
fn start_thruput_benchmark(conf_path: &String) {
    if true {
        panic!("THIS BENCHMARK IS BROKEN");
    }
    let path = Path::new(conf_path);
    let display = path.display();
    // let message_size_key = "CLIPPER_MESSAGE_SIZE";
    // let message_size = env::var(message_size_key).unwrap().parse::<i32>().unwrap();


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
    let mut parser = Parser::new(&toml_string);
    let pc = match parser.parse() {
        Some(pc) => pc,
        None => panic!(format!("TOML PARSE ERROR: {:?}", parser.errors)),
    };
    let results_path = pc.get("results_path").unwrap().as_str().unwrap().to_string();
    let num_requests = pc.get("num_benchmark_requests")
        .unwrap_or(&Value::Integer(100000))
        .as_integer()
        .unwrap() as usize;

    let message_size = pc.get("input_length")
        .unwrap()
        .as_integer()
        .unwrap() as i32;
    let salt_cache = true;
    let wait_to_end = pc.get("wait_to_end")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();

    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    // don't do batching
    // config.batch_strategy = BatchStrategy::Static { size: 1 };
    // config.input_type = InputType::Float(message_size);
    let instance_name = config.name.clone();
    let clipper = Arc::new(ClipperServer::<LogisticRegressionPolicy,
                                           LinearCorrectionState>::new(config));
    // let clipper = Arc::new(ClipperServer::<AveragePolicy,
    //                                        ()>::new(config));

    let report_interval_secs = 10;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();


    info!("starting benchmark");
    let (sender, receiver) = mpsc::channel::<(f64)>();
    let receiver_jh = thread::spawn(move || {
        for _ in 0..num_requests {
            let _ = receiver.recv().unwrap();
        }
    });

    let mut events_fired = 0;
    // let mut rng = thread_rng();
    // let num_users = 1;
    // let batch_size = 200;
    // let inter_batch_sleep_time_ms = 1000 / (target_qps / batch_size) as u64;

    thread::sleep(Duration::from_secs(10));
    let mut request_generator: Box<RequestGenerator> =
        Box::new(RandomSizeRequestGenerator::new(message_size as usize));

    let mut updates = Vec::new();

    let input0 = Input::Floats {
        f: request_generator.get_next_request(0).0,
        length: message_size,
    };
    updates.push(Update {
        query: Arc::new(input0),
        label: 1.0,
    });
    let input1 = Input::Floats {
        f: request_generator.get_next_request(0).0,
        length: message_size,
    };
    updates.push(Update {
        query: Arc::new(input1),
        label: 1.0,
    });
    // }
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

    // let mut load_gen = UniformLoadGenerator::new(batch_size, num_requests, target_qps);


    for _ in 0..num_requests {
        if events_fired % 20000 == 0 {
            info!("Submitted {} requests", events_fired);
        }

        let (input_data, _) = request_generator.get_next_request(events_fired);
        let input = Input::Floats {
            f: input_data,
            length: message_size,
        };
        // let user = rng.gen_range::<u32>(0, num_users);
        {
            let sender = sender.clone();
            // let req_num = events_fired;
            let on_pred = Box::new(move |pred_y| {
                match sender.send(pred_y) {
                    Ok(_) => {}
                    Err(e) => warn!("error in on_pred: {}", e.description()),
                };
            });
            let r = PredictionRequest::new(user, input, on_pred, salt_cache);
            clipper.schedule_prediction(r);
        }
        events_fired += 1;

    }

    thread::sleep(Duration::from_secs(5));

    // Record mid-workload metrics as soon as we finish sending requests,
    // instead of waiting for all requests to finish
    {
        let metrics_register = clipper.get_metrics();
        let m = metrics_register.read().unwrap();
        let final_metrics = m.report();
        // let timestamp = time::strftime("%Y%m%d-%H_%M_%S", &time::now()).unwrap();
        // let results_fname = format!("{}/{}_results.json", results_path, timestamp);
        let results_fname = format!("{}/{}_results.json", results_path, instance_name);
        info!("writing results to: {}", results_fname);
        let res_path = Path::new(&results_fname);
        let mut results_writer = BufWriter::new(File::create(res_path).unwrap());
        results_writer.write(&final_metrics.into_bytes()).unwrap();
    }
    if wait_to_end {
        receiver_jh.join().unwrap();
    } else {
        std::process::exit(0);
    }

}

/// /////////////////////////////////////////////////////////
#[allow(unused_variables)] // needed for metrics shutdown signal
fn start_cifar_benchmark(conf_path: &String) {
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
    let mut parser = Parser::new(&toml_string);
    let pc = match parser.parse() {
        Some(pc) => pc,
        None => panic!(format!("TOML PARSE ERROR: {:?}", parser.errors)),
    };
    let cifar_path = pc.get("cifar_path").unwrap().as_str().unwrap().to_string();
    let results_path = pc.get("results_path").unwrap().as_str().unwrap().to_string();
    let num_requests = pc.get("num_benchmark_requests")
        .unwrap_or(&Value::Integer(100000))
        .as_integer()
        .unwrap() as usize;
    let target_qps = pc.get("target_qps")
        .unwrap_or(&Value::Integer(1000))
        .as_integer()
        .unwrap() as usize;
    // let batch_size = pc.get("bench_batch_size")
    //     .unwrap_or(&Value::Integer(100))
    //     .as_integer()
    //     .unwrap() as usize;
    let salt_cache = pc.get("salt_cache")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();
    let wait_to_end = pc.get("wait_to_end")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();



    let input_data = digits::load_imagenet_dense(&cifar_path).unwrap();
    info!("Loaded CIFAR10 data");

    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    let instance_name = config.name.clone();
    let clipper = Arc::new(ClipperServer::<LogisticRegressionPolicy,
                                           LinearCorrectionState>::new(config));
    // let clipper = Arc::new(ClipperServer::<AveragePolicy,
    //                                        ()>::new(config));

    let report_interval_secs = 10;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();


    info!("starting benchmark");
    let (sender, receiver) = mpsc::channel::<(f64)>();


    let receiver_jh = thread::spawn(move || {
        for _ in 0..num_requests {
            let _ = receiver.recv().unwrap();
        }
    });

    let mut events_fired = 0;

    thread::sleep(Duration::from_secs(10));

    let mut updates = Vec::new();
    let input0 = Input::Floats {
        f: input_data[0].clone(),
        length: input_data[0].len() as i32,
    };
    updates.push(Update {
        query: Arc::new(input0),
        label: 1.0,
    });
    let input1 = Input::Floats {
        f: input_data[1].clone(),
        length: input_data[0].len() as i32,
    };
    updates.push(Update {
        query: Arc::new(input1),
        label: 1.0,
    });
    // }
    // let user = rng.gen_range::<u32>(0, num_users);
    let user = 1;
    let input_length = input_data[0].len() as i32;

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

    let mut load_gen = UniformLoadGenerator::new(num_requests, target_qps);

    let mut request_generator: Box<RequestGenerator> =
        Box::new(RandomRequestGenerator::new(input_data));

    while load_gen.next_request() {
        if events_fired % 20000 == 0 {
            info!("Submitted {} requests", events_fired);
        }

        let (input_data, _) = request_generator.get_next_request(events_fired);
        let input = Input::Floats {
            f: input_data,
            length: input_length,
        };
        // let user = rng.gen_range::<u32>(0, num_users);
        {
            let sender = sender.clone();
            // let req_num = events_fired;
            let on_pred = Box::new(move |pred_y| {
                match sender.send(pred_y) {
                    Ok(_) => {}
                    Err(e) => warn!("error in on_pred: {}", e.description()),
                };
            });
            let r = PredictionRequest::new(user, input, on_pred, salt_cache);
            clipper.schedule_prediction(r);
        }
        events_fired += 1;

    }

    // Record mid-workload metrics as soon as we finish sending requests,
    // instead of waiting for all requests to finish
    {
        let metrics_register = clipper.get_metrics();
        let m = metrics_register.read().unwrap();
        let final_metrics = m.report();
        // let timestamp = time::strftime("%Y%m%d-%H_%M_%S", &time::now()).unwrap();
        // let results_fname = format!("{}/{}_results.json", results_path, timestamp);
        let results_fname = format!("{}/{}_results.json", results_path, instance_name);
        info!("writing results to: {}", results_fname);
        let res_path = Path::new(&results_fname);
        let mut results_writer = BufWriter::new(File::create(res_path).unwrap());
        results_writer.write(&final_metrics.into_bytes()).unwrap();
    }
    if wait_to_end {
        receiver_jh.join().unwrap();
    } else {
        std::process::exit(0);

    }

}



/// //////////////////////////////////////////////////////////
#[allow(unused_variables)] // needed for metrics shutdown signal
fn start_imagenet_benchmark(conf_path: &String) {
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
    let mut parser = Parser::new(&toml_string);
    let pc = match parser.parse() {
        Some(pc) => pc,
        None => panic!(format!("TOML PARSE ERROR: {:?}", parser.errors)),
    };
    let imagenet_path = pc.get("imagenet_path").unwrap().as_str().unwrap().to_string();
    let results_path = pc.get("results_path").unwrap().as_str().unwrap().to_string();
    let num_requests = pc.get("num_benchmark_requests")
        .unwrap_or(&Value::Integer(100000))
        .as_integer()
        .unwrap() as usize;
    let target_qps = pc.get("target_qps")
        .unwrap_or(&Value::Integer(1000))
        .as_integer()
        .unwrap() as usize;
    let batch_size = pc.get("bench_batch_size")
        .unwrap_or(&Value::Integer(100))
        .as_integer()
        .unwrap() as usize;
    let salt_cache = pc.get("salt_cache")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();
    let wait_to_end = pc.get("wait_to_end")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();



    // info!("MNIST data loaded: {} points", norm_test_data.ys.len());
    // let input_type = InputType::Integer(784);
    // let input_data =
    let input_data = digits::load_imagenet_dense(&imagenet_path).unwrap();
    info!("Loaded imagenet data");

    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    let instance_name = config.name.clone();
    let clipper = Arc::new(ClipperServer::<LogisticRegressionPolicy,
                                           LinearCorrectionState>::new(config));
    // let clipper = Arc::new(ClipperServer::<AveragePolicy,
    //                                        ()>::new(config));

    let report_interval_secs = 10;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();


    info!("starting benchmark");
    let (sender, receiver) = mpsc::channel::<(f64)>();


    let receiver_jh = thread::spawn(move || {
        for _ in 0..num_requests {
            let _ = receiver.recv().unwrap();
        }
    });

    let mut events_fired = 0;
    // let mut rng = thread_rng();
    // let num_users = 1;
    // let batch_size = 200;
    // let inter_batch_sleep_time_ms = 1000 / (target_qps / batch_size) as u64;

    thread::sleep(Duration::from_secs(10));

    // // train correction policy
    // let mut num_pos_examples = 0;
    let mut updates = Vec::new();
    // while num_pos_examples < 200 {
    //     // let idx = rng.gen_range::<usize>(0, norm_test_data.ys.len());
    //
    //     let idx = if updates.len() % 2 == 0 {
    //         rng.gen_range::<usize>(last_three_pos as usize + 10, norm_test_data.ys.len())
    //     } else {
    //         rng.gen_range::<usize>(first_three_pos as usize, last_three_pos as usize)
    //     };
    //     let label = norm_test_data.ys[idx];
    //     let input_data = norm_test_data.xs[idx].clone();
    //     let y = if label == LABEL {
    //         num_pos_examples += 1;
    //         1.0
    //     } else {
    //         -1.0
    //     };
    let input0 = Input::Floats {
        f: input_data[0].clone(),
        length: 268203,
    };
    updates.push(Update {
        query: Arc::new(input0),
        label: 1.0,
    });
    let input1 = Input::Floats {
        f: input_data[1].clone(),
        length: 268203,
    };
    updates.push(Update {
        query: Arc::new(input1),
        label: 1.0,
    });
    // }
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

    let mut load_gen = UniformLoadGenerator::new(num_requests, target_qps);

    let mut request_generator: Box<RequestGenerator> =
        Box::new(RandomRequestGenerator::new(input_data));

    while load_gen.next_request() {
        if events_fired % 20000 == 0 {
            info!("Submitted {} requests", events_fired);
        }

        let (input_data, _) = request_generator.get_next_request(events_fired);
        let input = Input::Floats {
            f: input_data,
            length: 268203,
        };
        // let user = rng.gen_range::<u32>(0, num_users);
        {
            let sender = sender.clone();
            // let req_num = events_fired;
            let on_pred = Box::new(move |pred_y| {
                match sender.send(pred_y) {
                    Ok(_) => {}
                    Err(e) => warn!("error in on_pred: {}", e.description()),
                };
            });
            let r = PredictionRequest::new(user, input, on_pred, salt_cache);
            clipper.schedule_prediction(r);
        }
        events_fired += 1;

    }

    // Record mid-workload metrics as soon as we finish sending requests,
    // instead of waiting for all requests to finish
    {
        let metrics_register = clipper.get_metrics();
        let m = metrics_register.read().unwrap();
        let final_metrics = m.report();
        // let timestamp = time::strftime("%Y%m%d-%H_%M_%S", &time::now()).unwrap();
        // let results_fname = format!("{}/{}_results.json", results_path, timestamp);
        let results_fname = format!("{}/{}_results.json", results_path, instance_name);
        info!("writing results to: {}", results_fname);
        let res_path = Path::new(&results_fname);
        let mut results_writer = BufWriter::new(File::create(res_path).unwrap());
        results_writer.write(&final_metrics.into_bytes()).unwrap();
    }
    if wait_to_end {
        receiver_jh.join().unwrap();
    } else {
        std::process::exit(0);

    }

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
    let mut parser = Parser::new(&toml_string);
    let pc = match parser.parse() {
        Some(t) => t,
        None => panic!("ERROR Parsing toml: {:?}", parser.errors),
    };
    let mnist_path = pc.get("mnist_path").unwrap().as_str().unwrap().to_string();
    let results_path = pc.get("results_path").unwrap().as_str().unwrap().to_string();
    let num_requests = pc.get("num_benchmark_requests")
        .unwrap_or(&Value::Integer(100000))
        .as_integer()
        .unwrap() as usize;
    let target_qps = pc.get("target_qps")
        .unwrap_or(&Value::Integer(1000))
        .as_integer()
        .unwrap() as usize;
    let batch_size = pc.get("bench_batch_size")
        .unwrap_or(&Value::Integer(100))
        .as_integer()
        .unwrap() as usize;
    let report_interval_secs = pc.get("report_interval_secs")
        .unwrap_or(&Value::Integer(10))
        .as_integer()
        .unwrap() as u64;
    let salt_cache = pc.get("salt_cache")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();
    let wait_to_end = pc.get("wait_to_end")
        .unwrap_or(&Value::Boolean(true))
        .as_bool()
        .unwrap();
    let send_updates = pc.get("send_updates")
        .unwrap_or(&Value::Boolean(false))
        .as_bool()
        .unwrap();
    let load_generator_name = pc.get("load_generator")
        .unwrap_or(&Value::String("uniform".to_string()))
        .as_str()
        .unwrap()
        .to_string();
    let req_generator_name = pc.get("request_generator")
        .unwrap_or(&Value::String("balanced".to_string()))
        .as_str()
        .unwrap()
        .to_string();
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
    assert_eq!(all_test_data.ys[(first_three_pos - 1) as usize],
               LABEL - 1.0);
    assert_eq!(all_test_data.ys[last_three_pos as usize], LABEL);
    assert_eq!(all_test_data.ys[(last_three_pos + 1) as usize], LABEL + 1.0);


    info!("MNIST data loaded: {} points", norm_test_data.ys.len());
    // let input_type = InputType::Integer(784);


    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    let instance_name = config.name.clone();
    let clipper = Arc::new(ClipperServer::<LogisticRegressionPolicy,
                                           LinearCorrectionState>::new(config));
    // let clipper = Arc::new(ClipperServer::<AveragePolicy,
    //                                        ()>::new(config));

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
                accuracy_counter.incr(1, 1);
            } else {
                accuracy_counter.incr(0, 1);
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
    // let inter_batch_sleep_time_ms = 1000 / (target_qps / batch_size) as u64;

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
        updates.push(Update {
            query: Arc::new(input),
            label: y,
        });
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

    let mut load_gen = match load_generator_name.as_str() {
        "uniform" => UniformLoadGenerator::new(num_requests, target_qps),
        _ => panic!("{} is unsupported load generator type", load_generator_name),
    };

    let mut request_generator: Box<RequestGenerator> = match req_generator_name.as_str() {
        "balanced" => {
            Box::new(BalancedRequestGenerator::new(first_three_pos as usize,
                                                   last_three_pos as usize,
                                                   last_three_pos as usize + 1,
                                                   norm_test_data.ys.len(),
                                                   norm_test_data))
        }
        "cached_updates" => Box::new(CachedUpdatesRequestGenerator::new()),
        "cache_hits" => {
            Box::new(CacheHitRequestGenerator::new(pc.get("cache_hit_rate")
                .unwrap()
                .as_float()
                .unwrap()))
        }
        _ => {
            panic!("{} is unsupported request generator type",
                   req_generator_name)
        }

    };

    while load_gen.next_request() {
        if events_fired % 20000 == 0 {
            info!("Submitted {} requests", events_fired);
        }
        let mut send_update_req: bool = false;
        if send_updates {
            if rng.gen_range::<f64>(0.0, 1.0) >= 0.55 {
                send_update_req = true;
            }
        }

        if send_update_req {
            if let Some((input_data, y)) = request_generator.get_next_update() {
                let input = Input::Floats {
                    f: input_data,
                    length: 784,
                };
                let updates = vec![Update {
                                       query: Arc::new(input),
                                       label: y,
                                   }];
                let update_req = UpdateRequest::new(user, updates);
                clipper.schedule_update(update_req);
            }
        } else {

            let (input_data, y) = request_generator.get_next_request(events_fired);
            let input = Input::Floats {
                f: input_data,
                length: 784,
            };
            // let user = rng.gen_range::<u32>(0, num_users);
            {
                let sender = sender.clone();
                // let req_num = events_fired;
                let on_pred = Box::new(move |pred_y| {
                    match sender.send((pred_y, pred_y == y)) {
                        Ok(_) => {}
                        Err(e) => warn!("error in on_pred: {}", e.description()),
                    };
                    // if req_num % 100 == 0 {
                    //     info!("completed prediction {}", req_num);
                    // }
                });
                let r = PredictionRequest::new(user, input, on_pred, salt_cache);
                clipper.schedule_prediction(r);
            }
            events_fired += 1;
        }
    }

    // Record mid-workload metrics as soon as we finish sending requests,
    // instead of waiting for all requests to finish
    {
        let metrics_register = clipper.get_metrics();
        let m = metrics_register.read().unwrap();
        let final_metrics = m.report();
        // let timestamp = time::strftime("%Y%m%d-%H_%M_%S", &time::now()).unwrap();
        // let results_fname = format!("{}/{}_results.json", results_path, timestamp);
        let results_fname = format!("{}/{}_results.json", results_path, instance_name);
        info!("writing results to: {}", results_fname);
        let res_path = Path::new(&results_fname);
        let mut results_writer = BufWriter::new(File::create(res_path).unwrap());
        results_writer.write(&final_metrics.into_bytes()).unwrap();
    }
    if wait_to_end {
        receiver_jh.join().unwrap();
    } else {
        std::process::exit(0);

    }

}
