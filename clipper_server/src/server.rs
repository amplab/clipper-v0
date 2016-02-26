#![allow(unused_variables)]
// #![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_mut)]

// use gj;
// use gj::{EventLoop, Promise};
// use capnp;
use time;
// use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use std::net::{ToSocketAddrs, SocketAddr};
// use feature_capnp::feature;
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use num_cpus;
use linear_models::linalg;
use digits;
use features;
use features::FeatureHash;

pub const SLA: i64 = 100;



// fn anytime_features(features: &Vec<features::FeatureHandle>, input: &Vec<f64>) -> Vec<f64> {
//     // TODO check caches
//     // for f in features {
//     //     f.cache.read()
//     // }
//     vec![-3.2, 5.1]
// }


pub struct Request {
    start_time: time::PreciseTime,
    user: u32, // TODO: remove this because each feature has it's own hash
    input: Vec<f64>,
    true_label: Option<f64>, // used to evaluate accuracy,
    req_number: i32
}

impl Request {

    pub fn new(user: u32, input: Vec<f64>, req_num: i32) -> Request {
        Request { start_time: time::PreciseTime::now(), user: user, input: input, true_label: None, req_number: req_num}
    }

    pub fn new_with_label(user: u32, input: Vec<f64>, label: f64, req_num: i32) -> Request {
        Request {
            start_time: time::PreciseTime::now(),
            user: user,
            input: input,
            true_label: Some(label),
            req_number: req_num
        }
    }

}

struct Update {
    start_time: time::PreciseTime, // just for monitoring purposes
    user: u32,
    input: Vec<f64>,
    label: f32,
}

// fn start_update_worker(feature_handles: Vec<features::FeatureHandle<features::SimpleHasher>>) -> mpsc::Sender<Update> {
//     panic!("unimplemented method");
// }

pub trait TaskModel {
    /// Make a prediction with the available features
    fn predict(&self, mut fs: Vec<f64>, missing_fs: Vec<usize>, debug_str: &String) -> f64;

    fn rank_features(&self) -> Vec<usize>;


    // fn update_anytime_features(&mut self, fs: Vec<f64>, missing_fs: Vec<i32>);

    // fn train(&mut self,
}

struct DummyTaskModel;

impl TaskModel for DummyTaskModel {
    fn predict(&self, fs: Vec<f64>, missing_fs: Vec<usize>, debug_str: &String) -> f64 {
        0.3
    }

    fn rank_features(&self) -> Vec<usize> {
        return (0..10).into_iter().collect::<Vec<usize>>()
    }
}

// Because we don't have a good concurrent hash map, assume we know how many
// users there will be ahead of time. Then we can have a vec of RwLock
// and have row-level locking (because no inserts or deletes).
fn make_prediction<T, H>(feature_handles: &Vec<features::FeatureHandle<H>>,
                      input: &Vec<f64>,
                      task_model: &T,
                      req_id: i32) -> f64
                      where T: TaskModel,
                            H: features::FeatureHash + Send + Sync {

    
    let mut missing_feature_indexes: Vec<usize> = Vec::new();
    let mut features: Vec<f64> = Vec::new();
    let mut i = 0;
    for fh in feature_handles {
        let hash = fh.hasher.hash(input, Some(req_id));
        // println!("hash in prediction: {}", hash);
        let mut cache_reader = fh.cache.read().unwrap();
        let cache_entry = cache_reader.get(&hash);
        match cache_entry {
            Some(v) => features.push(*v),
            None => {
                features.push(0.0);
                missing_feature_indexes.push(i);
            }
        };
        i += 1
    }

    let debug_str = format!("req {}", req_id);
    task_model.predict(features, missing_feature_indexes, &debug_str)
}

fn start_prediction_worker<T, H>(worker_id: i32,
                           sla_millis: i64,
                           feature_handles: Vec<features::FeatureHandle<H>>,
                           user_models: Arc<Vec<RwLock<T>>>,
                           correct_counter: Arc<AtomicUsize>,
                           total_counter: Arc<AtomicUsize>, // tracks total number of requests WITH LABELS processed
                           processed_counter: Arc<AtomicUsize>, // tracks total number of requests processed
                           cum_latency_tracker_micros: Arc<AtomicUsize>, // find mean by doing cum_latency_tracker/processed_counter
                           max_latency_tracker_micros: Arc<AtomicUsize>,
                           ) -> mpsc::Sender<Request> 
                           where T: TaskModel + Send + Sync + 'static,
                                 H: features::FeatureHash + Send + Sync + 'static {

    let sla = time::Duration::milliseconds(sla_millis);
    let epsilon = time::Duration::milliseconds(sla_millis / 5);
    let (sender, receiver) = mpsc::channel::<Request>();
    let join_guard = thread::spawn(move || {
        println!("starting response worker {} with {} ms SLA", worker_id, sla_millis);
        loop {
            let req = receiver.recv().unwrap();
            // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
            let elapsed_time = req.start_time.to(time::PreciseTime::now());
            if elapsed_time < sla - epsilon {
                let sleep_time = ::std::time::Duration::new(
                    0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
                // println!("sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            // TODO: actually compute prediction
            // return result
            let pred_loop_start = time::PreciseTime::now();
            debug_assert!(req.user < user_models.len() as u32);
            let lock = (&user_models).get(*(&req.user) as usize).unwrap();
            let task_model: &T = &lock.read().unwrap();
            let pred = make_prediction(&feature_handles, &req.input, task_model, req.req_number);
            let end_time = time::PreciseTime::now();
            let latency = req.start_time.to(end_time).num_microseconds().unwrap();
            // This isn't consistent (max latency can change between the two calls,
            // but it doesn't if max_latency is totally consistent
            if latency > max_latency_tracker_micros.load(Ordering::Relaxed) as i64 {
                max_latency_tracker_micros.store(latency as usize, Ordering::Relaxed);
            }
            cum_latency_tracker_micros.fetch_add(latency as usize, Ordering::Relaxed);
            // println!("prediction latency: {} ms", latency);
            processed_counter.fetch_add(1, Ordering::Relaxed);
            if req.true_label.is_some() {
                total_counter.fetch_add(1, Ordering::Relaxed);
                if req.true_label.unwrap() == pred {
                    correct_counter.fetch_add(1, Ordering::Relaxed);
                }
            }
            let pred_loop_end = time::PreciseTime::now();
            let pred_loop_latency = pred_loop_start.to(pred_loop_end).num_microseconds().unwrap() as f64;
            // println!("pred LOOP latency: {} (ms)", pred_loop_latency / 1000.0);
        }
    });
    // (join_guard, sender)
    sender
}



pub struct Dispatcher<T: TaskModel + Send + Sync + 'static> {
    workers: Vec<mpsc::Sender<Request>>,
    next_worker: usize,
    feature_handles: Vec<features::FeatureHandle<features::SimpleHasher>>,
    user_models: Arc<Vec<RwLock<T>>>,
}

impl<T: TaskModel + Send + Sync + 'static> Dispatcher<T> {

    // pub fn new<T: TaskModel + Send + Sync + 'static>(num_workers: usize,
    pub fn new(num_workers: usize,
           sla_millis: i64,
           feature_handles: Vec<features::FeatureHandle<features::SimpleHasher>>,
           user_models: Arc<Vec<RwLock<T>>>,
           correct_counter: Arc<AtomicUsize>,
           total_counter: Arc<AtomicUsize>,
           processed_counter: Arc<AtomicUsize>,
           cum_latency_tracker_micros: Arc<AtomicUsize>, // find mean by doing cum_latency_tracker/processed_counter
           max_latency_tracker_micros: Arc<AtomicUsize>,
           ) -> Dispatcher<T> {
        println!("creating dispatcher with {} workers", num_workers);
        let mut worker_threads = Vec::new();
        for i in 0..num_workers {
            let worker = start_prediction_worker(i as i32,
                                                 sla_millis,
                                                 feature_handles.clone(),
                                                 user_models.clone(),
                                                 correct_counter.clone(),
                                                 total_counter.clone(),
                                                 processed_counter.clone(),
                                                 cum_latency_tracker_micros.clone(),
                                                 max_latency_tracker_micros.clone());
            worker_threads.push(worker);
        }
        Dispatcher {
            workers: worker_threads,
            next_worker: 0,
            feature_handles: feature_handles,
            user_models: user_models
        }
    }

    /// Dispatch a request.
    ///
    /// Requires self to be mutable so that we can increment `next_worker`
    pub fn dispatch(&mut self, req: Request, max_features: usize) {

        let mut features_indexes: Vec<usize> = (0..self.feature_handles.len()).into_iter().collect();
        if max_features < self.feature_handles.len() {
            let model = self.user_models[req.user as usize].read().unwrap();
            features_indexes = model.rank_features();
            while max_features < features_indexes.len() {
                features_indexes.pop();
            }
        }
        get_features(&self.feature_handles,
                     req.input.clone(),
                     features_indexes,
                     req.start_time.clone(),
                     Some(req.req_number));
        self.workers[self.next_worker].send(req).unwrap();
        self.increment_worker();
    }

    // for now do round robin scheduling
    fn increment_worker(&mut self) {
        self.next_worker = (self.next_worker + 1) % self.workers.len();
    }
}


fn init_user_models(num_users: usize, num_features: usize)
    -> Arc<Vec<RwLock<DummyTaskModel>>> {
    let mut rng = thread_rng();
    let mut models = Vec::with_capacity(num_users);
    for i in 0..num_users {
        let model = RwLock::new(DummyTaskModel);
        models.push(model);
    }
    Arc::new(models)
}




pub fn main(feature_addrs: Vec<(String, SocketAddr)>) {
    // let addr_vec = vec!["127.0.0.1:6001".to_string()];
    // let names = vec!["sklearn".to_string()];
    // let addr_vec = vec!["127.0.0.1:6001".to_string(), "127.0.0.1:6002".to_string(), "127.0.0.1:6003".to_string()];
    // let names = vec!["TEN_rf".to_string(), "HUNDRED_rf".to_string(), "FIVE_HUNDO_rf".to_string()];
    let num_features = feature_addrs.len();
    let num_users = 500;
    // let test_data_path = "/crankshaw-local/mnist/data/test.data";
    // let all_test_data = digits::load_mnist_dense(test_data_path).unwrap();
    // let norm_test_data = digits::normalize(&all_test_data);

    // println!("Test data loaded: {} points", norm_test_data.ys.len());

    // let (features, handles): (Vec<_>, Vec<_>) = addr_vec.into_iter()
    //                                                     .map(|a| features::get_addr(a))
    //                                                     .zip(names.into_iter())
    //                                                     .map(|(a, n)| create_feature_worker(a, n))
    //                                                     .unzip();
    let (features, handles): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                              .map(|(n, a)| features::create_feature_worker(n, a, 100))
                              .unzip();


    let correct_counter = Arc::new(AtomicUsize::new(0));
    let total_counter = Arc::new(AtomicUsize::new(0));
    let processed_counter = Arc::new(AtomicUsize::new(0));
    let num_events = 100;
    // let num_workers = num_cpus::get();
    let num_workers = 1;
    let cum_latency_tracker_micros = Arc::new(AtomicUsize::new(0));
    let max_latency_tracker_micros = Arc::new(AtomicUsize::new(0));
    let mut dispatcher = Dispatcher::new(num_workers,
                                         SLA,
                                         features.clone(),
                                         init_user_models(num_users, num_features),
                                         correct_counter.clone(),
                                         total_counter.clone(),
                                         processed_counter.clone(),
                                         cum_latency_tracker_micros.clone(),
                                         max_latency_tracker_micros.clone()
                                         );

    // create monitoring thread to check incremental thruput
    thread::sleep(::std::time::Duration::new(3, 0));
    let mon_thread_join_handle = launch_monitor_thread(processed_counter.clone(), num_events);

    let num_features = features.len();
    println!("sending batch with no delays");
    let mut rng = thread_rng();
    for i in 0..num_events {
        dispatcher.dispatch(Request::new(rng.gen_range(0, num_users as u32),
                            features::random_features(784), i), num_features);
    }

    println!("waiting for features to finish");
    mon_thread_join_handle.join().unwrap();
    for h in handles {
        h.join().unwrap();
    }
    // handle.join().unwrap();
    println!("done");
}

// TODO this is a lot of unnecessary copies of the input
/// Request the features for `input` from the feature servers indicated
/// by `feature_indexes`. This allows a prediction worker to request
/// only a subset of features to reduce the load on feature servers when
/// the system is under heavy load.
pub fn get_features(fs: &Vec<features::FeatureHandle<features::SimpleHasher>>,
                    input: Vec<f64>,
                    feature_indexes: Vec<usize>,
                    req_start: time::PreciseTime,
                    salt: Option<i32>) {
    for idx in feature_indexes.iter() {
        let f = &fs[*idx];
        let h = f.hasher.hash(&input, salt);
        let req = features::FeatureReq {
            hash_key: h,
            input: input.clone(),
            req_start_time: req_start.clone()
        };
        f.queue.send(req).unwrap();
    }
    // for f in fs {
    //     let h = f.hasher.hash(&input);
    //     f.queue.send((h, input.clone())).unwrap();
    // }
}

fn launch_monitor_thread(counter: Arc<AtomicUsize>,
                             num_events: i32) -> ::std::thread::JoinHandle<()> {

    // let counter = counter.clone();
    thread::spawn(move || {
        let bench_start = time::PreciseTime::now();
        let mut last_count = 0;
        let mut last_time = bench_start;
        loop {
            thread::sleep(::std::time::Duration::new(3, 0));
            let cur_time = last_time.to(time::PreciseTime::now()).num_milliseconds() as f64 / 1000.0;
            let cur_count = counter.load(Ordering::Relaxed);
            println!("processed {} events in {} seconds: {} preds/sec",
                     cur_count - last_count,
                     cur_time,
                     ((cur_count - last_count) as f64 / cur_time));

            if cur_count >= num_events as usize {
                println!("BENCHMARK FINISHED");
                break;
            }
            println!("sleeping...");
        }
    })
}




