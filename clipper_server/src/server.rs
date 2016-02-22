#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_mut)]

use gj;
use gj::{EventLoop, Promise};
use capnp;
use time;
use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use std::net::{ToSocketAddrs, SocketAddr};
use feature_capnp::feature;
use std::thread;
use std::sync::{RwLock, Arc};
use std::sync::mpsc;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use num_cpus;
use linear_models::linalg;
use digits;
use features;

const SLA: i64 = 20;



// fn anytime_features(features: &Vec<features::FeatureHandle>, input: &Vec<f64>) -> Vec<f64> {
//     // TODO check caches
//     // for f in features {
//     //     f.cache.read()
//     // }
//     vec![-3.2, 5.1]
// }


struct Request {
    start_time: time::PreciseTime,
    user: u32, // TODO: remove this because each feature has it's own hash
    input: Vec<f64>,
    true_label: Option<f64> // used to evaluate accuracy
}

impl Request {

    fn new(user: u32, input: Vec<f64>) -> Request {
        Request { start_time: time::PreciseTime::now(), user: user, input: input, true_label: None}
    }

    fn new_with_label(user: u32, input: Vec<f64>, label: f64) -> Request {
        Request {
            start_time: time::PreciseTime::now(),
            user: user,
            input: input,
            true_label: Some(label)
        }
    }

}

struct Update {
    start_time: time::PreciseTime, // just for monitoring purposes
    user: u32,
    input: Vec<f64>,
    label: f32,
}

fn start_update_worker(feature_handles: Vec<features::FeatureHandle>) -> mpsc::Sender<Update> {
    panic!("unimplemented method");
}

trait TaskModel {
    /// Make a prediction with the available features
    fn predict(&self, fs: Vec<f64>, missing_fs: Vec<i32>) -> f64;


    // fn update_anytime_features(&mut self, fs: Vec<f64>, missing_fs: Vec<i32>);

    // fn train(&mut self,
}

// Because we don't have a good concurrent hash map, assume we know how many
// users there will be ahead of time. Then we can have a vec of RwLock.
fn make_prediction(feature_handles: &Vec<features::FeatureHandle>, input: &Vec<f64>,
                   task_model: &TaskModel) -> f64 {

    
    let mut missing_feature_indexes: Vec<i32> = Vec::new();
    let mut features: Vec<f64> = Vec::capacity(features.len());
    let mut i = 0;
    for fh in feature_handles {
        let hash = fh.hasher.hash(input);
        let mut cache_reader = fh.cache.read().unwrap();
        let cache_entry = cache_reader.get(hash);
        match cache_entry {
            Some(v) => features.append(v),
            None => {
                features.append(0.0);
                missing_feature_indexes.append(i);
            }
        };
        i += 1
    }

    let anytime_features = task_model.predict(features, missing_feature_indexes);
    task_model.predict(anytime_features);
}

fn start_prediction_worker(worker_id: i32,
                           sla_millis: i64,
                           feature_handles: Vec<features::FeatureHandle>,
                           user_models: Arc<Vec<RwLock<TaskModel>>>,
                           correct_counter: Arc<AtomicUsize>,
                           total_counter: Arc<AtomicUsize>,
                           processed_counter: Arc<AtomicUsize>
                           ) -> mpsc::Sender<Request> {

    let sla = time::Duration::milliseconds(sla_millis);
    let epsilon = time::Duration::milliseconds(3);
    let (sender, receiver) = mpsc::channel::<Request>();
    let join_guard = thread::spawn(move || {
        println!("starting response worker {} with {}ms SLA", worker_id, sla_millis);
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
            debug_assert!(req.user < user_models.len() as u32);
            let lock = (&user_models).get(*(&req.user) as usize).unwrap();
            let task_model = lock.read().unwrap();
            let pred = make_prediction(&feature_handles, &req.input, &task_model);
            let end_time = time::PreciseTime::now();
            let latency = req.start_time.to(end_time).num_milliseconds();
            processed_counter.fetch_add(1, Ordering::Relaxed);
            if req.true_label.is_some() {
                total_counter.fetch_add(1, Ordering::Relaxed);
                if req.true_label.unwrap() == pred {
                    correct_counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });
    // (join_guard, sender)
    sender
}



struct Dispatcher {
    workers: Vec<mpsc::Sender<Request>>,
    next_worker: usize,
    features: Vec<features::FeatureHandle>,
}

impl Dispatcher {

    fn new(num_workers: usize,
           sla_millis: i64,
           features: Vec<features::FeatureHandle>,
           user_models: Arc<Vec<RwLock<TaskModel>>>,
           correct_counter: Arc<AtomicUsize>,
           total_counter: Arc<AtomicUsize>,
           processed_counter: Arc<AtomicUsize>
           ) -> Dispatcher {
        println!("creating dispatcher with {} workers", num_workers);
        let mut worker_threads = Vec::new();
        for i in 0..num_workers {
            let worker = start_prediction_worker(i as i32,
                                                 sla_millis,
                                                 features.clone(),
                                                 user_models.clone(),
                                                 correct_counter.clone(),
                                                 total_counter.clone(),
                                                 processed_counter.clone());
            worker_threads.push(worker);
        }
        Dispatcher {workers: worker_threads, next_worker: 0, features: features}
    }

    /// Dispatch a request.
    ///
    /// Requires self to be mutable so that we can increment `next_worker`
    fn dispatch(&mut self, req: Request) {
        get_features(&self.features, req.input.clone());
        self.workers[self.next_worker].send(req).unwrap();
        self.increment_worker();
    }

    // for now do round robin scheduling
    fn increment_worker(&mut self) {
        self.next_worker = (self.next_worker + 1) % self.workers.len();
    }
}

// fn init_user_models(num_users: usize, num_features: usize) -> Arc<Vec<RwLock<TaskModel>>> {
//     let mut rng = thread_rng();
//     let mut models = Vec::with_capacity(num_users);
//     for i in 0..num_users {
//         let model = RwLock::new(TaskModel {
//             w: rng.gen_iter::<f64>().take(num_features).collect::<Vec<f64>>()
//         });
//         models.push(model);
//     }
//     Arc::new(models)
// }




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
                              .map(|(n, a)| features::create_feature_worker(n, a))
                              .unzip();


    let correct_counter = Arc::new(AtomicUsize::new(0));
    let total_counter = Arc::new(AtomicUsize::new(0));
    let processed_counter = Arc::new(AtomicUsize::new(0));
    let num_events = 100;
    // let num_workers = num_cpus::get();
    let num_workers = 1;
    let mut dispatcher = Dispatcher::new(num_workers,
                                         SLA,
                                         features,
                                         init_user_models(num_users, num_features),
                                         correct_counter.clone(),
                                         total_counter.clone(),
                                         processed_counter.clone());

    // create monitoring thread to check incremental thruput
    thread::sleep(::std::time::Duration::new(3, 0));
    let mon_thread_join_handle = launch_monitor_thread(processed_counter.clone(), num_events);

    println!("sending batch with no delays");
    let mut rng = thread_rng();
    for _ in 0..num_events {
        dispatcher.dispatch(Request::new(rng.gen_range(0, num_users as u32),
                            features::random_features(784)));
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
fn get_features(fs: &Vec<features::FeatureHandle>, input: Vec<f64>) {
    for f in fs {
        let h = f.hasher.hash(&input);
        f.queue.send((h, input.clone())).unwrap();
    }
}

fn launch_monitor_thread(counter: Arc<AtomicUsize>,
                             num_events: i32) {

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

            if cur_count >= num_events {
                println!("BENCHMARK FINISHED");
                break;
            }
            println!("sleeping...");
        }
    })
}




