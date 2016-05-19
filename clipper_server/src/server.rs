// #![allow(unused_variables)]
// #![allow(dead_code)]
// #![allow(unused_mut)]

use time;
// use log;
// use std::net::SocketAddr;
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use rand::{thread_rng, Rng};
#[allow(unused_imports)]
use num_cpus;
use features;
use metrics;
use hashing::{FeatureHash, SimpleHasher};
use std::boxed::Box;
// use std::hash::{Hash, SipHasher, Hasher};

pub const SLA: i64 = 20;

pub type OnPredict = Fn(f64) -> () + Send;

pub type Output = Vec<f64>;

// #[derive(Hash, Clone, Debug)]
#[derive(Clone,Debug)]
pub enum Input {
    Str {s: String},
    Bytes {b: Vec<u8>, length: i32},
    Ints {i: Vec<i32>, length: i32},
    Floats {f: Vec<f32>, length: i32},
}



pub struct PredictRequest {
    start_time: time::PreciseTime,
    user: u32, // TODO: remove this because each feature has it's own hash
    input: Input,
    true_label: Option<f64>, // used to evaluate accuracy,
    req_number: i32,
    on_predict: Box<OnPredict>
}


impl PredictRequest {

    pub fn new(user: u32, input: Input, req_num: i32, on_predict: Box<OnPredict>) -> PredictRequest {
        PredictRequest {
            start_time: time::PreciseTime::now(),
            user: user,
            input: input,
            true_label: None,
            req_number: req_num,
            on_predict: on_predict,
        }
    }

    pub fn new_with_label(user: u32, input: Input, label: f64, req_num: i32) -> PredictRequest {
        PredictRequest {
            start_time: time::PreciseTime::now(),
            user: user,
            input: input,
            true_label: Some(label),
            req_number: req_num,
            on_predict: Box::new(|_| { }),
        }
    }
}

#[allow(dead_code)]
struct Update {
    start_time: time::PreciseTime, // just for monitoring purposes
    user: u32,
    input: Input,
    label: f32,
}

// fn start_update_worker(feature_handles: Vec<features::FeatureHandle<features::SimpleHasher>>) -> mpsc::Sender<Update> {
//     panic!("unimplemented method");
// }

pub trait TaskModel {
    /// Make a prediction with the available features
    fn predict(&self, mut fs: Output, missing_fs: Vec<usize>, debug_str: &String) -> f64;

    fn rank_features(&self) -> Vec<usize>;


    // fn update_anytime_features(&mut self, fs: Vec<f64>, missing_fs: Vec<i32>);

    // fn train(&mut self,
}

pub struct DummyTaskModel {
    num_features: usize
}

impl TaskModel for DummyTaskModel {
    #[allow(unused_variables)]
    fn predict(&self, fs: Output, missing_fs: Vec<usize>, debug_str: &String) -> f64 {
        fs.into_iter().fold(0.0, |acc, c| acc + c)
    }

    fn rank_features(&self) -> Vec<usize> {
        return (0..self.num_features).into_iter().collect::<Vec<usize>>()
    }
}

// Because we don't have a good concurrent hash map, assume we know how many
// users there will be ahead of time. Then we can have a vec of RwLock
// and have row-level locking (because no inserts or deletes).
fn make_prediction<T, H>(feature_handles: &Vec<features::FeatureHandle<H>>,
                      input: &Input,
                      task_model: &T,
                      req_id: i32) -> f64
                      where T: TaskModel,
                            H: FeatureHash + Send + Sync {

    
    let mut missing_feature_indexes: Vec<usize> = Vec::new();
    let mut features: Output = Vec::new();
    let mut i = 0;
    for fh in feature_handles {
        let hash = fh.hasher.hash(input, Some(req_id));
        let cache_reader = fh.cache.read().unwrap();
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
                           pred_metrics: PredictionMetrics
                           ) -> mpsc::Sender<PredictRequest> 
                           where T: TaskModel + Send + Sync + 'static,
                                 H: FeatureHash + Send + Sync + 'static {



    let sla = time::Duration::milliseconds(sla_millis);
    let epsilon = time::Duration::milliseconds(sla_millis / 5);
    let (sender, receiver) = mpsc::channel::<PredictRequest>();
    thread::spawn(move || {
        info!("starting response worker {} with {} ms SLA", worker_id, sla_millis);
        loop {
            let req = receiver.recv().unwrap();
            // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
            let elapsed_time = req.start_time.to(time::PreciseTime::now());
            if elapsed_time < sla - epsilon {
                let sleep_time = ::std::time::Duration::new(
                    0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
                debug!("prediction worker sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            // TODO: actually do something with the result
            debug_assert!(req.user < user_models.len() as u32);
            let lock = (&user_models).get(*(&req.user) as usize).unwrap();
            let task_model: &T = &lock.read().unwrap();
            let pred = make_prediction(&feature_handles, &req.input, task_model, req.req_number);
            (req.on_predict)(pred);
            let end_time = time::PreciseTime::now();
            let latency = req.start_time.to(end_time).num_microseconds().unwrap();
            pred_metrics.latency_hist.insert(latency);
            pred_metrics.thruput_meter.mark(1);
            pred_metrics.pred_counter.incr(1);
            if req.true_label.is_some() {
                if req.true_label.unwrap() == pred {
                    pred_metrics.accuracy_counter.incr(1, 1);
                } else {
                    pred_metrics.accuracy_counter.incr(0, 1);
                }
            }
        }
    });
    sender
}




#[derive(Clone)]
struct PredictionMetrics {
    latency_hist: Arc<metrics::Histogram>,
    pred_counter: Arc<metrics::Counter>,
    thruput_meter: Arc<metrics::Meter>,
    accuracy_counter: Arc<metrics::RatioCounter>
}

impl PredictionMetrics {

    pub fn new(metrics_register: Arc<RwLock<metrics::Registry>>) -> PredictionMetrics {

        let accuracy_counter = {
            let acc_counter_name = format!("prediction accuracy ratio");
            metrics_register.write().unwrap().create_ratio_counter(acc_counter_name)
        };
        
        let pred_counter = {
            let counter_name = format!("prediction_counter");
            metrics_register.write().unwrap().create_counter(counter_name)
        };

        let latency_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("prediction_latency");
            metrics_register.write().unwrap().create_histogram(metric_name, 2056)
        };

        let thruput_meter: Arc<metrics::Meter> = {
            let metric_name = format!("prediction_thruput");
            metrics_register.write().unwrap().create_meter(metric_name)
        };

        PredictionMetrics {
            latency_hist: latency_hist,
            pred_counter: pred_counter,
            thruput_meter: thruput_meter,
            accuracy_counter: accuracy_counter
        }
    }
}

pub struct Dispatcher<T>
where T: TaskModel + Send + Sync + 'static {
    workers: Vec<mpsc::Sender<PredictRequest>>,
    // next_worker: usize,
    feature_handles: Vec<features::FeatureHandle<SimpleHasher>>,
    user_models: Arc<Vec<RwLock<T>>>,
}

impl<T: TaskModel + Send + Sync + 'static> Dispatcher<T> {

    // pub fn new<T: TaskModel + Send + Sync + 'static>(num_workers: usize,
    pub fn new(num_workers: usize,
           sla_millis: i64,
           feature_handles: Vec<features::FeatureHandle<SimpleHasher>>,
           user_models: Arc<Vec<RwLock<T>>>,
           metrics_register: Arc<RwLock<metrics::Registry>>
           ) -> Dispatcher<T> {
        info!("creating dispatcher with {} workers", num_workers);
        let mut worker_threads = Vec::new();
        let pred_metrics = PredictionMetrics::new(metrics_register.clone());

        for i in 0..num_workers {
            let worker = start_prediction_worker(i as i32,
                                                 sla_millis,
                                                 feature_handles.clone(),
                                                 user_models.clone(),
                                                 pred_metrics.clone());
            worker_threads.push(worker);
        }
        Dispatcher {
            workers: worker_threads,
            // next_worker: 0,
            feature_handles: feature_handles,
            user_models: user_models
        }
    }

    /// Dispatch a request.
    ///
    /// Requires self to be mutable so that we can increment `next_worker`
    /// TODO(replace worker vec with spmc (http://seanmonstar.github.io/spmc/spmc/index.html)
    pub fn dispatch(&self, req: PredictRequest, max_features: usize) {

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
        let mut rng = thread_rng();
        // randomly pick a worker
        
        let w: usize = if self.workers.len() > 1 {
            rng.gen_range::<usize>(0, self.workers.len())
        } else {
            0
        };
        self.workers[w].send(req).unwrap();
        // self.increment_worker();
    }

    // for now do round robin scheduling
    // fn increment_worker(&self) {
    //     self.next_worker = (self.next_worker + 1) % self.workers.len();
    // }
}


pub fn init_user_models(num_users: usize, num_features: usize)
    -> Arc<Vec<RwLock<DummyTaskModel>>> {
    // let mut rng = thread_rng();
    let mut models = Vec::with_capacity(num_users);
    for _ in 0..num_users {
        let model = RwLock::new(DummyTaskModel { num_features: num_features });
        models.push(model);
    }
    Arc::new(models)
}



// TODO this is a lot of unnecessary copies of the input
/// PredictRequest the features for `input` from the feature servers indicated
/// by `feature_indexes`. This allows a prediction worker to request
/// only a subset of features to reduce the load on feature servers when
/// the system is under heavy load.
pub fn get_features(fs: &Vec<features::FeatureHandle<SimpleHasher>>,
                    input: Input,
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
        f.request_feature(req);
    }
}





// pub fn main(feature_addrs: Vec<(String, Vec<SocketAddr>)>) {
//     let num_features = feature_addrs.len();
//     let num_users = 500;
//     let metrics_register = Arc::new(RwLock::new(metrics::Registry::new("server main".to_string())));
//     // let test_data_path = "/crankshaw-local/mnist/data/test.data";
//     // let all_test_data = digits::load_mnist_dense(test_data_path).unwrap();
//     // let norm_test_data = digits::normalize(&all_test_data);
//
//     // info!("Test data loaded: {} points", norm_test_data.ys.len());
//
//     // let (features, handles): (Vec<_>, Vec<_>) = addr_vec.into_iter()
//     //                                                     .map(|a| features::get_addr(a))
//     //                                                     .zip(names.into_iter())
//     //                                                     .map(|(a, n)| create_feature_worker(a, n))
//     //                                                     .unzip();
//     let (features, handles): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
//                               .map(|(n, a)| features::create_feature_worker(
//                                       n, a, 100, metrics_register.clone())).unzip();
//
//
//     let num_events = 100;
//     let num_workers = num_cpus::get();
//     // let num_workers = 1;
//     let mut dispatcher = Dispatcher::new(num_workers,
//                                          SLA,
//                                          features.clone(),
//                                          init_user_models(num_users, num_features),
//                                          metrics_register.clone());
//
//     // create monitoring thread to check incremental thruput
//     thread::sleep(::std::time::Duration::new(3, 0));
//     // let mon_thread_join_handle = launch_monitor_thread(metrics_register.clone());
//
//     let num_features = features.len();
//     info!("sending batch with no delays");
//     let mut rng = thread_rng();
//     for i in 0..num_events {
//         dispatcher.dispatch(PredictRequest::new(rng.gen_range(0, num_users as u32),
//                             features::random_features(784), i), num_features);
//     }
//
//     info!("waiting for features to finish");
//     // mon_thread_join_handle.join().unwrap();
//     for h in handles {
//         for th in h {
//             th.join().unwrap();
//         }
//     }
//     // handle.join().unwrap();
//     info!("done");
// }

