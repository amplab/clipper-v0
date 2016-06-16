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
use cmt::{CorrectionModelTable, RedisCMT};
// use std::hash::{Hash, SipHasher, Hasher};

pub const SLA: i64 = 20;

pub type OnPredict = Fn(f64) -> () + Send;

pub type Output = f64;

/// Specifies the input type and expected length. A negative length indicates
/// a variable length input. `Str` does not have a length because Strings
/// are assumed to be always be variable length.
#[derive(Clone)]
pub enum InputType {
    Integer(i32),
    Float(i32),
    Str,
    Byte(i32),
}

// #[derive(Hash, Clone, Debug)]
#[derive(Clone,Debug)]
pub enum Input {
    Str {
        s: String,
    },
    Bytes {
        b: Vec<u8>,
        length: i32,
    },
    Ints {
        i: Vec<i32>,
        length: i32,
    },
    Floats {
        f: Vec<f64>,
        length: i32,
    },
}



pub struct PredictRequest {
    start_time: time::PreciseTime,
    user: u32, // TODO: remove this because each feature has it's own hash
    input: Input,
    true_label: Option<Output>, // used to evaluate accuracy,
    req_number: i32,
    on_predict: Box<OnPredict>,
}

pub struct UpdateRequest {
    start_time: time::PreciseTime,
    user: u32, // TODO: remove this because each feature has it's own hash
    input: Input,
    label: Output, // used to evaluate accuracy,
    req_number: i32,
}


impl PredictRequest {
    pub fn new(user: u32,
               input: Input,
               req_num: i32,
               on_predict: Box<OnPredict>)
               -> PredictRequest {
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
            on_predict: Box::new(|_| {
            }),
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

/// Correction policies are stateless, any required state is tracked separately
/// and stored in the `CorrectionModelTable`
pub trait CorrectionPolicy<S> where S: Serialize + Deserialize {

    fn new() -> S;

    fn predict(state: &S, ys: Vec<Output>, missing_ys: Vec<usize>, debug_str: &String) -> Output;

    fn rank_models(state: &S) -> Vec<usize>;

    fn train(state: &S) -> S;

}

pub struct DummyTaskModel {
    num_features: usize,
}

impl TaskModel for DummyTaskModel {
    #[allow(unused_variables)]
    fn predict(&self, fs: Vec<Output>, missing_fs: Vec<usize>, debug_str: &String) -> f64 {
        if missing_fs.len() > 0 {
            info!("missing fs: {:?}", missing_fs);
        }
        info!("Features: {:?}", fs);
        fs.into_iter().fold(0.0, |acc, c| acc + c)
    }

    fn rank_features(&self) -> Vec<usize> {
        return (0..self.num_features).into_iter().collect::<Vec<usize>>();
    }
}

// Because we don't have a good concurrent hash map, assume we know how many
// users there will be ahead of time. Then we can have a vec of RwLock
// and have row-level locking (because no inserts or deletes).
fn make_prediction<S, P, H>(feature_handles: &Vec<features::FeatureHandle<H>>,
                            input: &Input,
                            policy: &P<S>,
                            state: &S,
                            req_id: i32)
                            -> f64
    where P: CorrectionPolicy,
          H: FeatureHash + Send + Sync
{


    let mut missing_feature_indexes: Vec<usize> = Vec::new();
    let mut features: Vec<Output> = Vec::new();
    let mut i = 0;
    for fh in feature_handles {
        let hashed_input = fh.hasher.query_hash(input, Some(req_id));
        let cache_reader = fh.cache.read().unwrap();
        let cache_entry = cache_reader.get(&hashed_input);
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
                                 // user_models: Arc<Vec<RwLock<T>>>,
                                 pred_metrics: PredictionMetrics)
                                 -> mpsc::Sender<PredictRequest>
    where T: TaskModel + Send + Sync + 'static,
          H: FeatureHash + Send + Sync + 'static
{



    let sla = time::Duration::milliseconds(sla_millis);
    let epsilon = time::Duration::milliseconds(sla_millis / 5);
    let (sender, receiver) = mpsc::channel::<PredictRequest>();
    thread::spawn(move || {
        let cmt = RedisCMT::new_socket();
        info!("starting response worker {} with {} ms SLA",
              worker_id,
              sla_millis);
        loop {
            let req = receiver.recv().unwrap();
            // look up the task model before sleeping
            let correction_state: T = cmt.get(*(&req.user) as u32)
                                         .unwrap_or_else(cmt.get(0_u32).unwrap());


            // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
            let elapsed_time = req.start_time.to(time::PreciseTime::now());
            if elapsed_time < sla - epsilon {
                let sleep_time = ::std::time::Duration::new(
                    0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
                debug!("prediction worker sleeping for {:?} ms",
                       sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            // TODO: actually do something with the result
            // debug_assert!(req.user < user_models.len() as u32);

            // let lock = (&user_models).get(*(&req.user) as usize).unwrap();
            let pred = make_prediction(&feature_handles,
                                       &req.input,
                                       &correction_state,
                                       req.req_number);
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



fn start_update_worker<T, H>(worker_id: i32,
                             feature_handles: Vec<features::FeatureHandle<H>>,
                             user_models: Arc<Vec<RwLock<T>>>)
                             -> mpsc::Sender<PredictRequest>
    where T: TaskModel + Send + Sync + 'static,
          H: FeatureHash + Send + Sync + 'static
{
    let (sender, receiver) = mpsc::channel::<UpdateRequest>();


}

#[derive(Clone)]
struct PredictionMetrics {
    latency_hist: Arc<metrics::Histogram>,
    pred_counter: Arc<metrics::Counter>,
    thruput_meter: Arc<metrics::Meter>,
    accuracy_counter: Arc<metrics::RatioCounter>,
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
            accuracy_counter: accuracy_counter,
        }
    }
}

pub struct Dispatcher<T>
    where T: TaskModel + Send + Sync + 'static
{
    workers: Vec<mpsc::Sender<PredictRequest>>,
    update_workers: Vec<mpsc::Sender<UpdateRequest>>,
    // next_worker: usize,
    feature_handles: Vec<features::FeatureHandle<SimpleHasher>>, // user_models: Arc<Vec<RwLock<T>>>,
}

impl<T: TaskModel + Send + Sync + 'static> Dispatcher<T> {
    // pub fn new<T: TaskModel + Send + Sync + 'static>(num_workers: usize,
    pub fn new(num_workers: usize,
               sla_millis: i64,
               feature_handles: Vec<features::FeatureHandle<SimpleHasher>>,
               // user_models: Arc<Vec<RwLock<T>>>,
               metrics_register: Arc<RwLock<metrics::Registry>>)
               -> Dispatcher<T> {
        info!("creating dispatcher with {} workers", num_workers);
        let mut update_worker_threads = Vec::new();

        let num_update_workers = 1;
        for i in 0..num_update_workers {
            let update_worker = start_update_worker(i as i32,
                                                    feature_handles.clone(),
                                                    user_models.clone(),
                                                    metrics_register.clone());
            update_threads.push(worker);
        }

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
            update_worker: update_worker_threads,
            // next_worker: 0,
            feature_handles: feature_handles,
            user_models: user_models,
        }
    }

    /// Dispatch a request.
    ///
    /// TODO(replace worker vec with spmc (http://seanmonstar.github.io/spmc/spmc/index.html)
    pub fn dispatch(&self, req: PredictRequest, max_features: usize) {

        let mut features_indexes: Vec<usize> = (0..self.feature_handles.len())
                                                   .into_iter()
                                                   .collect();
        if max_features < self.feature_handles.len() {
            let model = self.user_models[req.user as usize].read().unwrap();
            features_indexes = model.rank_features();
            while max_features < features_indexes.len() {
                features_indexes.pop();
            }
        }
        get_predictions(&self.feature_handles,
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

    pub fn schedule_update(&self, update: UpdateRequest) {
        self.update_workers[0].send(update).unwrap();
    }
}


pub fn init_user_models(num_users: usize, num_features: usize) -> Arc<Vec<RwLock<DummyTaskModel>>> {
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
pub fn get_predictions(fs: &Vec<features::FeatureHandle<SimpleHasher>>,
                       input: Input,
                       feature_indexes: Vec<usize>,
                       req_start: time::PreciseTime,
                       salt: Option<i32>) {
    for idx in feature_indexes.iter() {
        let f = &fs[*idx];
        let h = f.hasher.query_hash(&input, salt);
        let req = features::FeatureReq {
            hash_key: h,
            input: input.clone(),
            req_start_time: req_start.clone(),
        };
        f.request_feature(req);
    }
}
