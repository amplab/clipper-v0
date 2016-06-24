
use time;
use cmt::{CorrectionModelTable, RedisCMT};
use std::marker::PhantomData;
use cache::{PredictionCache, SimplePredictionCache};
use configuration::{ClipperConf, ModelConf};
use hashing::{HashStrategy, EqualityHasher};
use batching::RpcPredictRequest;
use metrics;

// pub const SLO: i64 = 20;

pub type Output = f64;

pub type OnPredict = Fn(Output) -> () + Send;



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

pub struct PredictionRequest {
    recv_time: time::PreciseTime,
    uid: u32,
    query: Input,
    on_predict: Box<OnPredict>,
}

impl PredictionRequest {
    pub fn new(uid: u32, input: Input, on_predict: Box<OnPredict>) -> PredictionRequest {
        PredictionRequest {
            recv_time: time::PreciseTime::now(),
            uid: uid,
            input: input,
            on_predict: on_predict,
        }
    }
}

pub struct UpdateRequest {
    recv_time: time::PreciseTime,
    uid: u32,
    query: Input,
    label: Output,
}


// renamed dispatch, this is the external object we export from the library
// to run clipper
struct ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    prediction_workers: Vec<PredictionWorker<P, S>>,
    update_workers: Vec<UpdateWorker<P, S>>,
    // model_names: Vec<String>,
    cache: Arc<SimplePredictionCache<Output>>,
    metrics: Arc<RwLock<metrics::Registry>>,
    input_type: InputType,
}


impl<P, S> ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(conf: ClipperConf) -> ClipperServer<P, S> {

        // let cache_size = 10000;

        // TODO(#13): once LSH is implemented make cache type configurable
        let cache: Arc<SimplePredictionCache<Output, EqualityHasher>> =
            Arc::new(SimplePredictionCache::new(conf.cache_size));

        let mut model_batchers = HashMap::new();
        for m in conf.models.into_iter() {
            let b = PredictionBatcher::new(m.name.clone(),
                                           m.addresses,
                                           conf.input_type,
                                           conf.metrics.clone(),
                                           cache.clone(),
                                           conf.slo_millis);
            model_batchers.insert(m.name.clone(), b);
        }

        let models = Arc::new(model_batchers);




        let mut prediction_workers = Vec::with_capacity(conf.num_predict_workers);
        for i in 0..conf.num_predict_workers {
            prediction_workers.push(PredictionWorker::new(i,
                                                          conf.slo_millis,
                                                          cache.clone(),
                                                          models.clone()));
        }
        let mut update_workers = Vec::with_capacity(conf.num_update_workers);
        for i in 0..conf.num_update_workers {
            update_workers.push(UpdateWorker::new(i, cache.clone(), models.clone()));
        }
        ClipperServer {
            prediction_workers: prediction_workers,
            update_workers: update_workers,
            cache: cache,
            metrics: conf.metrics,
            input_type: conf.input_type,
        }
    }

    // TODO: replace worker vec with spmc (http://seanmonstar.github.io/spmc/spmc/index.html)
    pub fn schedule_prediction(&self, r: PredictionRequest) {
        let mut rng = thread_rng();
        // randomly pick a worker

        let w: usize = if self.prediction_workers.len() > 1 {
            rng.gen_range::<usize>(0, self.workers.len())
        } else {
            0
        };
        self.prediction_workers[w].predict(req, max_predictions);
    }

    pub fn get_metrics(&self) -> Arc<RwLock<metrics::Registry>> {
        self.metrics.clone()
    }

    pub fn get_input_type(&self) -> InputType {
        self.input_type.clone()
    }

    // TODO: make sure scheduling here hashes on uid so all updates for a single user get sent
    // to the same thread
    pub fn schedule_update() {
        unimplemented!();
    }

    pub fn shutdown() {
        unimplemented!();
    }
}


#[derive(Clone)]
struct PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    worker_id: i32,
    input_queue: mpsc::Sender<(PredictionRequest, i32)>,
    // cache: Arc<PredictionCache<Output>>,
    // models: Arc<HashMap<String, PredictionBatcher<SimplePredictionCache<Output>, Output>>>,
    _policy_marker: PhantomData<P>,
    _state_marker: PhantomData<S>,
}


impl<P, S> PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(worker_id: i32,
               slo_millis: u32,
               cache: Arc<PredictionCache<Output>>,
               models: Arc<HashMap<String,
                                   PredictionBatcher<SimplePredictionCache<Output>, Output>>>)
               -> PredictionWorker<P, S> {
        let (sender, receiver) = mpsc::channel::<(PredictionRequest, i32)>();
        thread::spawn(move || {
            PredictionWorker::run(worker_id, slo_millis, receiver, cache, models);
        });
        PredictionWorker {
            worker_id: worker_id,
            input_queue: sender,
            // cache: cache,
            // models: models,
            _policy_marker: PhantomData,
            _state_marker: PhantomData,
        }
    }

    fn run(worker_id: i32,
           slo_millis: u32,
           request_queue: mpsc::Receiver<(PredictionRequest, i32)>,
           cache: Arc<PredictionCache<Output>>,
           models: Arc<HashMap<String, PredictionBatcher<SimplePredictionCache<Output>, Output>>>) {
        let slo = time::Duration::milliseconds(slo_millis);
        // let epsilon = time::Duration::milliseconds(slo_millis / 5);
        let epsilon = time::Duration::milliseconds(1);
        let cmt = RedisCMT::new_socket_connection();
        info!("starting prediction worker {} with {} ms SLO",
              worker_id,
              SLO);
        while let Ok((req, max_preds)) = receiver.recv() {
            let correction_state: S = cmt.get(*(&req.user) as u32)
                                         .unwrap_or_else(cmt.get(0_u32).unwrap());
            let model_req_order = if max_preds < models.len() {
                P::rank_models_desc(correction_state)
            } else {
                models.keys().collect()
            };
            let mut num_requests = 0;
            let mut i = 0;
            while num_requests < max_preds && i < models.len() {
                // first check to see if the prediction is already cached
                if cache.fetch(model_req_order[i], req.input).is_none() {
                    // on cache miss, send to batching layer
                    models.get(model_req_order[i].unwrap()).request_prediction(RpcPredictRequest {
                        input: req.input,
                        recv_time: req.recv_time.clone(),
                    });
                    num_requests += 1;
                }
                i += 1;
            }

            let elapsed_time = req.recv_time.to(time::PreciseTime::now());
            // TODO: assumes SLA less than 1 second
            if elapsed_time < slo - epsilon {
                let sleep_time = ::std::time::Duration::new(
                    0, (slo - elapsed_time).num_nanoseconds().unwrap() as u32);
                debug!("prediction worker sleeping for {:?} ms",
                       sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            let mut ys = HashMap::new();
            let mut missing_ys = Vec::new();
            for model_name in models.keys() {
                match cache.fetch(model_name, &req.query) {
                    Some(v) => ys.insert(model_name.clone(), v),
                    None => missing_ys.push(model_name.clone()),
                }
            }

            // use correction policy to make the final prediction
            let prediction = P::predict(correction_state, ys, missing_ys);
            // execute the user's callback on this thread
            (req.on_predict)(prediction);
            let end_time = time::PreciseTime::now();
            let latency = req.start_time.to(end_time).num_microseconds().unwrap();
            // TODO: metrics
            // pred_metrics.latency_hist.insert(latency);
            // pred_metrics.thruput_meter.mark(1);
            // pred_metrics.pred_counter.incr(1);
        }
        info!("shutting down prediction worker {}", worker_id);
    }

    pub fn predict(&self, r: PredictionRequest) {
        self.input_queue.send(r).unwrap();
    }

    pub fn shutdown() {
        unimplemented!();
    }
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


struct UpdateWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    cache: Arc<PredictionCache<Output>>,
    models: Arc<HashMap<String, PredictionBatcher<SimplePredictionCache<Output>, Output>>>,
    _policy_marker: PhantomData<P>,
    _state_marker: PhantomData<S>,
}

impl<P, S> UpdateWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(worker_id: i32,
               cache: Arc<PredictionCache<Output>>,
               models: Arc<HashMap<String,
                                   PredictionBatcher<SimplePredictionCache<Output>, Output>>>)
               -> UpdateWorker<P, S> {
        unimplemented!();
    }

    // spawn new thread in here, return mpsc::sender?
    fn run() -> mpsc::Sender {
        unimplemented!();
    }
}
