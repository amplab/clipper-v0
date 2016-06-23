
use time;
use cmt::{CorrectionModelTable, RedisCMT};
use std::marker::PhantomData;
use cache::{PredictionCache, SimplePredictionCache};
use configuration::{ClipperConf, ModelConf};
use hashing::{HashStrategy, EqualityHasher};

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
}


impl<P, S> ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(conf: ClipperConf) -> ClipperServer {

        let cache_size = 10000;

        let cache: Arc<SimplePredictionCache<Output, EqualityHasher>> =
            Arc::new(SimplePredictionCache::new(cache_size));

        let mut model_batchers = HashMap::new();
        for m in conf.model_conf.into_iter() {
            let b = PredictionBatcher::new(m.name.clone(),
                                           m.addrs,
                                           m.input_type,
                                           m.metrics,
                                           cache.clone());
            model_batchers.push(m.name.clone(), b);
        }

        let models = Arc::new(model_batchers);




        let mut prediction_workers = Vec::with_capacity(conf.num_predict_workers);
        for _ in 0..conf.num_predict_workers {
            prediction_workers.push(PredictionWorker::new(models.clone()));
        }
        let mut update_workers = Vec::with_capacity(conf.num_update_workers);
        for _ in 0..conf.num_update_workers {
            update_workers.push(UpdateWorker::new(models.clone()));
        }
        ClipperServer {
            prediction_workers: prediction_workers,
            update_workers: update_workers,
        }
    }

    // asynchronous, takes a callback
    pub fn schedule_prediction() {}

    // TODO: make sure scheduling here hashes on uid so all updates for a single user get sent
    // to the same thread
    pub fn schedule_update() {}
}

struct PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    worker_id: i32,
    input_queue: mpsc::Sender<PredictionRequest>,
    cache: Arc<PredictionCache<Output>>,
    models: Arc<HashMap<String, PredictionBatcher<SimplePredictionCache<Output>, Output>>>,
    _policy_marker: PhantomData<P>,
    _state_marker: PhantomData<S>,
}


/// Workers don't need handle to features, prediction cache handles that
impl<P, S> PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(worker_id: i32,
               cache: Arc<PredictionCache<Output>>,
               models: Arc<HashMap<String,
                                   PredictionBatcher<SimplePredictionCache<Output>, Output>>>)
               -> PredictionWorker {
        let (sender, receiver) = mpsc::channel::<PredictionRequest>();
        thread::spawn(move || {
            PredictionWorker::run(worker_id, receiver);
        });
        PredictionWorker {
            worker_id: worker_id,
            input_queue: sender,
            cache: cache,
            models: models,
            _policy_marker: PhantomData,
            _state_marker: PhantomData,
        }
    }

    // spawn new thread in here, return mpsc::sender?
    fn run(worker_id: i32, request_queue: mpsc::Receiver) {
        let slo_millis = time::Duration::milliseconds(SLO);
        let epsilon = time::Duration::milliseconds(slo_millis / 5);
        let cmt = RedisCMT::new_socket_connection();
        info!("starting prediction worker {} with {} ms SLO",
              worker_id,
              SLO);
        while let Ok(req) = receiver.recv() {
            let correction_state: S = cmt.get(*(&req.user) as u32)
                                         .unwrap_or_else(cmt.get(0_u32).unwrap());

            let elapsed_time = req.start_time.to(time::PreciseTime::now());
            // TODO: assumes SLA less than 1 second
            if elapsed_time < slo_millis - epsilon {
                let sleep_time = ::std::time::Duration::new(
                    0, (slo - elapsed_time).num_nanoseconds().unwrap() as u32);
                debug!("prediction worker sleeping for {:?} ms",
                       sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            let mut ys = Vec::new();
            let mut missing_ys = Vec::new();




        }
        info!("shutting down prediction worker {}", worker_id);
    }

    // submit prediction, input is prediction and callback
    pub fn predict() {}

    // renamed make_predictions
    fn correct_predictions() -> Output {}

    // stop the thread for graceful shutdown
    pub fn shutdown() {}
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

impl<P: CorrectionPolicy> UpdateWorker<P> {
    pub fn new() -> UpdateWorker<P> {}

    // spawn new thread in here, return mpsc::sender?
    fn run() -> mpsc::Sender {}
}
