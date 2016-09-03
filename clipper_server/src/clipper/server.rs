
use time::{PreciseTime, Duration};
use std::marker::PhantomData;
use std::net::SocketAddr;
use serde::ser::Serialize;
use serde::de::Deserialize;
use serde_json;
use std::sync::{mpsc, RwLock, Arc};
use std::collections::{HashMap, HashSet};
use rand::{thread_rng, Rng};
use std::cmp;
use std::thread::{self, JoinHandle};
use std::time::Duration as StdDuration;

#[allow(unused_imports)]
use cmt::{CorrectionModelTable, RedisCMT, UpdateTable, RedisUpdateTable, REDIS_CMT_DB,
          REDIS_UPDATE_DB, DEFAULT_REDIS_SOCKET, REDIS_DEFAULT_PORT};
use cache::{PredictionCache, SimplePredictionCache};
use configuration::ClipperConf;
use hashing::EqualityHasher;
use batching::{RpcPredictRequest, PredictionBatcher, BatchStrategy};
use correction_policy::CorrectionPolicy;
use metrics;

// pub const SLO: i64 = 20;

pub type Output = f64;

pub type OnPredict = Fn(Output) -> () + Send;

// type Batcher = PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>;



/// Specifies the input type and expected length. A negative length indicates
/// a variable length input. `Str` does not have a length because Strings
/// are assumed to be always be variable length.
#[derive(Clone, PartialEq, Debug, Serialize)]
pub enum InputType {
    Integer(i32),
    Float(i32),
    Str,
    Byte(i32),
}

// #[derive(Hash, Clone, Debug)]
#[derive(Clone,Debug,Serialize,Deserialize, PartialEq, PartialOrd)]
pub enum Input {
    Str { s: String },
    Bytes { b: Vec<u8>, length: i32 },
    Ints { i: Vec<i32>, length: i32 },
    Floats { f: Vec<f64>, length: i32 },
}

#[allow(dead_code)]
pub struct PredictionRequest {
    recv_time: PreciseTime,
    uid: u32,
    query: Arc<Input>,
    salt: Option<i32>,
    on_predict: Box<OnPredict>,
    /// Specifies which offline models to use by name
    /// and version. If offline models is `None`, then the latest
    /// version of all models will be used.
    ///
    /// This allows Clipper to split requests between different
    /// versions of a model, and eventually will open the way
    /// to allow a single Clipper instance to serve multiple
    /// applications.
    offline_models: Option<Vec<VersionedModel>>, // correction_policy: Option<String>,
}

impl PredictionRequest {
    pub fn new(uid: u32,
               input: Input,
               on_predict: Box<OnPredict>,
               salt: bool)
               -> PredictionRequest {
        PredictionRequest {
            recv_time: PreciseTime::now(),
            uid: uid,
            query: Arc::new(input),
            salt: if salt {
                let mut rng = thread_rng();
                Some(rng.gen::<i32>())
            } else {
                None
            },
            on_predict: on_predict,
            offline_models: None, // correction_policy: None,
        }
    }
}



#[derive(Clone)]
pub struct Update {
    pub query: Arc<Input>,
    pub label: Output,
}

#[derive(Clone)]
pub struct UpdateRequest {
    recv_time: PreciseTime,
    uid: u32,
    updates: Vec<Update>, /* query: Input,
                           * label: Output */
    /// Specifies which offline models to use by name
    /// and version. If offline models is `None`, then the latest
    /// version of all models will be used.
    ///
    /// This allows Clipper to split requests between different
    /// versions of a model, and eventually will open the way
    /// to allow a single Clipper instance to serve multiple
    /// applications.
    offline_models: Option<Vec<VersionedModel>>, // correction_policy: Option<String>,
}

impl UpdateRequest {
    pub fn new(uid: u32, updates: Vec<Update>) -> UpdateRequest {
        UpdateRequest {
            recv_time: PreciseTime::now(),
            uid: uid,
            updates: updates,
            offline_models: None,
        }
    }
}


#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Eq)]
pub struct VersionedModel {
    pub name: String,
    /// If the version is `None`, the newest version will be used.
    pub version: Option<u32>,
}


#[derive(Clone)]
struct ModelSet {
    batchers: HashMap<VersionedModel,
                      PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>,
    input_type: InputType,
    metrics: Arc<RwLock<metrics::Registry>>,
    cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
    slo_micros: u32,
    batch_strategy: BatchStrategy,
}



impl ModelSet {
    // fn get_versions_for_model(&self, model_name: &String) -> Vec<u32> {}


    fn get_latest_version(&self, model_name: &String) -> u32 {
        let mut latest_version = 0;
        for mv in self.batchers.keys() {
            if &mv.name == model_name && mv.version.as_ref().unwrap() > &latest_version {
                latest_version = *mv.version.as_ref().unwrap();
            }
        }
        latest_version
    }

    fn num_models(&self) -> usize {
        let mut names: HashSet<&String> = HashSet::new();
        for mv in self.batchers.keys() {
            names.insert(&mv.name);
        }
        names.len()
    }

    fn version_exists(&self, model_name: &String, version: &u32) -> bool {
        for mv in self.batchers.keys() {
            if &mv.name == model_name && mv.version.as_ref().unwrap() == version {
                return true;
            }
        }
        false
    }

    pub fn from_conf(conf: &ClipperConf,
                     cache: Arc<SimplePredictionCache<Output, EqualityHasher>>)
                     -> ModelSet {


        let mut batchers = HashMap::new();
        for m in conf.models.iter() {
            let b = PredictionBatcher::new(m.name.clone(),
                                           m.addresses.clone(),
                                           conf.input_type.clone(),
                                           conf.metrics.clone(),
                                           cache.clone(),
                                           conf.slo_micros - 2,
                                           conf.batch_strategy.clone());
            batchers.insert(VersionedModel {
                                name: m.name.clone(),
                                version: Some(m.version),
                            },
                            b);
        }

        ModelSet {
            batchers: batchers,
            input_type: conf.input_type.clone(),
            metrics: conf.metrics.clone(),
            cache: cache,
            slo_micros: conf.slo_micros.clone(),
            batch_strategy: conf.batch_strategy.clone(),
        }

        // pub fn add_model
    }

    // TODO: Uggggh this is so hacky. I really need to find a better way to manage
    // consistent model updates.
    pub fn add_new_replica(&mut self,
                           model: VersionedModel,
                           address: SocketAddr)
                           -> Option<mpsc::Sender<RpcPredictRequest>> {
        match self.batchers.get_mut(&model) {
            Some(p) => Some(p.add_new_replica(address, self.metrics.clone())),
            None => {
                warn!("Trying to add a new replica to non-existent model version: {:?}",
                      model);
                None
            }
        }
    }

    pub fn incorporate_new_replica(&mut self,
                                   model: VersionedModel,
                                   sender: mpsc::Sender<RpcPredictRequest>) {
        match self.batchers.get_mut(&model) {
            Some(p) => p.incorporate_new_replica(sender),
            None => {
                warn!("Trying to incorporate a new replica to non-existent model version: {:?}",
                      model);
            }
        };
    }

    pub fn add_new_model(&mut self,
                         model: VersionedModel,
                         addresses: Vec<SocketAddr>)
                         -> PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>> {

        let new_batcher = PredictionBatcher::new(model.name.clone(),
                                                 addresses.clone(),
                                                 self.input_type.clone(),
                                                 self.metrics.clone(),
                                                 self.cache.clone(),
                                                 self.slo_micros - 2,
                                                 self.batch_strategy.clone());
        if self.batchers.insert(model.clone(), new_batcher.clone()).is_some() {
            warn!("ModelSet already contained a batcher for {:?}, did you mean to add a replica \
                   instead?",
                  model);

        };
        new_batcher
    }

    pub fn incorporate_new_model(&mut self,
                                 model: VersionedModel,
                                 new_batcher: PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>) {

        let warn_string = format!("ModelSet already contained a batcher for {:?}, did you mean \
                                   to add a replica instead?",
                                  model);
        if self.batchers.insert(model, new_batcher).is_some() {
            warn!("{}", warn_string);
        }
    }

    /// Ensures that the requested models and versions are available for
    /// querying, and resolves unspecified model versions to the latest
    /// version. Currently this function also ensures that only
    /// one version of each model will be queried.
    pub fn get_versioned_models(&self,
                                requested_names: &Option<Vec<VersionedModel>>)
                                -> Vec<VersionedModel> {

        match requested_names {
            &Some(ref vms) => {
                let mut model_versions: HashMap<String, Vec<u32>> = HashMap::new();
                for mv in vms.iter() {
                    let mut versions = model_versions.entry(mv.name.clone()).or_insert(Vec::new());
                    let version = mv.version
                        .as_ref()
                        .unwrap_or(&self.get_latest_version(&mv.name))
                        .clone();
                    if self.version_exists(&mv.name, &version) {
                        versions.push(version);
                    } else {
                        warn!("Requested unloaded or non-existent version {} for model {}",
                              version,
                              mv.name);
                    }
                }
                model_versions.iter()
                    .filter(|&(_, v)| v.len() > 0)
                    .map(|(name, versions)| {
                        let latest_version = if versions.len() > 1 {
                            let latest_version = versions.iter()
                                .fold(0, |acc, &x| cmp::max(acc, x));
                            warn!("Requested multiple versions ({:?}) of model: {}. Only
                        \
                                   using newest one {}",
                                  versions,
                                  name,
                                  latest_version);
                            latest_version
                        } else {
                            versions[0]
                        };
                        VersionedModel {
                            name: name.clone(),
                            version: Some(latest_version),
                        }
                    })
                    .collect::<Vec<VersionedModel>>()
            }
            &None => {
                let mut model_versions = HashSet::new();
                for mv in self.batchers.keys() {
                    let latest_model_version = self.get_latest_version(&mv.name);
                    model_versions.insert(VersionedModel {
                        name: mv.name.clone(),
                        version: Some(latest_model_version),
                    });
                }
                let mut mv_vec = Vec::new();
                mv_vec.extend(model_versions.into_iter());
                mv_vec
            }
        }
    }

    pub fn request_prediction(&self, offline_model: &VersionedModel, request: RpcPredictRequest) {
        match self.batchers.get(offline_model) {
            Some(b) => b.request_prediction(request),
            None => warn!("No batcher found for model {:?}", offline_model),
        }
    }
}

impl Drop for ModelSet {
    fn drop(&mut self) {
        self.batchers.clear();
    }
}



#[derive(Clone)]
struct ServerMetrics {
    pub num_queued_predictions_counter: Arc<metrics::Counter>,
    pub num_queued_updates_counter: Arc<metrics::Counter>,
}

#[derive(Clone)]
struct PredictionMetrics {
    pub num_queued_predictions_counter: Arc<metrics::Counter>,
    pub latency_hist: Arc<metrics::Histogram>,
    pub pred_counter: Arc<metrics::Counter>,
    pub thruput_meter: Arc<metrics::Meter>,
    pub accuracy_counter: Arc<metrics::RatioCounter>,
    pub cache_hit_counter: Arc<metrics::RatioCounter>,
    pub in_time_predictions_hist: Arc<metrics::Histogram>,
    pub cache_included_thruput_meter: Arc<RwLock<HashMap<String, Arc<metrics::Meter>>>>,
    pub registry: Arc<RwLock<metrics::Registry>>,
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
            metrics_register.write().unwrap().create_histogram(metric_name, 2056 * 2)
        };

        let thruput_meter: Arc<metrics::Meter> = {
            let metric_name = format!("prediction_thruput");
            metrics_register.write().unwrap().create_meter(metric_name)
        };

        let queued_predictions_counter: Arc<metrics::Counter> = {
            let metric_name = format!("queued_predictions");
            metrics_register.write().unwrap().create_counter(metric_name)
        };

        let cache_hit_counter: Arc<metrics::RatioCounter> = {
            let metric_name = format!("cache_hits");
            metrics_register.write().unwrap().create_ratio_counter(metric_name)
        };

        let in_time_predictions_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("in_time_predictions");
            metrics_register.write().unwrap().create_histogram(metric_name, 2056 * 2)
        };

        PredictionMetrics {
            num_queued_predictions_counter: queued_predictions_counter,
            latency_hist: latency_hist,
            pred_counter: pred_counter,
            thruput_meter: thruput_meter,
            accuracy_counter: accuracy_counter,
            cache_hit_counter: cache_hit_counter,
            in_time_predictions_hist: in_time_predictions_hist,
            cache_included_thruput_meter: Arc::new(RwLock::new(HashMap::new())),
            registry: metrics_register,
        }
    }
}

#[derive(Clone)]
struct UpdateMetrics {
    pub num_queued_updates_counter: Arc<metrics::Counter>,
    pub latency_hist: Arc<metrics::Histogram>,
    pub update_counter: Arc<metrics::Counter>,
    pub thruput_meter: Arc<metrics::Meter>, // accuracy_counter: Arc<metrics::RatioCounter>,
}

impl UpdateMetrics {
    pub fn new(metrics_register: Arc<RwLock<metrics::Registry>>) -> UpdateMetrics {

        let update_counter = {
            let counter_name = format!("update_counter");
            metrics_register.write().unwrap().create_counter(counter_name)
        };

        let latency_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("update_latency");
            metrics_register.write().unwrap().create_histogram(metric_name, 2056)
        };

        let thruput_meter: Arc<metrics::Meter> = {
            let metric_name = format!("update_thruput");
            metrics_register.write().unwrap().create_meter(metric_name)
        };

        let queued_updates_counter: Arc<metrics::Counter> = {
            let metric_name = format!("queued_updates");
            metrics_register.write().unwrap().create_counter(metric_name)
        };

        UpdateMetrics {
            num_queued_updates_counter: queued_updates_counter,
            latency_hist: latency_hist,
            update_counter: update_counter,
            thruput_meter: thruput_meter,
        }
    }
}

pub struct ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    prediction_workers: Vec<PredictionWorker<P, S>>,
    update_workers: Vec<UpdateWorker<P, S>>,
    // model_names: Vec<String>,
    // TODO(#13): Change cache type signature to be a trait object once LSH
    // is implemented
    // cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
    metrics: Arc<RwLock<metrics::Registry>>,
    input_type: InputType,
    // models: HashMap<String, PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>,
    models: ModelSet,
    server_metrics: ServerMetrics,
    cmt: RedisCMT<S>,
}

impl<P, S> ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(conf: ClipperConf) -> ClipperServer<P, S> {

        // Ensure configuration is valid

        assert!(P::accepts_input_type(&conf.input_type),
                format!("Invalid configuration: correction policy {} does not accept InputType \
                         {:?}",
                        P::get_name(),
                        conf.input_type));

        // let cache_size = 10000;

        // TODO(#13): once LSH is implemented make cache type configurable
        let cache: Arc<SimplePredictionCache<Output, EqualityHasher>> =
            Arc::new(SimplePredictionCache::with_models(&conf.models, conf.cache_size.clone()));

        let prediction_metrics = PredictionMetrics::new(conf.metrics.clone());

        let update_metrics = UpdateMetrics::new(conf.metrics.clone());

        let server_metrics = ServerMetrics {
            num_queued_predictions_counter: prediction_metrics.num_queued_predictions_counter
                .clone(),
            num_queued_updates_counter: update_metrics.num_queued_updates_counter.clone(),
        };


        let models = ModelSet::from_conf(&conf, cache.clone());

        let mut prediction_workers = Vec::with_capacity(conf.num_predict_workers.clone());
        for i in 0..conf.num_predict_workers {
            prediction_workers.push(PredictionWorker::new(i as i32,
                                                          conf.slo_micros.clone(),
                                                          cache.clone(),
                                                          models.clone(),
                                                          conf.redis_ip.clone(),
                                                          conf.redis_port,
                                                          prediction_metrics.clone()));
        }
        let mut update_workers = Vec::with_capacity(conf.num_update_workers.clone());
        for i in 0..conf.num_update_workers {
            update_workers.push(UpdateWorker::new(i as i32,
                                                  cache.clone(),
                                                  models.clone(),
                                                  conf.window_size,
                                                  conf.redis_ip.clone(),
                                                  conf.redis_port,
                                                  update_metrics.clone()));
        }
        ClipperServer {
            prediction_workers: prediction_workers,
            update_workers: update_workers,
            // cache: cache,
            metrics: conf.metrics,
            input_type: conf.input_type,
            models: models,
            server_metrics: server_metrics,
            cmt: RedisCMT::new_tcp_connection(&conf.redis_ip, conf.redis_port, REDIS_CMT_DB),
        }
    }

    // TODO: replace worker vec with spmc (http://seanmonstar.github.io/spmc/spmc/index.html)
    pub fn schedule_prediction(&self, req: PredictionRequest) {
        self.server_metrics.num_queued_predictions_counter.incr(1);
        let mut rng = thread_rng();
        // randomly pick a worker

        let w: usize = if self.prediction_workers.len() > 1 {
            rng.gen_range::<usize>(0, self.prediction_workers.len())
        } else {
            0
        };
        // let max_predictions = self.models.len() as i32;
        let max_predictions = match req.offline_models.as_ref() {
            Some(ref v) => v.len(),
            None => self.models.num_models(),

        };
        self.prediction_workers[w].predict(req, max_predictions as i32);
    }

    pub fn get_metrics(&self) -> Arc<RwLock<metrics::Registry>> {
        self.metrics.clone()
    }

    /// For now, this only returns the correction model for
    /// the latest set of model versions.
    pub fn get_correction_model(&self, uid: u32) -> String {
        let vms = self.models.get_versioned_models(&None);

        let model_names = vms.iter().map(|r| &r.name).collect::<Vec<&String>>();
        let correction_state: S = self.cmt.get(uid, &vms).unwrap_or(P::new(model_names));
        serde_json::ser::to_string_pretty(&correction_state).unwrap()
    }


    pub fn get_input_type(&self) -> InputType {
        self.input_type.clone()
    }

    pub fn schedule_update(&self, req: UpdateRequest) {
        self.server_metrics.num_queued_updates_counter.incr(1);
        // Ensure that all updates for a given user ID go to the same
        // update worker.
        self.update_workers[req.uid as usize % self.update_workers.len()].update(req);
    }


    pub fn add_new_replica(&mut self, name: String, version: u32, address: SocketAddr) {
        let vm = VersionedModel {
            name: name,
            version: Some(version),
        };
        let new_sender = self.models.add_new_replica(vm.clone(), address).unwrap();
        for p in self.prediction_workers.iter() {
            p.incorporate_new_replica(vm.clone(), new_sender.clone());
        }
        for u in self.update_workers.iter() {
            u.incorporate_new_replica(vm.clone(), new_sender.clone());
        }
    }

    pub fn add_new_model(&mut self, name: String, version: u32, addresses: Vec<SocketAddr>) {
        let vm = VersionedModel {
            name: name,
            version: Some(version),
        };
        let new_batcher = self.models.add_new_model(vm.clone(), addresses);
        for p in self.prediction_workers.iter() {
            p.incorporate_new_model(vm.clone(), new_batcher.clone());
        }
        for u in self.update_workers.iter() {
            u.incorporate_new_model(vm.clone(), new_batcher.clone());
        }
    }
}

impl<P, S> Drop for ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn drop(&mut self) {
        self.prediction_workers.clear();
        self.update_workers.clear();
        // self.models.clear();
        info!("Dropping ClipperServer");
    }
}

// struct SyncPredictionWorker<P,S>
//     where P: CorrectionPolicy<S>,
//           S: Serialize + Deserialize
// {
//     worker_id: i32,
//     input_queue: mpsc::Sender<(PredictionRequest, i32)>,
//     // cache: Arc<PredictionCache<Output>>,
//     // models: Arc<HashMap<String, PredictionBatcher<SimplePredictionCache<Output>, Output>>>,
//     _policy_marker: PhantomData<P>,
//     _state_marker: PhantomData<S>,
//     prediction_metrics: PredictionMetrics,
// }







#[derive(Clone)]
struct PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    worker_id: i32,
    input_queue: mpsc::Sender<InputChannelMessage>,
    output_queue: mpsc::Sender<OutputChannelMessage>,
    // cache: Arc<PredictionCache<Output>>,
    // models: Arc<HashMap<String, PredictionBatcher<SimplePredictionCache<Output>, Output>>>,
    _policy_marker: PhantomData<P>,
    _state_marker: PhantomData<S>,
    prediction_metrics: PredictionMetrics,
}

enum InputChannelMessage {
    Request(PredictionRequest, i32),
    IncorporateReplica(VersionedModel, mpsc::Sender<RpcPredictRequest>),
    IncorporateModel(VersionedModel,
                     PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>),
}

enum OutputChannelMessage {
    Request(PredictionRequest), /* IncorporateReplica(VersionedModel, mpsc::Sender<RpcPredictRequest>), */
}

impl<P, S> PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(worker_id: i32,
               slo_micros: u32,
               cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
               models: ModelSet,
               redis_ip: String,
               redis_port: u16,
               prediction_metrics: PredictionMetrics)
               -> PredictionWorker<P, S> {
        let (input_sender, input_receiver) = mpsc::channel::<InputChannelMessage>();
        let (output_sender, output_receiver) = mpsc::channel::<OutputChannelMessage>();
        {
            let cache = cache.clone();
            let models = models.clone();
            let redis_ip = redis_ip.clone();
            let prediction_metrics = prediction_metrics.clone();
            let output_sender = output_sender.clone();
            thread::spawn(move || {
                PredictionWorker::<P, S>::run_input_thread(worker_id,
                                                           input_receiver,
                                                           output_sender,
                                                           cache,
                                                           models,
                                                           redis_ip,
                                                           redis_port,
                                                           prediction_metrics);
            });
        }
        {
            let prediction_metrics = prediction_metrics.clone();
            thread::spawn(move || {
                PredictionWorker::<P, S>::run_output_thread(worker_id,
                                                            slo_micros,
                                                            output_receiver,
                                                            cache,
                                                            // models,
                                                            redis_ip,
                                                            redis_port,
                                                            prediction_metrics);


            });
        }
        PredictionWorker {
            worker_id: worker_id,
            input_queue: input_sender,
            output_queue: output_sender,
            // cache: cache,
            // models: models,
            _policy_marker: PhantomData,
            _state_marker: PhantomData,
            prediction_metrics: prediction_metrics,
        }
    }

    fn run_input_thread(worker_id: i32,
                        request_queue: mpsc::Receiver<InputChannelMessage>,
                        send_queue: mpsc::Sender<OutputChannelMessage>,
                        cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
                        mut models: ModelSet,
                        redis_ip: String,
                        redis_port: u16,
                        prediction_metrics: PredictionMetrics) {
        let cmt = RedisCMT::new_tcp_connection(&redis_ip, redis_port, REDIS_CMT_DB);
        info!("starting input prediction worker {} ", worker_id);
        // while let Ok((mut req, max_preds)) = request_queue.recv() {
        while let Ok(message) = request_queue.recv() {
            match message {
                InputChannelMessage::IncorporateReplica(vm, s) => {
                    models.incorporate_new_replica(vm, s);
                }
                InputChannelMessage::IncorporateModel(vm, b) => {
                    models.incorporate_new_model(vm, b);
                }
                InputChannelMessage::Request(mut req, max_preds) => {
                    {
                        req.offline_models = Some(models.get_versioned_models(&req.offline_models));

                        // let model_names = models.get_names(&req.offline_models)
                        //                         .map(|vm| vm.name)
                        //                         .collect::<Vec<_>>();
                        // TODO: take into account model versions
                        let model_names = req.offline_models
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|r| &r.name)
                            .collect::<Vec<&String>>();
                        let model_req_order: Vec<String> =
                            if max_preds < req.offline_models.as_ref().unwrap().len() as i32 {
                                let correction_state: S = cmt.get(*(&req.uid) as u32,
                                         req.offline_models
                                             .as_ref()
                                             .unwrap())
                                    .unwrap_or({
                                        info!("INPUT THREAD: creating new correction model for \
                                               user: {}",
                                              req.uid);
                                        P::new(model_names.clone())
                                    });
                                P::rank_models_desc(&correction_state, model_names.clone())
                            } else {
                                // models.keys().map(|r| r.clone()).collect::<Vec<String>>()
                                model_names.iter().map(|m| (*m).clone()).collect::<Vec<String>>()
                            };
                        let mut num_requests = 0;
                        let mut i = 0;
                        let mut model_name_version_map: HashMap<&String, &u32> = HashMap::new();
                        for vm in req.offline_models.as_ref().unwrap() {
                            model_name_version_map.insert(&vm.name, vm.version.as_ref().unwrap());
                        }
                        while num_requests < max_preds && i < model_req_order.len() {
                            // first check to see if the prediction is already cached
                            if cache.fetch(&model_req_order[i], &req.query, req.salt.clone())
                                .is_none() {
                                let vm = VersionedModel {
                                    name: model_req_order[i].clone(),
                                    version: Some(**model_name_version_map.get(&model_req_order[i])
                                        .unwrap()),
                                };
                                models.request_prediction(&vm,
                                                          RpcPredictRequest {
                                                              input: req.query.clone(),
                                                              recv_time: req.recv_time.clone(),
                                                              salt: req.salt.clone(),
                                                          });
                                num_requests += 1;
                                prediction_metrics.cache_hit_counter.incr(0, 1);
                            } else {
                                prediction_metrics.cache_hit_counter.incr(1, 1);
                            }
                            i += 1;
                        }
                    }
                    send_queue.send(OutputChannelMessage::Request(req)).unwrap();

                }
            }
        }
        info!("ending input loop: prediction worker {}", worker_id);
    }

    #[allow(unused_variables)]
    fn run_output_thread(worker_id: i32,
                         slo_micros: u32,
                         request_queue: mpsc::Receiver<OutputChannelMessage>,
                         cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
                         // models: ModelSet,
                         redis_ip: String,
                         redis_port: u16,
                         prediction_metrics: PredictionMetrics) {
        let slo = Duration::microseconds(slo_micros as i64);
        // let epsilon = time::Duration::milliseconds(slo_micros / 5.0 * 1000.0);
        let epsilon = Duration::milliseconds(3);
        // let mut cmt = RedisCMT::new_socket_connection(DEFAULT_REDIS_SOCKET, REDIS_CMT_DB);
        let cmt = RedisCMT::new_tcp_connection(&redis_ip, redis_port, REDIS_CMT_DB);
        info!("starting prediction worker {} output thread with {} ms SLO",
              worker_id,
              slo_micros as f64 / 1000.0);


        while let Ok(message) = request_queue.recv() {
            match message {
                // OutputChannelMessage::IncorporateReplica(_, _) => {
                //     // NOOP for now
                //
                //
                //
                // }
                OutputChannelMessage::Request(req) => {


                    let model_names = req.offline_models
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|r| &r.name)
                        .collect::<Vec<&String>>();
                    let correction_state: S =
                        cmt.get(*(&req.uid) as u32, req.offline_models.as_ref().unwrap())
                            .unwrap_or_else(|e| {
                                info!("OUTPUT THREAD: user: {}, error: {}", req.uid, e);
                                P::new(model_names.clone())
                            });
                    let elapsed_time = req.recv_time.to(PreciseTime::now());
                    // NOTE: assumes SLA less than 1 second
                    if elapsed_time < slo - epsilon {
                        let sleep_time =
                    StdDuration::new(0, (slo - elapsed_time).num_nanoseconds().unwrap() as u32);
                        debug!("prediction worker sleeping for {:?} ms",
                               sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                        thread::sleep(sleep_time);
                    }
                    let mut ys = HashMap::new();
                    let mut missing_ys = Vec::new();
                    for model_name in model_names {
                        match cache.fetch(model_name, &req.query, req.salt.clone()) {
                            Some(v) => {
                                ys.insert(model_name.clone(), v);
                                let mut need_write_lock: bool = false;
                                {
                                    let read_map = prediction_metrics.cache_included_thruput_meter
                                        .read()
                                        .unwrap();
                                    match read_map.get(model_name) {
                                        Some(m) => m.mark(1),
                                        None => need_write_lock = true,
                                    }
                                }
                                if need_write_lock {
                                    let mut write_map =
                                        prediction_metrics.cache_included_thruput_meter
                                            .write()
                                            .unwrap();
                                    let meter = write_map.entry(model_name.clone())
                                        .or_insert({
                                            let metric_name = format!("{}:cache_included_thruput",
                                                                      model_name);
                                            prediction_metrics.registry
                                                .write()
                                                .unwrap()
                                                .create_meter(metric_name)
                                        });
                                    meter.mark(1);
                                }
                            }
                            None => missing_ys.push(model_name.clone()),
                        }
                    }
                    prediction_metrics.in_time_predictions_hist.insert(ys.len() as i64);

                    // use correction policy to make the final prediction
                    let prediction = P::predict(&correction_state, ys, missing_ys);
                    // execute the user's callback on this thread
                    (req.on_predict)(prediction);
                    let end_time = PreciseTime::now();
                    // TODO: metrics
                    let latency = req.recv_time.to(end_time).num_microseconds().unwrap();
                    prediction_metrics.latency_hist.insert(latency);
                    prediction_metrics.thruput_meter.mark(1);
                    prediction_metrics.pred_counter.incr(1);
                    prediction_metrics.num_queued_predictions_counter.decr(1);

                }
            }
        }
        info!("ending output loop: prediction worker {}", worker_id);
    }

    /// Execute the provided prediction request,
    /// allowing at most `max_preds` requests to offline models. This
    /// allows Clipper to do some load-shedding under heavy load, while
    /// allowing individual correction policies to rank which offline models
    /// should be evaluated.
    pub fn predict(&self, r: PredictionRequest, max_preds: i32) {
        self.input_queue.send(InputChannelMessage::Request(r, max_preds)).unwrap();
    }

    pub fn incorporate_new_replica(&self,
                                   vm: VersionedModel,
                                   sender: mpsc::Sender<RpcPredictRequest>) {
        self.input_queue.send(InputChannelMessage::IncorporateReplica(vm, sender)).unwrap();
    }

    pub fn incorporate_new_model(&self,
                                 vm: VersionedModel,
                                 batcher: PredictionBatcher<SimplePredictionCache<Output,
                                                                                  EqualityHasher>>) {
        self.input_queue.send(InputChannelMessage::IncorporateModel(vm, batcher)).unwrap();
    }

    // pub fn add_model(&self) {}

    // #[allow(dead_code)]
    // pub fn shutdown() {
    //     unimplemented!();
    // }
}

impl<P, S> Drop for PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn drop(&mut self) {
        // info!("DROPPING PREDICTION WORKER {}", self.worker_id);
    }
}



#[allow(dead_code)]
struct UpdateWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    worker_id: i32,
    input_queue: mpsc::Sender<UpdateMessage>,
    _policy_marker: PhantomData<P>,
    _state_marker: PhantomData<S>,
    runner_handle: Option<JoinHandle<()>>,
    update_metrics: UpdateMetrics,
}

impl<P, S> UpdateWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(worker_id: i32,
               cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
               models: ModelSet,
               window_size: isize,
               redis_ip: String,
               redis_port: u16,
               update_metrics: UpdateMetrics)
               -> UpdateWorker<P, S> {

        let (sender, receiver) = mpsc::channel::<UpdateMessage>();
        let jh;
        {
            let update_metrics = update_metrics.clone();
            jh = thread::spawn(move || {
                UpdateWorker::<P, S>::run(worker_id,
                                          receiver,
                                          cache.clone(),
                                          models,
                                          window_size,
                                          redis_ip,
                                          redis_port,
                                          update_metrics);
            });
        }
        UpdateWorker {
            worker_id: worker_id,
            input_queue: sender,
            // cache: cache,
            // models: models,
            _policy_marker: PhantomData,
            _state_marker: PhantomData,
            runner_handle: Some(jh),
            update_metrics: update_metrics,
        }
    }


    /// A note on the implementation here: an `UpdateRequest` can contain
    /// multiple pieces of training data which will be aggregated together
    /// and result in a single call to `CorrectionPolicy::train()`.
    /// However, each call to this function will result in a fetch of
    /// the training data from the UpdateTable and a call to retrain
    /// the correction policy, even if these requests overlap.
    pub fn update(&self, r: UpdateRequest) {
        self.input_queue.send(UpdateMessage::Request(r)).unwrap();
    }

    #[allow(unused_variables, unused_mut)]
    fn run(worker_id: i32,
           request_queue: mpsc::Receiver<UpdateMessage>,
           cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
           mut models: ModelSet,
           window_size: isize,
           redis_ip: String,
           redis_port: u16,
           update_metrics: UpdateMetrics) {
        // let mut cmt: RedisCMT<S> = RedisCMT::new_socket_connection(DEFAULT_REDIS_SOCKET,
        //                                                            REDIS_CMT_DB);

        let mut cmt: RedisCMT<S> =
            RedisCMT::new_tcp_connection(&redis_ip, redis_port, REDIS_CMT_DB);
        let mut update_table: RedisUpdateTable =
            RedisUpdateTable::new_tcp_connection(&redis_ip, redis_port, REDIS_UPDATE_DB);
        // RedisUpdateTable::new_socket_connection(DEFAULT_REDIS_SOCKET, REDIS_UPDATE_DB);
        info!("starting update worker {}", worker_id);

        // Max number of updates to perform from ready_updates before
        // checking for new updates whose prediction's need to be computed.
        let update_batch_size: usize = 10;

        let mut waiting_updates: Vec<Arc<RwLock<UpdateDependencies>>> = Vec::new();
        // let mut ready_updates: HashMap<u32, Vec<UpdateDependencies>> = HashMap::new();
        let mut ready_updates: Vec<UpdateDependencies> = Vec::new();
        // Track the order of updates by user ID
        // let mut update_order = VecDeque::new();
        let mut consecutive_sleeps = 0;

        loop {
            let try_req = request_queue.try_recv();
            let mut sleep = false;
            match try_req {
                Ok(message) => {
                    match message {
                        UpdateMessage::Request(mut req) => {

                            req.offline_models =
                                Some(models.get_versioned_models(&req.offline_models));
                            UpdateWorker::<P, S>::stage_update(req,
                                                               cache.clone(),
                                                               &mut waiting_updates,
                                                               &models,
                                                               &mut update_table,
                                                               window_size)
                        }
                        UpdateMessage::Shutdown => {
                            // info!("Update worker {} got shutdown message and is executing break",
                            //       worker_id);
                            break;
                        }
                        UpdateMessage::IncorporateReplica(vm, s) => {
                            models.incorporate_new_replica(vm, s);
                        }
                        UpdateMessage::IncorporateModel(vm, b) => {
                            models.incorporate_new_model(vm, b);
                        }
                    }
                    // models.keys().collect::<Vec<&String>>())
                }
                Err(mpsc::TryRecvError::Empty) => sleep = true,
                Err(mpsc::TryRecvError::Disconnected) => {
                    // info!("Update worker {} detected disconnect", worker_id);
                    break;
                }
            }

            UpdateWorker::<P, S>::check_for_ready_updates(&mut waiting_updates, &mut ready_updates);

            UpdateWorker::<P, S>::execute_updates(update_batch_size,
                                                  &mut ready_updates,
                                                  &mut cmt,
                                                  &update_metrics);


            if sleep {
                thread::sleep(StdDuration::from_millis(5));
                consecutive_sleeps += 1;
            } else {
                consecutive_sleeps = 0;
            }


        }
        info!("Ending loop: update worker {}", worker_id);
    }

    // #[allow(unused_variables)]
    fn stage_update(mut req: UpdateRequest,
                    cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
                    waiting_updates: &mut Vec<Arc<RwLock<UpdateDependencies>>>,
                    models: &ModelSet,
                    update_table: &mut RedisUpdateTable,
                    window_size: isize) {

        // let window_size = 20;
        let num_new_updates = req.updates.len();
        // TODO: better error handling
        let mut old_updates = update_table.get_updates(req.uid, window_size)
            .unwrap()
            .into_iter()
            .map(|u| {
                Update {
                    query: Arc::new(u.0),
                    label: u.1,
                }
            })
            .collect::<Vec<_>>();
        req.updates.append(&mut old_updates);

        // now write new updates to UpdateTable
        for i in 0..num_new_updates {
            // TODO: better error handling
            update_table.add_update(req.uid, &req.updates[i].query, &req.updates[i].label)
                .unwrap();

        }



        // TODO: move to more fine-grained locking
        let update_dependencies = Arc::new(RwLock::new(UpdateDependencies::new(req.clone())));
        // Get predictions first as those are likely to incur the most latency
        let mut idx = 0;
        for update in req.updates.iter() {
            for m in req.offline_models.as_ref().unwrap() {
                let u = update_dependencies.clone();
                let model_name = m.name.clone();
                cache.add_listener(&m.name,
                                   &update.query,
                                   None,
                                   Box::new(move |o| {
                                       let mut deps = u.write().unwrap();
                                       deps.predictions[idx].insert(model_name.clone(), o);
                                   }));
                if cache.fetch(&m.name, &update.query, None).is_none() {

                    models.request_prediction(&m,
                                              RpcPredictRequest {
                                                  input: update.query.clone(),
                                                  recv_time: req.recv_time.clone(),
                                                  salt: None,
                                              });
                }
            }
            idx += 1;
        }
        waiting_updates.push(update_dependencies);
    }

    fn check_for_ready_updates(waiting_updates: &mut Vec<Arc<RwLock<UpdateDependencies>>>,
                               ready_updates: &mut Vec<UpdateDependencies>) {

        // determine which waiting updates have all of their predictions
        // available
        let mut ready_update_indexes = Vec::new();
        for i in 0..waiting_updates.len() {
            let wu = waiting_updates.get(i).unwrap().read().unwrap();
            let ref current_update_deps = wu.predictions;
            let mut ready = true;
            for preds in current_update_deps.iter() {
                if preds.len() < wu.num_predictions {
                    ready = false;
                    break;
                }
            }
            if ready {
                ready_update_indexes.push(i);
            }
        }


        // Move updates from the waiting queue to the ready queue.
        // As we remove elements from waiting_updates, everything behind
        // that index is shifted an element to the left so the index changes. For this reason,
        // we track the offset (how many items we've removed from earlier in the vector) to
        // make sure remove the correct items. We iterate forward through the vec because updates
        // get added to the vec in the order they arrive, so given two updates that are ready to be
        // processed, we want to process the earlier one first.

        let mut offset = 0;
        for i in ready_update_indexes {
            // Remove from Arc and RwLock

            let update_dep: UpdateDependencies =
                match Arc::try_unwrap(waiting_updates.remove(i - offset)) {
                    Ok(u) => u.into_inner().unwrap(),
                    Err(_) => panic!("Uh oh"),
                };
            ready_updates.push(update_dep);
            offset += 1;

        }
    }

    fn execute_updates(max_updates: usize,
                       ready_updates: &mut Vec<UpdateDependencies>,
                       // update_order: &mut VecDeque<usize>,
                       cmt: &mut RedisCMT<S>,
                       update_metrics: &UpdateMetrics) {
        let num_updates = ready_updates.len();
        for mut update_dep in ready_updates.drain(0..cmp::min(max_updates, num_updates)) {
            let uid = update_dep.req.uid;
            // let mut update_deps = ready_updates.remove(&uid).unwrap();
            let correction_state: S = match cmt.get(uid,
                                                    update_dep.req
                                                        .offline_models
                                                        .as_ref()
                                                        .unwrap()) {
                Ok(s) => s,
                Err(e) => {
                    info!("Error in getting correction state for update: {}", e);
                    info!("Creating model state for new user: {}", uid);
                    P::new(update_dep.req
                        .offline_models
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|mv| &mv.name)
                        .collect::<Vec<&String>>())
                }
            };
            let mut collected_inputs: Vec<Arc<Input>> = Vec::new();
            let mut collected_predictions: Vec<HashMap<String, Output>> = Vec::new();
            let mut collected_labels: Vec<Output> = Vec::new();
            for (preds, update) in update_dep.predictions
                .drain(..)
                .zip(update_dep.req.updates.drain(..)) {
                collected_inputs.push(update.query);
                collected_predictions.push(preds);
                collected_labels.push(update.label);
            }
            let new_state = P::train(&correction_state,
                                     collected_inputs,
                                     collected_predictions,
                                     collected_labels);
            match cmt.put(uid,
                          &new_state,
                          update_dep.req
                              .offline_models
                              .as_ref()
                              .unwrap()) {
                Ok(_) => {

                    // info!("putting new state for {}", uid);
                }
                Err(e) => warn!("{}", e),
            }

            let end_time = PreciseTime::now();
            let latency = update_dep.req.recv_time.to(end_time).num_microseconds().unwrap();
            update_metrics.latency_hist.insert(latency);
            update_metrics.thruput_meter.mark(1);
            update_metrics.update_counter.incr(1);
            update_metrics.num_queued_updates_counter.decr(1);
        }
    }

    pub fn incorporate_new_replica(&self,
                                   vm: VersionedModel,
                                   sender: mpsc::Sender<RpcPredictRequest>) {
        self.input_queue.send(UpdateMessage::IncorporateReplica(vm, sender)).unwrap();
    }

    pub fn incorporate_new_model(&self,
                                 vm: VersionedModel,
                                 batcher: PredictionBatcher<SimplePredictionCache<Output,
                                                                                  EqualityHasher>>) {
        self.input_queue.send(UpdateMessage::IncorporateModel(vm, batcher)).unwrap();
    }
}

impl<P, S> Drop for UpdateWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn drop(&mut self) {
        self.input_queue.send(UpdateMessage::Shutdown).unwrap();
        self.runner_handle.take().unwrap().join().unwrap();
        // info!("DROPPING UPDATE WORKER {}", self.worker_id);
    }
}


enum UpdateMessage {
    Request(UpdateRequest),
    Shutdown,
    IncorporateReplica(VersionedModel, mpsc::Sender<RpcPredictRequest>),
    IncorporateModel(VersionedModel,
                     PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>),
}

struct UpdateDependencies {
    // state: Option<S>,
    req: UpdateRequest,
    predictions: Vec<HashMap<String, Output>>,
    num_predictions: usize,
}


impl UpdateDependencies {
    pub fn new(req: UpdateRequest) -> UpdateDependencies {
        let num_updates = req.updates.len();
        UpdateDependencies {
            // state: None,
            num_predictions: req.offline_models.as_ref().unwrap().len(),
            req: req,
            predictions: vec![HashMap::new(); num_updates],
        }
    }
}
