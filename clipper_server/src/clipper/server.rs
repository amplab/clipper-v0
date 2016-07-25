
use time::{PreciseTime, Duration};
use std::marker::PhantomData;
use serde::ser::Serialize;
use serde::de::Deserialize;
use std::sync::{mpsc, RwLock, Arc};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::cmp;
use std::thread::{self, JoinHandle};
use std::time::Duration as StdDuration;

#[allow(unused_imports)]
use cmt::{CorrectionModelTable, RedisCMT, UpdateTable, RedisUpdateTable, REDIS_CMT_DB,
          REDIS_UPDATE_DB, DEFAULT_REDIS_SOCKET};
use cache::{PredictionCache, SimplePredictionCache};
use configuration::ClipperConf;
use hashing::EqualityHasher;
use batching::{RpcPredictRequest, PredictionBatcher};
use correction_policy::CorrectionPolicy;
use metrics;

// pub const SLO: i64 = 20;

pub type Output = f64;

pub type OnPredict = Fn(Output) -> () + Send;



/// Specifies the input type and expected length. A negative length indicates
/// a variable length input. `Str` does not have a length because Strings
/// are assumed to be always be variable length.
#[derive(Clone, PartialEq, Debug)]
pub enum InputType {
    Integer(i32),
    Float(i32),
    Str,
    Byte(i32),
}

// #[derive(Hash, Clone, Debug)]
#[derive(Clone,Debug,Serialize,Deserialize, PartialEq, PartialOrd)]
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

#[allow(dead_code)]
pub struct PredictionRequest {
    recv_time: PreciseTime,
    uid: u32,
    query: Input,
    on_predict: Box<OnPredict>,
}

impl PredictionRequest {
    pub fn new(uid: u32, input: Input, on_predict: Box<OnPredict>) -> PredictionRequest {
        PredictionRequest {
            recv_time: PreciseTime::now(),
            uid: uid,
            query: input,
            on_predict: on_predict,
        }
    }
}



#[derive(Clone)]
pub struct Update {
    pub query: Input,
    pub label: Output,
}

#[derive(Clone)]
pub struct UpdateRequest {
    recv_time: PreciseTime,
    uid: u32,
    updates: Vec<Update>, /* query: Input,
                           * label: Output */
}

impl UpdateRequest {
    pub fn new(uid: u32, updates: Vec<Update>) -> UpdateRequest {
        UpdateRequest {
            recv_time: PreciseTime::now(),
            uid: uid,
            updates: updates,
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
    models: HashMap<String, PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>,
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
            Arc::new(SimplePredictionCache::new(&conf.models, conf.cache_size.clone()));

        let mut model_batchers = HashMap::new();
        for m in conf.models.into_iter() {
            let b = PredictionBatcher::new(m.name.clone(),
                                           m.addresses.clone(),
                                           conf.input_type.clone(),
                                           conf.metrics.clone(),
                                           cache.clone(),
                                           conf.slo_micros.clone());
            model_batchers.insert(m.name.clone(), b);
        }

        // let models = Arc::new(model_batchers);
        let models = model_batchers;




        let mut prediction_workers = Vec::with_capacity(conf.num_predict_workers.clone());
        for i in 0..conf.num_predict_workers {
            prediction_workers.push(PredictionWorker::new(i as i32,
                                                          conf.slo_micros.clone(),
                                                          cache.clone(),
                                                          models.clone()));
        }
        let mut update_workers = Vec::with_capacity(conf.num_update_workers.clone());
        for i in 0..conf.num_update_workers {
            update_workers.push(UpdateWorker::new(i as i32,
                                                  cache.clone(),
                                                  models.clone(),
                                                  conf.window_size));
        }
        ClipperServer {
            prediction_workers: prediction_workers,
            update_workers: update_workers,
            // cache: cache,
            metrics: conf.metrics,
            input_type: conf.input_type,
            models: models.clone(),
        }
    }

    // TODO: replace worker vec with spmc (http://seanmonstar.github.io/spmc/spmc/index.html)
    pub fn schedule_prediction(&self, req: PredictionRequest) {
        let mut rng = thread_rng();
        // randomly pick a worker

        let w: usize = if self.prediction_workers.len() > 1 {
            rng.gen_range::<usize>(0, self.prediction_workers.len())
        } else {
            0
        };
        let max_predictions = self.models.len() as i32;
        self.prediction_workers[w].predict(req, max_predictions);
    }

    pub fn get_metrics(&self) -> Arc<RwLock<metrics::Registry>> {
        self.metrics.clone()
    }

    pub fn get_input_type(&self) -> InputType {
        self.input_type.clone()
    }

    pub fn schedule_update(&self, req: UpdateRequest) {
        // Ensure that all updates for a given user ID go to the same
        // update worker. If there are
        self.update_workers[req.uid as usize % self.update_workers.len()].update(req);
    }

    // pub fn shutdown(&mut self) {
    //     for p in self.prediction_workers.iter_mut() {
    //         p.input_queue
    //
    //
    //     }
    // }
}

impl<P, S> Drop for ClipperServer<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn drop(&mut self) {
        self.prediction_workers.clear();
        self.update_workers.clear();
        self.models.clear();
        info!("Dropping ClipperServer");
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
               slo_micros: u32,
               cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
               models: HashMap<String,
                               PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>)
               -> PredictionWorker<P, S> {
        let (sender, receiver) = mpsc::channel::<(PredictionRequest, i32)>();
        thread::spawn(move || {
            PredictionWorker::<P, S>::run(worker_id, slo_micros, receiver, cache.clone(), models);
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

    #[allow(unused_variables)]
    fn run(worker_id: i32,
           slo_micros: u32,
           request_queue: mpsc::Receiver<(PredictionRequest, i32)>,
           cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
           models: HashMap<String,
                           PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>) {
        let slo = Duration::microseconds(slo_micros as i64);
        // let epsilon = time::Duration::milliseconds(slo_micros / 5.0 * 1000.0);
        let epsilon = Duration::milliseconds(1);
        let mut cmt = RedisCMT::new_socket_connection(DEFAULT_REDIS_SOCKET, REDIS_CMT_DB);
        info!("starting prediction worker {} with {} ms SLO",
              worker_id,
              slo_micros as f64 / 1000.0);
        if worker_id == 0 {
            cmt.put(0 as u32, &P::new(models.keys().collect::<Vec<_>>())).unwrap();
        }

        while let Ok((req, max_preds)) = request_queue.recv() {
            let correction_state: S = cmt.get(*(&req.uid) as u32)
                                         .unwrap_or(cmt.get(0_u32).unwrap());
            let model_req_order: Vec<String> = if max_preds < models.len() as i32 {
                P::rank_models_desc(&correction_state, models.keys().collect::<Vec<&String>>())
            } else {
                models.keys().map(|r| r.clone()).collect::<Vec<String>>()
            };
            let mut num_requests = 0;
            let mut i = 0;
            while num_requests < max_preds && i < models.len() {
                // first check to see if the prediction is already cached
                if cache.fetch(&model_req_order[i], &req.query).is_none() {
                    // on cache miss, send to batching layer
                    // TODO: can we avoid copying the input for each model?
                    models.get(&model_req_order[i])
                          .unwrap()
                          .request_prediction(RpcPredictRequest {
                              input: req.query.clone(),
                              recv_time: req.recv_time.clone(),
                          });
                    num_requests += 1;
                }
                i += 1;
            }

            let elapsed_time = req.recv_time.to(PreciseTime::now());
            // TODO: assumes SLA less than 1 second
            if elapsed_time < slo - epsilon {
                let sleep_time =
                    StdDuration::new(0, (slo - elapsed_time).num_nanoseconds().unwrap() as u32);
                debug!("prediction worker sleeping for {:?} ms",
                       sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
                thread::sleep(sleep_time);
            }
            let mut ys = HashMap::new();
            let mut missing_ys = Vec::new();
            for model_name in models.keys() {
                match cache.fetch(model_name, &req.query) {
                    Some(v) => {
                        ys.insert(model_name.clone(), v);
                    }
                    None => missing_ys.push(model_name.clone()),
                }
            }

            // use correction policy to make the final prediction
            let prediction = P::predict(&correction_state, ys, missing_ys);
            // execute the user's callback on this thread
            (req.on_predict)(prediction);
            let end_time = PreciseTime::now();
            // TODO: metrics
            let latency = req.recv_time.to(end_time).num_microseconds().unwrap();
            // pred_metrics.latency_hist.insert(latency);
            // pred_metrics.thruput_meter.mark(1);
            // pred_metrics.pred_counter.incr(1);
        }
        info!("ending loop: prediction worker {}", worker_id);
    }

    pub fn predict(&self, r: PredictionRequest, max_preds: i32) {
        self.input_queue.send((r, max_preds)).unwrap();
    }

    #[allow(dead_code)]
    pub fn shutdown() {
        unimplemented!();
    }
}

impl<P, S> Drop for PredictionWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn drop(&mut self) {
        // info!("DROPPING PREDICTION WORKER {}", self.worker_id);
    }
}


#[derive(Clone)]
#[allow(dead_code)]
struct PredictionMetrics {
    latency_hist: Arc<metrics::Histogram>,
    pred_counter: Arc<metrics::Counter>,
    thruput_meter: Arc<metrics::Meter>,
    accuracy_counter: Arc<metrics::RatioCounter>,
}

#[allow(dead_code)]
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
}

impl<P, S> UpdateWorker<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    pub fn new(worker_id: i32,
               cache: Arc<SimplePredictionCache<Output, EqualityHasher>>,
               models: HashMap<String,
                               PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>,
               window_size: isize)
               -> UpdateWorker<P, S> {

        let (sender, receiver) = mpsc::channel::<UpdateMessage>();
        let jh = thread::spawn(move || {
            UpdateWorker::<P, S>::run(worker_id, receiver, cache.clone(), models, window_size);
        });
        UpdateWorker {
            worker_id: worker_id,
            input_queue: sender,
            // cache: cache,
            // models: models,
            _policy_marker: PhantomData,
            _state_marker: PhantomData,
            runner_handle: Some(jh),
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
           models: HashMap<String,
                           PredictionBatcher<SimplePredictionCache<Output, EqualityHasher>>>,
           window_size: isize) {
        let mut cmt: RedisCMT<S> = RedisCMT::new_socket_connection(DEFAULT_REDIS_SOCKET,
                                                                   REDIS_CMT_DB);
        let mut update_table: RedisUpdateTable =
            RedisUpdateTable::new_socket_connection(DEFAULT_REDIS_SOCKET, REDIS_UPDATE_DB);
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
                        UpdateMessage::Request(req) => {
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
                    }
                    // models.keys().collect::<Vec<&String>>())
                }
                Err(mpsc::TryRecvError::Empty) => sleep = true,
                Err(mpsc::TryRecvError::Disconnected) => {
                    // info!("Update worker {} detected disconnect", worker_id);
                    break;
                }
            }

            UpdateWorker::<P, S>::check_for_ready_updates(&mut waiting_updates,
                                                          &mut ready_updates,
                                                          // &mut update_order,
                                                          models.len());

            UpdateWorker::<P, S>::execute_updates(update_batch_size,
                                                  &mut ready_updates,
                                                  // &mut update_order,
                                                  models.keys().collect::<Vec<&String>>(),
                                                  &mut cmt);


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
                    models: &HashMap<String,
                                     PredictionBatcher<SimplePredictionCache<Output,
                                                                             EqualityHasher>>>,
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
                                                  query: u.0,
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
            for (m, b) in models.iter() {
                let u = update_dependencies.clone();
                let model_name = m.clone();
                cache.add_listener(m,
                                   &update.query,
                                   Box::new(move |o| {
                                       let mut deps = u.write().unwrap();
                                       deps.predictions[idx].insert(model_name.clone(), o);
                                   }));
                if cache.fetch(m, &update.query).is_none() {
                    b.request_prediction(RpcPredictRequest {
                        input: update.query.clone(),
                        recv_time: req.recv_time.clone(),
                    });
                }
            }
            idx += 1;
        }
        waiting_updates.push(update_dependencies);
    }


    fn check_for_ready_updates(waiting_updates: &mut Vec<Arc<RwLock<UpdateDependencies>>>,
                               ready_updates: &mut Vec<UpdateDependencies>,
                               // update_order: &mut VecDeque<usize>,
                               num_models: usize) {

        // determine which waiting updates have all of their predictions
        // available
        let mut ready_update_indexes = Vec::new();
        for i in 0..waiting_updates.len() {
            let wu = waiting_updates.get(i).unwrap();
            let ref current_update_deps = wu.read().unwrap().predictions;
            let mut ready = true;
            for preds in current_update_deps.iter() {
                if preds.len() < num_models {
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

            // let uid = update_dep.req.uid;


            // let mut done = false;
            // if let Some(u) = ready_updates.get_mut(&uid) {
            //     u.push(update_dep);
            //     done = true;
            // }
            // if !done {
            //     // If no entries for this user exist, add their user ID to the back of
            //     // the update queue. If they have an existing vec of updates, they are already
            //     // in the update queue and so we don't need to add them.
            //     update_order.push_back(uid as usize);
            //     ready_updates.insert(uid, vec![update_dep]);
            // }

            // let mut entry = ready_updates.entry(uid).or_insert(Vec::new());
            // if (*entry).len() == 0 {
            //     update_order.push_back(uid as usize);
            // }
            // (*entry).push(update_dep);

            // match ready_updates.get_mut(&uid) {
            //     Some(u) => u.push(update_dep),
            //     None => {
            //         // If no entries for this user exist, add their user ID to the back of
            //         // the update queue. If they have an existing vec of updates, they are already
            //         // in the update queue and so we don't need to add them.
            //         update_order.push_back(uid as usize);
            //         ready_updates.insert(uid, vec![update_dep]);
            //     }
            // }
        }
    }

    fn execute_updates(max_updates: usize,
                       ready_updates: &mut Vec<UpdateDependencies>,
                       // update_order: &mut VecDeque<usize>,
                       model_names: Vec<&String>,
                       cmt: &mut RedisCMT<S>) {
        let num_updates = ready_updates.len();
        for mut update_dep in ready_updates.drain(0..cmp::min(max_updates, num_updates)) {
            let uid = update_dep.req.uid;
            // let mut update_deps = ready_updates.remove(&uid).unwrap();
            let correction_state: S = match cmt.get(uid) {
                Ok(s) => s,
                Err(e) => {
                    info!("Error in getting correction state for update: {}", e);
                    info!("Creating model state for new user: {}", uid);
                    P::new(model_names.clone())
                }
            };
            let mut collected_inputs: Vec<Input> = Vec::new();
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
            match cmt.put(uid, &new_state) {
                Ok(_) => {
                    // info!("putting new state for {}", uid);
                }
                Err(e) => warn!("{}", e),
            }
        }
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
}

struct UpdateDependencies {
    // state: Option<S>,
    req: UpdateRequest,
    predictions: Vec<HashMap<String, Output>>,
}


impl UpdateDependencies {
    pub fn new(req: UpdateRequest) -> UpdateDependencies {
        let num_updates = req.updates.len();
        UpdateDependencies {
            // state: None,
            req: req,
            predictions: vec![HashMap::new(); num_updates],
        }
    }
}
