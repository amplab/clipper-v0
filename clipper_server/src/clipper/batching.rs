use time;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration as StdDuration;
// use std::net::Shutdown
use std::thread::{self, JoinHandle};
use std::sync::{RwLock, Arc, mpsc, Mutex};
use std::cmp;
use std::boxed::Box;
use rand::{thread_rng, Rng};
use server::{self, InputType, Output};
use metrics;
use rpc;
use cache::PredictionCache;
// use serde::ser::Serialize;
use serde_json;
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write, Cursor};
use std::mem;


#[derive(Clone)]
pub struct RpcPredictRequest {
    pub input: Arc<server::Input>,
    pub recv_time: time::PreciseTime,
    pub salt: Option<i32>,
    pub ttl: bool, // okay to drop if this prediction is too old
}

#[derive(Debug, Serialize, PartialEq, Eq, Clone)]
struct VariableLoadBatchResponseMetrics {
    pub measurement_time_nanos: u64,
    pub total_latency_nanos: u64,
}

pub struct PredictionBatcher<C>
    where C: PredictionCache<Output>
{
    name: String,
    input_queues: Vec<mpsc::Sender<RpcPredictRequest>>,
    cache: Arc<C>,
    latency_hist: Arc<metrics::Histogram>,
    batch_size_hist: Arc<metrics::Histogram>,
    thruput_meter: Arc<metrics::Meter>,
    predictions_counter: Arc<metrics::Counter>,
    input_type: InputType,
    slo_micros: u32,
    batch_strategy: BatchStrategy,
    join_handles: Option<Arc<Mutex<Vec<Option<JoinHandle<()>>>>>>,
}

impl<C> Drop for PredictionBatcher<C>
    where C: PredictionCache<Output>
{
    fn drop(&mut self) {
        self.input_queues.clear();
        match Arc::try_unwrap(self.join_handles.take().unwrap()) {
            Ok(u) => {
                let mut raw_handles = u.into_inner().unwrap();
                raw_handles.drain(..)
                    .map(|mut jh| jh.take().unwrap().join().unwrap())
                    .collect::<Vec<_>>();
            }
            Err(_) => debug!("PredictionBatcher still has outstanding references"),
        }
    }
}

impl<C> Clone for PredictionBatcher<C>
    where C: PredictionCache<Output>
{
    fn clone(&self) -> PredictionBatcher<C> {
        PredictionBatcher {
            name: self.name.clone(),
            input_queues: self.input_queues.clone(),
            cache: self.cache.clone(),
            latency_hist: self.latency_hist.clone(),
            batch_size_hist: self.batch_size_hist.clone(),
            thruput_meter: self.thruput_meter.clone(),
            predictions_counter: self.predictions_counter.clone(),
            input_type: self.input_type.clone(),
            slo_micros: self.slo_micros,
            batch_strategy: self.batch_strategy.clone(),
            join_handles: self.join_handles.clone(),
        }
    }
}




impl<C> PredictionBatcher<C>
    where C: PredictionCache<Output> + 'static + Send + Sync
{
    pub fn new(name: String,
               addrs: Vec<SocketAddr>,
               input_type: InputType,
               metric_register: Arc<RwLock<metrics::Registry>>,
               cache: Arc<C>,
               slo_micros: u32,
               batch_strategy: BatchStrategy)
               -> PredictionBatcher<C> {

        let latency_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:model_latency", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        let thruput_meter: Arc<metrics::Meter> = {
            let metric_name = format!("{}:model_thruput", name);
            metric_register.write().unwrap().create_meter(metric_name)
        };

        let batch_size_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:model_batch_size", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        let predictions_counter: Arc<metrics::Counter> = {
            let metric_name = format!("{}:prediction_counter", name);
            metric_register.write().unwrap().create_counter(metric_name)
        };

        let input_queues = Vec::with_capacity(addrs.len());
        let join_handles = Vec::with_capacity(addrs.len());
        let mut prediction_batcher = PredictionBatcher {
            name: name,
            input_queues: input_queues,
            cache: cache,
            latency_hist: latency_hist.clone(),
            batch_size_hist: batch_size_hist.clone(),
            thruput_meter: thruput_meter.clone(),
            predictions_counter: predictions_counter.clone(),
            input_type: input_type.clone(),
            slo_micros: slo_micros,
            batch_strategy: batch_strategy.clone(),
            join_handles: Some(Arc::new(Mutex::new(join_handles))),
        };
        // info!("starting batchers encoding {} times", num_encodes);
        for a in addrs.into_iter() {
            prediction_batcher.add_new_replica(a, metric_register.clone());
        }
        prediction_batcher
    }

    fn run(name: String,
           receiver: mpsc::Receiver<RpcPredictRequest>,
           addr: SocketAddr,
           latency_hist: Arc<metrics::Histogram>,
           batch_size_hist: Arc<metrics::Histogram>,
           thruput_meter: Arc<metrics::Meter>,
           predictions_counter: Arc<metrics::Counter>,
           input_type: InputType,
           cache: Arc<C>,
           slo_micros: u32,
           batch_strategy: BatchStrategy,
           metric_register: Arc<RwLock<metrics::Registry>>) {


        let mut stream: TcpStream;
        loop {
            match TcpStream::connect(addr) {
                Ok(s) => {
                    info!("Connected to {} model wrapper at {:?}", name, addr);
                    stream = s;
                    break;
                }
                Err(_) => {
                    info!("Couldn't connect to {} model wrapper. Sleeping 1 second",
                          name);
                    thread::sleep(StdDuration::from_millis(500));
                }
            }
        }
        stream.set_nodelay(true).unwrap();
        stream.set_read_timeout(None).unwrap();
        // let max_batch_size = batch_size;
        let mut cur_batch_size = 1;
        let mut var_load_batch_latency_logger: Vec<VariableLoadBatchResponseMetrics> =
            Vec::with_capacity(11000);
        let mut batcher: Box<Batcher> = match batch_strategy {
            BatchStrategy::Static { size: s } => {
                Box::new(StaticBatcher { batch_size: s }) as Box<Batcher>
            }
            BatchStrategy::AIMD => Box::new(AIMDBatcher {}) as Box<Batcher>,
            BatchStrategy::Learned { sample_size, opt_addr } => {
                let alpha_gauge = {
                    let metric_name = format!("{}:{}:alpha_gauge", name, addr);
                    metric_register.write().unwrap().create_counter(metric_name)
                };
                let beta_gauge = {
                    let metric_name = format!("{}:{}:beta_gauge", name, addr);
                    metric_register.write().unwrap().create_counter(metric_name)
                };
                let max_batch_size_gauge = {
                    let metric_name = format!("{}:{}:max_batch_size_gauge", name, addr);
                    metric_register.write().unwrap().create_counter(metric_name)
                };
                let max_thru_gauge = {
                    let metric_name = format!("{}:{}:max_thru_gauge", name, addr);
                    metric_register.write().unwrap().create_counter(metric_name)
                };
                let base_thru_gauge = {
                    let metric_name = format!("{}:{}:base_thru_gauge", name, addr);
                    metric_register.write().unwrap().create_counter(metric_name)
                };
                let gauges = BatcherGauges {
                    alpha_gauge: alpha_gauge,
                    beta_gauge: beta_gauge,
                    max_batch_size_gauge: max_batch_size_gauge,
                    max_thru_gauge: max_thru_gauge,
                    base_thru_gauge: base_thru_gauge,
                };
                Box::new(LearnedBatcher::new(name.clone(),
                                             sample_size,
                                             gauges,
                                             opt_addr)) as Box<Batcher>
            }
        };

        let batch_setup_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:batch_setup_hist:", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        let rpc_time_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:rpc_time_hist:", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        let ser_time_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:ser_time_hist:", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        let send_time_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:send_time_hist:", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        let recv_time_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}:recv_time_hist:", name);
            metric_register.write().unwrap().create_histogram(metric_name, 8224)
        };

        // block until new request, then try to get more requests
        while let Ok(first_req) = receiver.recv() {
            // Drop predictions we have no hope of evaluating in time.
            // This is a crude way of implementing a TTL.
            // NOTE: We only check the first request, because the request
            // queue is in FIFO order so this is guaranteed to be the oldest
            // request in the batch.
            let t1 = time::precise_time_ns();
            let delay =
                first_req.recv_time.to(time::PreciseTime::now()).num_microseconds().unwrap();
            if first_req.ttl && delay > slo_micros as i64 {
                continue;
            }
            let mut batch: Vec<RpcPredictRequest> = Vec::new();
            batch.push(first_req);
            let start_time = time::PreciseTime::now();
            // let max_batch_size =  cur_batch_size;
            assert!(cur_batch_size >= 1);
            while batch.len() < cur_batch_size {
                if let Ok(req) = receiver.try_recv() {
                    // let req_latency = req.req_start_time.to(time::PreciseTime::now()).num_microseconds().unwrap();
                    // println!("req->features latency {} (ms)", (req_latency as f64 / 1000.0));
                    batch.push(req);
                } else {
                    break;
                }
            }
            assert!(batch.len() > 0);

            let t2 = time::precise_time_ns();
            let batch_setup_time = t2 - t1;
            batch_setup_hist.insert(batch_setup_time as i64);
            let (response_floats, ser_time, send_time, recv_time) =
                rpc::send_batch(&mut stream, &batch, &input_type);

            ser_time_hist.insert(ser_time as i64);
            send_time_hist.insert(send_time as i64);
            recv_time_hist.insert(recv_time as i64);

            let end_time = time::PreciseTime::now();
            let latency = start_time.to(end_time).num_microseconds().unwrap();
            for _ in 0..batch.len() {
                latency_hist.insert(latency);
            }

            let measurement_time = time::precise_time_ns();
            let rpc_time = measurement_time - t2;
            rpc_time_hist.insert(rpc_time as i64);
            for b in batch.iter() {
                let var_load_batch_metric = VariableLoadBatchResponseMetrics {
                    measurement_time_nanos: measurement_time,
                    total_latency_nanos: b.recv_time.to(end_time).num_nanoseconds().unwrap() as u64,
                };
                var_load_batch_latency_logger.push(var_load_batch_metric);
            }
            if var_load_batch_latency_logger.len() >= 10000 {
                let lat_string = serde_json::ser::to_string(&var_load_batch_latency_logger)
                    .unwrap();
                debug!("GREPTHISVARLOAD{}: XXXXXX {}", name, lat_string);
                var_load_batch_latency_logger.clear();
            }


            thruput_meter.mark(batch.len());
            predictions_counter.incr(batch.len() as isize);
            batch_size_hist.insert(batch.len() as i64);
            if latency > slo_micros as i64 {
                trace!("latency: {}, batch size: {}",
                       (latency as f64 / 1000.0),
                       batch.len());
            }
            // only try to increase the batch size if we actually sent a batch of maximum size
            cur_batch_size = batcher.update_batch_size(LatencyMeasurement {
                                                           latency: latency as u64,
                                                           batch_size: batch.len(),
                                                       },
                                                       slo_micros as u64);
            for r in 0..batch.len() {
                cache.put(name.clone(),
                          &batch[r].input,
                          response_floats[r],
                          batch[r].salt.clone());
            }
        }
        if !rpc::shutdown(&mut stream) {
            warn!("Connection to model: {} did not shut down cleanly", name);
        }

        // Log the remaining latencies measurements
        if var_load_batch_latency_logger.len() > 0 {
            let lat_string = serde_json::ser::to_string(&var_load_batch_latency_logger).unwrap();
            debug!("GREPTHISVARLOAD{}: XXXXXX {}", name, lat_string);
            var_load_batch_latency_logger.clear();
        }
        // stream.shutdown(Shutdown::Both).unwrap();
    }


    pub fn request_prediction(&self, req: RpcPredictRequest) {
        // TODO: this could be optimized with
        // https://doc.rust-lang.org/rand/rand/distributions/range/struct.Range.html
        let mut rng = thread_rng();
        let replica: usize = rng.gen_range(0, self.input_queues.len());
        self.input_queues[replica].send(req).unwrap();
    }

    pub fn incorporate_new_replica(&mut self, new_replica: mpsc::Sender<RpcPredictRequest>) {
        self.input_queues.push(new_replica);
    }

    pub fn add_new_replica(&mut self,
                           addr: SocketAddr,
                           metric_register: Arc<RwLock<metrics::Registry>>)
                           -> mpsc::Sender<RpcPredictRequest> {

        let (sender, receiver) = mpsc::channel::<RpcPredictRequest>();
        self.input_queues.push(sender.clone());
        let name = self.name.clone();
        // let addr = a.clone();
        let latency_hist = self.latency_hist.clone();
        let batch_size_hist = self.batch_size_hist.clone();
        let thruput_meter = self.thruput_meter.clone();
        let predictions_counter = self.predictions_counter.clone();
        let input_type = self.input_type.clone();
        let cache = self.cache.clone();
        let batch_strategy = self.batch_strategy.clone();
        let slo_micros = self.slo_micros;
        let jh = thread::spawn(move || {
            PredictionBatcher::run(name,
                                   receiver,
                                   addr,
                                   latency_hist,
                                   batch_size_hist,
                                   thruput_meter,
                                   predictions_counter,
                                   input_type,
                                   cache,
                                   slo_micros,
                                   batch_strategy,
                                   metric_register);
        });
        let mut lock = self.join_handles.as_ref().unwrap().lock().unwrap();
        lock.push(Some(jh));
        sender

    }
}


#[derive(Debug, Serialize, PartialEq, Eq, Clone)]
pub enum BatchStrategy {
    Static { size: usize },
    AIMD,
    Learned {
        sample_size: usize,
        opt_addr: SocketAddr,
    },
}

trait Batcher {
    fn update_batch_size(&mut self, measurement: LatencyMeasurement, max_latency: u64) -> usize;
}

struct AIMDBatcher { }

struct StaticBatcher {
    batch_size: usize,
}

impl Batcher for StaticBatcher {
    fn update_batch_size(&mut self, _: LatencyMeasurement, _: u64) -> usize {
        self.batch_size
    }
}

impl Batcher for AIMDBatcher {
    fn update_batch_size(&mut self, measurement: LatencyMeasurement, max_latency: u64) -> usize {
        update_batch_size_aimd(&measurement, max_latency)
    }
}


fn update_batch_size_aimd(measurement: &LatencyMeasurement, max_time_micros: u64) -> usize {
    let cur_batch = measurement.batch_size;
    let cur_time_micros = measurement.latency;
    let batch_increment = 2;
    let backoff = 0.9;
    // TODO TODO TODO: change back to 0.9
    // let backoff = 0.05;
    let epsilon = (0.1 * max_time_micros as f64).ceil() as u64;
    if cur_time_micros < (max_time_micros - epsilon) {
        let new_batch = cur_batch + batch_increment;
        debug!("increasing batch to {}", new_batch);
        new_batch as usize
    } else if cur_time_micros < max_time_micros {
        cur_batch
    } else {
        // don't try to set the batch size below 1
        let new_batch = cmp::max((cur_batch as f64 * backoff).floor() as u64, 1);
        debug!("decreasing batch to {}", new_batch);
        new_batch as usize
    }
}

// Making this a struct so data is labeled and because
// we may want to measure
#[derive(Serialize, Deserialize, Clone, Debug)]
struct LatencyMeasurement {
    latency: u64,
    batch_size: usize,
}

struct LearnedBatcher {
    pub measurements: Vec<LatencyMeasurement>,
    pub alpha: f64,
    pub beta: f64,
    pub num_samples: usize,
    pub batch_size: usize,
    // try an intermediate period of exploration
    pub calibrated: bool,
    pub name: String,
    batcher_gauges: BatcherGauges,
    opt_addr: SocketAddr,
}

#[derive(Serialize, Deserialize)]
struct SerializeBatcher {
    measurements: Vec<LatencyMeasurement>,
    alpha: f64,
    beta: f64,
    batch_size: usize,
    name: String,
}

impl SerializeBatcher {
    pub fn from_learned_batcher(lb: &LearnedBatcher) -> SerializeBatcher {
        SerializeBatcher {
            measurements: lb.measurements.clone(),
            alpha: lb.alpha,
            beta: lb.beta,
            batch_size: lb.batch_size,
            name: lb.name.clone(),
        }
    }
}


struct BatcherGauges {
    alpha_gauge: Arc<metrics::Counter>,
    beta_gauge: Arc<metrics::Counter>,
    max_batch_size_gauge: Arc<metrics::Counter>,
    max_thru_gauge: Arc<metrics::Counter>,
    base_thru_gauge: Arc<metrics::Counter>,
}

impl LearnedBatcher {
    pub fn new(name: String,
               num_samples: usize,
               batcher_gauges: BatcherGauges,
               opt_addr: SocketAddr)
               -> LearnedBatcher {
        LearnedBatcher {
            measurements: Vec::with_capacity(num_samples),
            alpha: 1.0,
            beta: 1.0,
            num_samples: num_samples,
            batch_size: 1,
            calibrated: false,
            name: name,
            batcher_gauges: batcher_gauges,
            opt_addr: opt_addr,
        }
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.alpha = 1.0;
        self.beta = 1.0;
        self.batch_size = 1;
        self.calibrated = false;
    }

    fn throughput(&self, batch_size: usize) -> f64 {
        let s = batch_size as f64;
        s / (self.alpha * s + self.beta)
    }
}

#[derive(Serialize, Deserialize)]
struct OptimizationBatch {
    batch_size: Vec<usize>,
    latencies: Vec<u64>,
}

impl OptimizationBatch {
    fn from_latency_measurements(measurements: &Vec<LatencyMeasurement>) -> OptimizationBatch {
        let mut batch_size = Vec::with_capacity(measurements.len());
        let mut latencies = Vec::with_capacity(measurements.len());
        for m in measurements.iter() {
            batch_size.push(m.batch_size);
            latencies.push(m.latency);
        }
        OptimizationBatch {
            batch_size: batch_size,
            latencies: latencies,
        }
    }
}

#[allow(dead_code)]
fn ols_estimator(measurements: &Vec<LatencyMeasurement>) -> (f64, f64) {

    let n = measurements.len() as u64;
    let mut sum_xy: u64 = 0;
    let mut sum_x: u64 = 0;
    let mut sum_y: u64 = 0;
    let mut sum_x_squared: u64 = 0;
    for m in measurements.iter() {
        sum_xy += m.latency * m.batch_size as u64;
        sum_x += m.batch_size as u64;
        sum_x_squared += (m.batch_size * m.batch_size) as u64;
        sum_y += m.latency;
    }
    // NOTE: y = alpha * x + beta
    let alpha = (n * sum_xy - sum_x * sum_y) as f64 / (n * sum_x_squared - (sum_x * sum_x)) as f64;
    let beta = (sum_y as f64 - alpha * sum_x as f64) / (n as f64);
    (alpha, beta)
}


fn quantile_estimator(opt_server_addr: &SocketAddr,
                      measurements: &Vec<LatencyMeasurement>)
                      -> (f64, f64) {

    let optimization_batch = OptimizationBatch::from_latency_measurements(measurements);
    let json_string = serde_json::ser::to_string(&optimization_batch).unwrap();

    let mut stream: TcpStream;
    loop {
        match TcpStream::connect(opt_server_addr.clone()) {
            Ok(s) => {
                info!("Connected to opt server at {:?}", opt_server_addr);
                stream = s;
                break;
            }
            Err(_) => {
                info!("Couldn't connect to opt server at {:?}. Sleeping 1 second",
                      opt_server_addr);
                thread::sleep(StdDuration::from_millis(500));
            }
        }
    }
    stream.set_nodelay(true).unwrap();
    stream.set_read_timeout(None).unwrap();
    let mut message = Vec::new();
    let mut bytes = json_string.into_bytes();
    message.write_u32::<LittleEndian>(bytes.len() as u32).unwrap();
    message.append(&mut bytes);
    stream.write_all(&message[..]).unwrap();
    stream.flush().unwrap();

    let floatsize = mem::size_of::<f64>();
    let num_response_bytes = floatsize * 2; // alpha and beta
    let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
    stream.read_exact(&mut response_buffer).unwrap();
    let mut cursor = Cursor::new(response_buffer);
    // let mut responses: Vec<f64> = Vec::with_capacity(inputs.len());
    // alpha
    let alpha = cursor.read_f64::<LittleEndian>().unwrap();
    // beta
    let beta = cursor.read_f64::<LittleEndian>().unwrap();
    (alpha, beta)
}

impl Batcher for LearnedBatcher {
    fn update_batch_size(&mut self, measurement: LatencyMeasurement, max_latency: u64) -> usize {
        if !self.calibrated {
            self.measurements.push(measurement);
            if self.measurements.len() >= self.num_samples {
                // remove first measurement
                self.measurements.remove(0);
                // let (alpha, beta) = ols_estimator(&self.measurements);
                let (alpha, beta) = quantile_estimator(&self.opt_addr, &self.measurements);
                self.alpha = alpha;
                self.beta = beta;
                // lat = (alpha * batch_size) + beta;
                let max_batch_size = ((max_latency as f64 - beta) / alpha).round() as usize;
                info!("{} BATCHER: Updated batching model: max_batch_size: {}, alpha = {}, beta \
                       = {}",
                      self.name,
                      max_batch_size,
                      alpha,
                      beta);
                let baseline_thruput = self.throughput(1);
                let max_thruput = self.throughput(max_batch_size);
                info!("{} BATCHER: Predicted max thruput: {}, baseline: {}, increase: {}",
                      self.name,
                      max_thruput * 10.0_f64.powi(6),
                      baseline_thruput * 10.0_f64.powi(6),
                      max_thruput / baseline_thruput);
                // TODO TODO TODO: maximize throughput with minimal batch size, rather than
                // taking the max batch size here
                self.batch_size = max_batch_size;
                self.calibrated = true;
                self.batcher_gauges.alpha_gauge.incr((alpha * 1000.0).round() as isize);
                self.batcher_gauges.beta_gauge.incr((beta * 1000.0).round() as isize);
                self.batcher_gauges.max_batch_size_gauge.incr(max_batch_size as isize);
                self.batcher_gauges
                    .max_thru_gauge
                    .incr((max_thruput * 10.0_f64.powi(9)).round() as isize);
                self.batcher_gauges
                    .base_thru_gauge
                    .incr((baseline_thruput * 10.0_f64.powi(9)).round() as isize);
                let sb = SerializeBatcher::from_learned_batcher(&self);
                let measurements_string = serde_json::ser::to_string(&sb).unwrap();
                debug!("GREPTHIS{}: XXXXXX {}", self.name, measurements_string);


                // self.measurements.clear();

            } else {
                // do an intermediate period of exploration to get a range of measurements
                // without wildly exceeding latency SLOs
                self.batch_size = update_batch_size_aimd(self.measurements.last().unwrap(),
                                                         max_latency);

                // let mut rng = thread_rng();
                // self.batch_size = rng.gen_range::<usize>(500);
            }
        }
        self.batch_size
    }
}



pub fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}
