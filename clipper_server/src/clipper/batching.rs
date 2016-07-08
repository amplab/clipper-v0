use time;
use std::net::{SocketAddr, TcpStream};
// use std::net::Shutdown
use std::thread::{self, JoinHandle};
use std::sync::{RwLock, Arc, mpsc, Mutex};
use std::cmp;
use rand::{thread_rng, Rng};
use server::{self, InputType, Output};
use metrics;
use rpc;
use cache::PredictionCache;


#[derive(Clone)]
pub struct RpcPredictRequest {
    pub input: server::Input,
    pub recv_time: time::PreciseTime,
}

pub struct PredictionBatcher<C>
    where C: PredictionCache<Output>
{
    name: String,
    input_queues: Vec<mpsc::Sender<RpcPredictRequest>>,
    cache: Arc<C>,
    join_handles: Option<Arc<Mutex<Vec<Option<JoinHandle<()>>>>>>,
}

impl<C> Drop for PredictionBatcher<C> where C: PredictionCache<Output>
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

impl<C> Clone for PredictionBatcher<C> where C: PredictionCache<Output>
{
    fn clone(&self) -> PredictionBatcher<C> {
        PredictionBatcher {
            name: self.name.clone(),
            input_queues: self.input_queues.clone(),
            cache: self.cache.clone(),
            join_handles: self.join_handles.clone(),
        }
    }
}




impl<C> PredictionBatcher<C> where C: PredictionCache<Output> + 'static + Send + Sync
{
    pub fn new(name: String,
               addrs: Vec<SocketAddr>,
               input_type: InputType,
               metric_register: Arc<RwLock<metrics::Registry>>,
               cache: Arc<C>,
               slo_micros: u32)
               -> PredictionBatcher<C> {

        let latency_hist: Arc<metrics::Histogram> = {
            let metric_name = format!("{}_model_latency", name);
            metric_register.write().unwrap().create_histogram(metric_name, 2056)
        };

        let thruput_meter: Arc<metrics::Meter> = {
            let metric_name = format!("{}_model_thruput", name);
            metric_register.write().unwrap().create_meter(metric_name)
        };

        let predictions_counter: Arc<metrics::Counter> = {
            let metric_name = format!("{}_prediction_counter", name);
            metric_register.write().unwrap().create_counter(metric_name)
        };

        let mut input_queues = Vec::with_capacity(addrs.len());
        let mut join_handles = Vec::with_capacity(addrs.len());
        for a in addrs.iter() {
            let (sender, receiver) = mpsc::channel::<RpcPredictRequest>();
            input_queues.push(sender);
            let name = name.clone();
            let addr = a.clone();
            let latency_hist = latency_hist.clone();
            let thruput_meter = thruput_meter.clone();
            let predictions_counter = predictions_counter.clone();
            let input_type = input_type.clone();
            let cache = cache.clone();
            let jh = thread::spawn(move || {
                PredictionBatcher::run(name,
                                       receiver,
                                       addr,
                                       latency_hist,
                                       thruput_meter,
                                       predictions_counter,
                                       input_type,
                                       cache,
                                       slo_micros);
            });
            join_handles.push(Some(jh));
        }
        PredictionBatcher {
            name: name,
            input_queues: input_queues,
            cache: cache,
            join_handles: Some(Arc::new(Mutex::new(join_handles))),
        }
    }

    fn run(name: String,
           receiver: mpsc::Receiver<RpcPredictRequest>,
           addr: SocketAddr,
           latency_hist: Arc<metrics::Histogram>,
           thruput_meter: Arc<metrics::Meter>,
           predictions_counter: Arc<metrics::Counter>,
           input_type: InputType,
           cache: Arc<C>,
           slo_micros: u32) {

        let dynamic_batching = true;

        let mut stream: TcpStream = TcpStream::connect(addr).unwrap();
        stream.set_nodelay(true).unwrap();
        stream.set_read_timeout(None).unwrap();
        // let max_batch_size = batch_size;
        let mut cur_batch_size = 1;

        // block until new request, then try to get more requests
        while let Ok(first_req) = receiver.recv() {
            let mut batch: Vec<RpcPredictRequest> = Vec::new();
            batch.push(first_req);
            let start_time = time::PreciseTime::now();
            let max_batch_size = if dynamic_batching {
                cur_batch_size
            } else {
                5
            };
            assert!(max_batch_size >= 1);
            // while batch.len() < cur_batch_size {
            while batch.len() < max_batch_size {
                if let Ok(req) = receiver.try_recv() {
                    // let req_latency = req.req_start_time.to(time::PreciseTime::now()).num_microseconds().unwrap();
                    // println!("req->features latency {} (ms)", (req_latency as f64 / 1000.0));
                    batch.push(req);
                } else {
                    break;
                }
            }
            assert!(batch.len() > 0);

            let response_floats: Vec<f64> = rpc::send_batch(&mut stream, &batch, &input_type);
            let end_time = time::PreciseTime::now();
            let latency = start_time.to(end_time).num_microseconds().unwrap();
            for _ in 0..batch.len() {
                latency_hist.insert(latency);
            }
            thruput_meter.mark(batch.len());
            predictions_counter.incr(batch.len() as isize);
            if latency > slo_micros as i64 {
                debug!("latency: {}, batch size: {}",
                       (latency as f64 / 1000.0),
                       batch.len());
            }
            if dynamic_batching {
                // only try to increase the batch size if we actually sent a batch of maximum size
                if batch.len() == cur_batch_size {
                    cur_batch_size = update_batch_size_aimd(cur_batch_size,
                                                            latency as u64,
                                                            slo_micros as u64);
                    // debug!("{} updated batch size to {}", name, cur_batch_size);
                }
            }
            for r in 0..batch.len() {
                cache.put(name.clone(), &batch[r].input, response_floats[r]);
            }
        }
        if !rpc::shutdown(&mut stream) {
            warn!("Connection to model: {} did not shut down cleanly", name);
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
}






fn update_batch_size_aimd(cur_batch: usize, cur_time_micros: u64, max_time_micros: u64) -> usize {
    let batch_increment = 2;
    let backoff = 0.9;
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

pub fn random_features(d: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    rng.gen_iter::<f64>().take(d).collect::<Vec<f64>>()
}


// #[cfg(test)]
// #[cfg_attr(rustfmt, rustfmt_skip)]
// mod tests {
//     use super::*;
// // use std::sync::mpsc;
// // use thread;
//     use cache::PredictionCache;
//     use std::clone::Clone;
//
//     #[derive(Clone)]
//     struct TestPredictionCache {}
//
//     impl<i32> PredictionCache<i32> for TestPredictionCache {}
//
//
//
// }

// pub struct FeatureReq {
//     pub hash_key: HashKey,
//     pub input: server::Input,
//     pub req_start_time: time::PreciseTime,
// }
//
// #[derive(Clone)]
// pub struct FeatureHandle<H: FeatureHash + Send + Sync> {
//     pub name: String,
//     // TODO: need a better concurrent hashmap: preferable lock free wait free
//     // This should actually be reasonably simple, because we don't need to resize
//     // (fixed size cache) and things never get evicted. Neither of these is strictly
//     // true but it's a good approximation for now.
//     pub cache: Arc<RwLock<HashMap<HashKey, f64>>>,
//     pub hasher: Arc<H>,
//     queues: Vec<mpsc::Sender<FeatureReq>>,
//     next_instance: Arc<AtomicUsize>,
// }
//
//
// impl<H: FeatureHash + Send + Sync> FeatureHandle<H> {
//     pub fn request_feature(&self, req: FeatureReq) {
//         // let inst = self.next_instance.fetch_add(1, Ordering::Relaxed) % self.queues.len();
//         let inst = req.hash_key as usize % self.queues.len();
//         self.queues[inst].send(req).unwrap();
//     }
// }
//
// pub fn create_feature_worker(name: String,
//                              addrs: Vec<SocketAddr>,
//                              batch_size: usize,
//                              metric_register: Arc<RwLock<metrics::Registry>>,
//                              input_type: InputType)
//                              -> (FeatureHandle<SimpleHasher>,
//                                  Vec<::std::thread::JoinHandle<()>>) {
//
//     let latency_hist: Arc<metrics::Histogram> = {
//         let metric_name = format!("{}_faas_latency", name);
//         metric_register.write().unwrap().create_histogram(metric_name, 2056)
//     };
//
//     let thruput_meter: Arc<metrics::Meter> = {
//         let metric_name = format!("{}_faas_thruput", name);
//         metric_register.write().unwrap().create_meter(metric_name)
//     };
//
//     let predictions_counter: Arc<metrics::Counter> = {
//         let metric_name = format!("{}_prediction_counter", name);
//         metric_register.write().unwrap().create_counter(metric_name)
//     };
//
//     let feature_cache: Arc<RwLock<HashMap<HashKey, f64>>> = Arc::new(RwLock::new(HashMap::new()));
//     // let latencies: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
//     let mut handles = Vec::new();
//     let mut queues = Vec::new();
//     for a in addrs.iter() {
//         let (tx, rx) = mpsc::channel();
//         queues.push(tx);
//         let handle = {
//             // cache is shared
//             let thread_cache = feature_cache.clone();
//             // latency tracking is shared
//             // let latencies = latencies.clone();
//             let name = name.clone();
//             let addr = a.clone();
//             let latency_hist = latency_hist.clone();
//             let thruput_meter = thruput_meter.clone();
//             let predictions_counter = predictions_counter.clone();
//             let input_type = input_type.clone();
//             thread::spawn(move || {
//                 feature_worker(name,
//                                rx,
//                                thread_cache,
//                                addr,
//                                latency_hist,
//                                thruput_meter,
//                                predictions_counter,
//                                batch_size,
//                                input_type);
//             })
//         };
//         handles.push(handle);
//     }
//     info!("Creating feature worker with {} replicas", queues.len());
//     (FeatureHandle {
//         name: name.clone(),
//         queues: queues,
//         cache: feature_cache,
//         hasher: Arc::new(SimpleHasher),
//         next_instance: Arc::new(AtomicUsize::new(0)), // thread_handle: handle,
//     },
//      handles)
// }

// fn feature_worker(name: String,
//                   rx: mpsc::Receiver<FeatureReq>,
//                   cache: Arc<RwLock<HashMap<HashKey, f64>>>,
//                   address: SocketAddr,
//                   latency_hist: Arc<metrics::Histogram>,
//                   thruput_meter: Arc<metrics::Meter>,
//                   predictions_counter: Arc<metrics::Counter>,
//                   batch_size: usize,
//                   input_type: InputType) {
//
//     // let input_type = InputType::Float(7);
//     // if the batch_size is less than 1 (these are unsigned
//     // integers, so that means batch size == 0), we assume dynamic batching
//     let dynamic_batching = batch_size < 1;
//     if dynamic_batching {
//         info!("using dynamic batch size for {}", name);
//     }
//     // println!("starting worker: {}", name);
//     let mut stream: TcpStream = TcpStream::connect(address).unwrap();
//     stream.set_nodelay(true).unwrap();
//     stream.set_read_timeout(None).unwrap();
//     // let max_batch_size = batch_size;
//     let mut cur_batch_size = 1;
//     // let mut bench_latencies = Vec::new();
//     // let mut loop_counter = 0;
//     // let mut epoch_start = time::PreciseTime::now();
//     // let mut epoch_count = 0;
//
//     loop {
//         let mut batch: Vec<FeatureReq> = Vec::new();
//         // block until new request, then try to get more requests
//         let first_req = rx.recv().unwrap();
//         batch.push(first_req);
//         let start_time = time::PreciseTime::now();
//         let max_batch_size = if dynamic_batching {
//             cur_batch_size
//         } else {
//             batch_size
//         };
//         assert!(max_batch_size >= 1);
//         // while batch.len() < cur_batch_size {
//         while batch.len() < max_batch_size {
//             if let Ok(req) = rx.try_recv() {
//                 // let req_latency = req.req_start_time.to(time::PreciseTime::now()).num_microseconds().unwrap();
//                 // println!("req->features latency {} (ms)", (req_latency as f64 / 1000.0));
//                 batch.push(req);
//             } else {
//                 break;
//             }
//         }
//         assert!(batch.len() > 0);
//         // send batch
//         // let mut header_wtr: Vec<u8> = vec![];
//         // header_wtr.write_u16::<LittleEndian>(batch.len() as u16).unwrap();
//         // stream.write_all(&header_wtr).unwrap();
//         // for r in batch.iter() {
//         //     match r.input {
//         //         server::Input::Floats { ref f, length: _ } => {
//         //             stream.write_all(floats_to_bytes(f.clone())).unwrap()
//         //         }
//         //         _ => panic!("unimplemented input type"),
//         //     }
//         // }
//         // stream.flush().unwrap();
//         // read response: assumes 1 f64 for each entry in batch
//         // let num_response_bytes = batch.len() * mem::size_of::<f64>();
//         // let mut response_buffer: Vec<u8> = vec![0; num_response_bytes];
//         // stream.read_exact(&mut response_buffer).unwrap();
//         // // make immutable
//         // let response_buffer = response_buffer;
//         // let response_floats = bytes_to_floats(&response_buffer);
//         let response_floats: Vec<f64> = rpc::send_batch(&mut stream, &batch, &input_type);
//         let end_time = time::PreciseTime::now();
//         let latency = start_time.to(end_time).num_microseconds().unwrap();
//         for _ in 0..batch.len() {
//             latency_hist.insert(latency);
//         }
//         thruput_meter.mark(batch.len());
//         predictions_counter.incr(batch.len() as isize);
//         if latency > server::SLA * 1000 {
//             debug!("latency: {}, batch size: {}",
//                    (latency as f64 / 1000.0),
//                    batch.len());
//         }
//         if dynamic_batching {
//             // only try to increase the batch size if we actually sent a batch of maximum size
//             if batch.len() == cur_batch_size {
//                 cur_batch_size = update_batch_size(cur_batch_size,
//                                                    latency as u64,
//                                                    server::SLA as u64 * 1000 as u64);
//                 // debug!("{} updated batch size to {}", name, cur_batch_size);
//             }
//         }
//
//         // let mut l = latencies.write().unwrap();
//         // l.push(latency);
//         let mut w = cache.write().unwrap();
//         for r in 0..batch.len() {
//             let hashed_query = batch[r].hash_key;
//             if !w.contains_key(&hashed_query) {
//                 w.insert(hashed_query, response_floats[r]);
//             } else {
//                 // println!("CACHE HIT");
//                 // let existing_res = w.get(&hash).unwrap();
//                 // if result != *existing_res {
//                 //     // println!("{} CACHE ERR: existing: {}, new: {}",
//                 //     //          name, existing_res, result);
//                 // } else {
//                 //     println!("{} CACHE HIT", name);
//                 // }
//             }
//         }
//     }
// }
