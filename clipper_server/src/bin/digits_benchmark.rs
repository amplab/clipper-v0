use time;
use std::ptr;
use std::sync::{RwLock, Arc};
use std::thread;
use std::net::SocketAddr;
use linear_models::{linalg, linear};
use clipper::{digits, features, metrics};
use clipper::hashing::{FeatureHash, SimpleHasher};
use clipper::server::{self, TaskModel};
// use quickersort;

#[derive(Debug)]
pub struct DigitsBenchConfig {
    pub num_users: usize,
    pub num_train_examples: usize,
    pub num_test_examples: usize,
    pub mnist_path: String,
    pub num_events: usize,
    pub num_workers: usize,
    pub target_qps: usize,
    pub query_batch_size: usize,
    pub max_features: usize,
    pub salt_hash: bool, // do we want cache hits or not
    pub feature_batch_size: usize,
}


pub fn run(feature_addrs: Vec<(String, Vec<SocketAddr>)>, dc: DigitsBenchConfig) {

    info!("starting digits");
    info!("Config: {:?}", dc);

    let metrics_register = Arc::new(RwLock::new(metrics::Registry::new("digits_bench"
                                                                           .to_string())));

    let all_test_data = digits::load_mnist_dense(&dc.mnist_path).unwrap();
    let norm_test_data = digits::normalize(&all_test_data);
    // let norm_test_data = all_test_data;

    info!("Test data loaded: {} points", norm_test_data.ys.len());

    let tasks = digits::create_online_dataset(&norm_test_data,
                                              &norm_test_data,
                                              dc.num_train_examples,
                                              0,
                                              dc.num_test_examples,
                                              dc.num_users as usize);


    let (features, _): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                                                       .map(|(n, a)| {
                                                           features::create_feature_worker(n,
                                                         a,
                                                         dc.feature_batch_size,
                                                         metrics_register.clone())
                                                       })
                                                       .unzip();
    let trained_tasks = pretrain_task_models(tasks, &features);
    // let num_events = num_users * num_test_examples;
    let num_events = dc.num_events;

    // TODO (function call is so this won't compile)
    // clear_cache();

    // reset the metrics after training models
    {
        metrics_register.read().unwrap().reset();
    }
    info!("clearing caches");
    for f in features.iter() {
        let mut w = f.cache.write().unwrap();
        w.clear();
    }

    let num_workers = dc.num_workers;

    let dispatcher = server::Dispatcher::new(num_workers,
                                             server::SLA,
                                             features.clone(),
                                             trained_tasks.clone(),
                                             metrics_register.clone());

    thread::sleep(::std::time::Duration::new(3, 0));
    let report_interval_secs = 15;
    let mon_thread_join_handle = launch_monitor_thread(metrics_register.clone(),
                                                       report_interval_secs);

    info!("Starting test");
    let target_qps = dc.target_qps;
    let query_batch_size = dc.query_batch_size;
    let mut events_fired = 0;
    let batches_per_sec = target_qps / query_batch_size + 1;
    let ms_per_sec = 1000; // just labeling my magic number
    let sleep_time_ms = ms_per_sec / batches_per_sec as u64;
    let mut cur_user: usize = 0;
    let mut cur_index: usize = 0;
    while events_fired < num_events {
        for _ in 0..query_batch_size {
            // let user: usize = rng.gen_range(0, dc.num_users as usize);
            // let example_idx: usize = rng.gen_range(0, dc.num_test_examples);
            let input = (*(&trained_tasks)[cur_user].read().unwrap().test_x[cur_index]).clone();
            let true_label = (&trained_tasks)[cur_user].read().unwrap().test_y[cur_index];
            // let max_features = features.len();
            let max_features = dc.max_features;
            let r = server::PredictRequest::new_with_label(cur_user as u32,
                                                           server::Input::Floats {
                                                               f: input,
                                                               length: 784,
                                                           },
                                                           true_label,
                                                           events_fired as i32);
            // dispatcher.dispatch(r, features.len());
            // NOOP callback
            dispatcher.dispatch(r, max_features);
            cur_index += 1;
            if cur_index == dc.num_test_examples {
                cur_index = 0;
                cur_user = (cur_user + 1) % dc.num_users;
                if cur_user == 0 {
                    debug!("restarting test");
                }
            }
            events_fired += 1;
        }
        thread::sleep(::std::time::Duration::from_millis(sleep_time_ms));
    }

    info!("waiting for features to finish");
    mon_thread_join_handle.join().unwrap();
    println!("done");
}

fn launch_monitor_thread(metrics_register: Arc<RwLock<metrics::Registry>>,
                         report_interval_secs: u64)
                         -> ::std::thread::JoinHandle<()> {

    thread::spawn(move || {
        loop {
            thread::sleep(::std::time::Duration::new(report_interval_secs, 0));
            let m = metrics_register.read().unwrap();
            info!("{}", m.report());
            m.reset();
            // thread::sleep(::std::time::Duration::new(report_interval_secs, 0));
            // info!("{}", metrics_register.read().unwrap().report());
        }
    })
}


/// Benchmark struct containing the trained model for
/// a task as well as some test data to evaluate the task on

struct TrainedTask {
    pub task_id: usize,
    pub pref: f64,
    // raw inputs for training data
    pub test_x: Vec<Arc<Vec<f64>>>,
    pub test_y: Vec<f64>,
    model: linear::LogisticRegressionModel,
    /// anytime estimator for each feature in case we don't have it in time
    pub anytime_estimators: Arc<RwLock<Vec<f64>>>,
}

impl TrainedTask {
    /// Note that `xs` contains the featurized training data,
    /// while `test_x` contains the raw test inputs.
    pub fn train(tid: usize,
                 pref: f64,
                 xs: &Vec<Arc<Vec<f64>>>,
                 ys: &Vec<f64>,
                 test_x: Vec<Arc<Vec<f64>>>,
                 test_y: Vec<f64>)
                 -> TrainedTask {
        let params = linear::Struct_parameter {
            // solver_type: linear::L2R_LR,
            solver_type: linear::L1R_LR,
            eps: 0.0001,
            // C: 1.0f64,
            C: 100.0f64,
            nr_weight: 0,
            weight_label: ptr::null_mut(),
            weight: ptr::null_mut(),
            p: 0.1,
            init_sol: ptr::null_mut(),
        };
        let prob = linear::Problem::from_training_data(xs, ys);
        let model = linear::train_logistic_regression(prob, params);
        let (anytime_estimators, _) = linalg::mean_and_var(xs);
        // let mut nnz = 0;
        // let mut biggest: f64 = 0.0;
        // let mut sec_biggest: f64 = 0.0;
        // for wi in model.w.iter() {
        //   if (*wi).abs() > sec_biggest.abs() {
        //     if (*wi).abs() > biggest.abs() {
        //       biggest = (*wi);
        //     } else {
        //       sec_biggest = (*wi);
        //     }
        //   }
        // }


        // println!("biggest: {}, sec_biggest: {}", biggest, sec_biggest);

        TrainedTask {
            task_id: tid,
            pref: pref,
            test_x: test_x,
            test_y: test_y,
            model: model,
            anytime_estimators: Arc::new(RwLock::new(anytime_estimators)),
        }
    }

    // pub fn predict(&self, x: &Vec<f64>) -> f64 {
    //     self.model.logistic_regression_predict(x)
    // }
}


impl TaskModel for TrainedTask {
    #[allow(unused_variables)]
    fn predict(&self, fs: Vec<f64>, missing_fs: Vec<usize>, debug_str: &String) -> f64 {
        // if missing_fs.len() > 5 {
        //     println!("{}: missing {} features", debug_str, missing_fs.len());
        // }
        let mut fs = fs.clone();
        let estimators = self.anytime_estimators.read().unwrap();
        for i in &missing_fs {
            fs[*i] = estimators[*i];
        }
        self.model.logistic_regression_predict(&fs)
        // self.model.predict(fs)
    }

    fn rank_features(&self) -> Vec<usize> {
        let mut ws = self.model.w.clone();
        // this is a very slow sort, but we expect the
        // number of features to be pretty small so this is
        // probably okay.

        let mut feature_order = Vec::new();
        while !ws.is_empty() {
            // TODO: rank by highest val or absolute value?
            let mut next_highest = ws[0].abs();
            let mut highest_index = 0;
            for (i, v) in ws.iter().enumerate() {
                if v.abs() > next_highest {
                    next_highest = v.abs();
                    highest_index = i;
                }
            }
            feature_order.push(highest_index);
            ws.remove(highest_index);
        }
        feature_order
    }
}


/// Wait until all features for all tasks have been computed asynchronously
fn get_all_train_features(tasks: &Vec<digits::DigitsTask>,
                          feature_handles: &Vec<features::FeatureHandle<SimpleHasher>>) {
    let num_features = feature_handles.len();
    for t in tasks.iter() {

        // let ttt: &digits::DigitsTask = t;

        for x in t.offline_train_x.iter() {
            let mut xv: Vec<f64> = Vec::new();
            for i in x.iter() {
                xv.push(*i);
            }
            let input = server::Input::Floats {
                f: xv,
                length: 784,
            };
            server::get_features(feature_handles,
                                 input,
                                 (0..num_features).collect(),
                                 time::PreciseTime::now(),
                                 None);
        }
    }
    // println!("requesting all training features");
    loop {
        let sleep_secs = 10;
        // println!("Sleeping {} seconds...", sleep_secs);
        thread::sleep(::std::time::Duration::new(sleep_secs, 0));
        let mut done = true;
        for t in tasks.iter() {
            for x in t.offline_train_x.iter() {
                for fh in feature_handles {

                    let input = server::Input::Floats {
                        f: (**x).clone(),
                        length: 784,
                    };
                    let hash = fh.hasher.query_hash(&input, None);
                    let cache_reader = fh.cache.read().unwrap();
                    let cache_entry = cache_reader.get(&hash);
                    if cache_entry.is_none() {
                        done = false;
                        break;
                    }
                }
                if done == false {
                    break;
                }
            }
            if done == false {
                break;
            }
        }
        // if we reach the end and done is still true, we have all features
        if done == true {
            break;
        }
    }
    // println!("all features have been cached");
    thread::sleep(::std::time::Duration::new(10, 0));
}

fn pretrain_task_models(tasks: Vec<digits::DigitsTask>,
                        feature_handles: &Vec<features::FeatureHandle<SimpleHasher>>)
                        -> Arc<Vec<RwLock<TrainedTask>>> {

    // println!("pretraining task models");
    get_all_train_features(&tasks, feature_handles);
    let mut trained_tasks = Vec::new();
    let mut cur_tid = 0;
    for t in tasks.iter() {
        let mut training_data: Vec<Arc<Vec<f64>>> = Vec::new();
        for x in t.offline_train_x.iter() {
            let mut x_features: Vec<f64> = Vec::new();
            let input = server::Input::Floats {
                f: (**x).clone(),
                length: 784,
            };
            for fh in feature_handles {
                let hash = fh.hasher.query_hash(&input, None);
                let cache_reader = fh.cache.read().unwrap();
                x_features.push(*cache_reader.get(&hash).unwrap());
            }
            training_data.push(Arc::new(x_features));
        }
        let new_trained_task = TrainedTask::train(cur_tid,
                                                  t.pref,
                                                  &training_data,
                                                  &t.offline_train_y,
                                                  t.test_x.clone(),
                                                  t.test_y.clone());
        trained_tasks.push(RwLock::new(new_trained_task));
        cur_tid += 1;
    }
    Arc::new(trained_tasks)
}
