

use time;
use std::ptr;
use std::thread;
use std::sync::{RwLock, Arc};
use std::sync::mpsc;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::net::{ToSocketAddrs, SocketAddr};
use server;
use digits;
use features;
use features::FeatureHash;
use linear_models::{linalg, linear};
use server::TaskModel;
// use quickersort;

pub fn run(feature_addrs: Vec<(String, SocketAddr)>,
       num_users: usize,
       num_train_examples: usize,
       num_test_examples: usize,
       mnist_path: String) {

    println!("starting digits");
    let all_test_data = digits::load_mnist_dense(&mnist_path).unwrap();
    let norm_test_data = digits::normalize(&all_test_data);

    println!("Test data loaded: {} points", norm_test_data.ys.len());

    let tasks = digits::create_online_dataset(&norm_test_data,
                                              &norm_test_data,
                                              num_train_examples,
                                              0,
                                              num_test_examples,
                                              num_users as usize);


    let (features, handles): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                              .map(|(n, a)| features::create_feature_worker(n, a))
                              .unzip();
    let trained_tasks = pretrain_task_models(tasks, &features);
    let num_events = num_users * num_test_examples;

    let num_workers = 2;
    let correct_counter = Arc::new(AtomicUsize::new(0));
    let total_counter = Arc::new(AtomicUsize::new(0));
    let processed_counter = Arc::new(AtomicUsize::new(0));
    let mut dispatcher = server::Dispatcher::new(num_workers,
                                         server::SLA,
                                         features.clone(),
                                         trained_tasks.clone(),
                                         correct_counter.clone(),
                                         total_counter.clone(),
                                         processed_counter.clone());

    thread::sleep(::std::time::Duration::new(3, 0));
    let mon_thread_join_handle = launch_monitor_thread(correct_counter.clone(),
                                                       total_counter.clone(),
                                                       processed_counter.clone(),
                                                       num_events as i32);

    let mut rng = thread_rng();
    for _ in 0..num_events {
        // TODO dispatch events
        let user: usize = rng.gen_range(0, num_users as usize);
        let example_idx: usize = rng.gen_range(0, num_test_examples);
        let input = (*(&trained_tasks)[user].read().unwrap().test_x[example_idx]).clone();
        let true_label = (&trained_tasks)[user].read().unwrap().test_y[example_idx];
        // random delay
        
        let sleep_time_ms: u32 = rng.gen_range(0, 100);
        let ms_to_ns_factor = 1000*1000;
        // let sleep_time = ::std::time::Duration::new(0,sleep_time_ms * ms_to_ns_factor);
        let sleep_time = ::std::time::Duration::new(2, 0);
        // println!("sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
        thread::sleep(sleep_time);
        let max_features = features.len();
        // println!("dispatching request");
        dispatcher.dispatch(server::Request::new_with_label(user as u32, input, true_label),
                            features.len());
    }

    println!("waiting for features to finish");
    mon_thread_join_handle.join().unwrap();

    // TODO: do something with latencies
    for fh in &features {
        let cur_lats: &Vec<i64> = &fh.latencies.read().unwrap();
        // println!("{}, {:?}", fh.name, cur_lats);
    }

    for h in handles {
        h.join().unwrap();
    }
    // handle.join().unwrap();
    println!("done");
}

fn launch_monitor_thread(correct_counter: Arc<AtomicUsize>,
                         total_counter: Arc<AtomicUsize>,
                         processed_counter: Arc<AtomicUsize>,
                         num_events: i32) -> ::std::thread::JoinHandle<()> {

    // let counter = counter.clone();
    thread::spawn(move || {
        let bench_start = time::PreciseTime::now();
        let mut last_count = 0;
        let mut last_time = bench_start;
        loop {
            thread::sleep(::std::time::Duration::new(3, 0));
            let cur_time = last_time.to(time::PreciseTime::now()).num_milliseconds() as f64 / 1000.0;
            let cur_count = processed_counter.load(Ordering::Relaxed);
            println!("processed {} events in {} seconds: {} preds/sec",
                     cur_count - last_count,
                     cur_time,
                     ((cur_count - last_count) as f64 / cur_time));

            let cur_correct = correct_counter.load(Ordering::Relaxed);
            let cur_total = total_counter.load(Ordering::Relaxed);
            println!("current accuracy: {} out of {} correct, ({}%)",
                     cur_correct, cur_total, (cur_correct as f64/cur_total as f64)*100.0);
            if cur_count >= num_events as usize {
                println!("BENCHMARK FINISHED");
                break;
            }
            println!("sleeping...");
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
    pub anytime_estimators: Arc<RwLock<Vec<f64>>>
}

impl TrainedTask {

    /// Note that `xs` contains the featurized training data,
    /// while `test_x` contains the raw test inputs.
    pub fn train(tid: usize,
                 pref: f64,
                 xs: &Vec<Arc<Vec<f64>>>,
                 ys: &Vec<f64>,
                 test_x: Vec<Arc<Vec<f64>>>,
                 test_y: Vec<f64>) -> TrainedTask {
        let params = linear::Struct_parameter {
            solver_type: linear::L2R_LR,
            eps: 0.0001,
            C: 1.0f64,
            nr_weight: 0,
            weight_label: ptr::null_mut(),
            weight: ptr::null_mut(),
            p: 0.1,
            init_sol: ptr::null_mut()
        };
        let prob = linear::Problem::from_training_data(xs, ys);
        let model = linear::train_logistic_regression(prob, params);
        let (anytime_estimators, _) = linalg::mean_and_var(xs);

        TrainedTask {
            task_id: tid,
            pref: pref,
            test_x: test_x,
            test_y: test_y,
            model: model,
            anytime_estimators: Arc::new(RwLock::new(anytime_estimators))
        }
    }

    // pub fn predict(&self, x: &Vec<f64>) -> f64 {
    //     self.model.logistic_regression_predict(x)
    // }
}


impl TaskModel for TrainedTask {
    fn predict(&self, fs: Vec<f64>, missing_fs: Vec<usize>) -> f64 {
        if missing_fs.len() > 0 {
            println!("missing {} features", missing_fs.len());
        }
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
                          feature_handles: &Vec<features::FeatureHandle<features::SimpleHasher>>) {
    let num_features = feature_handles.len();
    for t in tasks.iter() {

        // let ttt: &digits::DigitsTask = t;

        for x in t.offline_train_x.iter() {
            let mut xv: Vec<f64> = Vec::new();
            for i in x.iter() {
                xv.push(*i);
            }
            server::get_features(feature_handles,
                                 xv,
                                 (0..num_features).collect(),
                                 time::PreciseTime::now());
        }
    }
    println!("requesting all training features");
    loop {
        let sleep_secs = 10;
        println!("Sleeping {} seconds...", sleep_secs);
        thread::sleep(::std::time::Duration::new(sleep_secs, 0));
        let mut done = true;
        for t in tasks.iter() {
            for x in t.offline_train_x.iter() {
                for fh in feature_handles {
                    let hash = fh.hasher.hash(&x);
                    let mut cache_reader = fh.cache.read().unwrap();
                    let cache_entry = cache_reader.get(&hash);
                    if cache_entry.is_none() {
                        done = false;
                        break;
                    }
                }
                if done == false { break; }
            }
            if done == false { break; }
        }
        // if we reach the end and done is still true, we have all features
        if done == true { break; }
    }
    println!("all features have been cached");
    thread::sleep(::std::time::Duration::new(10, 0));
}

fn pretrain_task_models(tasks: Vec<digits::DigitsTask>,
                        feature_handles: &Vec<features::FeatureHandle<features::SimpleHasher>>) 
    -> Arc<Vec<RwLock<TrainedTask>>> {

    println!("pretraining task models");
    get_all_train_features(&tasks, feature_handles);
    let mut trained_tasks = Vec::new();
    let mut cur_tid = 0;
    for t in tasks.iter() {
        let mut training_data: Vec<Arc<Vec<f64>>> = Vec::new();
        for x in t.offline_train_x.iter() {
            let mut x_features: Vec<f64> = Vec::new();
            for fh in feature_handles {
                let hash = fh.hasher.hash(&x);
                let mut cache_reader = fh.cache.read().unwrap();
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


//
//     
//     let mut trained_tasks = Vec::new();
//
//     let tasks_with_models = tasks.into_iter().enumerate().map(|t| {
//         let mut x_features: Vec<Arc<Vec<f64>>> = Vec::new();
//         for x in t.offline_train_x {
//             server::get_features(feature_handles, (&x).clone());
//         }
//         // (t, TaskModel::train(i, t.pref, t.offline_train_x, t.offline_train_y))
//     }).collect::<Vec<_>>();
//
// }

