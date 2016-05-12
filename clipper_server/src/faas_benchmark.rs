use features;
use metrics;
use std::sync::{Arc, RwLock};
use std::thread;
use digits;
use time;
use rand::{thread_rng, Rng};



pub fn feature_batch_latency(batch_size: usize) {

    let metrics_register = Arc::new(RwLock::new(metrics::Registry::new("faas".to_string())));


    let mnist_path = "/crankshaw-local/mnist/data/test.data".to_string();
    let all_test_data = digits::load_mnist_dense(&mnist_path).unwrap();
    // let addr_vec = vec![vec!["169.229.49.167:6001".to_string(),
    let addr_vec = vec![vec!["127.0.0.1:6001".to_string(),
                             // "127.0.0.1:7001".to_string(),
                             // "127.0.0.1:8001".to_string(),
                                 ]];
    let names = vec!["tensorflow_conv".to_string()];

    let replicas_counter = {
        metrics_register.write().unwrap().create_counter("feature replicas counter".to_string())
    };

    replicas_counter.incr(addr_vec.first().unwrap().len() as isize);




    let (mut features, mut handles): (Vec<_>, Vec<_>) = addr_vec.into_iter()
                    .map(|a| features::get_addrs_str(a))
                    .zip(names.into_iter())
                    .map(|(a, n)| features::create_feature_worker(
                            n, a, batch_size, metrics_register.clone()))
                    .unzip();

    assert!(features.len() == 1);
    let feat = features.pop().unwrap();
    let hand = handles.pop().unwrap();
    
    let mut rng = thread_rng();
    // let num_trials = 10000;
    let num_reqs = 5000000;
    info!("making {} requests", num_reqs);
    for i in 0..num_reqs {
        let example_idx: usize = rng.gen_range(0, all_test_data.xs.len());
        let input = (*all_test_data.xs[example_idx]).clone();
        let req = features::FeatureReq {
            hash_key: i as u64,
            input: input,
            req_start_time: time::PreciseTime::now(),
        };
        feat.request_feature(req);
    }

    let report_interval_secs = 15;
    launch_monitor_thread(metrics_register.clone(), report_interval_secs);
    thread::sleep(::std::time::Duration::new(40, 0));

    // let l = feat.latencies.read().unwrap();
    // let mut avg_t: f64 = 0.0;
    // let mut avg_l: f64 = 0.0;
    // for i in l.iter() {
    //     avg_t += (batch_size as f64) / (*i as f64) * 1000.0 * 1000.0;
    //     avg_l += *i as f64 / 1000.0;
    // }
    // info!("replicas: {}, batch size: {}, trials: {}, average thruput (pred/s): {}, average lat (ms): {}",
    //         hand.len(),
    //         batch_size,
    //          l.len(),
    //          avg_t / (l.len() as f64),
    //          avg_l / (l.len() as f64));

    for h in hand.into_iter() {
        h.join().unwrap();
    }
    // hand.pop().unwrap().join().unwrap();
}


fn launch_monitor_thread(metrics_register: Arc<RwLock<metrics::Registry>>,
                         report_interval_secs: u64) -> ::std::thread::JoinHandle<()> {

    // let counter = counter.clone();
    thread::spawn(move || {
        loop {
            thread::sleep(::std::time::Duration::new(report_interval_secs, 0));
            let m = metrics_register.read().unwrap();
            info!("{}", m.report());
            m.reset();
            // metrics_register.write().unwrap().reset();
        }
    })
}


