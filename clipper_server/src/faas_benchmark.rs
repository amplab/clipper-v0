use features;
use log;
use metrics;




pub fn feature_batch_latency(batch_size: usize) {

    let mut metric_register = Arc::new(RwLock::new(metrics::Registry::new("faas")));



    let mnist_path = "/crankshaw-local/mnist/data/test.data".to_string();
    let all_test_data = digits::load_mnist_dense(&mnist_path).unwrap();
    let addr_vec = vec![vec!["127.0.0.1:6001".to_string(),
                             // "127.0.0.1:7001".to_string(),
                             // "127.0.0.1:8001".to_string(),
                                 ]];
    let names = vec!["pyspark-svm".to_string()];




    let (mut features, mut handles): (Vec<_>, Vec<_>) = addr_vec.into_iter().map(|a| get_addrs_str(a))
                    .zip(names.into_iter())
                    .map(|(a, n)| create_feature_worker(n, a, batch_size, metric_register.clone()))
                    .unzip();

    assert!(features.len() == 1);
    let feat = features.pop().unwrap();
    let hand = handles.pop().unwrap();

    let mut rng = thread_rng();
    let num_trials = 10000;
    let num_reqs = num_trials * batch_size;
    for i in 0..num_reqs {
        let example_idx: usize = rng.gen_range(0, all_test_data.xs.len());
        let input = (*all_test_data.xs[example_idx]).clone();
        let req = FeatureReq {
            hash_key: 11,
            input: input,
            req_start_time: time::PreciseTime::now(),
        };
        feat.request_feature(req);
    }

    thread::sleep(::std::time::Duration::new(20, 0));

    // let l = feat.latencies.read().unwrap();
    // let mut avg_t: f64 = 0.0;
    // let mut avg_l: f64 = 0.0;
    // for i in l.iter() {
    //     avg_t += (batch_size as f64) / (*i as f64) * 1000.0 * 1000.0;
    //     avg_l += *i as f64 / 1000.0;
    // }
    info!("replicas: {}, batch size: {}, trials: {}, average thruput (pred/s): {}, average lat (ms): {}",
            hand.len(),
            batch_size,
             l.len(),
             avg_t / (l.len() as f64),
             avg_l / (l.len() as f64));

    for h in hand.into_iter() {
        h.join().unwrap();
    }
    // hand.pop().unwrap().join().unwrap();
}
