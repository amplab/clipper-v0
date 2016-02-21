

use server;
use digits;
use features;

fn run(features: Vec<(String, SocketAddr)>,
       num_users: isize,
       num_train_examples: usize,
       num_test_examples: usize,
       mnist_path: String) {

    println!("starting digits");
    let all_test_data = digits::load_mnist_dense(mnist_path).unwrap();
    let norm_test_data = digits::normalize(&all_test_data);

    println!("Test data loaded: {} points", norm_test_data.ys.len());

    let tasks = digits::create_online_dataset(&norm_test_data,
                                              &norm_test_data,
                                              num_train_examples,
                                              0,
                                              num_test_examples,
                                              num_users);

    let (features, handles): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                              .map(|(n, a)| features::create_feature_worker(n, a))
                              .unzip();
}
