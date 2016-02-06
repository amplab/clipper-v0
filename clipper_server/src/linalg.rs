use rand;
use std::sync::Arc;
// use rand::distributions::normal;
use rand::distributions::{Normal, IndependentSample};

pub fn mean_and_var(xs: &Vec<Arc<Vec<f64>>>) -> (Vec<f64>, Vec<f64>) {

    let mut sums: Vec<f64> = vec![0.0; xs[0].len()];
    let mut sum_squares: Vec<f64> = vec![0.0; xs[0].len()];
    let ks = xs[0].clone();
    let n: f64 = xs.len() as f64;
    for d in xs {
        let mut idx: usize = 0;
        for f in d.iter() {
            let k = ks[idx];
            sums[idx] += f - k;
            sum_squares[idx] += (f - k).powi(2);
            idx += 1;
        }
    }

    let mut vars: Vec<f64> = vec![0.0; xs[0].len()];
    let mut means: Vec<f64> = vec![0.0; xs[0].len()];
    for idx in 0..sums.len() {
        vars[idx] = (sum_squares[idx] - sums[idx].powi(2) / n) / n;
        let k_correction = n*ks[idx];
        means[idx] = (sums[idx] + k_correction) / n;
    }
    (means, vars)
}

/// Generate a vector of size `dim` whose elements
/// are normally distributed with mu=0 and sigma=1.
pub fn gen_normal_vec(dim: usize) -> Vec<f64> {
    #![allow(unused_variables)]
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0);
    let mut w: Vec<f64> = Vec::with_capacity(dim);
    for i in 0..dim {
        w.push(normal.ind_sample(&mut rng));
    }
    w
}


pub fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {

    // assert_eq!(a.len(), b.len());
    let alen = a.len();
    let blen = b.len();
    // if alen != blen {
    //     println!("Warning: dotting vectors of different lengths: {} and {}", alen, blen);
    // }
    let max_idx = if alen > blen { blen } else { alen };

    let mut sum = 0.0_f64;
    for i in 0..max_idx {
        sum += a[i] * b[i];
    }
    sum
}

