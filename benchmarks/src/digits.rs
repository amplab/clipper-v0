#![allow(dead_code)]
use std::error::Error;
use std::result::Result;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
// use std::sync::Arc;
// use std::collections::HashSet;
// use std::iter::FromIterator;
// use std::fmt;
// use rand;
// use rand::Rng;
// use time::PreciseTime;

use clipper::ml::linalg;
// use mtl;

pub const NUM_DIGITS: usize = 10;

pub struct TrainingData {
    pub xs: Vec<Vec<f64>>,
    pub ys: Vec<f64>,
}

pub fn load_mnist_dense(fpath: &String) -> Result<TrainingData, String> {
    let path = Path::new(fpath);
    let display = path.display();

    let file = match File::open(&path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => {
            return Err(format!("couldn't open {}: REASON: {}",
                               display,
                               Error::description(&why)))
        }
        Ok(file) => BufReader::new(file),
    };

    // pointer to first feature_node struct in each example
    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for line in file.lines().filter_map(|result| result.ok()) {
        let mut split = line.split(",").collect::<Vec<&str>>();
        // println!("{:?}", split);
        let label = match split.first() {
            Some(l) => {
                match l.trim().parse::<f64>() {
                    Ok(n) => n,
                    Err(why) => return Err(format!("couldn't parse {}", Error::description(&why))),
                }
            }
            None => return Err("malformed label".to_string()),
        };
        let mut features: Vec<f64> = Vec::new();
        split.remove(0); // remove the label, only features remaining
        for f in split.iter().map(|x| x.trim().parse::<f64>().unwrap()) {
            features.push(f);
        }
        xs.push(features);
        ys.push(label);
    }
    Ok(TrainingData { xs: xs, ys: ys })
}

pub fn normalize(training_data: &TrainingData) -> TrainingData {
    let zs = normalize_matrix(&training_data.xs);
    TrainingData {
        xs: zs,
        ys: training_data.ys.clone(),
    }
}

pub fn normalize_matrix(xs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    let (means, vars) = linalg::mean_and_var(&xs);

    let mut zs: Vec<Vec<f64>> = Vec::with_capacity(xs.len());
    for point in xs {
        let mut idx = 0;
        let mut new_x: Vec<f64> = vec![0.0; point.len()];
        for feature in point.iter() {
            let sigma = if vars[idx] > 0.0 {
                vars[idx].powf(0.5)
            } else {
                1.0
            };
            new_x[idx] = (feature - means[idx]) / sigma;
            idx += 1;
        }
        zs.push(new_x);
    }
    zs
}


// // #[derive(Debug)]
// pub struct DigitsTask {
//     pub pref: f64,
//     pub offline_train_x: Vec<Vec<f64>>,
//     pub online_train_x: Vec<Vec<f64>>,
//     pub offline_train_y: Vec<f64>,
//     pub online_train_y: Vec<f64>,
//     pub test_x: Vec<Vec<f64>>,
//     pub test_y: Vec<f64>, // pub model: Option<mtl::LogisticRegression>
// }
//
//
// impl DigitsTask {
//     // pub fn get_model(&self) -> &mtl::LogisticRegression {
//     //     self.model.as_ref().unwrap()
//     // }
//
//     pub fn to_strings(&self, id: i32) -> (String, String, String) {
//         let mut train_str = String::new();
//         let mut test_str = String::new();
//         let pref_str = format!("{}, {}\n", id, self.pref);
//         for i in 0..self.offline_train_y.len() {
//             train_str.push_str(&format!("{}, {}, {}\n",
//                                         id,
//                                         self.offline_train_y[i],
//                                         vec_to_csv(&self.offline_train_x[i])));
//         }
//         for i in 0..self.online_train_y.len() {
//             train_str.push_str(&format!("{}, {}, {}\n",
//                                         id,
//                                         self.online_train_y[i],
//                                         vec_to_csv(&self.online_train_x[i])));
//         }
//         for i in 0..self.test_y.len() {
//             test_str.push_str(&format!("{}, {}, {}\n",
//                                        id,
//                                        self.test_y[i],
//                                        vec_to_csv(&self.test_x[i])));
//         }
//         (pref_str, train_str, test_str)
//     }
// }
//
// pub fn vec_to_csv(v: &Vec<f64>) -> String {
//     let mut vec_str = String::new();
//     for i in 0..v.len() {
//         vec_str.push_str(&format!("{}", v[i]));
//         if i < v.len() - 1 {
//             vec_str.push_str(", ");
//         }
//     }
//     vec_str
// }
//
// impl Clone for DigitsTask {
//     fn clone(&self) -> DigitsTask {
//         DigitsTask {
//             pref: self.pref,
//             offline_train_x: self.offline_train_x.clone(),
//             online_train_x: self.online_train_x.clone(),
//             offline_train_y: self.offline_train_y.clone(),
//             online_train_y: self.online_train_y.clone(),
//             test_x: self.test_x.clone(),
//             test_y: self.test_y.clone(), // model: None
//         }
//     }
// }

// impl fmt::Display for DigitsTask {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         // The `f` value implements the `Write` trait, which is what the
//         // write! macro is expecting. Note that this formatting ignores the
//         // various flags provided to format strings.
//         write!(f,
//                "(pref: {}, train_x: {}, train_y: {})",
//                self.pref,
//                self.train_x,
//                self.train_y)
//     }
// }


// pub fn create_online_dataset(training_data: &TrainingData,
//                              test_data: &TrainingData,
//                              offline_task_size: usize,
//                              online_task_size: usize,
//                              test_size: usize,
//                              num_tasks: usize)
//                              -> Vec<DigitsTask> {
//
//     let mut rng = rand::thread_rng();
//     let mut tasks: Vec<DigitsTask> = Vec::with_capacity(num_tasks as usize);
//     let total_train_size = offline_task_size + online_task_size;
//
//     for i in 0..num_tasks {
//         if i % 50 == 0 {
//             // println!("Making task {}", i);
//         }
//         let pref_digit: f64 = (rng.gen_range(0, NUM_DIGITS) + 1) as f64;
//         let (train_x, train_y) = select_datapoints(pref_digit, total_train_size, training_data);
//         let (test_x, test_y) = select_datapoints(pref_digit, test_size, test_data);
//
//         let (offline_train_x, online_train_x) = train_x.split_at(offline_task_size);
//         let (offline_train_y, online_train_y) = train_y.split_at(offline_task_size);
//
//         // cloning okay because these are all just refs anyway
//         // all_train_xs.extend(train_x.clone());
//         // all_train_ys.extend(train_y.clone());
//         // all_test_xs.extend(test_x.clone());
//         // all_test_ys.extend(test_y.clone());
//         // all_train_ts.extend(vec![i; task_size]);
//
//         tasks.push(DigitsTask {
//             pref: pref_digit,
//             offline_train_x: offline_train_x.to_vec(),
//             online_train_x: online_train_x.to_vec(),
//             offline_train_y: offline_train_y.to_vec(),
//             online_train_y: online_train_y.to_vec(),
//             test_x: test_x,
//             test_y: test_y, // model: None
//         });
//     }
//     tasks
// }



// pub fn create_mtl_datasets(training_data: &TrainingData,
//                            test_data: &TrainingData,
//                            task_size: usize,
//                            test_size: usize,
//                            num_tasks: usize)
//                            -> (Vec<DigitsTask>, // list of tasks
//                                Vec<Arc<Vec<f64>>>, // train_xs
//                                Vec<f64>, // train_ys
//                                Vec<usize>, // train_ts
//                                Vec<Arc<Vec<f64>>>, // test_xs
//                                Vec<f64> #<{(| test_ys |)}>#) {
//
//     let mut rng = rand::thread_rng();
//     let mut tasks: Vec<DigitsTask> = Vec::with_capacity(num_tasks as usize);
//     let mut all_train_xs: Vec<Arc<Vec<f64>>> = Vec::with_capacity(num_tasks * task_size as usize);
//     let mut all_train_ys: Vec<f64> = Vec::with_capacity(num_tasks * task_size as usize);
//     let mut all_train_ts: Vec<usize> = Vec::with_capacity(num_tasks * task_size as usize);
//     let mut all_test_xs: Vec<Arc<Vec<f64>>> = Vec::with_capacity(num_tasks * test_size as usize);
//     let mut all_test_ys: Vec<f64> = Vec::with_capacity(num_tasks * test_size as usize);
//
//     for i in 0..num_tasks {
//         if i % 50 == 0 {
//             println!("Making task {}", i);
//         }
//         let pref_digit: f64 = (rng.gen_range(0, NUM_DIGITS) + 1) as f64;
//         let (train_x, train_y) = select_datapoints(pref_digit, task_size, training_data);
//         let (test_x, test_y) = select_datapoints(pref_digit, test_size, test_data);
//
//         // cloning okay because these are all just refs anyway
//         all_train_xs.extend(train_x.clone());
//         all_train_ys.extend(train_y.clone());
//         all_test_xs.extend(test_x.clone());
//         all_test_ys.extend(test_y.clone());
//         all_train_ts.extend(vec![i; task_size]);
//
//         tasks.push(DigitsTask {
//             pref: pref_digit,
//             offline_train_x: train_x,
//             online_train_x: Vec::new(),
//             offline_train_y: train_y,
//             online_train_y: Vec::new(),
//             test_x: test_x,
//             test_y: test_y, // model: None
//         });
//     }
//     (tasks,
//      all_train_xs,
//      all_train_ys,
//      all_train_ts,
//      all_test_xs,
//      all_test_ys)
// }

// fn select_datapoints(pref_digit: f64,
//                      num_points: usize,
//                      data: &TrainingData)
//                      -> (Vec<Arc<Vec<f64>>>, Vec<f64>) {
//     let mut rng = rand::thread_rng();
//     let num_false = (num_points / 2) as usize;
//     // handle odd numbers by picking one more true example than false
//     let num_true = if num_false * 2 < num_points {
//         num_false + 1
//     } else {
//         num_false
//     };
//
//     let mut perm: Vec<usize> = (0..data.ys.len()).collect::<Vec<usize>>();
//     rng.shuffle(&mut perm); // in-place shuffle
//     let true_ind = perm.iter().filter(|idx| pref_digit == data.ys[**idx]).collect::<Vec<_>>();
//     let false_ind = perm.iter().filter(|idx| pref_digit != data.ys[**idx]).collect::<Vec<_>>();
//     let mut ord_x: Vec<Arc<Vec<f64>>> = Vec::with_capacity(num_points as usize);
//     let mut ord_y: Vec<f64> = Vec::with_capacity(num_points as usize);
//     // print!("pref digit: {}, true: ", pref_digit);
//     for ind in true_ind[0..num_true].iter() {
//         ord_x.push(data.xs[**ind].clone());
//         // print!("{}, ", &data.ys[**ind]);
//         ord_y.push(1.0);
//     }
//     // print!("false: ");
//     for ind in false_ind[0..num_false].iter() {
//         // print!("{}, ", &data.ys[**ind]);
//         ord_x.push(data.xs[**ind].clone());
//         ord_y.push(0.0);
//     }
//     // println!("");
//
//     let mut selection_perm: Vec<usize> = (0..num_points).collect::<Vec<_>>();
//     rng.shuffle(&mut selection_perm); // in-place shuffle
//     let selection_perm = selection_perm;
//     let mut x: Vec<Arc<Vec<f64>>> = Vec::with_capacity(num_points as usize);
//     let mut y: Vec<f64> = Vec::with_capacity(num_points as usize);
//     for idx in selection_perm {
//         x.push(ord_x[idx].clone());
//         y.push(ord_y[idx]);
//     }
//     (x, y)
// }
//
// // needed because file.write(), .flush()
// // require a mutable ref but the variables are never mutated
// #[allow(unused_mut)]
// pub fn save_tasks(tasks: &Vec<DigitsTask>, fname: &str) {
//     let pf = fname.to_string() + ".pref";
//     let trf = fname.to_string() + ".train";
//     let tef = fname.to_string() + ".test";
//
//     let pref_path = Path::new(&pf);
//     let train_path = Path::new(&trf);
//     let test_path = Path::new(&tef);
//     let mut options = OpenOptions::new();
//     // We want to write to our file as well as append new data to it.
//     options.write(true).truncate(true).create(true);
//
//     let mut pref_file = match options.open(&pref_path) {
//         // The `description` method of `io::Error` returns a string that
//         // describes the error
//         Err(why) => panic!(format!("couldn't open. REASON: {}", Error::description(&why))),
//         Ok(file) => BufWriter::new(file),
//     };
//
//     let mut train_file = match options.open(&train_path) {
//         // The `description` method of `io::Error` returns a string that
//         // describes the error
//         Err(why) => panic!(format!("couldn't open. REASON: {}", Error::description(&why))),
//         Ok(file) => BufWriter::new(file),
//     };
//
//     let mut test_file = match options.open(&test_path) {
//         // The `description` method of `io::Error` returns a string that
//         // describes the error
//         Err(why) => panic!(format!("couldn't open. REASON: {}", Error::description(&why))),
//         Ok(file) => BufWriter::new(file),
//     };
//
//     for (i, t) in tasks.iter().enumerate() {
//         let (p, trn, tst) = t.to_strings(i as i32);
//         pref_file.write(p.as_bytes()).unwrap();
//         train_file.write(trn.as_bytes()).unwrap();
//         test_file.write(tst.as_bytes()).unwrap();
//     }
//     pref_file.flush().unwrap();
//     train_file.flush().unwrap();
//     test_file.flush().unwrap();
// }

// #[allow(unused_mut)]
// pub fn load_tasks(fname: &str) -> (
//         Vec<DigitsTask>, // list of tasks
//         Vec<Arc<Vec<f64>>>, // train_xs
//         Vec<f64>, // train_ys
//         Vec<usize>, //train_ts
//         Vec<Arc<Vec<f64>>>, // test_xs
//         Vec<f64> // test_ys
//         ) {
//
//     // Load files
//     let pf = fname.to_string() + ".pref";
//     let trf = fname.to_string() + ".train";
//     let tef = fname.to_string() + ".test";
//
//     let pref_path = Path::new(&pf);
//     let train_path = Path::new(&trf);
//     let test_path = Path::new(&tef);
//     let mut options = OpenOptions::new();
//     options.read(true);
//
//     let mut pref_file = BufReader::new(options.open(&pref_path).unwrap());
//     let mut train_file = BufReader::new(options.open(&train_path).unwrap());
//     let mut test_file = BufReader::new(options.open(&test_path).unwrap());
//
//     let (train_tids, train_xs, train_ys) = parse_data_file(train_file);
//     let (test_tids, test_xs, test_ys) = parse_data_file(test_file);
//     let prefs = parse_prefs(pref_file);
//     println!("loading {} tasks", prefs.len());
//
//
//     // extract tasks from loaded data
//     let mut tasks: Vec<DigitsTask> = Vec::new();
//     let mut train_idx = 0;
//     let mut test_idx = 0;
//     for (tid, pref) in prefs.iter().enumerate() {
//         let mut cur_train_xs: Vec<Arc<Vec<f64>>> = Vec::new();
//         let mut cur_train_ys: Vec<f64> = Vec::new();
//         let mut cur_test_xs: Vec<Arc<Vec<f64>>> = Vec::new();
//         let mut cur_test_ys: Vec<f64> = Vec::new();
//         loop {
//             assert!(train_tids[train_idx] == tid);
//             cur_train_xs.push(train_xs[train_idx].clone());
//             cur_train_ys.push(train_ys[train_idx]);
//             train_idx += 1;
//             if train_tids.len() <= train_idx || train_tids[train_idx] > tid {
//                 break;
//             }
//         }
//         loop {
//             assert!(test_tids[test_idx] == tid);
//             cur_test_xs.push(test_xs[test_idx].clone());
//             cur_test_ys.push(test_ys[test_idx]);
//             test_idx += 1;
//             if test_tids.len() <= test_idx ||  test_tids[test_idx] > tid {
//                 break;
//             }
//         }
//
//         tasks.push(DigitsTask {
//             pref: *pref,
//             train_x: cur_train_xs,
//             train_y: cur_train_ys,
//             test_x: cur_test_xs,
//             test_y: cur_test_ys,
//             // model: None,
//         });
//     }
//
//     (tasks, train_xs, train_ys, train_tids, test_xs, test_ys)
// }

// fn parse_prefs(r: BufReader<File>) -> Vec<f64> {
//     let mut prefs: Vec<f64> = Vec::new();
//
//     let allowable_prefs: HashSet<i32> = HashSet::from_iter((1..11));
//     let mut idx = 0;
//     for line in r.lines().filter_map(|result| result.ok()) {
//         let split = line.split(",").collect::<Vec<&str>>();
//         let tid = match split.first() {
//             Some(l) => {
//                 match l.trim().parse::<i32>() {
//                     Ok(n) => n,
//                     Err(why) => panic!(format!("couldn't parse {}", Error::description(&why))),
//                 }
//             }
//             None => panic!("malformed label".to_string()),
//         };
//         let pref = match split[1].trim().parse::<i32>() {
//             Ok(n) => n,
//             Err(why) => panic!(format!("couldn't parse {}", Error::description(&why))),
//         };
//         assert!(idx == tid);
//         assert!(allowable_prefs.contains(&pref));
//         prefs.push(pref as f64);
//         idx += 1;
//     }
//     prefs
// }

pub fn parse_data_file(r: BufReader<File>)
                       -> (Vec<usize> /* task ids */, Vec<Vec<f64>> /* xs */, Vec<f64>) {
    // ys

    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut tids: Vec<usize> = Vec::new();
    for line in r.lines().filter_map(|result| result.ok()) {
        let mut split = line.split(",").collect::<Vec<&str>>();
        // println!("{:?}", split);
        let tid = match split.first() {
            Some(l) => {
                match l.trim().parse::<usize>() {
                    Ok(n) => n,
                    Err(why) => panic!(format!("couldn't parse {}", Error::description(&why))),
                }
            }
            None => panic!("malformed label".to_string()),
        };
        split.remove(0); // remove the tid, only features remaining
        let label = match split.first() {
            Some(l) => {
                match l.trim().parse::<f64>() {
                    Ok(n) => n,
                    Err(why) => panic!(format!("couldn't parse {}", Error::description(&why))),
                }
            }
            None => panic!("malformed label".to_string()),
        };
        split.remove(0); // remove the label, only features remaining
        let mut features: Vec<f64> = Vec::new();
        for f in split.iter().map(|x| x.trim().parse::<f64>().unwrap()) {
            features.push(f);
        }
        xs.push(features);
        ys.push(label);
        tids.push(tid);
    }

    (tids, xs, ys)
}


// #[cfg(test)]
// mod tests {
//     extern crate float_cmp;
//     use self::float_cmp::*;
//     use super::*;
//
//     fn round(x: f64, places: i32) -> f64 {
//         let r = (10.0_f64).powi(places);
//         (x*r).round() / r
//     }
//
//     #[test]
//     fn test_normalize() {
//
//         let x: Vec<Vec<f64>> = vec![vec![1.1, 6., -3.1], vec![-7., -1.5, 11.2], vec![0., 1., -2.4], vec![0.1, 0., 9.7]];
//         let computed_norm = normalize_matrix(&x);
//         // println!("Computed: {:?}", computed_norm);
//
//         // computed from python
//         let expected_norm = vec![vec![ 0.78873315,  1.64322769, -1.04891472], vec![-1.71665449, -1.02146586,  1.10928391], vec![ 0.44849532, -0.13323468, -0.94326863], vec![ 0.47942603, -0.48852715,  0.88289944]];
//         // println!("Expected: {:?}", expected_norm);
//         for i in 0..x.len() {
//             for j in 0..x[i].len() {
//                 println!("{}, {}", computed_norm[i][j], expected_norm[i][j]);
//
//                 assert!(round(computed_norm[i][j], 8).approx_eq_ulps(&expected_norm[i][j], 2));
//             }
//         }
//     }
// }
//
//
//
//
//
//
//
