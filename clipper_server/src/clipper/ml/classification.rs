use std::ptr;
use std::sync::Arc;
// use std::sync::mpsc;
// use threadpool::ThreadPool;
// use std::collections::HashMap;

use ml::{linalg, linear};
// use svm;


#[allow(dead_code)]
#[derive(Debug)]
pub struct TaskModel {
    w: Vec<f64>,
    tid: usize, // model/task ID
    k: usize,
    model: Option<linear::LogisticRegressionModel>,
}

impl TaskModel {
    /// Constructs a new `TaskModel`
    /// `k` is the dimension of the model.
    /// `tid` is the task id assigned to this task (e.g. user ID).
    pub fn new(k: usize, tid: usize) -> TaskModel {
        TaskModel {
            w: linalg::gen_normal_vec(k),
            tid: tid,
            k: k,
            model: None,
        }
    }

    pub fn get_id(&self) -> usize {
        self.tid
    }

    pub fn get_wi(&self, i: usize) -> f64 {
        if i >= self.w.len() {
            0.0
        } else {
            self.w[i]
        }
    }

    pub fn get_labels(&self) -> (f64, f64) {
        match self.model.as_ref() {
            Some(m) => m.get_labels(),
            None => (1.0, 0.0),
        }
    }

    pub fn get_w(&self) -> &Vec<f64> {
        &self.w
    }

    pub fn train(k: usize, tid: usize, xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> TaskModel {
        let params = linear::Struct_parameter {
            solver_type: linear::L2R_LR,
            eps: 0.0001,
            C: 1.0f64,
            nr_weight: 0,
            weight_label: ptr::null_mut(),
            weight: ptr::null_mut(),
            p: 0.1,
            init_sol: ptr::null_mut(),
        };
        let prob = linear::Problem::from_training_data(xs, ys);
        let model = linear::train_logistic_regression(prob, params);

        TaskModel {
            w: model.w.clone(),
            tid: tid,
            k: k,
            model: Some(model),
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        match self.model.as_ref() {
            Some(m) => m.logistic_regression_predict(x),
            None => {
                if linalg::dot(&self.w, x) > 0_f64 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// pub const NUM_THREADS: usize = 8;

// pub trait CanonicalModel {
//
//     fn predict(&self, x: &Vec<f64>) -> f64;
//     fn train(xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> Self;
// }
//
// impl CanonicalModel for svm::SVMModel {
//
//     fn predict(&self, x: &Vec<f64>) -> f64 {
//         self.svm_predict(x)
//     }
//
//     #[must_use]
//     fn train(xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> svm::SVMModel {
//         // settings taken from defaults in sklearn:
//         // http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC
//         // and libsvm:
//         // https://github.com/cjlin1/libsvm/blob/8f3d96edce6e9363ad40cefe6e135650a10e53fb/svm-train.c#L167
//         let params = svm::Struct_svm_parameter {
//             svm_type: svm::C_SVC,
//             kernel_type: svm::get_kernel_type(svm::Kernel::Rbf),
//             degree: 3,
//             gamma: 1.0_f64 / (xs[0].len() as f64), //sklearn default
//             coef0: 0.0,
//             cache_size: 200_f64,
//             eps: 0.001,
//             C: 1_f64,
//             nr_weight: 0,
//             weight_label: ptr::null_mut(),
//             weight: ptr::null_mut(),
//             nu: 0.5,
//             p: 0.1,
//             shrinking: 1,
//             probability: 0
//         };
//
//         let prob = svm::SVMProblem::from_training_data(xs, ys);
//         svm::train_svm(prob, params)
//     }
// }
//
// impl CanonicalModel for linear::LogisticRegressionModel {
//
//     fn predict(&self, x: &Vec<f64>) -> f64 {
//         self.logistic_regression_predict(x)
//     }
//
//     #[must_use]
//     fn train(xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> linear::LogisticRegressionModel {
//         let params = linear::Struct_parameter {
//             solver_type: linear::L2R_LR,
//             eps: 0.0001,
//             C: 1.0f64,
//             nr_weight: 0,
//             weight_label: ptr::null_mut(),
//             weight: ptr::null_mut(),
//             p: 0.1,
//             init_sol: ptr::null_mut()
//         };
//         let prob = linear::Problem::from_training_data(xs, ys);
//         linear::train_logistic_regression(prob, params)
//     }
//
//     // fn get_sv_indices(&self) -> Option<Vec<i32>> { None }
//
//     // fn get_w(&self) -> Option<Vec<f64>> {
//     //     Some(self.w.clone())
//     // }
//
//     // fn check_model_debug(&self) {} // no-op
// }


// pub struct BinaryClassifier<B: CanonicalModel + Send> {
//
//     // pub xs: Vec<Arc<Vec<f64>>>, //takes ownership of training data
//     // pub ys: Arc<Vec<f64>>, // takes ownership
//     // pub ts: Arc<Vec<usize>>, // records assignment of training data to task
//     pub fs: Option<Vec<Arc<B>>>,
//     pub task_models: HashMap<usize, TaskModel>,
//     pub k: usize, // number of base models (dimension of w)
//     // pub d: usize // dimension of base models (|w|)
//
// }
//
// enum BaseInitStrategy {
//     RandomSubset, // randomly pick a subset to initialize
//     Oracle, //
//     All, // no initialization, just randomly init ws and start training fs
//     Cluster, // use kmeans to group similar tasks together. Train an initial f for each model.
// }
//
// impl <B> BinaryClassifier<B> where B: 'static + CanonicalModel + Send + Sync {
//
//     /// Create a new BinaryClassifier from the given training dataset
//     pub fn new(num_tasks: usize, k: usize) -> BinaryClassifier<B> {
//
//         // let num_tasks = ts.iter().max().unwrap();
//         let mut tasks = HashMap::new();
//         for i in 0..num_tasks {
//             tasks.insert(i, TaskModel::new(k, i));
//         }
//
//         let fs = None;
//         BinaryClassifier {
//             fs: fs,
//             task_models: tasks,
//             k: k,
//         }
//     }
//
//     // pub fn train_mtl_model(k: usize,
//     //                        xs: Vec<Arc<Vec<f64>>>,
//     //                        ys: Vec<f64>,
//     //                        ts: Vec<usize>,
//     //                        iters: usize,
//     //
//     //                        ) -> BinaryClassifier<B> {
//     //
//     //     let mut b = BinaryClassifier::new();
//     //     for i in 0..iters {
//     //         println!("Starting training iter: {}", i);
//     //
//     //     }
//     // }
//
//     pub fn compute_features(&self, x: &Vec<f64>) -> Result<Vec<f64>, &str> {
//         // self.fs.iter().map(|f| f.predict(x)).collect::<Vec<f64>>()
//         match self.fs.as_ref() {
//             None => Err("feature functions not initialized"),
//             Some(fs) => Ok(BinaryClassifier::<B>::featurize(&fs, x))
//         }
//     }
//
//     pub fn featurize(fs: &Vec<Arc<B>>, x: &Vec<f64>) -> Vec<f64> {
//         fs.iter().map(|f| f.predict(x)).collect::<Vec<f64>>()
//     }
//
//     // TODO: parallelize
//     // pub fn train_ws(&mut self) {
//     pub fn train_task_models(&mut self, xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>, ts: &Vec<usize>) {
//
//         assert!(self.fs.is_some(), "cannot train task models without canonical models");
//         let mut train_data = group_by_task(xs, ys, ts);
//         let num_tasks = train_data.len();
//         let (tx, rx) = mpsc::channel();
//         let pool = ThreadPool::new(NUM_THREADS);
//
//         for (tid, data) in train_data.drain() {
//             let (tx, local_fs) = (tx.clone(), self.fs.as_ref().unwrap().clone());
//             let train_x = data.0;
//             let train_y = data.1;
//             let k = self.k;
//             pool.execute(move || {
//                 let transformed_xs = train_x
//                     .iter()
//                     .map(|x| Arc::new(BinaryClassifier::<B>::featurize(&local_fs, &x)))
//                     .collect::<Vec<Arc<Vec<f64>>>>();
//                 let new_model = TaskModel::train(k, tid, &transformed_xs, &train_y);
//                 tx.send((tid, new_model)).unwrap();
//             });
//         }
//
//         for _ in 0..num_tasks {
//             let (tid, m) = rx.recv().unwrap();
//             self.task_models.insert(tid, m);
//         }
//     }
//
//     pub fn train_single_task_model(&mut self, tid: usize, xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) {
//         assert!(self.fs.is_some(), "cannot train task models without canonical models");
//         let transformed_xs = xs
//             .iter()
//             .map(|x| Arc::new(BinaryClassifier::<B>::featurize(self.fs.as_ref().unwrap(), &x)))
//             .collect::<Vec<Arc<Vec<f64>>>>();
//         let new_model = TaskModel::train(self.k, tid, &transformed_xs, &ys);
//         self.task_models.insert(tid, new_model);
//     }
//
//     pub fn train_canonical_models(&mut self, xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>, ts: &Vec<usize>) {
//         let pool = ThreadPool::new(NUM_THREADS);
//         let (tx, rx) = mpsc::channel();
//         for fi in 0..self.k {
//             let (tx, xs, ys, ts) = (tx.clone(), xs.clone(), ys.clone(), ts.clone());
//
//             // TODO: if we haven't trained a model for the user yet, set wi[t] to 0
//             let wis: HashMap<usize, f64> = self.collect_wis(fi);
//             pool.execute(move || {
//                 let new_ys = transform_ys(&wis, &ys, &ts);
//                 let new_f = B::train(&xs, &new_ys);
//                 tx.send((fi, new_f)).unwrap();
//             });
//         }
//
//
//
//         let mut results = Vec::new();
//         for _ in 0..self.k {
//             let result = rx.recv().unwrap();
//             results.push(result);
//         }
//         results.sort_by(|a, b| (a.0).cmp(&b.0));
//         let new_fs = results.drain(..).map(|r| Arc::new(r.1)).collect::<_>(); // sort by fi
//         self.fs = Some(new_fs);
//     }
//
//     pub fn set_canonical_models(&mut self, new_fs: Vec<Arc<B>>) {
//         assert!(new_fs.len() == self.k);
//         self.fs = Some(new_fs);
//     }
//
//     pub fn set_task_models(&mut self, new_tasks: HashMap<usize, TaskModel>) {
//         self.task_models = new_tasks;
//     }
//
//     pub fn set_task_model(&mut self, tid: usize, model: TaskModel) {
//         self.task_models.insert(tid, model);
//     }
//
//     fn collect_wis(&self, idx: usize) -> HashMap<usize, f64> {
//         let mut wis: HashMap<usize, f64> = HashMap::with_capacity(self.task_models.len());
//         for (tid, m) in self.task_models.iter() {
//             let (p, _) = m.get_labels();
//             let wi = m.get_wi(idx);
//             if p == 1.0 {
//                 wis.insert(*tid, wi);
//             } else if p == 0.0 {
//                 wis.insert(*tid, -1.0*wi);
//             } else {
//                 panic!("Invalid value for label, must be either 0.0 or 1.0 but found {}", p);
//             }
//         }
//         wis
//     }
//
//     // /// Identify the positive and negative labels
//     // /// associated with each task's model.
//     // ///
//     // /// This is necessary because liblinear treats
//     // /// binary classification tasks as multi-class
//     // /// classification with two tasks, and so maps
//     // /// labels to pos/neg examples based on the first
//     // /// label encountered in the training data set, rather
//     // /// than the actual value of the label.
//     // ///
//     // /// Return value is a tuple of hashmaps that map task ID to pos and
//     // /// negative label values respectively. First map is positive labels,
//     // /// second is negative labels.
//     // fn collect_labels(&self) -> (Arc<HashMap<usize, f64>>, Arc<HashMap<usize, f64>>) {
//     //     let mut pos_labels = HashMap::new();
//     //     let mut neg_labels = HashMap::new();
//     //     for (tid, m) in &self.tasks.iter() {
//     //         let (p, n) = m.get_labels();
//     //         pos_labels.insert(tid, p);
//     //         neg_labels.insert(tid, n);
//     //     }
//     //     (Arc::new(pos_labels), Arc::new(neg_labels))
//     // }
//
//     pub fn predict(&self, x: &Vec<f64>, tid: usize) -> f64 {
//         let features = match self.compute_features(x) {
//             Ok(fs) => fs,
//             Err(e) => panic!(format!("Error in prediction: {}", e))
//         };
//         match self.task_models.get(&tid) {
//             Some(m) => m.predict(&features),
//             None => {
//                 println!("Warning: attempting to make a prediction for unknown task: {}", tid);
//                 0.0
//             }
//         }
//     }
// }



// fn transform_ys(
//     wis: &HashMap<usize, f64>,
//     ys: &Vec<f64>,
//     ts: &Vec<usize>) -> Vec<f64> {
//     let mut new_ys: Vec<f64> = Vec::with_capacity(ys.len());
//     for (i, y) in ys.iter().enumerate() {
//         let wi = *wis.get(&ts[i]).unwrap_or(&0.0_f64);
//         let yhat = if *y == 1.0 && wi > 0.0 {
//             1.0
//         } else if *y == 1.0 && wi <= 0.0 {
//             0.0
//         } else if *y == 0.0 && wi > 0.0 {
//             0.0
//         } else if *y == 0.0 && wi <= 0.0 {
//             1.0
//         } else {
//             panic!("invalid value for label: {}", y);
//         };
//         new_ys.push(yhat);
//     }
//     new_ys
// }
//
// fn group_by_task(
//     xs: &Vec<Arc<Vec<f64>>>,
//     ys: &Vec<f64>,
//     ts: &Vec<usize>) -> HashMap<usize, (Vec<Arc<Vec<f64>>>, Vec<f64>)> {
//     assert!(xs.len() == (&ys).len() && xs.len() == ts.len());
//     let mut grouped_data = HashMap::new();
//     for i in 0..ts.len() {
//         let entry = grouped_data.entry(ts[i]).or_insert((Vec::new(), Vec::new()));
//         (*entry).0.push(xs[i].clone());
//         (*entry).1.push(ys[i].clone());
//     }
//     grouped_data
// }



// pub struct ClusterClassifier<B: CanonicalModel + Send> {
//
//     pub xs: Vec<Arc<Vec<f64>>>, //takes ownership of training data
//     pub ys: Arc<Vec<f64>>, // takes ownership
//     pub ts: Arc<Vec<usize>>, // records assignment of training data to task
//     pub clusters: Vec<Arc<B>>,
//     pub tasks: Vec<digits::DigitsTask>, // index corresponds to task ID
//     pub k: usize, // number of base models (dimension of w)
//     pub d: usize // dimension of base models (|w|)
//
// }
//
