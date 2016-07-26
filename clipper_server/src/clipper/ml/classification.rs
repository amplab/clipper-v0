use std::ptr;
// use std::sync::Arc;
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

    pub fn train(k: usize, tid: usize, xs: &Vec<Vec<f64>>, ys: &Vec<f64>) -> TaskModel {
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
