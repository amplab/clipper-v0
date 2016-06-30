use std::slice;
use std::sync::Arc;

use linear_raw;
pub use linear_raw::{Struct_parameter, L2R_LR, L1R_LR, common};
use ml::linalg;
// use common;
// use util;

mod util {

    use std::marker::PhantomData;
    use linear_raw::common;
    use std::sync::Arc;



    pub struct PtrVec<'a, T: 'a> {
        pub vec: Vec<*const T>,
        phantom: PhantomData<&'a Vec<T>>,
    }

    // TODO figure out how to use lifetimes
    pub fn vec_to_ptrs<'a, T>(examples: &'a Vec<Vec<T>>) -> PtrVec<'a, T> {

        // let all_x_vec: Vec<*mut Struct_feature_node> = Vec::new();
        let mut first_x_vec: Vec<*const T> = Vec::with_capacity(examples.len());

        for i in 0..examples.len() {
            first_x_vec.push((&examples[i][..]).as_ptr());
        }
        PtrVec {
            vec: first_x_vec,
            phantom: PhantomData,
        }
    }

    pub fn make_sparse_matrix(xs: &Vec<Arc<Vec<f64>>>)
                              -> (Vec<Vec<common::Struct_feature_node>>, i32) {

        let mut examples: Vec<Vec<common::Struct_feature_node>> = Vec::with_capacity(xs.len());

        let mut max_index = 1;
        for example in xs {
            let mut features: Vec<common::Struct_feature_node> = Vec::new();
            let mut idx = 1; // liblinear is 1-based indexing
            for f in example.iter() {
                if *f != 0.0 {
                    if idx > max_index {
                        max_index = idx;
                    }
                    let f_node = common::Struct_feature_node {
                        index: idx,
                        value: *f,
                    };
                    features.push(f_node);
                }
                idx += 1;
            }
            features.push(common::Struct_feature_node {
                index: -1,
                value: 0.0,
            }); // -1 indicates end of feature vector
            examples.push(features);
        }
        (examples, max_index)
    }

    #[allow(dead_code)]
    pub fn make_sparse_vector(x: &Vec<f64>) -> Vec<common::Struct_feature_node> {
        let mut sparse_x: Vec<common::Struct_feature_node> = Vec::with_capacity(x.len());
        for (i, f) in x.iter().enumerate() {
            if *f != 0.0_f64 {
                sparse_x.push(common::Struct_feature_node {
                    index: i as i32 + 1,
                    value: *f,
                });
            }
        }
        sparse_x.push(common::Struct_feature_node {
            index: -1,
            value: 0.0_f64,
        });
        sparse_x
    }
}

// #[derive(Default, Debug)]
pub struct Problem {
    pub num_examples: i32,
    pub max_index: i32,
    pub labels: Vec<f64>,
    pub examples: Vec<Vec<common::Struct_feature_node>>,
    pub example_ptrs: Vec<*const common::Struct_feature_node>,
    pub bias: f64,
    pub raw: linear_raw::Struct_problem,
}


impl Problem {
    pub fn from_training_data(xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> Problem {
        let (examples, max_index) = util::make_sparse_matrix(xs);
        let example_ptrs = util::vec_to_ptrs(&examples).vec;
        let labels = ys.clone();
        let raw = linear_raw::Struct_problem {
            l: ys.len() as i32,
            n: max_index,
            y: labels.as_ptr(),
            x: (&example_ptrs[..]).as_ptr(),
            bias: -1.0,
        };
        Problem {
            num_examples: ys.len() as i32,
            max_index: max_index,
            labels: labels,
            examples: examples,
            example_ptrs: example_ptrs,
            bias: -1.0,
            raw: raw,
        }
    }
}


#[derive(Default, Debug)]
#[allow(non_snake_case)] // to better match liblinear names
pub struct Parameters {
    pub solver_type: u32,
    pub eps: f64,
    pub C: f64,
    pub nr_weight: i32,
    pub weight_label: Option<Vec<i32>>,
    pub weight: Option<Vec<f64>>,
    pub p: f64,
}

impl Parameters {
    #[allow(dead_code)]
    fn from_raw(mut param: linear_raw::Struct_parameter) -> Parameters {

        let mut safe_params: Parameters = Parameters::default();
        unsafe {
            safe_params.solver_type = param.solver_type;
            safe_params.eps = param.eps;
            safe_params.C = param.C;
            safe_params.nr_weight = param.nr_weight;
            // TODO weight_label, weight could be null
            if !param.weight_label.is_null() {
                safe_params.weight_label =
                    Some(slice::from_raw_parts(param.weight_label, safe_params.nr_weight as usize)
                             .to_vec());
            } else {
                safe_params.weight_label = None;
            }
            if !param.weight.is_null() {
                safe_params.weight = Some(slice::from_raw_parts(param.weight,
                                                                safe_params.nr_weight as usize)
                                              .to_vec());
            } else {
                safe_params.weight = None;
            }
            safe_params.p = param.p;
            linear_raw::destroy_param(&mut param as *mut linear_raw::Struct_parameter);
        }
        safe_params
    }
}

#[derive(Default, Debug)]
pub struct LogisticRegressionModel {
    pub params: Parameters,
    pub nr_class: i32,
    pub nr_feature: i32,
    pub w: Vec<f64>,
    pub label: Option<Vec<i32>>,
    pub bias: f64,
}

impl LogisticRegressionModel {
    #[allow(dead_code)]
    fn from_raw(model: *const linear_raw::Struct_model) -> LogisticRegressionModel {
        let safe_model: LogisticRegressionModel = unsafe {
            let _model = LogisticRegressionModel {
                params: Parameters::from_raw((*model).param),
                nr_class: (*model).nr_class,
                nr_feature: (*model).nr_feature,
                // w: slice::from_raw_parts((*model).w, max_index as usize).to_vec(),
                w: slice::from_raw_parts((*model).w, (*model).nr_feature as usize).to_vec(),
                label: if !(*model).label.is_null() {
                    Some(slice::from_raw_parts((*model).label, (*model).nr_class as usize).to_vec())
                } else {
                    None
                },
                bias: (*model).bias,
            };
            linear_raw::free_and_destroy_model(&model as *const *const linear_raw::Struct_model);
            _model
        };
        safe_model
    }

    pub fn logistic_regression_predict(&self, x: &Vec<f64>) -> f64 {
        let dot = linalg::dot(&self.w, x);
        let (pos, neg) = self.get_labels();
        let pred = if dot > 0_f64 {
            pos
        } else {
            neg
        };
        pred as f64
    }

    pub fn get_labels(&self) -> (f64, f64) {
        let l = self.label.as_ref().unwrap();
        if l.len() == 1 {
            if l[0] == 1 {
                (1.0, 0.0)
            } else if l[0] == 0 {
                (0.0, 1.0)
            } else {
                panic!(format!("invalid label: {}", l[0]));
            }
        } else if l.len() == 2 {
            (l[0] as f64, l[1] as f64)
        } else {
            panic!(format!("strange number of labels: {}", l.len()));
        }
    }
}

pub fn train_logistic_regression(prob: Problem,
                                 params: linear_raw::Struct_parameter)
                                 -> LogisticRegressionModel {

    let model: LogisticRegressionModel = unsafe {
        // params.C = if find_c {
        //     let start_c = params.C;
        //     let nr_fold = 4;
        //     let max_c = 1024;
        //     let mut best_c = 0.0f64;
        //     let mut best_rate = 0.0f64;
        //     params.C
        // } else {
        //     params.C
        // };
        let _model = linear_raw::train(&prob.raw as *const linear_raw::Struct_problem,
                                       &params as *const linear_raw::Struct_parameter);
        LogisticRegressionModel::from_raw(_model)
    };
    model
}
