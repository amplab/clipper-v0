use std::ptr;
use std::slice;
use std::sync::Arc;
use svm_raw;
pub use svm_raw::{Struct_svm_parameter, C_SVC, NU_SVC, common};
// use util;
// use common;

mod util {

    use std::marker::PhantomData;
    use svm_raw::common;
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

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Kernel {
    Linear,
    Poly,
    Rbf,
    Sigmoid,
}

pub fn get_kernel_type(k: Kernel) -> i32 {
    match k {
        Kernel::Linear => svm_raw::LINEAR,
        Kernel::Poly => svm_raw::POLY,
        Kernel::Rbf => svm_raw::RBF,
        Kernel::Sigmoid => svm_raw::SIGMOID,
    }
}

#[allow(dead_code)]
pub fn to_kernel(t: i32) -> Kernel {
    match t {
        svm_raw::LINEAR => Kernel::Linear,
        svm_raw::POLY => Kernel::Poly,
        svm_raw::RBF => Kernel::Rbf,
        svm_raw::SIGMOID => Kernel::Sigmoid,
        _ => panic!("invalid value for svm kernel enum"),
    }
}

#[allow(non_snake_case)]
struct SVMParams {
    pub svm_type: i32,
    pub kernel_type: i32,
    pub degree: i32,
    pub gamma: f64,
    pub coef0: f64,
    pub cache_size: f64,
    pub eps: f64,
    pub C: f64,
    pub nr_weight: i32,
    pub nu: f64,
    pub p: f64,
    pub shrinking: i32,
    pub probability: i32,
}

impl SVMParams {
    fn from_raw(raw: svm_raw::Struct_svm_parameter) -> SVMParams {
        SVMParams {
            svm_type: raw.svm_type,
            kernel_type: raw.kernel_type,
            degree: raw.degree,
            gamma: raw.gamma,
            coef0: raw.coef0,
            cache_size: raw.cache_size,
            eps: raw.eps,
            C: raw.C,
            nu: raw.nu,
            p: raw.p,
            shrinking: raw.shrinking,
            probability: raw.probability,
            nr_weight: 0,
        }
    }

    fn to_raw(&self) -> svm_raw::Struct_svm_parameter {
        // let iii: *mut f64 = ptr::null_mut();
        // let jjj: *mut i32 = ptr::null_mut();
        svm_raw::Struct_svm_parameter {
            svm_type: self.svm_type,
            kernel_type: self.kernel_type,
            degree: self.degree,
            gamma: self.gamma,
            coef0: self.coef0,
            cache_size: self.cache_size,
            eps: self.eps,
            C: self.C,
            nu: self.nu,
            p: self.p,
            shrinking: self.shrinking,
            probability: self.probability,
            nr_weight: 0,
            weight: ptr::null_mut() as *mut f64,
            weight_label: ptr::null_mut() as *mut i32,
        }
    }
}


#[allow(dead_code)]
struct RawSVM<'a> {
    support_vectors: &'a Vec<Vec<svm_raw::Struct_svm_node>>, // the support vectors
    sv_ptrs: Option<Vec<*const svm_raw::Struct_svm_node>>, // ptrs to svs
    sv_coefs: &'a Vec<Vec<f64>>,
    sv_coef_ptrs: Option<Vec<*const f64>>, // ptrs to coefs
    raw: Option<svm_raw::Struct_svm_model>,
}

#[allow(non_snake_case, dead_code)]
pub struct SVMModel {
    param: SVMParams,
    nr_class: i32, // number of classes
    l: i32, // total number of support vectors
    support_vectors: Vec<Vec<svm_raw::Struct_svm_node>>, // the support vectors
    // sv_ptrs: Option<Vec<*const svm_raw::Struct_svm_node>>, // ptrs to svs
    sv_coefs: Vec<Vec<f64>>, // coef for each support vector
    // sv_coef_ptrs: Option<Vec<*const f64>>, // ptrs to coefs
    nSV: Vec<i32>, // number of SV in each class
    free_sv: i32, // bool whether to free or not, set to 1
    rho: Vec<f64>,
    label: Vec<i32>,
    sv_indices: Vec<i32>, /* raw: svm_raw::Struct_svm_model,
                           *
                           *
                           * TODO make sure these are unneeded fields
                           * probA: Vec<f64>,
                           * probB: Vec<f64>, */
}

impl SVMModel {
    unsafe fn extract_coefs(m: *const svm_raw::Struct_svm_model) -> Vec<Vec<f64>> {
        let nr_class = (*m).nr_class as isize;
        let l = (*m).l as usize;
        let mut sv_coefs: Vec<Vec<f64>> = Vec::with_capacity(nr_class as usize - 1);
        let sv_coef_ptr: *const *const f64 = (*m).sv_coef;
        for i in 0..nr_class - 1 {
            // let c = slice::from_raw_parts((*(*m).sv_coef.offset(i)), l).to_vec();
            let sv_cur_coefs: *const f64 = *(sv_coef_ptr.offset(i));
            let c = slice::from_raw_parts(sv_cur_coefs, l).to_vec();
            sv_coefs.push(c);
        }
        sv_coefs
    }

    unsafe fn extract_svs(m: *const svm_raw::Struct_svm_model) -> Vec<Vec<svm_raw::Struct_svm_node>> {
        let l = (*m).l as usize;
        let mut svs: Vec<Vec<svm_raw::Struct_svm_node>> = Vec::with_capacity(l);
        let sv_ptr_ptrs: *const *const svm_raw::Struct_svm_node = (*m).SV;
        for i in 0..l {
            // println!("extracting sv {} out of {}", i, l);
            let mut cur_sv: Vec<svm_raw::Struct_svm_node> = Vec::new();
            let sv_ptr_offset: *const svm_raw::Struct_svm_node = *(sv_ptr_ptrs.offset(i as isize));
            let mut j = 0;
            loop {
                // TODO I want to be able to free this memory. If I just deref the pointer,
                // will the struct be copied into the vector?
                let sv_node_offset: *const svm_raw::Struct_svm_node =
                    sv_ptr_offset.offset(j as isize);
                // println!("ptr_offset: {:?}, node_offset: {:?}", sv_ptr_offset, sv_node_offset);
                let sv: svm_raw::Struct_svm_node = *sv_node_offset;
                let node = svm_raw::Struct_svm_node {
                    index: sv.index,
                    value: sv.value,
                };
                assert!(sv_node_offset != &node as *const svm_raw::Struct_svm_node);
                j += 1;
                cur_sv.push(node);
                if node.index == -1 {
                    break;
                }
            }
            svs.push(cur_sv);
        }
        svs
    }

    #[allow(non_snake_case)]
    pub fn from_raw(m: *const svm_raw::Struct_svm_model) -> SVMModel {
        unsafe {
            let svs = SVMModel::extract_svs(m);
            let sv_coefs = SVMModel::extract_coefs(m);
            let nr_class = (*m).nr_class;
            let nSV = slice::from_raw_parts((*m).nSV, (*m).nr_class as usize).to_vec();
            let rho = slice::from_raw_parts((*m).rho, (nr_class * (nr_class - 1) / 2) as usize)
                          .to_vec();
            let label = slice::from_raw_parts((*m).label, nr_class as usize).to_vec();
            let sv_indices = slice::from_raw_parts((*m).sv_indices, (*m).l as usize).to_vec();
            let _model = SVMModel {
                param: SVMParams::from_raw((*m).param),
                nr_class: (*m).nr_class,
                l: (*m).l,
                support_vectors: svs,
                // sv_ptrs: None,
                sv_coefs: sv_coefs,
                nSV: nSV,
                rho: rho,
                label: label,
                free_sv: (*m).free_sv,
                sv_indices: sv_indices,
            };

            // _model.sv_ptrs = Some(util::vec_to_ptrs(&_model.support_vectors).vec);
            // _model.sv_coef_ptrs = Some(util::vec_to_ptrs(&_model.sv_coefs).vec);
            // let raw = svm_raw::Struct_svm_model {
            //         param: _model.param.to_raw(),
            //         nr_class: nr_class,
            //         l: (*m).l,
            //         SV: (_model.sv_ptrs.as_ref().unwrap()[..]).as_ptr(),
            //         sv_coef: (_model.sv_coef_ptrs.as_ref().unwrap()[..]).as_ptr(),
            //         rho: (&_model.rho[..]).as_ptr(),
            //         label: (&_model.label[..]).as_ptr(),
            //         probA: ptr::null(),
            //         probB: ptr::null(),
            //         sv_indices: ptr::null(),
            //         nSV: (&_model.nSV[..]).as_ptr(),
            //         free_sv: (*m).free_sv,
            //     };

            // TODO free mem
            // svm_raw::svm_free_and_destroy_model(&mut (m as *mut svm_raw::Struct_svm_model) as *mut *mut svm_raw::Struct_svm_model);
            _model
        }
    }

    #[allow(unused_variables)]
    fn to_raw<'a>(&'a self) -> RawSVM<'a> {
        let sv_ptrs = util::vec_to_ptrs(&self.support_vectors).vec;
        let sv_coef_ptrs = util::vec_to_ptrs(&self.sv_coefs).vec;

        let mut raw_model = RawSVM {
            support_vectors: &self.support_vectors,
            sv_coefs: &self.sv_coefs,
            sv_ptrs: None,
            sv_coef_ptrs: None,
            raw: None,
        };
        raw_model.sv_ptrs = Some(util::vec_to_ptrs(&self.support_vectors).vec);
        raw_model.sv_coef_ptrs = Some(util::vec_to_ptrs(&self.sv_coefs).vec);

        // sv_ptrs: None,
        // sv_coef_ptrs: None,
        // raw: svm_raw::Struct_svm_model::default(),
        raw_model.raw = Some(svm_raw::Struct_svm_model {
            param: self.param.to_raw(),
            nr_class: self.nr_class,
            l: self.l,
            SV: (raw_model.sv_ptrs.as_ref().unwrap()[..]).as_ptr(),
            sv_coef: (raw_model.sv_coef_ptrs.as_ref().unwrap()[..]).as_ptr(),
            rho: (&self.rho[..]).as_ptr(),
            label: (&self.label[..]).as_ptr(),
            probA: ptr::null(),
            probB: ptr::null(),
            sv_indices: ptr::null(),
            nSV: (&self.nSV[..]).as_ptr(),
            free_sv: self.free_sv,
        });
        raw_model
    }


    pub fn svm_predict(&self, x: &Vec<f64>) -> f64 {
        let sparse_x = util::make_sparse_vector(x);
        // let m: *const svm_raw::Struct_svm_model = &self.to_raw() as *const svm_raw::Struct_svm_model;
        self.predict_sparse(&sparse_x)
    }

    // #[allow(dead_code)]
    // pub fn predict_all(&self, xs: &Vec<Arc<Vec<f64>>>) -> Vec<f64> {
    //     let m = &self.to_raw() as *const svm_raw::Struct_svm_model;
    //     let mut predictions: Vec<f64> = Vec::with_capacity(xs.len());
    //     for t in xs {
    //         predictions.push(self.predict_single(&util::make_sparse_vector(&t), m));
    //     }
    //     predictions
    // }

    fn predict_sparse(&self, sparse_x: &Vec<svm_raw::Struct_svm_node>) -> f64 {
        let result = unsafe {
            let raw_model = self.to_raw();
            let m = raw_model.raw.as_ref().unwrap() as *const svm_raw::Struct_svm_model;
            svm_raw::svm_predict(m, (&sparse_x[..]).as_ptr())
        };
        result
    }

    // pub fn check_model(&self) {
    //     let m: *const svm_raw::Struct_svm_model = &self.to_raw() as *const svm_raw::Struct_svm_model;
    //     let other_svs = &self.support_vectors;
    //     let extracted_svs = unsafe { SVMModel::extract_svs(m) };
    //     // assert!(extracted_svs.len() == other_svs.len());
    //     for i in 0..extracted_svs.len() {
    //         let ex_sv = &extracted_svs[i];
    //         let ot_sv = &other_svs[i];
    //         // assert!(ex_sv.len() == ot_sv.len(), "number of support vectors different");
    //         // assert!(ex_sv[ex_sv.len() - 1].index == -1, "extracted sv doesn't end in -1");
    //         // assert!(ot_sv[ot_sv.len() - 1].index == -1, "other sv doesn't end in -1");
    //         for j in 0..ex_sv.len() {
    //             let ex_node = ex_sv[j];
    //             let ot_node = ot_sv[j];
    //             // assert!(ex_node.index == ot_node.index, "indices don't match");
    //             // assert!(ex_node.value == ot_node.value, "values don't match");
    //         }
    //     }
    //     println!("checked model");
    // }
}

pub fn train_svm(prob: SVMProblem, params: svm_raw::Struct_svm_parameter) -> SVMModel {
    // let unsafe_prob = prob.to_unsafe_svm_problem();
    let model: SVMModel = unsafe {
        // Warning:
        let _model = svm_raw::svm_train(&prob.raw as *const svm_raw::Struct_svm_problem,
                                        &params as *const svm_raw::Struct_svm_parameter);
        let wrapped_model = SVMModel::from_raw(_model);
        // TODO: Is this necessary? NO
        // svm_raw::svm_free_and_destroy_model(_model as *mut *mut svm_raw::Struct_svm_model);
        wrapped_model
    };
    model
}

pub struct SVMProblem {
    pub num_examples: i32,
    pub max_index: i32,
    pub labels: Vec<f64>,
    pub examples: Vec<Vec<common::Struct_feature_node>>,
    pub example_ptrs: Vec<*const common::Struct_feature_node>,
    pub bias: f64,
    pub raw: svm_raw::Struct_svm_problem,
}

impl SVMProblem {
    pub fn from_training_data(xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> SVMProblem {
        let (examples, max_index) = util::make_sparse_matrix(xs);
        let example_ptrs = util::vec_to_ptrs(&examples).vec;
        let labels = ys.clone();
        let raw = svm_raw::Struct_svm_problem {
            l: ys.len() as i32,
            y: labels.as_ptr(),
            x: (&example_ptrs[..]).as_ptr(),
        };
        SVMProblem {
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
