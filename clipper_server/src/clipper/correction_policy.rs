use server::{Input, InputType, Output};
use std::collections::HashMap;
use serde::ser::Serialize;
use serde::de::Deserialize;
use ml::{linalg, linear};
use std::ptr;

/// Correction policies are stateless, any required state is tracked separately
/// and stored in the `CorrectionModelTable`
pub trait CorrectionPolicy<S> where S: Serialize + Deserialize {

    /// Initialize new correction state without any training data.
    /// `models` is a list of the new model IDs.
    fn new(models: Vec<&String>) -> S;

    /// Returns true if this correction policy accepts the provided input type.
    /// Used to check for valid configurations at runtime.
    fn accepts_input_type(input_type: &InputType) -> bool;

    /// Each correction policy must provide a name, used for logging and debugging
    /// purposes.
    fn get_name() -> &'static str;

    // need to think about the semantics of this method first
    // fn add_model(state: &S) -> S;

    /// Make a correction to the available model predictions.
    fn predict(state: &S,
               predictions: HashMap<String, Output>,
               missing_predictions: Vec<String>)
               -> Output;

    /// Prioritize the importance of models from highest to lowest priority.
    fn rank_models_desc(state: &S, model_names: Vec<&String>) -> Vec<String>;

    /// Update the correction model state with newly observed, labeled
    /// training data.
    fn train(state: &S,
             inputs: Vec<Input>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> S;
}

pub struct LogisticRegressionPolicy {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearCorrectionState {
    linear_model: Vec<f64>,
    anytime_estimators: Vec<f64>,
    offline_model_order: Vec<String>,
}

impl CorrectionPolicy<LinearCorrectionState> for LogisticRegressionPolicy {
    fn new(models: Vec<&String>) -> LinearCorrectionState {
        LinearCorrectionState {
            // initialize correction model to zero vector when we have no training data
            linear_model: vec![0.0; models.len()],
            anytime_estimators: vec![0.0; models.len()],
            offline_model_order: models.iter().map(|m| (*m).clone()).collect::<Vec<String>>(),
        }
    }

    /// The linear correction policy ignores the original input
    /// and so it can accept all input types.
    #[allow(unused_variables)]
    fn accepts_input_type(input_type: &InputType) -> bool {
        true
    }

    fn get_name() -> &'static str {
        "logistic_regression_correction_policy"
    }

    fn predict(state: &LinearCorrectionState,
               predictions: HashMap<String, Output>,
               missing_predictions: Vec<String>)
               -> Output {
        // TODO track anytime estimators for each prediction, but this API is too restrictive
        // to do that because we don't allow updates to the correction state in this method.
        let mut x = Vec::with_capacity(state.linear_model.len());
        for i in 0..state.linear_model.len() {
            let pred = match predictions.get(&state.offline_model_order[i]) {
                Some(p) => *p,
                None => {
                    debug_assert!(missing_predictions.contains(&state.offline_model_order[i]));
                    state.anytime_estimators[i]
                }
            };
            x.push(pred);
        }
        if linalg::dot(&state.linear_model, &x) > 0_f64 {
            1.0
        } else {
            0.0
        }
    }

    #[allow(unused_variables)]
    fn rank_models_desc(state: &LinearCorrectionState, model_names: Vec<&String>) -> Vec<String> {
        let mut weight_magnitudes = state.linear_model
                                         .iter()
                                         .map(|w| w.abs())
                                         .zip(state.offline_model_order
                                                   .iter())
                                         .collect::<Vec<_>>();
        weight_magnitudes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let ordered_names = weight_magnitudes.iter().map(|pair| pair.1.clone()).collect::<Vec<_>>();
        ordered_names
    }

    #[allow(unused_variables)]
    fn train(state: &LinearCorrectionState,
             inputs: Vec<Input>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> LinearCorrectionState {
        let mut xs = Vec::with_capacity(predictions.len());

        for map in predictions {
            let mut x = Vec::with_capacity(state.linear_model.len());
            for i in 0..state.linear_model.len() {
                let pred = map.get(&state.offline_model_order[i]).unwrap();
                x.push(*pred);
            }
            xs.push(x);
        }

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
        let prob = linear::Problem::from_training_data(&xs, &labels);
        let model = linear::train_logistic_regression(prob, params);
        unimplemented!();
        // LinearCorrectionState {
        //     linear_model: model.w,
        //     anytime_estimators: linalg::mean_and_var(&xs).0,
        //     offline_model_order: state.offline_model_order.clone(),
        // }
    }
}

pub struct LinearRegressionPolicy {}

pub struct DummyCorrectionPolicy {}

impl CorrectionPolicy<Vec<f64>> for DummyCorrectionPolicy {
    #[allow(dead_code, unused_variables)]
    fn new(models: Vec<&String>) -> Vec<f64> {
        let mut v = Vec::with_capacity(models.len());
        for i in 0..models.len() {
            v.push(i as f64);
        }
        v
    }

    #[allow(unused_variables)]
    fn accepts_input_type(input_type: &InputType) -> bool {
        true
    }

    #[allow(dead_code)]
    fn get_name() -> &'static str {
        "dummy_correction_model"
    }

    #[allow(dead_code, unused_variables)]
    fn predict(state: &Vec<f64>,
               predictions: HashMap<String, Output>,
               missing_predictions: Vec<String>)
               -> Output {
        predictions.len() as Output
    }

    #[allow(dead_code, unused_variables)]
    fn rank_models_desc(state: &Vec<f64>, model_names: Vec<&String>) -> Vec<String> {
        model_names.iter().map(|s| (*s).clone()).collect::<Vec<_>>()
    }

    #[allow(dead_code, unused_variables)]
    fn train(state: &Vec<f64>,
             inputs: Vec<Input>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> Vec<f64> {
        let mut new_state = Vec::new();
        for i in state {
            new_state.push(i + 2.0);
        }
        info!("num inputs: {}, new state: {:?}", inputs.len(), new_state);
        new_state
    }
}
