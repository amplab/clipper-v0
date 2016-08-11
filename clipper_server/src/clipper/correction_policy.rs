use server::{Input, InputType, Output};
use std::collections::HashMap;
use std::sync::Arc;
use serde::ser::Serialize;
use serde::de::Deserialize;
use ml::{linalg, linear};
use ml::linear::LogisticRegressionModel;
use std::ptr;
use rand::{thread_rng, Rng};
// use backtrace::Backtrace;

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

    // /// Update the correction model state when an offline model is added.
    // ///
    // /// For example, a linear model may expand the size of the
    // /// vector and initialize the weight for the new model to 0.
    // fn add_models(old_state: &S, new_model_name: &String) -> S;

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
             inputs: Vec<Arc<Input>>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> S;
}

pub struct AveragePolicy {}

impl CorrectionPolicy<()> for AveragePolicy {

    #[allow(unused_variables)]
    fn new(models: Vec<&String>) -> () { () }

    #[allow(unused_variables)]
    fn accepts_input_type(input_type: &InputType) -> bool {
        true
    }

    fn get_name() -> &'static str {
        "average_correction_policy"
    }

    #[allow(unused_variables)]
    fn predict(state: &(),
               predictions: HashMap<String, Output>,
               missing_predictions: Vec<String>)
               -> Output {
        let pred = if predictions.len() == 0 {
            let mut rng = thread_rng();
            rng.next_f64() - 0.5
        } else {
            let mut sum = 0.0;
            for (_, p) in predictions.iter() {
                sum += *p;
            }
            sum / (predictions.len() as f64)
        };

        if pred > 0_f64 {
            1.0
        } else {
            -1.0
        }

    }

    #[allow(unused_variables)]
    fn rank_models_desc(state: &(), model_names: Vec<&String>) -> Vec<String> {
        unimplemented!();
    }

    #[allow(unused_variables)]
    fn train(state: &(),
             inputs: Vec<Arc<Input>>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> () {
        ()
    }
}

pub struct LogisticRegressionPolicy {}

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearCorrectionState {
    linear_model: LogisticRegressionModel,
    anytime_estimators: Vec<f64>,
    offline_model_order: Vec<String>,
}

impl CorrectionPolicy<LinearCorrectionState> for LogisticRegressionPolicy {
    fn new(models: Vec<&String>) -> LinearCorrectionState {
        // let bt = Backtrace::new();
        // info!("CREATING NEW CORRECTION MODEL);
        LinearCorrectionState {
            // initialize correction model to zero vector when we have no training data
            linear_model: LogisticRegressionModel::new(models.len()),
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
        let mut x = Vec::with_capacity(state.linear_model.w.len());
        for i in 0..state.linear_model.w.len() {
            let pred = match predictions.get(&state.offline_model_order[i]) {
                Some(p) => *p,
                None => {
                    debug_assert!(missing_predictions.contains(&state.offline_model_order[i]));
                    // info!("using anytime estimator for model: {}", state.offline_model_order[i]);
                    state.anytime_estimators[i]
                }
            };
            x.push(pred);
        }
        state.linear_model.logistic_regression_predict(&x)
        // let dot = linalg::dot(&state.linear_model, &x);
        // let mut rng = thread_rng();
        // if rng.next_f64() < 0.01 {
        //     info!("sample prediction vector: {:?}, dot product: {}", x, dot);
        // }

        // if dot > 0_f64 {
        //     1.0
        // } else {
        //     -1.0
        // }
    }

    #[allow(unused_variables)]
    fn rank_models_desc(state: &LinearCorrectionState, model_names: Vec<&String>) -> Vec<String> {
        let mut weight_magnitudes = state.linear_model.w
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
             inputs: Vec<Arc<Input>>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> LinearCorrectionState {
        let mut xs = Vec::with_capacity(predictions.len());

        for map in predictions {
            let mut x = Vec::with_capacity(state.linear_model.w.len());
            for i in 0..state.linear_model.w.len() {
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
        info!("OLD correction state: {:?}, offline_model_order: {:?}",
              state.linear_model.w,
              state.offline_model_order);
        info!("New correction state: {:?}, offline_model_order: {:?}, labels: {:?}",
              model.w,
              state.offline_model_order,
              model.label);
        // assert_eq!(model.w.len(), state.offline_model_order.len());
        LinearCorrectionState {
            linear_model: model,
            anytime_estimators: linalg::mean_and_var(&xs).0,
            offline_model_order: state.offline_model_order.clone(),
        }
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
             inputs: Vec<Arc<Input>>,
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
