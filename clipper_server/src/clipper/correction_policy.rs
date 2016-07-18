use server::{Input, Output};
use std::collections::HashMap;
use serde::ser::Serialize;
use serde::de::Deserialize;
// use ml::linear::LogisticRegressionModel;

/// Correction policies are stateless, any required state is tracked separately
/// and stored in the `CorrectionModelTable`
pub trait CorrectionPolicy<S> where S: Serialize + Deserialize {

    /// Initialize new correction state without any training data.
    /// `models` is a list of the new model IDs.
    fn new(models: Vec<&String>) -> S;

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
    fn rank_models_desc<'a>(state: &S, model_names: Vec<&'a String>) -> Vec<&'a String>;

    /// Update the correction model state with newly observed, labeled
    /// training data. Each training example is used to train the model
    /// exactly once, but multiple examples may be batched together into a
    /// single call to `train()`.
    fn train(state: &S,
             inputs: Vec<Input>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> S;
}

pub struct LogisticRegressionPolicy {}

pub struct LinearRegressionPolicy {}

// impl CorrectionPolicy<LogisticRegressionModel> for LogisticRegressionPolicy {
//     fn new(models: Vec<&String>) -> LogisticRegressionModel {}
//
//     fn get_name() -> &'static str {}
//
//     fn predict(state: &LogisticRegressionModel,
//                predictions: HashMap<String, Output>,
//                missing_predictions: Vec<String>)
//                -> Output {
//     }
//
//     /// Prioritize the importance of models from highest to lowest priority.
//     fn rank_models_desc<'a>(state: &LogisticRegressionModel,
//                             model_names: Vec<&'a String>)
//                             -> Vec<&'a String> {
//         // sort by magnitude of the weights
//
//
//     }
//
//     #[allow(unused_variables)]
//     fn train(state: &LogisticRegressionModel,
//              inputs: Vec<Input>,
//              predictions: Vec<HashMap<String, Output>>,
//              labels: Vec<Output>)
//              -> LogisticRegressionModel {
//
//     }
// }




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
    fn rank_models_desc<'a>(state: &Vec<f64>, model_names: Vec<&'a String>) -> Vec<&'a String> {
        model_names
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
        new_state
    }
}
