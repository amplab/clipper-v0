use server::{Input, Output};
use std::collections::HashMap;
use serde::ser::Serialize;
use serde::de::Deserialize;

/// Correction policies are stateless, any required state is tracked separately
/// and stored in the `CorrectionModelTable`
pub trait CorrectionPolicy<S> where S: Serialize + Deserialize {

    /// Initialize new correction state without any training data.
    /// `models` is a list of the new model IDs.
    fn new(models: Vec<String>) -> S;

    /// Each correction policy must provide a name, used for logging and debugging
    /// purposes.
    fn get_name() -> &'static str;

    // need to think about the semantics of this method first
    // fn add_model(state: &S) -> S;

    /// Make a correction to the available model predictions.
    fn predict(state: &S, ys: HashMap<String, Output>, missing_ys: Vec<String>) -> Output;

    /// Prioritize the importance of models from highest to lowest priority.
    fn rank_models_desc<'a>(state: &S, model_names: Vec<&'a String>) -> Vec<&'a String>;

    fn train(state: &S,
             inputs: Vec<&Input>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> S;
}

struct DummyCorrectionPolicy {}

impl CorrectionPolicy<Vec<f64>> for DummyCorrectionPolicy {
    fn new(models: Vec<String>) -> Vec<f64> {
        unimplemented!();
    }

    fn get_name() -> &'static str {
        unimplemented!();
    }

    fn predict(state: &Vec<f64>, ys: HashMap<String, Output>, missing_ys: Vec<String>) -> Output {
        unimplemented!();
    }

    fn rank_models_desc<'a>(state: &Vec<f64>, model_names: Vec<&'a String>) -> Vec<&'a String> {
        unimplemented!();
    }

    fn train(state: &Vec<f64>,
             inputs: Vec<&Input>,
             predictions: Vec<HashMap<String, Output>>,
             labels: Vec<Output>)
             -> Vec<f64> {
        unimplemented!();
    }
}

// impl CorrectionPolicy<Vec<f64>> for DummyCorrectionPolicy {
//     fn new(num_models: usize) -> Vec<f64> {
//         let mut v = Vec::with_capacity(num_models);
//         for i in 0..num_models {
//             v.push(i as f64);
//         }
//         v
//     }
//
//     fn get_name() -> &'static str {
//         "dummy policy"
//     }
//
//
//     #[allow(unused_variables)]
//     fn predict(state: &Vec<f64>, ys: Vec<Output>, missing_ys: Vec<usize>) -> Output {
//         missing_ys.len() as Output
//     }
//
//     fn rank_models_desc(state: &Vec<f64>) -> Vec<usize> {
//         (0..state.len()).into_iter().collect::<Vec<usize>>()
//     }
//
//     fn train(state: &Vec<f64>) -> Vec<f64> {
//         state.iter().map(|x| x + 1.5).collect()
//     }
// }
