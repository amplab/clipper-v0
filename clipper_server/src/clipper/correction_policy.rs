
/// Correction policies are stateless, any required state is tracked separately
/// and stored in the `CorrectionModelTable`
pub trait CorrectionPolicy<S> where S: Serialize + Deserialize {

    fn new(num_models: usize) -> S;

    // need to think about the semantics of this method first
    // fn add_model(state: &S) -> S;

    fn predict(state: &S, ys: Vec<Output>, missing_ys: Vec<usize>, debug_str: &String) -> Output;

    fn rank_models(state: &S) -> Vec<usize>;

    fn train(state: &S) -> S;

}

struct DummyCorrectionPolicy {}

impl CorrectionPolicy<Vec<f64>> for DummyCorrectionPolicy {
    fn new(num_models: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(num_models);
        for i in 0..num_models {
            v.push(i as f64);
        }
        v
    }

    #[allow(unused_variables)]
    fn predict(state: &Vec<f64>,
               ys: Vec<Output>,
               missing_ys: Vec<usize>,
               debug_str: &String)
               -> Output {
        missing_ys.len() as Output
    }

    fn rank_models(state: &Vec<f64>) -> Vec<usize> {
        (0..state.len()).into_iter().collect::<Vec<usize>>()
    }

    fn train(state: &Vec<f64>) -> Vec<f64> {
        state.iter().map(|x| x + 1.5).collect()
    }
}
