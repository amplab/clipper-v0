use std::net::{ToSocketAddrs, SocketAddr, TcpStream, Shutdown};


pub struct ClipperConf {
    // General configuration
    pub slo_micros: u32,
    pub policy_name: String,
    pub models: HashMap<String, ModelConf>,
    pub use_lsh: bool,

    // Internal system settings
    pub num_predict_workers: u32,
    pub num_observe_workers: u32,
}


pub struct ModelConf {
    pub addresses: Vec<SocketAddr>,
    pub name: String,
    /// The dimension of the output vector this model produces
    pub num_outputs: usize,
}



impl ClipperConf {
    pub fn from_toml(fname: &str) -> ClipperConf {}
}

pub struct ClipperConfBuilder {
    // General configuration
    pub sla_micros: u32,
    pub policy_name: String,
    pub models: HashMap<String, ModelConf>,

    // Internal system settings
    pub num_predict_workers: u32,
    pub num_observe_workers: u32,
}
