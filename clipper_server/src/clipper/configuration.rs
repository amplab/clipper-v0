use std::net::{ToSocketAddrs, SocketAddr};
use std::sync::{RwLock, Arc};
use toml::{Parser, Table, Value};
use metrics;
use server::InputType;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
use std::collections::HashSet;
use std::cmp::PartialEq;
use std::fmt;
// use serde::ser::Serialize;
use serde_json;

use batching::BatchStrategy;

#[derive(Serialize)]
pub struct ClipperConf {
    // General configuration
    pub name: String,
    pub slo_micros: u32,
    pub num_message_encodes: usize,
    pub policy_name: String,
    pub models: Vec<ModelConf>,
    pub use_lsh: bool,
    pub salt_update_cache: bool,
    pub input_type: InputType,
    // pub batch_size: i32,
    pub batch_strategy: BatchStrategy,

    // Internal system settings
    pub num_predict_workers: usize,
    pub num_update_workers: usize,
    pub cache_size: usize,
    pub window_size: isize,
    #[serde(skip_serializing)]
    pub metrics: Arc<RwLock<metrics::Registry>>,
    pub redis_ip: String,
    pub redis_port: u16,
}

impl fmt::Debug for ClipperConf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::ser::to_string_pretty(&self).unwrap())
    }
}

impl PartialEq<ClipperConf> for ClipperConf {
    fn eq(&self, other: &ClipperConf) -> bool {
        self.name == other.name && self.slo_micros == other.slo_micros &&
        self.policy_name == other.policy_name && self.models == other.models &&
        self.use_lsh == other.use_lsh && self.input_type == other.input_type &&
        self.num_predict_workers == other.num_predict_workers &&
        self.num_update_workers == other.num_update_workers &&
        self.cache_size == other.cache_size && self.window_size == other.window_size &&
        self.redis_ip == other.redis_ip &&
        self.redis_port == other.redis_port &&
        self.batch_strategy == other.batch_strategy &&
        self.num_message_encodes == other.num_message_encodes &&
        self.salt_update_cache == other.salt_update_cache
    }
}

impl ClipperConf {
    pub fn parse_from_toml(fname: &String) -> ClipperConf {
        let path = Path::new(fname);
        let display = path.display();

        let mut file = match File::open(&path) {
            // The `description` method of `io::Error` returns a string that
            // describes the error
            Err(why) => {
                panic!(format!("couldn't open {}: REASON: {}",
                               display,
                               Error::description(&why)))
            }
            Ok(file) => BufReader::new(file),
        };

        let mut toml_string = String::new();
        match file.read_to_string(&mut toml_string) {
            Err(why) => panic!("couldn't read {}: {}", display, Error::description(&why)),
            Ok(_) => print!("{} contains:\n{}", display, toml_string),
        }
        ClipperConf::parse_toml_string(&toml_string)
    }

    fn parse_toml_string(toml_string: &String) -> ClipperConf {

        let pc = Parser::new(&toml_string).parse().unwrap();

        let provided_input_name = pc.get("input_type").unwrap().as_str().unwrap();
        let lc_input_name = provided_input_name.to_lowercase();

        let mut int_keywords = HashSet::new();
        int_keywords.insert("int".to_string());
        int_keywords.insert("ints".to_string());
        int_keywords.insert("integer".to_string());
        int_keywords.insert("integers".to_string());
        int_keywords.insert("i32".to_string());

        let mut float_keywords = HashSet::new();
        float_keywords.insert("float".to_string());
        float_keywords.insert("floats".to_string());
        float_keywords.insert("f64".to_string());

        let mut str_keywords = HashSet::new();
        str_keywords.insert("str".to_string());
        str_keywords.insert("string".to_string());

        let mut byte_keywords = HashSet::new();
        byte_keywords.insert("byte".to_string());
        byte_keywords.insert("bytes".to_string());
        byte_keywords.insert("u8".to_string());


        let input_type = if int_keywords.contains(&lc_input_name) {

            let length = pc.get("input_length")
                .unwrap_or(&Value::Integer(-1))
                .as_integer()
                .unwrap() as i32;
            InputType::Integer(length)
        } else if float_keywords.contains(&lc_input_name) {
            let length = pc.get("input_length")
                .unwrap_or(&Value::Integer(-1))
                .as_integer()
                .unwrap() as i32;
            InputType::Float(length)
        } else if str_keywords.contains(&lc_input_name) {
            InputType::Str
        } else if byte_keywords.contains(&lc_input_name) {
            let length = pc.get("input_length")
                .unwrap_or(&Value::Integer(-1))
                .as_integer()
                .unwrap() as i32;
            InputType::Byte(length)
        } else {
            panic!("Invalid input type: {}", provided_input_name);
        };
        let conf = ClipperConf {
            name: pc.get("name")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
            num_message_encodes: pc.get("num_message_encodes")
                .unwrap_or(&Value::Integer(1))
                .as_integer()
                .unwrap() as usize,
            slo_micros: pc.get("slo_micros")
                .unwrap_or(&Value::Integer(20000))
                .as_integer()
                .unwrap() as u32,
            policy_name: pc.get("correction_policy")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
            models: ClipperConf::parse_model_confs(pc.get("models")
                .unwrap_or(&Value::Array(Vec::new()))
                .as_slice()
                .unwrap()),

            batch_strategy: ClipperConf::parse_batcher_toml(pc.get("batching")
                .unwrap()
                .as_table()
                .unwrap()),
            use_lsh: pc.get("use_lsh")
                .unwrap_or(&Value::Boolean(false))
                .as_bool()
                .unwrap(),
            salt_update_cache: pc.get("salt_update_cache")
                .unwrap_or(&Value::Boolean(false))
                .as_bool()
                .unwrap(),

            input_type: input_type,

            num_predict_workers: pc.get("num_predict_workers")
                .unwrap_or(&Value::Integer(2))
                .as_integer()
                .unwrap() as usize,
            num_update_workers: pc.get("num_update_workers")
                .unwrap_or(&Value::Integer(1))
                .as_integer()
                .unwrap() as usize,
            cache_size: pc.get("cache_size")
                .unwrap_or(&Value::Integer(49999))
                .as_integer()
                .unwrap() as usize,
            window_size: pc.get("window_size")
                .unwrap_or(&Value::Integer(-1))
                .as_integer()
                .unwrap() as isize,
            redis_port: pc.get("redis_port")
                .unwrap_or(&Value::Integer(6379))
                .as_integer()
                .unwrap() as u16,
            redis_ip: pc.get("redis_ip")
                .unwrap_or(&Value::String("127.0.0.1".to_string()))
                .as_str()
                .unwrap()
                .to_string(),
            metrics: Arc::new(RwLock::new(metrics::Registry::new(pc.get("name")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string()))),
        };
        conf
    }

    fn parse_model_confs(model_confs: &[Value]) -> Vec<ModelConf> {
        let mut models = Vec::new();
        for m in model_confs.iter() {
            let mt = m.as_table().unwrap();
            models.push(ModelConf::from_toml(mt));
        }
        models
    }

    fn parse_batcher_toml(bt: &Table) -> BatchStrategy {
        let strategy = bt.get("strategy").unwrap().as_str().unwrap();
        match strategy {
            "learned" => {
                let sample_size = bt.get("sample_size")
                    .unwrap_or(&Value::Integer(1000))
                    .as_integer()
                    .unwrap() as usize;
                let opt_addr = get_addr(bt.get("opt_addr")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string());
                BatchStrategy::Learned {
                    sample_size: sample_size,
                    opt_addr: opt_addr,
                }
            }
            "static" => {

                let batch_size = bt.get("batch_size")
                    .unwrap_or(&Value::Integer(-1))
                    .as_integer()
                    .unwrap() as usize;
                BatchStrategy::Static { size: batch_size }
            }
            "aimd" => BatchStrategy::AIMD,
            _ => panic!("Invalid batch strategy: {}", strategy),
        }
    }
}




#[derive(PartialEq,Debug,Serialize)]
pub struct ModelConf {
    pub name: String,
    pub addresses: Vec<SocketAddr>,
    /// The dimension of the output vector this model produces
    pub num_outputs: usize,
    pub version: u32,
}

// impl PartialEq<ModelConf> for ModelConf {
//     fn eq(&self, other: &ModelConf) -> bool {
//         self.name == other.name && self.addresses == other.addresses &&
//         self.num_outputs == other.num_outputs
//     }
// }


impl ModelConf {
    pub fn from_toml(mt: &Table) -> ModelConf {
        ModelConf {
            name: mt.get("name")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
            num_outputs: mt.get("num_outputs")
                .unwrap_or(&Value::Integer(1))
                .as_integer()
                .unwrap() as usize,

            addresses: get_addrs(mt.get("addresses")
                .unwrap()
                .as_slice()
                .unwrap()
                .to_vec()),

            version: mt.get("version")
                .unwrap_or(&Value::Integer(1))
                .as_integer()
                .unwrap() as u32,
        }
    }

    pub fn new(name: String,
               addresses: Vec<String>,
               num_outputs: usize,
               version: u32)
               -> ModelConf {
        ModelConf {
            name: name,
            addresses: get_addrs_str(addresses),
            num_outputs: num_outputs,
            version: version,
        }
    }
}

pub fn get_addr(a: String) -> SocketAddr {
    info!("{}", a);
    a.to_socket_addrs().unwrap().next().unwrap()
}

pub fn get_addrs(addrs: Vec<Value>) -> Vec<SocketAddr> {
    addrs.into_iter().map(|a| get_addr(a.as_str().unwrap().to_string())).collect::<Vec<_>>()
    // a.to_socket_addrs().unwrap().next().unwrap()
}

pub fn get_addrs_str(addrs: Vec<String>) -> Vec<SocketAddr> {
    addrs.into_iter().map(|a| get_addr(a)).collect::<Vec<_>>()
    // a.to_socket_addrs().unwrap().next().unwrap()
}


pub struct ClipperConfBuilder {
    // General configuration
    pub name: String,
    pub slo_micros: u32,
    pub num_message_encodes: usize,
    pub policy_name: String,
    pub models: Vec<ModelConf>,
    pub use_lsh: bool,
    pub salt_update_cache: bool,
    pub input_type: InputType,
    pub window_size: isize,
    pub redis_ip: String,
    pub redis_port: u16,
    // pub batch_size: i32,
    pub batch_strategy: BatchStrategy,

    // Internal system settings
    pub num_predict_workers: usize,
    pub num_update_workers: usize,
    pub cache_size: usize,
}

impl ClipperConfBuilder {
    pub fn new() -> ClipperConfBuilder {
        ClipperConfBuilder {
            name: "DEFAULT".to_string(),
            slo_micros: 20 * 1000,
            num_message_encodes: 1,
            policy_name: "default".to_string(),
            models: Vec::new(),
            use_lsh: false,
            salt_update_cache: false,
            input_type: InputType::Integer(-1),
            num_predict_workers: 2,
            num_update_workers: 1,
            cache_size: 49999,
            window_size: -1,
            redis_ip: "127.0.0.1".to_string(),
            redis_port: 6379,
            batch_strategy: BatchStrategy::Learned {
                sample_size: 1000,
                opt_addr: get_addr("127.0.0.1:7777".to_string()),
            },
        }
    }

    pub fn cache_size(&mut self, s: usize) -> &mut ClipperConfBuilder {
        self.cache_size = s;
        self
    }

    pub fn window_size(&mut self, w: isize) -> &mut ClipperConfBuilder {
        self.window_size = w;
        self
    }

    pub fn slo_micros(&mut self, m: u32) -> &mut ClipperConfBuilder {
        self.slo_micros = m;
        self
    }

    pub fn num_message_encodes(&mut self, m: usize) -> &mut ClipperConfBuilder {
        self.num_message_encodes = m;
        self
    }

    pub fn batch_strategy(&mut self, b: BatchStrategy) -> &mut ClipperConfBuilder {
        self.batch_strategy = b;
        self
    }

    pub fn policy_name(&mut self, name: String) -> &mut ClipperConfBuilder {
        self.policy_name = name;
        self
    }

    pub fn use_lsh(&mut self, l: bool) -> &mut ClipperConfBuilder {
        self.use_lsh = l;
        self
    }

    pub fn add_model(&mut self, m: ModelConf) -> &mut ClipperConfBuilder {
        self.models.push(m);
        self
    }

    pub fn num_predict_workers(&mut self, n: usize) -> &mut ClipperConfBuilder {
        self.num_predict_workers = n;
        self
    }

    pub fn num_update_workers(&mut self, n: usize) -> &mut ClipperConfBuilder {
        self.num_update_workers = n;
        self
    }

    pub fn name(&mut self, n: String) -> &mut ClipperConfBuilder {
        self.name = n;
        self
    }

    pub fn redis_ip(&mut self, ri: String) -> &mut ClipperConfBuilder {
        self.redis_ip = ri;
        self
    }

    pub fn redis_port(&mut self, rp: u16) -> &mut ClipperConfBuilder {
        self.redis_port = rp;
        self
    }

    pub fn input_type(&mut self, name: String, length: Option<i32>) -> &mut ClipperConfBuilder {
        let lc_name = name.to_lowercase();

        let mut int_keywords = HashSet::new();
        int_keywords.insert("int".to_string());
        int_keywords.insert("ints".to_string());
        int_keywords.insert("integer".to_string());
        int_keywords.insert("integers".to_string());
        int_keywords.insert("i32".to_string());

        let mut float_keywords = HashSet::new();
        float_keywords.insert("float".to_string());
        float_keywords.insert("floats".to_string());
        float_keywords.insert("f64".to_string());

        let mut str_keywords = HashSet::new();
        str_keywords.insert("str".to_string());
        str_keywords.insert("string".to_string());

        let mut byte_keywords = HashSet::new();
        byte_keywords.insert("byte".to_string());
        byte_keywords.insert("bytes".to_string());
        byte_keywords.insert("u8".to_string());

        let input_type = if int_keywords.contains(&lc_name) {
            match length {
                Some(l) => InputType::Integer(l),
                None => InputType::Integer(-1),
            }
        } else if float_keywords.contains(&lc_name) {
            match length {
                Some(l) => InputType::Float(l),
                None => InputType::Float(-1),
            }
        } else if str_keywords.contains(&lc_name) {
            if length.is_some() {
                info!("length arg provided for string is ignored");
            }
            InputType::Str
        } else if byte_keywords.contains(&lc_name) {
            match length {
                Some(l) => InputType::Byte(l),
                None => InputType::Byte(-1),
            }
        } else {
            panic!("Invalid input type: {}", name);
        };
        self.input_type = input_type;
        self
    }

    /// Takes ownership of builder and moves built items into finalized ClipperConf.
    pub fn finalize(&mut self) -> ClipperConf {
        // let models = self.models.drain(..).collect()
        ClipperConf {
            name: self.name.clone(),
            slo_micros: self.slo_micros,
            num_message_encodes: self.num_message_encodes,
            policy_name: self.policy_name.clone(),
            models: self.models.drain(..).collect(),
            use_lsh: self.use_lsh,
            salt_update_cache: self.salt_update_cache,
            num_predict_workers: self.num_predict_workers,
            num_update_workers: self.num_update_workers,
            cache_size: self.cache_size,
            input_type: self.input_type.clone(),
            window_size: self.window_size,
            redis_ip: self.redis_ip.clone(),
            redis_port: self.redis_port,
            batch_strategy: self.batch_strategy.clone(),
            metrics: Arc::new(RwLock::new(metrics::Registry::new(self.name.clone()))),
        }
    }
}



#[cfg(test)]
#[cfg_attr(rustfmt, rustfmt_skip)]
mod tests {
    use super::*;

    #[test]
    fn toml_parse() {
        let toml_string = "
name = \"clipper-test\"
slo_micros = 10000
num_message_encodes = 2
correction_policy = \"hello_world\"
use_lsh = true
input_type = \"int\"
input_length = -1
window_size = -1

num_predict_workers = 4
num_update_workers = 2
# largest prime less than 50000
cache_size = 49999

[batching]
strategy = \"learned\"
sample_size = 2000
opt_addr = \"localhost:7777\"


[[models]]
name = \"m1\"
addresses = [\"127.0.0.1:6002\", \"127.0.0.1:7002\", \"127.0.0.1:8002\"]
num_outputs = 3
version = 7

[[models]]
name = \"m2\"
addresses = [\"127.0.0.1:6004\"]
version = 2
".to_string();

    let toml_conf = ClipperConf::parse_toml_string(&toml_string);
    let mut builder_conf = ClipperConfBuilder::new();
    let m1 = ModelConf::new("m1".to_string(),
                vec!["127.0.0.1:6002".to_string(),
                     "127.0.0.1:7002".to_string(),
                     "127.0.0.1:8002".to_string()], 3, 7);
    let m2 = ModelConf::new("m2".to_string(),
                vec!["127.0.0.1:6004".to_string()], 1, 2);

    let built_conf = builder_conf.cache_size(49999)
                                 .slo_micros(10000)
                                 .num_message_encodes(2)
                                 .name("clipper-test".to_string())
                                 .policy_name("hello_world".to_string())
                                 .use_lsh(true)
                                 .input_type("int".to_string(), Some(-1))
                                 .num_predict_workers(4)
                                 .num_update_workers(2)
                                 .add_model(m1)
                                 .add_model(m2)
                                 .window_size(-1)
                                 .batch_strategy(BatchStrategy::Learned {sample_size: 2000, opt_addr: get_addr("localhost:7777".to_string()) })
                                 .finalize();

    assert_eq!(toml_conf, built_conf);
    }


}
