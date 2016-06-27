use std::net::{ToSocketAddrs, SocketAddr, TcpStream, Shutdown};
use std::sync::{RwLock, Arc, mpsc};
use toml::{Parser, Array, Table, Value};
use metrics;
use server::InputType;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
use std::collections::HashSet;


pub struct ClipperConf {
    // General configuration
    pub name: String,
    pub slo_micros: u32,
    pub policy_name: String,
    pub models: Vec<ModelConf>,
    pub use_lsh: bool,
    pub input_type: InputType,
    // TODO training data
    // pub training_data_file: Option<String>,
    // TODO configurable output type
    //
    //
    // Internal system settings
    pub num_predict_workers: usize,
    pub num_update_workers: usize,
    pub cache_size: usize,
    pub metrics: Arc<RwLock<metrics::Registry>>,
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
        ClipperConf::parse_toml_string(toml_string)
    }

    fn parse_toml_string(toml_string: &String) -> ClipperConf {

        let pc = Parser::new(&toml_string).parse().unwrap();
        let conf = ClipperConf {
            name: pc.get("name")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
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
                                                     .unwrap()
                                                     .as_slice()
                                                     .unwrap()),
            use_lsh: pc.get("use_lsh")
                       .unwrap_or(&Value::Boolean(false))
                       .as_bool()
                       .unwrap(),
            input_type: match pc.get("input_type").unwrap().as_str().unwrap().as_slice() {
                "int" | "ints" | "integer" | "integers" | "i32" => {
                    let length = pc.get("input_length")
                                   .unwrap_or(&Value::Integer(-1))
                                   .as_integer()
                                   .unwrap() as i32;
                    InputType::Integer(length)
                }
                "float" | "floats" | "f64" => {
                    let length = pc.get("input_length")
                                   .unwrap_or(&Value::Integer(-1))
                                   .as_integer()
                                   .unwrap() as i32;
                    InputType::Float(length)
                }
                "str" | "string" => InputType::Str,
                "byte" | "bytes" | "u8" => {
                    let length = pc.get("input_length")
                                   .unwrap_or(&Value::Integer(-1))
                                   .as_integer()
                                   .unwrap() as i32;
                    InputType::Byte(length)
                }
            },
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
}




pub struct ModelConf {
    pub name: String,
    pub addresses: Vec<SocketAddr>,
    /// The dimension of the output vector this model produces
    pub num_outputs: usize,
}

impl ModelConf {
    fn from_toml(mt: &Table) -> ModelConf {
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
        }
    }
}

pub fn get_addr(a: String) -> SocketAddr {
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


impl ClipperConf {
    pub fn from_toml(fname: &str) -> ClipperConf {}
}

pub struct ClipperConfBuilder {
    // General configuration
    pub name: String,
    pub slo_micros: u32,
    pub policy_name: String,
    pub models: Vec<ModelConf>,
    pub use_lsh: bool,
    pub input_type: InputType,

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
            policy_name: "default".to_string(),
            models: Vec::new(),
            use_lsh: false,
            input_type: InputType::Integer(-1),
            num_predict_workers: 2,
            num_update_workers: 1,
            cache_size: 49999,
        }
    }

    pub fn cache_size(&mut self, s: usize) -> &mut ClipperConfBuilder {
        self.cache_size = s;
        self
    }

    pub fn slo_micros(&mut self, m: u32) -> &mut ClipperConfBuilder {
        self.slo_micros = m;
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
    pub fn finalize(self) -> ClipperConf {
        let name = self.name;
        ClipperConf {
            name: name.clone(),
            slo_micros: self.slo_micros,
            policy_name: self.policy_name,
            models: self.models,
            use_lsh: self.use_lsh,
            num_predict_workers: self.num_predict_workers,
            num_update_workers: self.num_update_workers,
            cache_size: self.cache_size,
            input_type: self.input_type,
            metrics: Arc::new(RwLock::new(metrics::Registry::new(name.clone()))),
        }
    }
}
