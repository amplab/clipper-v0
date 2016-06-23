use std::net::{ToSocketAddrs, SocketAddr, TcpStream, Shutdown};


pub struct ClipperConf {
    // General configuration
    pub name: String,
    pub slo_micros: u32,
    pub policy_name: String,
    pub models: Vec<ModelConf>,
    pub use_lsh: bool,
    pub input_type: InputType,

    // Internal system settings
    pub num_predict_workers: u32,
    pub num_observe_workers: u32,
    pub metrics: metrics::Registry,
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

        let conf = toml::Parser::new(&toml_string).parse().unwrap();
        unimplemented!();
    }
}


pub struct ModelConf {
    pub name: String,
    pub addresses: Vec<SocketAddr>,
    /// The dimension of the output vector this model produces
    pub num_outputs: usize,
}

impl ModelConf {
    fn from_toml_table() {
        unimplemented!();
    }
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
    pub num_predict_workers: u32,
    pub num_observe_workers: u32,
}

impl ClipperConfBuilder {
    pub fn new() -> ClipperConfBuilder {
        ClipperConfBuilder {
            slo_micros: 20 * 1000,
            policy_name: "default",
            models: Vec::new(),
            use_lsh: false,
            input_type: InputType::Integer(-1),
            num_predict_workers: 2,
            num_observe_workers: 1,
        }
    }

    pub fn slo_micros(&mut self, m: u32) -> &mut ClipperConfBuilder {
        self.slo_micros = m;
        self
    }

    pub fn policy_name(&mut self, name: String) -> &mut ClipperConfBuilder {
        self.policy_name = name;
        self
    }

    pub fn use_lsh(&mut self, m: bool) -> &mut ClipperConfBuilder {
        self.use_lsh = l;
        self
    }

    pub fn add_model(&mut self, m: ModelConf) -> &mut ClipperConfBuilder {
        self.models.push(m);
        self
    }

    pub fn num_predict_workers(&mut self, n: u32) -> &mut ClipperConfBuilder {
        self.num_predict_workers = n;
        self
    }

    pub fn num_update_workers(&mut self, n: u32) -> &mut ClipperConfBuilder {
        self.num_update_workers = n;
        self
    }

    pub fn name(&mut self, n: String) -> &mut ClipperConfBuilder {
        self.name = n;
        self
    }

    pub fn input_type(&mut self, name: String, length: Option<i32>) -> &mut ClipperConfBuilder {
        let lc_name = name.to_lowercase();
        let input_type = match lc_name {
            "int" | "ints" | "integer" | "integers" | "i32" => {
                match length {
                    Some(l) => InputType::Integer(l),
                    None => InputType::Integer(-1),
                }
            }
            "float" | "floats" | "f64" => {
                match length {
                    Some(l) => InputType::Float(l),
                    None => InputType::Float(-1),
                }
            }
            "str" | "string" => {
                if length.is_some() {
                    info!("length arg provided for string is ignored");
                }
                InputType::Str
            }
            "byte" | "bytes" | "u8" => {
                match length {
                    Some(l) => InputType::Byte(l),
                    None => InputType::Byte(-1),
                }
            }
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
            num_observe_workers: self.num_update_workers,
            metrics: metrics::Registry::new(name.clone()),
        }
    }
}
