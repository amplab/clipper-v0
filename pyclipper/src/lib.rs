#![deny(warnings)]

extern crate libc;
extern crate clipper;
extern crate toml;
#[macro_use]
extern crate log;

// use std::io::{Read, Write};
// use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::boxed::Box;
use std::slice;
use std::str;
use std::ffi::CStr;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;
use std::net::SocketAddr;

use libc::{uint32_t, c_char};

use clipper::server::{self, TaskModel, InputType};
use clipper::{features, metrics};


#[allow(dead_code)]
pub struct PyClipper {
    metrics: Arc<RwLock<metrics::Registry>>,
    dispatcher: Arc<server::Dispatcher<server::DummyTaskModel>>,
    result_rx: mpsc::Receiver<f64>,
    result_tx: mpsc::Sender<f64>,
    num_features: usize,
    input_type: InputType,
}

impl PyClipper {
    pub fn new(feature_addrs: Vec<(String, Vec<SocketAddr>)>, input_type: InputType) -> PyClipper {
        let metrics_register = Arc::new(RwLock::new(metrics::Registry::new("PyClipper"
                                                                               .to_string())));

        let (features, _): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                                                           .map(|(n, a)| {
                                                               features::create_feature_worker(
                                  n, a, 0, metrics_register.clone(), input_type.clone())
                                                           })
                                                           .unzip();

        let num_workers = 2;
        let num_features = features.len();
        let num_users = 100;

        let dispatcher = Arc::new(server::Dispatcher::new(num_workers,
                                                          server::SLA,
                                                          features.clone(),
                                                          server::init_user_models(num_users,
                                                                                   num_features),
                                                          metrics_register.clone()));
        let (tx, rx) = mpsc::channel::<f64>();

        PyClipper {
            metrics: metrics_register,
            dispatcher: dispatcher,
            result_rx: rx,
            result_tx: tx,
            num_features: num_features,
            input_type: input_type,
        }
    }

    pub fn predict(&self, uid: u32, input: server::Input) -> f64 {
        let tx = self.result_tx.clone();
        let on_pred = Box::new(move |y| {
            tx.send(y).unwrap();
        });
        let r = server::PredictRequest::new(uid, input, 0_i32, on_pred);
        self.dispatcher.dispatch(r, self.num_features);
        let prediction = self.result_rx.recv().unwrap();
        info!("Predicted: {}", prediction);
        prediction
    }
}

#[no_mangle]
pub extern "C" fn init_clipper(config: *const c_char) -> *mut PyClipper {
    let config_loc = unsafe {
        assert!(!config.is_null());
        CStr::from_ptr(config)
    };
    let config_str = str::from_utf8(config_loc.to_bytes()).unwrap().to_string();
    let model_wrappers = parse_feature_config(&config_str);
    Box::into_raw(Box::new(PyClipper::new(model_wrappers, InputType::Float(7))))
}

#[no_mangle]
pub extern "C" fn pyclipper_free(ptr: *mut PyClipper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
pub extern "C" fn pyclipper_predict(ptr: *mut PyClipper, input: *const f64, len: uint32_t) -> f64 {
    let clipper = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let x: Vec<f64> = unsafe {
        assert!(!input.is_null());
        let slice: &[f64] = slice::from_raw_parts(input, len as usize);
        let mut v = vec![0.0_f64; len as usize];
        v[..].clone_from_slice(slice);
        v
    };
    clipper.predict(0_u32,
                    server::Input::Floats {
                        f: x,
                        length: len as i32,
                    })
}

fn parse_feature_config(fname: &String) -> Vec<(String, Vec<SocketAddr>)> {

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

    let features = toml::Parser::new(&toml_string).parse().unwrap();
    let fs = features.get("features")
                     .unwrap()
                     .as_table()
                     .unwrap()
                     .iter()
                     .map(|(k, v)| {
                         (k.clone(),
                          features::get_addrs(v.as_slice().unwrap().to_vec()))
                     })
                     .collect::<Vec<_>>();
    info!("{:?}", fs);
    fs
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {}
// }
