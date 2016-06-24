
use std::io::{Read, Write};
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::net::SocketAddr;
use std::boxed::Box;
use serde::ser::Serialize;
use serde::de::Deserialize;

#[allow(unused_imports)]
use hyper::{Get, Post, StatusCode, RequestUri, Decoder, Encoder, Next, Control};
use hyper::header::ContentLength;
use hyper::net::HttpStream;
use hyper::server::{Server, Handler, Request, Response};

use clipper::server::{self, ClipperServer, InputType};
use clipper::{batching, metrics, configuration};
use clipper::correction_policy::CorrectionPolicy;






const PREDICT: &'static str = "/predict";
const UPDATE: &'static str = "/update";


struct PredictHandler<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    clipper: Arc<ClipperServer<P, S>>,
    result_string: String,
    result_channel: Option<mpsc::Receiver<String>>,
    // num_features: usize,
    ctrl: Control,
    uid: u32,
    input_type: InputType,
}

struct UpdateHandler<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    clipper: Arc<ClipperServer<P, S>>,
    // num_features: usize,
    ctrl: Control,
    uid: u32,
    input_type: InputType,
}

impl<P, S> PredictHandler<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn new(clipper: Arc<ClipperServer<P, S>>,
           num_features: usize,
           ctrl: Control,
           input_type: InputType)
           -> PredictHandler<P, S> {
        PredictHandler {
            clipper: clipper,
            result_string: "NO RESULT YET".to_string(),
            result_channel: None,
            num_features: num_features,
            ctrl: ctrl,
            uid: 0,
            input_type: input_type,
        }
    }
}


fn parse_to_floats(transport: &mut Decoder<HttpStream>,
                   length: i32)
                   -> Result<server::Input, String> {
    let mut request_str = String::new();
    transport.read_to_string(&mut request_str).unwrap();
    let parsed_floats: Vec<f64> = request_str.split(", ")
                                             .map(|x| x.trim().parse::<f64>().unwrap())
                                             .collect();
    if length >= 0 && parsed_floats.len() as i32 != length {
        Err(format!("input wrong length: expected {}, found {}",
                    length,
                    parsed_floats.len()))
    } else {
        Ok(server::Input::Floats {
            f: parsed_floats,
            length: length,
        })
    }
}

fn parse_to_ints(transport: &mut Decoder<HttpStream>,
                 length: i32)
                 -> Result<server::Input, String> {
    let mut request_str = String::new();
    info!("AAA");
    transport.read_to_string(&mut request_str).unwrap();
    info!("BBB");
    let splits = request_str.split(", ").collect::<Vec<&str>>();
    info!("{:?}", splits);
    let parsed_ints: Vec<i32> = request_str.split(", ")
                                           .map(|x| x.trim().parse::<i32>().unwrap())
                                           .collect();
    if length >= 0 && parsed_ints.len() as i32 != length {
        Err(format!("input wrong length: expected {}, found {}",
                    length,
                    parsed_ints.len()))
    } else {
        Ok(server::Input::Ints {
            i: parsed_ints,
            length: length,
        })
    }
}

fn parse_to_string(transport: &mut Decoder<HttpStream>) -> Result<server::Input, String> {
    let mut request_str = String::new();
    transport.read_to_string(&mut request_str).unwrap();
    Ok(server::Input::Str { s: request_str })
}


impl<T: TaskModel + Send + Sync + 'static> Handler<HttpStream> for PredictHandler<T> {
    fn on_request(&mut self, req: Request) -> Next {
        match *req.uri() {
            RequestUri::AbsolutePath(ref path) => {
                match (req.method(), &path[0..PREDICT.len()]) {
                    (&Post, PREDICT) => {
                        let query_str: Vec<&str> = path.split("?").collect();
                        self.uid = if query_str.len() > 1 {
                            let q = query_str[1];
                            let sp: Vec<&str> = q.split("=").collect();
                            assert!(sp.len() == 2 && sp[0] == "uid");
                            let provided_uid = sp[1].parse::<u32>().unwrap();
                            assert!(provided_uid != 0, "Cannot provide user ID 0");
                            provided_uid
                        } else {
                            0
                        };
                        info!("UID: {}", self.uid);
                        Next::read()
                    }
                    _ => Next::write(),
                }
            }
            _ => Next::write(),
        }
    }

    fn on_request_readable(&mut self, transport: &mut Decoder<HttpStream>) -> Next {
        let input = match self.input_type {
            InputType::Integer(length) => parse_to_ints(transport, length),
            InputType::Float(length) => parse_to_floats(transport, length),
            InputType::Byte(_) => panic!("unsupported input type"),
            InputType::Str => parse_to_string(transport),
        };
        match input {
            Ok(i) => {
                // info!("query input: {:?}", i);
                let ctrl = self.ctrl.clone();
                let (tx, rx) = mpsc::channel::<String>();
                self.result_channel = Some(rx);
                let on_pred = Box::new(move |y| {
                    tx.send(format!("predict: {}", y).to_string()).unwrap();
                    ctrl.ready(Next::write()).unwrap();
                });
                let r = server::PredictRequest::new(self.uid, i, 0_i32, on_pred);
                self.dispatcher.dispatch(r, self.num_features);
                Next::wait()
            }
            Err(e) => {
                self.result_string = e.to_string();
                Next::write()
            }
        }
    }


    fn on_response(&mut self, res: &mut Response) -> Next {
        self.result_string = match self.result_channel {
            Some(ref c) => c.recv().unwrap(),
            None => {
                warn!("query failed for some reason");
                self.result_string.clone()
            }
        };
        // self.result_string = self.result_channel.recv().unwrap();
        res.headers_mut().set(ContentLength(self.result_string.as_bytes().len() as u64));
        Next::write()
    }

    fn on_response_writable(&mut self, transport: &mut Encoder<HttpStream>) -> Next {
        transport.write(self.result_string.as_bytes()).unwrap();
        info!("{}", self.result_string);
        Next::end()
    }
}

#[allow(dead_code)]
fn launch_monitor_thread(metrics_register: Arc<RwLock<metrics::Registry>>,
                         report_interval_secs: u64)
                         -> ::std::thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            thread::sleep(::std::time::Duration::new(report_interval_secs, 0));
            let m = metrics_register.read().unwrap();
            info!("{}", m.report());
            m.reset();
            // thread::sleep(::std::time::Duration::new(report_interval_secs, 0));
            // info!("{}", metrics_register.read().unwrap().report());
        }
    })
}


fn start_listening<P, S>(clipper: Arc<ClipperServer<P, S>>)
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{

    let rest_server = Server::http(&"127.0.0.1:1337".parse().unwrap()).unwrap();

    let report_interval_secs = 15;
    let _ = launch_monitor_thread(metrics.clone(), report_interval_secs);


    let (listening, server) = rest_server.handle(|ctrl| {
                                             PredictHandler::new(clipper_server,
                                                                 ctrl,
                                                                 input_type.clone())
                                         })
                                         .unwrap();
    println!("Listening on http://{}", listening);
    server.run();

}

// pub fn start_listening(feature_addrs: Vec<(String, Vec<SocketAddr>)>, input_type: InputType) {
pub fn start(conf_path: &String) {

    let config = ClipperConf::parse_from_toml(conf_path);
    let metrics = config.metrics.clone();
    let input_type = config.input_type.clone();

    match config.policy_name {
        "hello_world" => {
            start_listening(Arc::new(ClipperServer::<DummyCorrectionPolicy, Vec<f64>>::new(config)))
        }
        _ => panic!("Unknown correction policy"),
    }
}
