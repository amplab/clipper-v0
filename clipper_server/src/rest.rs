
use std::io::{Read, Write};
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::net::SocketAddr;
use std::boxed::Box;

#[allow(unused_imports)]
use hyper::{Get, Post, StatusCode, RequestUri, Decoder, Encoder, Next, Control};
use hyper::header::ContentLength;
use hyper::net::HttpStream;
use hyper::server::{Server, Handler, Request, Response};

use server::{self,TaskModel};
use features;
use metrics;




enum InputType {
    Integer(usize),
    Float(usize),
    Str,
    Bytes(usize),
}


const PREDICT: &'static str = "/predict";
// const UPDATE: &'static str = "/update";


struct PredictHandler<T>
where T: TaskModel + Send + Sync + 'static {
    dispatcher: Arc<server::Dispatcher<T>>,
    result_string: String,
    result_channel: Option<mpsc::Receiver<String>>,
    num_features: usize,
    ctrl: Control,
    uid: u32,
    input_type: InputType
}

impl<T> PredictHandler<T>
where T: TaskModel + Send + Sync + 'static {
    fn new(dispatcher: Arc<server::Dispatcher<T>>,
           num_features: usize, ctrl: Control, input_type: InputType) -> PredictHandler<T> {
        PredictHandler {
            dispatcher: dispatcher,
            result_string: "NO RESULT YET".to_string(),
            result_channel: None,
            num_features: num_features,
            ctrl: ctrl,
            uid: 0,
            input_type: input_type,
        }
    }
}


fn parse_to_floats(transport: &mut Decoder<HttpStream>, length: i32) -> Result<server::Input, String> {
    let mut request_str = String::new();
    try!(transport.read_to_string(&mut request_str));
    let parsed_floats = request_str.split(", ").map(|x| x.parse::<f64>().unwrap()).collect();
    if length >= 0 && parsed_floats.len() != length {
        Err(format!("input wrong length: expected {}, found {}",
                    length,
                    parsed_floats.len()));
    } else {
        Ok(server::Input::Floats {f: parsed_floats, length: length})
    }
}

fn parse_to_ints(transport: &mut Decoder<HttpStream>, length: i32) -> Result<server::Input, String> {
    let mut request_str = String::new();
    try!(transport.read_to_string(&mut request_str));
    let parsed_ints = request_str.split(", ").map(|x| x.parse::<i32>().unwrap()).collect();
    if length >= 0 && parsed_ints.len() != length {
        Err(format!("input wrong length: expected {}, found {}",
                    length,
                    parsed_ints.len()));
    } else {
        Ok(server::Input::Ints {i: parsed_ints, length: length})
    }
}

fn parse_to_string(transport: &mut Decoder<HttpStream>, length: i32) -> Result<server::Input, String> {
    let mut request_str = String::new();
    try!(transport.read_to_string(&mut request_str));
    Ok(server::Input::Str {s: request_str})
}


impl<T: TaskModel + Send + Sync + 'static> Handler<HttpStream> for PredictHandler<T> {
    fn on_request(&mut self, req: Request) -> Next {
        match *req.uri() {
            RequestUri::AbsolutePath(ref path) => match (req.method(), &path[0..PREDICT.len()]) {
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
                _ => Next::write()
            },
            _ => Next::write()
        }
    }

    fn on_request_readable(&mut self, transport: &mut Decoder<HttpStream>) -> Next {
        let mut request_str = String::new();
        let input = match self.input_type {
            InputType::Integer(length) => parse_to_ints(transport, length),
            InputType::Float(length) => parse_to_floats(transport, length),
            InputType::Bytes(length) => panic!("unsupported input type"),
            InputType::Str => parse_to_string(transport),
        };
        match input {
            Ok(i) => {
                info!("query input: {:?}", i);
                let ctrl = self.ctrl.clone();
                let (tx, rx) = mpsc::channel::<String>();
                self.result_channel = Some(rx);
                let on_pred = Box::new(move |y| {
                    tx.send(format!("predict: {}", y).to_string()).unwrap();
                    ctrl.ready(Next::write()).unwrap();
                });
                let r = server::PredictRequest::new(self.uid,
                                                    i,
                                                    0_i32,
                                                    on_pred);
                self.dispatcher.dispatch(r, self.num_features);
                Next::wait()
            },
            Err(e) => {
                self.result_string = e;
                Next::write()
            }
        }
    }


    fn on_response(&mut self, res: &mut Response) -> Next {
        self.result_string = match self.result_channel {
            Some(ref c) => c.recv().unwrap(),
            None => {
                warn!("query failed for some reason");
                self.result_string
            },
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


fn launch_monitor_thread(metrics_register: Arc<RwLock<metrics::Registry>>,
                         report_interval_secs: u64) -> ::std::thread::JoinHandle<()> {
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


pub fn start_listening(feature_addrs: Vec<(String, Vec<SocketAddr>)>, input_type: InputType) {
    let server = Server::http(&"127.0.0.1:1337".parse().unwrap()).unwrap();

    let metrics_register = 
      Arc::new(RwLock::new(metrics::Registry::new("CLIPPER REST API".to_string())));

    let (features, _): (Vec<_>, Vec<_>) = feature_addrs.into_iter()
                              .map(|(n, a)| features::create_feature_worker(
                                  n, a, 0, metrics_register.clone()))
                              .unzip();

    let num_workers = 2;
    let num_features = features.len();
    let num_users = 100;
    let report_interval_secs = 15;

    let dispatcher = Arc::new(server::Dispatcher::new(num_workers,
                                         server::SLA,
                                         features.clone(),
                                         server::init_user_models(num_users, num_features),
                                         metrics_register.clone()));

    // let _ = launch_monitor_thread(metrics_register.clone(),
    //                                                    report_interval_secs);

    let (listening, server) = server.handle(|ctrl| 
        PredictHandler::new(dispatcher.clone(),
                            num_features, ctrl,
                            input_type)).unwrap();
    println!("Listening on http://{}", listening);
    server.run();
}

