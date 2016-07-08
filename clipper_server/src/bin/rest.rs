
use std::io::{Read, Write};
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::boxed::Box;
use serde::ser::Serialize;
use serde::de::Deserialize;
use std::error::Error;
use std::time::Duration;
use serde_json;

#[allow(unused_imports)]
use hyper::{Get, Post, StatusCode, RequestUri, Decoder, Encoder, Next, Control};
use hyper::header::{ContentType, ContentLength};
use hyper::mime::{Mime, TopLevel, SubLevel};
use hyper::net::HttpStream;
use hyper::server::{Server, Handler, Request, Response};

use clipper::server::{Input, ClipperServer, InputType, PredictionRequest, UpdateRequest, Output};
use clipper::{metrics, configuration};
use clipper::correction_policy::{CorrectionPolicy, DummyCorrectionPolicy};







const PREDICT: &'static str = "/predict";
const UPDATE: &'static str = "/update";


struct RequestHandler<P, S>
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
    request_type: Option<RequestType>,
}

enum RequestType {
    Predict,
    Update,
}

#[derive(Serialize,Deserialize)]
struct IntsInput {
    uid: u32,
    input: Vec<i32>,
    label: Option<Output>,
}
#[derive(Serialize,Deserialize)]
struct FloatsInput {
    uid: u32,
    input: Vec<f64>,
    label: Option<Output>,
}
#[derive(Serialize,Deserialize)]
struct StrInput {
    uid: u32,
    input: String,
    label: Option<Output>,
}
#[derive(Serialize,Deserialize)]
struct BytesInput {
    uid: u32,
    input: Vec<u8>,
    label: Option<Output>,
}


impl<P, S> RequestHandler<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn new(clipper: Arc<ClipperServer<P, S>>,
           ctrl: Control,
           input_type: InputType)
           -> RequestHandler<P, S> {
        RequestHandler {
            clipper: clipper,
            result_string: "NO RESULT YET".to_string(),
            result_channel: None,
            // num_features: num_features,
            ctrl: ctrl,
            uid: 0,
            input_type: input_type,
            request_type: None,
        }
    }
}


// fn parse_to_floats(transport: &mut Decoder<HttpStream>, length: i32) -> Result<Input, String> {
//     let mut request_str = String::new();
//     transport.read_to_string(&mut request_str).unwrap();
//     let parsed_floats: Vec<f64> = request_str.split(", ")
//                                              .map(|x| x.trim().parse::<f64>().unwrap())
//                                              .collect();
//     if length >= 0 && parsed_floats.len() as i32 != length {
//         Err(format!("input wrong length: expected {}, found {}",
//                     length,
//                     parsed_floats.len()))
//     } else {
//         Ok(Input::Floats {
//             f: parsed_floats,
//             length: length,
//         })
//     }
// }

// fn parse_to_ints(transport: &mut Decoder<HttpStream>, length: i32) -> Result<Input, String> {
//     let mut request_str = String::new();
//     transport.read_to_string(&mut request_str).unwrap();
//     let splits = request_str.split(", ").collect::<Vec<&str>>();
//     info!("{:?}", splits);
//     let parsed_ints: Vec<i32> = request_str.split(", ")
//                                            .map(|x| x.trim().parse::<i32>().unwrap())
//                                            .collect();
//     if length >= 0 && parsed_ints.len() as i32 != length {
//         Err(format!("input wrong length: expected {}, found {}",
//                     length,
//                     parsed_ints.len()))
//     } else {
//         Ok(Input::Ints {
//             i: parsed_ints,
//             length: length,
//         })
//     }
// }

// fn parse_to_string(transport: &mut Decoder<HttpStream>) -> Result<Input, String> {
//     let mut request_str = String::new();
//     transport.read_to_string(&mut request_str).unwrap();
//     Ok(Input::Str { s: request_str })
// }

// fn extract_uid_from_path(path: &String) -> u32 {
//
//     let query_str: Vec<&str> = path.split("?").collect();
//     let uid = if query_str.len() > 1 {
//         let q = query_str[1];
//         let sp: Vec<&str> = q.split("=").collect();
//         assert!(sp.len() == 2 && sp[0] == "uid");
//         let provided_uid = sp[1].parse::<u32>().unwrap();
//         assert!(provided_uid != 0, "Cannot provide user ID 0");
//         provided_uid
//     } else {
//         0
//     };
//     uid
// }


fn decode_predict_input(input_type: &InputType,
                        json_string: &String)
                        -> Result<(u32, Input), String> {
    match input_type {
        &InputType::Integer(length) => {
            let i: IntsInput = try!(serde_json::from_str(&json_string)
                                        .map_err(|e| format!("{}", e.description())));
            if length >= 0 && i.input.len() != length as usize {
                return Err(format!("Wrong input length: expected {}, received {}",
                                   length,
                                   i.input.len()));
            }
            Ok((i.uid,
                Input::Ints {
                i: i.input,
                length: length,
            }))
        }
        &InputType::Float(length) => {
            let i: FloatsInput = try!(serde_json::from_str(&json_string)
                                          .map_err(|e| format!("{}", e.description())));
            if length >= 0 && i.input.len() != length as usize {
                return Err(format!("Wrong input length: expected {}, received {}",
                                   length,
                                   i.input.len()));
            }
            Ok((i.uid,
                Input::Floats {
                f: i.input,
                length: length,
            }))
        }
        &InputType::Byte(length) => {
            let i: BytesInput = try!(serde_json::from_str(&json_string)
                                         .map_err(|e| format!("{}", e.description())));
            if length >= 0 && i.input.len() != length as usize {
                return Err(format!("Wrong input length: expected {}, received {}",
                                   length,
                                   i.input.len()));
            }
            Ok((i.uid,
                Input::Bytes {
                b: i.input,
                length: length,
            }))
        }
        &InputType::Str => {
            let i: StrInput = try!(serde_json::from_str(&json_string)
                                       .map_err(|e| format!("{}", e.description())));
            Ok((i.uid, Input::Str { s: i.input }))
        }
    }
}

fn decode_update_input(input_type: &InputType,
                       json_string: &String)
                       -> Result<(u32, Input, Output), String> {
    match input_type {
        &InputType::Integer(length) => {
            let i: IntsInput = try!(serde_json::from_str(&json_string)
                                        .map_err(|e| format!("{}", e.description())));
            if length >= 0 && i.input.len() != length as usize {
                return Err(format!("Wrong input length: expected {}, received {}",
                                   length,
                                   i.input.len()));
            }
            if i.label.is_none() {
                return Err(format!("No label for update"));
            }
            Ok((i.uid,
                Input::Ints {
                i: i.input,
                length: length,
            },
                i.label.unwrap()))
        }
        &InputType::Float(length) => {
            let i: FloatsInput = try!(serde_json::from_str(&json_string)
                                          .map_err(|e| format!("{}", e.description())));
            if length >= 0 && i.input.len() != length as usize {
                return Err(format!("Wrong input length: expected {}, received {}",
                                   length,
                                   i.input.len()));
            }
            if i.label.is_none() {
                return Err(format!("No label for update"));
            }
            Ok((i.uid,
                Input::Floats {
                f: i.input,
                length: length,
            },
                i.label.unwrap()))
        }
        &InputType::Byte(length) => {
            let i: BytesInput = try!(serde_json::from_str(&json_string)
                                         .map_err(|e| format!("{}", e.description())));
            if length >= 0 && i.input.len() != length as usize {
                return Err(format!("Wrong input length: expected {}, received {}",
                                   length,
                                   i.input.len()));
            }
            if i.label.is_none() {
                return Err(format!("No label for update"));
            }
            Ok((i.uid,
                Input::Bytes {
                b: i.input,
                length: length,
            },
                i.label.unwrap()))
        }
        &InputType::Str => {
            let i: StrInput = try!(serde_json::from_str(&json_string)
                                       .map_err(|e| format!("{}", e.description())));
            if i.label.is_none() {
                return Err(format!("No label for update"));
            }
            Ok((i.uid, Input::Str { s: i.input }, i.label.unwrap()))
        }
    }
}

impl<P, S> Handler<HttpStream> for RequestHandler<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    fn on_request(&mut self, req: Request<HttpStream>) -> Next {

        // check content type
        let headers = req.headers();
        match headers.get::<ContentType>() {
            Some(&ContentType(ref mime)) => {
                match mime {
                    &Mime(TopLevel::Application, SubLevel::Json, _) => {}
                    _ => {
                        self.result_string = format!("Incorrect mime type. Expected \
                                                      application/json, found: {}",
                                                     mime)
                    }
                }
            }
            None => warn!("no ContentType header found. Assuming application/json."),
        }

        match *req.uri() {
            RequestUri::AbsolutePath(ref path) => {
                match (req.method(), &path[..]) {
                    (&Post, PREDICT) => {
                        // self.uid = extract_uid_from_path(path);
                        self.request_type = Some(RequestType::Predict);
                        Next::read()
                    }
                    (&Post, UPDATE) => {
                        // self.uid = extract_uid_from_path(path);
                        self.request_type = Some(RequestType::Update);
                        Next::read()
                    }
                    _ => Next::write(),
                }
            }
            _ => Next::write(),
        }
    }

    fn on_request_readable(&mut self, transport: &mut Decoder<HttpStream>) -> Next {
        let mut json_string = String::new();
        transport.read_to_string(&mut json_string).unwrap();
        match self.request_type {
            Some(RequestType::Predict) => {
                match decode_predict_input(&self.input_type, &json_string) {
                    Ok((uid, input)) => {
                        self.uid = uid;
                        info!("/predict for user: {}", self.uid);
                        let ctrl = self.ctrl.clone();
                        let (tx, rx) = mpsc::channel::<String>();
                        self.result_channel = Some(rx);
                        let on_pred = Box::new(move |y| {
                            tx.send(format!("predict: {}", y).to_string()).unwrap();
                            ctrl.ready(Next::write()).unwrap();
                        });
                        let r = PredictionRequest::new(self.uid, input, on_pred);
                        self.clipper.schedule_prediction(r);
                        Next::wait()
                    }
                    Err(e) => {
                        self.result_string = e.to_string();
                        Next::write()
                    }
                }
            }
            Some(RequestType::Update) => {
                match decode_update_input(&self.input_type, &json_string) {
                    Ok((uid, input, label)) => {
                        self.uid = uid;
                        info!("/update for user: {}", self.uid);
                        let u = UpdateRequest::new(self.uid, input, label);
                        self.clipper.schedule_update(u);
                        self.result_string = "Update scheduled".to_string();
                        Next::write()
                    }
                    Err(e) => {
                        self.result_string = e.to_string();
                        Next::write()
                    }
                }
            }
            None => Next::write(),
        }
    }


    fn on_response(&mut self, res: &mut Response) -> Next {
        match self.request_type {
            Some(RequestType::Predict) => {
                self.result_string = match self.result_channel {
                    Some(ref c) => c.recv().unwrap(),
                    None => {
                        warn!("query failed for some reason");
                        self.result_string.clone()
                    }
                };
            }
            _ => {}
        }
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
                         report_interval_secs: u64,
                         shutdown_signal_rx: mpsc::Receiver<()>)
                         -> ::std::thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            match shutdown_signal_rx.try_recv() {
                Ok(_) | Err(mpsc::TryRecvError::Empty) => {
                    thread::sleep(Duration::new(report_interval_secs, 0));
                    let m = metrics_register.read().unwrap();
                    info!("{}", m.report());
                    m.reset();
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        info!("Shutting down metrics thread");
    })
}


#[allow(unused_variables)]
fn start_listening<P, S>(shutdown_signal: mpsc::Receiver<()>, clipper: Arc<ClipperServer<P, S>>)
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{

    let rest_server = Server::http(&"127.0.0.1:1337".parse().unwrap()).unwrap();

    // TODO: add admin server to update models
    // let admin_server = Server::http(&"127.0.0.1:1338".parse().unwrap()).unwrap();

    let report_interval_secs = 15;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();
    let _ = launch_monitor_thread(clipper.get_metrics(),
                                  report_interval_secs,
                                  metrics_signal_rx);
    let input_type = clipper.get_input_type();

    let (listening, server) = rest_server.handle(|ctrl| {
                                             RequestHandler::new(clipper.clone(),
                                                                 ctrl,
                                                                 input_type.clone())
                                         })
                                         .unwrap();

    let jh = thread::spawn(move || {
        println!("Listening on http://{}", listening);
        shutdown_signal.recv().unwrap();
        metrics_signal_tx.send(()).unwrap();
        listening.close();
    });
    server.run();
    println!("Done running");
    jh.join().unwrap();

}

// pub fn start_listening(feature_addrs: Vec<(String, Vec<SocketAddr>)>, input_type: InputType) {
pub fn start(shutdown_signal: mpsc::Receiver<()>, conf_path: &String) {

    let config = configuration::ClipperConf::parse_from_toml(conf_path);
    // let metrics = config.metrics.clone();
    // let input_type = config.input_type.clone();

    if config.policy_name == "hello world".to_string() {
        start_listening(shutdown_signal,
                        Arc::new(ClipperServer::<DummyCorrectionPolicy, Vec<f64>>::new(config)));
    } else {
        panic!("Unknown correction policy");
    }
}
