
// use std::io::{Read, Write};
use std::io::Read;
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
// use hyper::header::{ContentType, ContentLength};
use hyper::header::ContentLength;
// use hyper::mime::{Mime, TopLevel, SubLevel};
use hyper::net::HttpStream;
use hyper::server::{Server, Handler, Request, Response};

use clipper::server::{Input, ClipperServer, InputType, PredictionRequest, UpdateRequest, Update,
                      Output};
use clipper::{metrics, configuration};
use clipper::configuration::get_addrs_str;
use clipper::correction_policy::{CorrectionPolicy, DummyCorrectionPolicy,
                                 LogisticRegressionPolicy, LinearCorrectionState};








const PREDICT: &'static str = "/predict";
const UPDATE: &'static str = "/update";

// const ADMIN: &'static str = "/admin";
const ADDMODEL: &'static str = "/addmodel";
const ADDREPLICA: &'static str = "/addreplica";
const GETMETRICS: &'static str = "/metrics";



struct RequestHandler<P, S>
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{
    clipper: Arc<RwLock<ClipperServer<P, S>>>,
    result_string: String,
    result_channel: Option<mpsc::Receiver<String>>,
    // num_features: usize,
    ctrl: Control,
    uid: u32,
    input_type: InputType,
    request_type: Option<RequestType>,
    command: Option<AdminCommand>,
}

enum RequestType {
    Predict,
    Update,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
struct NewModelWrapperData {
    name: String,
    version: u32,
    addrs: Vec<String>,
}


enum AdminCommand {
    AddModel,
    AddReplica,
    GetMetrics,
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
    fn new(clipper: Arc<RwLock<ClipperServer<P, S>>>,
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
            command: None,
        }
    }
}


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

        // // check content type
        // let headers = req.headers();
        // match headers.get::<ContentType>() {
        //     Some(&ContentType(ref mime)) => {
        //         match mime {
        //             &Mime(TopLevel::Application, SubLevel::Json, _) => {}
        //             _ => {
        //                 self.result_string = format!("Incorrect mime type. Expected \
        //                                               application/json, found: {}",
        //                                              mime)
        //             }
        //         }
        //     }
        //     None => warn!("no ContentType header found. Assuming application/json."),
        // }

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
                    (&Post, ADDMODEL) => {
                        // self.uid = extract_uid_from_path(path);
                        self.command = Some(AdminCommand::AddModel);
                        Next::read()
                    }
                    (&Post, ADDREPLICA) => {
                        // self.uid = extract_uid_from_path(path);
                        self.command = Some(AdminCommand::AddReplica);
                        Next::read()
                    }
                    (&Get, GETMETRICS) => {
                        // self.uid = extract_uid_from_path(path);
                        self.command = Some(AdminCommand::GetMetrics);
                        Next::write()
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
        if self.request_type.is_some() {
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
                            // Don't salt the cache during normal operations
                            let salt = false;
                            let r = PredictionRequest::new(self.uid, input, on_pred, salt);
                            self.clipper.read().unwrap().schedule_prediction(r);
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
                            let u = UpdateRequest::new(self.uid,
                                                       vec![Update {
                                                                query: Arc::new(input),
                                                                label: label,
                                                            }]);
                            self.clipper.read().unwrap().schedule_update(u);
                            self.result_string = "Update scheduled".to_string();
                            Next::write()
                        }
                        Err(e) => {
                            self.result_string = e.to_string();
                            Next::write()
                        }
                    }
                }
                None => unreachable!(),
            }
        } else if self.command.is_some() {
            match self.command {
                Some(AdminCommand::AddModel) => {
                    match serde_json::from_str::<NewModelWrapperData>(&json_string) {
                        Ok(add_model_data) => {
                            let resolved_addrs = get_addrs_str(add_model_data.addrs);
                            let mut clipper_lock = self.clipper.write().unwrap();
                            clipper_lock.add_new_model(add_model_data.name,
                                                       add_model_data.version,
                                                       resolved_addrs);
                            self.result_string = "Success!".to_string();
                            Next::write()
                        }
                        Err(e) => {
                            self.result_string = e.description().to_string();
                            Next::write()
                        }
                    }
                }
                Some(AdminCommand::AddReplica) => {
                    match serde_json::from_str::<NewModelWrapperData>(&json_string) {
                        Ok(add_model_data) => {
                            let resolved_addrs = get_addrs_str(add_model_data.addrs);
                            let mut clipper_lock = self.clipper.write().unwrap();
                            for a in resolved_addrs {
                                clipper_lock.add_new_replica(add_model_data.name.clone(),
                                                             add_model_data.version,
                                                             a);
                            }
                            self.result_string = "Success!".to_string();
                            Next::write()
                        }
                        Err(e) => {
                            self.result_string = e.description().to_string();
                            Next::write()
                        }
                    }
                }
                _ => Next::write(),
            }
        } else {
            Next::write()
        }
    }


    fn on_response(&mut self, res: &mut Response) -> Next {
        if self.request_type.is_some() {
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
        } else if self.command.is_some() {
            match self.command {
                Some(AdminCommand::GetMetrics) => {
                    let clipper_read_lock = self.clipper.read().unwrap();
                    let metrics_register = clipper_read_lock.get_metrics();
                    let m = metrics_register.read().unwrap();
                    self.result_string = m.report();
                }
                _ => {}
            }
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


// struct AdminHandler<P, S>
//     where P: CorrectionPolicy<S>,
//           S: Serialize + Deserialize
// {
//     clipper: Arc<RwLock<ClipperServer<P, S>>>,
//     command: Option<AdminCommand>,
//     result_string: String,
// }
//
// impl<P, S> AdminHandler<P, S>
//     where P: CorrectionPolicy<S>,
//           S: Serialize + Deserialize
// {
//     fn new(clipper: Arc<RwLock<ClipperServer<P, S>>>) -> AdminHandler<P, S> {
//         AdminHandler {
//             clipper: clipper,
//             command: None,
//             result_string: "NO RESULT YET".to_string(),
//         }
//     }
// }


// impl<P, S> Handler<HttpStream> for AdminHandler<P, S>
//     where P: CorrectionPolicy<S>,
//           S: Serialize + Deserialize
// {
// fn on_request(&mut self, req: Request<HttpStream>) -> Next {
//     // let headers = req.headers();
//     // match headers.get::<ContentType>() {
//     //     Some(&ContentType(ref mime)) => {
//     //         match mime {
//     //             &Mime(TopLevel::Application, SubLevel::Json, _) => {}
//     //             _ => {
//     //                 self.result_string = format!("Incorrect mime type. Expected \
//     //                                               application/json, found: {}",
//     //                                              mime)
//     //             }
//     //         }
//     //     }
//     //     None => warn!("no ContentType header found. Assuming application/json."),
//     // }
//
//     match *req.uri() {
//         RequestUri::AbsolutePath(ref path) => {
//             match (req.method(), &path[..]) {
//                 (&Post, ADDMODEL) => {
//                     // self.uid = extract_uid_from_path(path);
//                     self.command = Some(AdminCommand::AddModel);
//                     Next::read()
//                 }
//                 (&Post, ADDREPLICA) => {
//                     // self.uid = extract_uid_from_path(path);
//                     self.command = Some(AdminCommand::AddReplica);
//                     Next::read()
//                 }
//                 (&Get, GETMETRICS) => {
//                     // self.uid = extract_uid_from_path(path);
//                     self.command = Some(AdminCommand::GetMetrics);
//                     Next::write()
//                 }
//                 _ => Next::write(),
//             }
//         }
//         _ => Next::write(),
//     }
// }

// fn on_request_readable(&mut self, transport: &mut Decoder<HttpStream>) -> Next {
//     let mut json_string = String::new();
//     transport.read_to_string(&mut json_string).unwrap();
//     match self.command {
//         Some(AdminCommand::AddModel) => {
//             match serde_json::from_str::<NewModelWrapperData>(&json_string) {
//                 Ok(add_model_data) => {
//                     let resolved_addrs = get_addrs_str(add_model_data.addrs);
//                     let mut clipper_lock = self.clipper.write().unwrap();
//                     clipper_lock.add_new_model(add_model_data.name,
//                                                add_model_data.version,
//                                                resolved_addrs);
//                     self.result_string = "Success!".to_string();
//                     Next::write()
//                 }
//                 Err(e) => {
//                     self.result_string = e.description().to_string();
//                     Next::write()
//                 }
//             }
//         }
//         Some(AdminCommand::AddReplica) => {
//             match serde_json::from_str::<NewModelWrapperData>(&json_string) {
//                 Ok(add_model_data) => {
//                     let resolved_addrs = get_addrs_str(add_model_data.addrs);
//                     let mut clipper_lock = self.clipper.write().unwrap();
//                     for a in resolved_addrs {
//                         clipper_lock.add_new_replica(add_model_data.name.clone(),
//                                                      add_model_data.version,
//                                                      a);
//                     }
//                     self.result_string = "Success!".to_string();
//                     Next::write()
//                 }
//                 Err(e) => {
//                     self.result_string = e.description().to_string();
//                     Next::write()
//                 }
//             }
//         }
//         _ => Next::write(),
//     }
// }

//     fn on_response(&mut self, res: &mut Response) -> Next {
//         match self.command {
//             Some(AdminCommand::GetMetrics) => {
//                 let clipper_read_lock = self.clipper.read().unwrap();
//                 let metrics_register = clipper_read_lock.get_metrics();
//                 let m = metrics_register.read().unwrap();
//                 self.result_string = m.report();
//             }
//             _ => {}
//         }
//         res.headers_mut().set(ContentLength(self.result_string.as_bytes().len() as u64));
//         Next::write()
//     }
// }

#[allow(dead_code)]
fn launch_monitor_thread(metrics_register: Arc<RwLock<metrics::Registry>>,
                         report_interval_secs: u64,
                         shutdown_signal_rx: mpsc::Receiver<()>)
                         -> ::std::thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            match shutdown_signal_rx.try_recv() {
                Ok(_) |
                Err(mpsc::TryRecvError::Empty) => {
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




#[allow(unused_variables)] // needed for metrics shutdown signal
fn start_listening<P, S>(shutdown_signal: mpsc::Receiver<()>,
                         clipper: Arc<RwLock<ClipperServer<P, S>>>)
    where P: CorrectionPolicy<S>,
          S: Serialize + Deserialize
{

    let rest_server = Server::http(&"0.0.0.0:1337".parse().unwrap()).unwrap();

    let report_interval_secs = 15;
    let (metrics_signal_tx, metrics_signal_rx) = mpsc::channel::<()>();
    let _ = launch_monitor_thread(clipper.read().unwrap().get_metrics(),
                                  report_interval_secs,
                                  metrics_signal_rx);
    let input_type = clipper.read().unwrap().get_input_type();

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
                        Arc::new(RwLock::new(ClipperServer::<DummyCorrectionPolicy, Vec<f64>>::new(config))));
    } else if config.policy_name == "logistic_regression".to_string() {
        start_listening(shutdown_signal,
                        Arc::new(RwLock::new(ClipperServer::<LogisticRegressionPolicy,
                                                             LinearCorrectionState>::new(config))));
    } else {
        panic!("Unknown correction policy");
    }
}
