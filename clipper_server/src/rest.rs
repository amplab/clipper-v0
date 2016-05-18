
use std::io::{Read, Write};
use std::thread;
use std::sync::{mpsc, RwLock, Arc};
use std::net::SocketAddr;
use std::boxed::Box;
// use std::cell::Cell;

#[allow(unused_imports)]
use hyper::{Get, Post, StatusCode, RequestUri, Decoder, Encoder, Next, Control};
use hyper::header::ContentLength;
use hyper::net::HttpStream;
use hyper::server::{Server, Handler, Request, Response};

use server::{self,TaskModel};
use features;
use metrics;

const PREDICT: &'static str = "/predict";
// const UPDATE: &'static str = "/update";


struct RequestHandler<T>
where T: TaskModel + Send + Sync + 'static {
    dispatcher: Arc<server::Dispatcher<T>>,
    result_string: String,
    result_channel: Option<mpsc::Receiver<String>>,
    num_features: usize,
    ctrl: Control,
}

impl<T> RequestHandler<T>
where T: TaskModel + Send + Sync + 'static {
    fn new(dispatcher: Arc<server::Dispatcher<T>>,
           num_features: usize, ctrl: Control) -> RequestHandler<T> {
        RequestHandler {
            dispatcher: dispatcher,
            result_string: "NO RESULT YET".to_string(),
            result_channel: None,
            num_features: num_features,
            ctrl: ctrl
        }
    }
}


impl<T: TaskModel + Send + Sync + 'static> Handler<HttpStream> for RequestHandler<T> {
    fn on_request(&mut self, req: Request) -> Next {
        match *req.uri() {
            RequestUri::AbsolutePath(ref path) => match (req.method(), &path[0..PREDICT.len()]) {
                (&Get, PREDICT) => {
                    let query_str: Vec<&str> = path.split("?").collect();
                    let uid = if query_str.len() > 1 {
                        let q = query_str[1];
                        let sp: Vec<&str> = q.split("=").collect();
                        assert!(sp.len() == 2 && sp[0] == "uid");
                        let provided_uid = sp[1].parse::<u32>().unwrap();
                        assert!(provided_uid != 0, "Cannot provide user ID 0");
                        provided_uid
                    } else {
                        0
                    };
                    info!("UID: {}", uid);
                    let ctrl = self.ctrl.clone();
                    let (tx, rx) = mpsc::channel::<String>();
                    self.result_channel = Some(rx);
    // let (q_tx, q_rx) = mpsc::channel::<(time::PreciseTime, Control)>();
                    let on_pred = Box::new(move |y| {
                        tx.send(format!("predict: {}", y).to_string()).unwrap();
                        ctrl.ready(Next::write()).unwrap();
                    });
                    let r = server::PredictRequest::new(uid,
                                                        features::random_features(784),
                                                        0_i32,
                                                        on_pred,
                                                        );

                    self.dispatcher.dispatch(r, self.num_features);
                    // self.ctrl.ready(Next::write());
                    // self.queue.send((self.start_time.clone(), ctrl)).unwrap();
                    Next::wait()
                }
                _ => Next::write()
            },
            _ => Next::write()
        }
    }

    fn on_request_readable(&mut self, _: &mut Decoder<HttpStream>) -> Next {
        Next::write()
    }



    fn on_response(&mut self, res: &mut Response) -> Next {
        self.result_string = match self.result_channel {
            Some(ref c) => c.recv().unwrap(),
            None => panic!("no channel"),
        };
        // self.result_string = self.result_channel.recv().unwrap();
        res.headers_mut().set(ContentLength(self.result_string.as_bytes().len() as u64));
        Next::write()
    }

    fn on_response_writable(&mut self, transport: &mut Encoder<HttpStream>) -> Next {
        transport.write(self.result_string.as_bytes()).unwrap();
        info!("{}", self.result_string);
        // info!("{} in {}", self.result_string, self.start_time.to(time::PreciseTime::now()).num_milliseconds());
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


pub fn start_listening(feature_addrs: Vec<(String, Vec<SocketAddr>)>) {
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

    let _ = launch_monitor_thread(metrics_register.clone(),
                                                       report_interval_secs);

    // let (q_tx, q_rx) = mpsc::channel::<(time::PreciseTime, Control)>();
    let (listening, server) = server.handle(|ctrl| 
        RequestHandler::new(dispatcher.clone(), num_features, ctrl)).unwrap();
    println!("Listening on http://{}", listening);
    // thread::spawn(move || {
    //     let sla_millis = 2000;
    //     let sla = time::Duration::milliseconds(sla_millis);
    //     let epsilon = time::Duration::milliseconds(sla_millis / 5);
    //     loop {
    //         let (start_time, ctrl) = q_rx.recv().unwrap();
    //         // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
    //         let elapsed_time = start_time.to(time::PreciseTime::now());
    //         if elapsed_time < sla - epsilon {
    //             let sleep_time = ::std::time::Duration::new(
    //                 0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
    //             info!("prediction worker sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
    //             thread::sleep(sleep_time);
    //         }
    //         ctrl.ready(Next::write());
    //     }
    // });
    server.run();
}

