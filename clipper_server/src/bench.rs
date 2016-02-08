
use gj;
// use gj::{EventLoop, Promise};
// use gj::io::Timer;
use time;
use eventual;
use mio;
use eventual::Async;
use std::thread;
// use mio::util::Slab;
// use mio::{EventLoop, Handler};

const SLA: u64 = 20;

/// gj benchmarking
pub fn gj_timers(num_events: u32) {

    gj::EventLoop::top_level(|wait_scope| {
        // let ts = gj::TaskSet::new(Box::new(Reporter));
        gj::Promise::all((0..num_events).map(|h| start_request())).wait(wait_scope);
        Ok(())
    });


}

fn start_request() -> gj::Promise<f64, ::std::io::Error> {

    // let sla = 20;
    let start_time = time::PreciseTime::now();
    // get_features(&features, hash_id, input);
    gj::io::Timer.after_delay(::std::time::Duration::from_millis(20)).then(move |()| {
        // println!("responding to request");
        // let fs = anytime_features(hash_id, &features);
        let end_time = time::PreciseTime::now();
        let latency = start_time.to(end_time).num_milliseconds();
        println!("latency: {} ms", latency);
        // counter.fetch_add(1, Ordering::Relaxed);
        gj::Promise::ok(1.2_f64)
    })

}


/// eventual benchmarking

// TODO: needs a threadpool to work with

// pub fn eventual_timers(num_events: u32) {
//     println!("starting ev bench");
//     let t = eventual::Timer::new();
//     let start_time = time::PreciseTime::now();
//     // let futures: Vec<eventual::Future<f64, ()>> = (0..num_events).into_iter().map(|_| start_ev_request()).collect();
//     let futures = (0..num_events).into_iter().map(|_| start_ev_request()).collect::<Vec<_>>();
//     let res = eventual::join(futures).await();
//     println!("{:?}", res);
//     let end_time = time::PreciseTime::now();
//     let latency = start_time.to(end_time).num_milliseconds();
//     println!("runtime: {} ms", (latency as f64) / 1000.0);
//
//     // .await().unwrap();
// }
//
// // extend timer to also take a closure, execute closure after completed
//
// fn start_ev_request(t: eventual::Timer) -> eventual::Future<f64, ()> {
//     let start_time = time::PreciseTime::now();
//     let tm = t.timeout_ms(SLA as u32);
//     tm.and_then(move |()| {
//         let end_time = time::PreciseTime::now();
//         let latency = start_time.to(end_time).num_milliseconds();
//         println!("latency: {} ms", latency);
//         Ok(1.1)
//     })
// }
//





/// Mio benchmarking


struct LatencyHandler;

impl mio::Handler for LatencyHandler {
    type Timeout = time::PreciseTime;
    type Message = u32;

    fn timeout(&mut self,
               event_loop: &mut mio::EventLoop<LatencyHandler>,
               start_time: time::PreciseTime) {
        let end_time = time::PreciseTime::now();
        let latency = start_time.to(end_time).num_milliseconds();
        println!("latency: {} ms", latency);
        event_loop.shutdown();
    }

    fn notify(&mut self, event_loop: &mut mio::EventLoop<LatencyHandler>, msg: u32) {
        // println!("starting timeout");
        let timeout = event_loop.timeout_ms(time::PreciseTime::now(), SLA).unwrap();
    }

}

pub fn mio_timers(num_events: u32) {

    let mut event_loop = mio::EventLoop::new().unwrap();
    let sender = event_loop.channel();
    thread::spawn(move || {
        thread::sleep(::std::time::Duration::new(3, 0));
        for i in (0..num_events) {
            let _ = sender.send(123);
        }
    });
    let _ = event_loop.run(&mut LatencyHandler);
}











