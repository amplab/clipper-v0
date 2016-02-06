
use gj;
use gj::{EventLoop, Promise};
// use gj::io::Timer;
use time;
use eventual;
use eventual::Async;

const SLA: u32 = 20;

/// gj benchmarking
pub fn gj_timers(num_events: u32) {

    EventLoop::top_level(|wait_scope| {
        // let ts = gj::TaskSet::new(Box::new(Reporter));
        Promise::all((0..num_events).map(|h| start_request())).wait(wait_scope);
        Ok(())
    });


}

fn start_request() -> Promise<f64, ::std::io::Error> {

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
        Promise::ok(1.2_f64)
    })

}


/// eventual benchmarking

// TODO: needs a threadpool to work with

pub fn eventual_timers(num_events: u32) {
    println!("starting ev bench");
    let start_time = time::PreciseTime::now();
    // let futures: Vec<eventual::Future<f64, ()>> = (0..num_events).into_iter().map(|_| start_ev_request()).collect();
    let futures = (0..num_events).into_iter().map(|_| start_ev_request()).collect::<Vec<_>>();
    let res = eventual::join(futures).await();
    println!("{:?}", res);
    let end_time = time::PreciseTime::now();
    let latency = start_time.to(end_time).num_milliseconds();
    println!("runtime: {} ms", (latency as f64) / 1000.0);

    // .await().unwrap();
}

fn start_ev_request() -> eventual::Future<f64, ()> {
    let start_time = time::PreciseTime::now();
    let t = eventual::Timer::new();
    t.timeout_ms(SLA).and_then(move |()| {
        let end_time = time::PreciseTime::now();
        let latency = start_time.to(end_time).num_milliseconds();
        println!("latency: {} ms", latency);
        Ok(1.1)
    })
}
