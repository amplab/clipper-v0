
// #![allow(unused_variables)]
// #![allow(unused_imports)]
// #![allow(unused_must_use)]
// #![allow(dead_code)]
// use gj;
// // use gj::{EventLoop, Promise};
// // use gj::io::Timer;
// use time;
// use eventual;
// use mio;
// use eventual::Async;
// use std::thread;
// use std::sync::mpsc;
// use rand::{thread_rng, Rng};
// use std::sync::atomic::{AtomicUsize, Ordering};
// use std::sync::Arc;
// // use mio::util::Slab;
// // use mio::{EventLoop, Handler};
//
// const SLA: u64 = 20;
//
// /// gj benchmarking
// pub fn gj_timers(num_events: u32) {
//
//     gj::EventLoop::top_level(|wait_scope| {
//         // let ts = gj::TaskSet::new(Box::new(Reporter));
//         gj::Promise::all((0..num_events).map(|h| start_request())).wait(wait_scope);
//         Ok(())
//     });
//
//
// }
//
// fn start_request() -> gj::Promise<f64, ::std::io::Error> {
//
//     // let sla = 20;
//     let start_time = time::PreciseTime::now();
//     // get_features(&features, hash_id, input);
//     gj::io::Timer.after_delay(::std::time::Duration::from_millis(20)).then(move |()| {
//         // println!("responding to request");
//         // let fs = anytime_features(hash_id, &features);
//         let end_time = time::PreciseTime::now();
//         let latency = start_time.to(end_time).num_milliseconds();
//         println!("latency: {} ms", latency);
//         // counter.fetch_add(1, Ordering::Relaxed);
//         gj::Promise::ok(1.2_f64)
//     })
//
// }
//
//
// /// eventual benchmarking
//
// // TODO: needs a threadpool to work with
//
// // pub fn eventual_timers(num_events: u32) {
// //     println!("starting ev bench");
// //     let t = eventual::Timer::new();
// //     let start_time = time::PreciseTime::now();
// //     // let futures: Vec<eventual::Future<f64, ()>> = (0..num_events).into_iter().map(|_| start_ev_request()).collect();
// //     let futures = (0..num_events).into_iter().map(|_| start_ev_request()).collect::<Vec<_>>();
// //     let res = eventual::join(futures).await();
// //     println!("{:?}", res);
// //     let end_time = time::PreciseTime::now();
// //     let latency = start_time.to(end_time).num_milliseconds();
// //     println!("runtime: {} ms", (latency as f64) / 1000.0);
// //
// //     // .await().unwrap();
// // }
// //
// // // extend timer to also take a closure, execute closure after completed
// //
// // fn start_ev_request(t: eventual::Timer) -> eventual::Future<f64, ()> {
// //     let start_time = time::PreciseTime::now();
// //     let tm = t.timeout_ms(SLA as u32);
// //     tm.and_then(move |()| {
// //         let end_time = time::PreciseTime::now();
// //         let latency = start_time.to(end_time).num_milliseconds();
// //         println!("latency: {} ms", latency);
// //         Ok(1.1)
// //     })
// // }
// //
//
//
//
//
//
// /// Mio benchmarking
//
//
// struct LatencyHandler;
//
// impl mio::Handler for LatencyHandler {
//     type Timeout = time::PreciseTime;
//     type Message = u32;
//
//     fn timeout(&mut self,
//                event_loop: &mut mio::EventLoop<LatencyHandler>,
//                start_time: time::PreciseTime) {
//         let end_time = time::PreciseTime::now();
//         let latency = start_time.to(end_time).num_milliseconds();
//         println!("latency: {} ms", latency);
//         event_loop.shutdown();
//     }
//
//     fn notify(&mut self, event_loop: &mut mio::EventLoop<LatencyHandler>, msg: u32) {
//         // println!("starting timeout");
//         let timeout = event_loop.timeout_ms(time::PreciseTime::now(), SLA).unwrap();
//     }
//
// }
//
// pub fn mio_timers(num_events: u32) {
//
//     let mut event_loop = mio::EventLoop::new().unwrap();
//     let sender = event_loop.channel();
//     thread::spawn(move || {
//         thread::sleep(::std::time::Duration::new(3, 0));
//         for i in 0..num_events {
//             let _ = sender.send(123);
//         }
//     });
//     let _ = event_loop.run(&mut LatencyHandler);
// }
//
//
//
// /// hand-rolled queue
//
//
// fn sleep_queue(counter: Arc<AtomicUsize>) -> mpsc::Sender<time::PreciseTime> {
//
//     let (sender, receiver) = mpsc::channel::<time::PreciseTime>();
//     let epsilon = time::Duration::milliseconds(3);
//     let sla = time::Duration::milliseconds(20);
//
//     thread::spawn(move|| {
//         loop {
//             // NOTE: this is a blocking call
//             let req_time = receiver.recv().unwrap();
//             // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
//             let elapsed_time = req_time.to(time::PreciseTime::now());
//             if elapsed_time < sla - epsilon {
//                 let sleep_time = ::std::time::Duration::new(
//                     0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
//                 println!("sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
//                 thread::sleep(sleep_time);
//             }
//             // return result
//             counter.fetch_add(1, Ordering::Relaxed);
//             let end_time = time::PreciseTime::now();
//             let latency = req_time.to(end_time).num_milliseconds();
//             // if latency > 25 {
//             //     println!("latency: {} ms", latency);
//             // }
//             // println!("latency: {} ms", latency);
//         }
//         // receiver.unwrap();
//     });
//     sender
// }
//
// pub fn clipper_timers(num_events: u32, num_threads: usize) {
//     // let (sender, receiver) = mpsc::channel::<(time::PreciseTime, u32)>();
//
//     // let num_threads = 1;
//     println!("running benchmark with {} events, {} threads", num_events, num_threads);
//     let counter = Arc::new(AtomicUsize::new(0));
//     let mut senders = Vec::new();
//     for _ in 0..num_threads {
//         senders.push(sleep_queue(counter.clone()));
//     }
//
//     // thread::spawn(move|| {
//     //     loop {
//     //         // NOTE: this is a blocking call
//     //         let (req_time, hash) = receiver.recv().unwrap();
//     //         // if elapsed_time is less than SLA (+- epsilon wiggle room) then wait
//     //         let elapsed_time = req_time.to(time::PreciseTime::now());
//     //         if elapsed_time < sla - epsilon {
//     //             let sleep_time = ::std::time::Duration::new(
//     //                 0, (sla - elapsed_time).num_nanoseconds().unwrap() as u32);
//     //             println!("sleeping for {:?} ms",  sleep_time.subsec_nanos() as f64 / (1000.0 * 1000.0));
//     //             thread::sleep(sleep_time);
//     //         }
//     //         // return result
//     //         let end_time = time::PreciseTime::now();
//     //         let latency = req_time.to(end_time).num_milliseconds();
//     //         // println!("latency: {} ms", latency);
//     //     }
//     //     // receiver.unwrap();
//     // });
//
//     thread::sleep(::std::time::Duration::new(3, 0));
//     let mon_thread_join_handle = {
//         let counter = counter.clone();
//         thread::spawn(move || {
//             let bench_start = time::PreciseTime::now();
//             let mut last_count = 0;
//             let mut last_time = bench_start;
//             loop {
//                 thread::sleep(::std::time::Duration::new(1, 0));
//                 let cur_time = last_time.to(time::PreciseTime::now()).num_milliseconds() as f64 / 1000.0;
//                 let cur_count = counter.load(Ordering::Relaxed);
//                 println!("processed {} events in {} seconds: {} preds/sec",
//                          cur_count - last_count,
//                          cur_time,
//                          ((cur_count - last_count) as f64 / cur_time));
//
//                 if cur_count >= num_events as usize {
//                     println!("BENCHMARK FINISHED");
//                     break;
//                 }
//                 println!("sleeping...");
//             }
//         })
//     };
//
//     println!("sending batch with no delays");
//     let mut cur_sender = 0;
//     for i in 0..num_events {
//         senders[cur_sender].send(time::PreciseTime::now()).unwrap();
//         cur_sender = (cur_sender + 1) % num_threads;
//     }
//
//     //
//     // println!("sleeping...");
//     // thread::sleep(::std::time::Duration::new(10, 0));
//     //
//     // println!("sending batch with random delays");
//     // let mut rng = thread_rng();
//     // for i in 0..num_events {
//     //     let max_delay_millis = 10;
//     //     let delay = rng.gen_range(0, max_delay_millis*1000*1000);
//     //     thread::sleep(::std::time::Duration::new(0, delay));
//     //     sender.send((time::PreciseTime::now(), 14)).unwrap();
//     // }
//
//     mon_thread_join_handle.join().unwrap();
// }
//
//
//
//
//
//
//
