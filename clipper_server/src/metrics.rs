
use time;
use log;
use std::sync::atomic::{AtomicUsize, AtomicIsize, Ordering};
use rand::{thread_rng, Rng};

const NUM_MICROS_PER_SEC: usize = 1_000_000;

trait Reportable {

    fn report(&self) -> String;

    fn clear(&mut self);

}


/// Counts a value
pub struct Counter {
    pub name: String,
    count: AtomicIsize
}

impl Counter {
    pub fn new(name: String, start_count: isize) -> Counter {
        Counter {
            name: name,
            count: AtomicIsize::new(start_count)
        }
    }

    pub fn incr(&self, increment: isize) {
        self.count.fetch_add(increment);
    }

    pub fn decr(&self, decrement: isize) {
        self.count.fetch_sub(decrement);
    }

    pub fn clear(&self) {
        self.count.store(0, Ordering::SeqCst);
    }
}

// impl Reportable for Counter {
//
// }


// /// Returns the sum of several
// /// simple `Counter` objects.
// pub struct SumCounter {
//     pub name: String,
//     counters: Vec<Arc<Counter>>
// }


pub struct RatioCounter {
    pub name: String,
    numerator: AtomicUsize,
    denominator: AtomicUsize,
}

impl RatioCounter {

    pub fn new(name: String, n: usize, d: usize) -> Ratio {
        Ratio {
            name: name,
            numerator: n,
            denominator: d
        }
    }

    pub fn incr(&self, n_incr: usize, d_incr: usize) {
        self.numerator.fetch_add(n_incr, Ordering::Relaxed);
        self.denominator.fetch_add(d_incr, Ordering::Relaxed);
    }

    pub fn get_ratio(&self) -> f64 {
        let n = self.numerator.load(Ordering::Relaxed);
        let d = self.denominator.load(Ordering::Relaxed);
        if d == 0 {
            warn!("{} ratio denominator is 0", self.name);
            f64::NAN
        } else {
            n as f64 / d as f64
        }
    }

    // TODO: This has a race condition. 
    pub fn clear(&self) {
        self.denominator.store(0, Ordering::SeqCst);
        self.numerator.store(0, Ordering::SeqCst);
    }

}






/// Measures the rate of an event occurring
// TODO: add support for exponentially weighted moving averages. See
// https://github.com/dropwizard/metrics/blob/4286d49c7f0da1d9cdb90b4fdc15dd3c91e7a22c/metrics-core/src/main/java/io/dropwizard/metrics/Meter.java
// for details.
pub struct Meter {
    pub name: String,
    start_time: RwLock<time::PreciseTime>,
    count: AtomicUsize
}

impl Meter {
    pub fn new(name: String) -> Meter {
        Meter {
            name: name,
            start_time: RwLock::new(time::PreciseTime::now()),
            count: AtomicUsize::new(0)
        }
    }

    pub fn mark(&self, num: usize) {
        self.count.fetch_add(num, Ordering::Relaxed);
    }

    fn get_rate_micros(&self) -> f64 {
        let cur_time = time::PreciseTime::now();
        let count = self.count.load(Ordering::SeqCst);
        let dur = (cur_time - self.start_time.read().unwrap());
        let whole_secs = Duration::seconds(dur.num_seconds());
        let sub_sec_micros = (dur - whole_secs).num_microseconds().unwrap();
        assert!(sub_sec_micros <= NUM_MICROS_PER_SEC);
        let total_micros = whole_secs*NUM_MICROS_PER_SEC + sub_sec_micros;
        let rate = (count as f64 / total_micros as f64);
        rate
    }

    /// Returns the rate of this meter in
    /// events per second.
    pub fn get_rate_secs(&self) -> f64 {
        self.get_rate_micros() / NUM_MICROS_PER_SEC
    }

    pub fn clear(&self) {
        let mut t = self.start_time.write().unwrap();
        t = time::PreciseTime::now();
        self.count.store(0, Ordering::SeqCst);
    }
}

pub struct HistStats {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
    p95: f64,
    p99: f64,
    p50: f64
}

// This gives me latency distribution, min, mean, max, etc.
pub struct Histogram {
    pub name: String,
    sample: RwLock<ReservoirSampler>
}

impl Histogram {

    pub fn new(name: String, sample_size: usize) -> Histogram {
        Histogram {
            name: name,
            sample: RwLock::new(ReservoirSampler::new(sample_size))
        }
    }

    pub fn insert(&self, item: f64) {
        let mut res = self.sample.write().unwrap();
        res.sample(item);
    }

    pub fn stats(&self) -> HistStats {
        let mut snapshot = {
            self.sample.read().unwrap().clone();
        };
        let sample_size = snapshot.len();
        assert!(sample_size > 0, "cannot compute stats of empty histogram");
        snapshot.sort();
        let min = snapshot.first();
        let max = snapshot.last();
        let p99 = Histogram::percentile(&snapshot, 99);
        let p95 = Histogram::percentile(&snapshot, 95);
        let p50 = Histogram::percentile(&snapshot, 50);
        let mean = snapshot.iter().sum() / snapshot.len() as f64;
        let mut var: f64 = snapshot.iter().fold(0.0, |acc, &x| acc + (x - mean).powi(2));
        var = var / (sample_size - 1) as f64;
        HistStats {
            min: min,
            max: max,
            mean: mean,
            std: var.sqrt(),
            p95: p95,
            p99: p99,
            p50: p50
        }
    }

    fn percentile(snapshot: &Vec<f64>, p: usize) -> f64 {
        assert!(p >= 0 && p <= 100, "must supply a percentile between 0 and 100");
        let per = if sample_size < 100 {
            warn!("computing p{} of sample size smaller than 100", p); 
            snapshot[(sample_size - 1 - (100 - p)) as usize] as f64
        } else if (sample_size % 100) == 0 {
          let per_index: usize = sample_size * p / 100;
          snapshot[per_index as usize] as f64
        } else {
          let per_index: f64 = (sample_size as f64) * (p as f64 / 100.0);
          let per_below = per_index.floor() as usize;
          let per_above = per_index.ceil() as usize;
          (snapshot[per_below] as f64 + snapshot[per_above] as f64)  / 2.0_f64
        };
    }

    pub fn clear(&self) {
        let mut res = self.sample.write().unwrap();
        res.clear();
    }
}




struct ReservoirSampler {
    reservoir: Vec<f64>,
    sample_size: usize,
    n: usize,
}

impl ReservoirSampler {
    
    pub fn new(sample_size: usize) -> ReservoirSampler {
        ReservoirSampler {
            reservoir: Vec::with_capacity(sample_size),
            sample_size: sample_size,
            n: 0
        }
    }

    pub fn sample(&mut self, value: f64) {
        if self.n < self.sample_size {
            self.reservoir.push(value);
        } else {
            assert!(self.reservoir.len() == self.sample_size);
            let mut rng = thread_rng();
            let j = rng.gen_range(0, self.n + 1); // exclusive
            if j < self.sample_size {
                self.reservoir[j] = value;
            }
        }
        self.n += 1;
    }

    pub fn clear(&mut self) {
        self.reservoir.clear();
        self.n = 0;
    }
}


pub struct Registry {
    pub name: String,
    counters: Vec<Arc<Counter>>,
    // sum_counters: Vec<SumCounter>,
    ratio_counters: Vec<Arc<RatioCounter>>,
    histograms: Vec<Arc<Histogram>>,
    meters: Vec<Arc<Meters>>
}

impl Registry {

    pub fn new(name: String) -> Registry<R> {
        Registry {
            name: name,
            reporters: Vec::new()
        }
    }

    pub fn create_histogram(&mut self, name: String, sample_size: usize) -> Arc<Histogram> {
        let hist = Arc::new(Histogram::new(name, sample_size));
        self.histograms.push(hist.clone());
        hist
    }

    pub fn create_meter(&mut self, name: String) -> Arc<Meter> {
        let meter = Arc::new(Meter::new(name));
        self.meters.push(meter.clone());
        meter
    }

    pub fn create_ratio_counter(&mut self, name: String) -> Arc<RatioCounter> {
        let counter = Arc::new(RatioCounter::new(name, 0, 0));
        self.ratio_counters.push(counter.clone());
        counter
    }

    pub fn create_counter(&mut self, name: String) -> Arc<Counter> {
        let counter = Arc::new(Counter::new(name, 0));
        self.counters.push(counter.clone());
        counter
    }

    pub fn report(&self) -> String {
        warn!("metrics reporting is unimplemented");
        "report placeholder".to_string();
    }

    // pub fn report_and_reset(&self) -> String {
    //
    // }

    pub fn reset(&self) {
        for x in counters.iter() {
            x.clear();
        }
        for x in ratio_counters.iter() {
            x.clear();
        }

        for x in histograms.iter() {
            x.clear();
        }

        for x in meters.iter() {
            x.clear();
        }
    }
}


