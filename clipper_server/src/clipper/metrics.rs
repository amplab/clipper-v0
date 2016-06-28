
use time;
use std::sync::atomic::{AtomicUsize, AtomicIsize, Ordering};
use rand::{thread_rng, Rng};
use std::sync::{RwLock, Arc};

const NUM_MICROS_PER_SEC: i64 = 1_000_000;

trait Metric {

    fn report(&self) -> String;

    /// Must have way to atomically clear state
    fn clear(&self);

}


/// Counts a value
pub struct Counter {
    pub name: String,
    count: AtomicIsize,
}

#[derive(Debug)]
struct CounterStats {
    name: String,
    count: isize,
}

impl Metric for Counter {
    fn clear(&self) {
        self.count.store(0, Ordering::SeqCst);
    }

    fn report(&self) -> String {
        let stats = CounterStats {
            name: self.name.clone(),
            count: self.count.load(Ordering::SeqCst),
        };
        format!("{:?}", stats)
    }
}

impl Counter {
    pub fn new(name: String, start_count: isize) -> Counter {
        Counter {
            name: name,
            count: AtomicIsize::new(start_count),
        }
    }

    pub fn incr(&self, increment: isize) {
        self.count.fetch_add(increment, Ordering::Relaxed);
    }

    pub fn decr(&self, decrement: isize) {
        self.count.fetch_sub(decrement, Ordering::Relaxed);
    }
}

pub struct RatioCounter {
    pub name: String,
    numerator: AtomicUsize,
    denominator: AtomicUsize,
}

#[derive(Debug)]
struct RatioStats {
    name: String,
    ratio: f64,
}

impl Metric for RatioCounter {
    // TODO: This has a race condition.
    fn clear(&self) {
        self.denominator.store(0, Ordering::SeqCst);
        self.numerator.store(0, Ordering::SeqCst);
    }

    fn report(&self) -> String {
        let ratio = self.numerator.load(Ordering::SeqCst) as f64 /
                    self.denominator.load(Ordering::SeqCst) as f64;
        let stats = RatioStats {
            name: self.name.clone(),
            ratio: ratio,
        };
        format!("{:?}", stats)
    }
}

impl RatioCounter {
    pub fn new(name: String, n: usize, d: usize) -> RatioCounter {
        RatioCounter {
            name: name,
            numerator: AtomicUsize::new(n),
            denominator: AtomicUsize::new(d),
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
            ::std::f64::NAN
        } else {
            n as f64 / d as f64
        }
    }
}






/// Measures the rate of an event occurring
// TODO: add support for exponentially weighted moving averages. See
// https://github.com/dropwizard/metrics/blob/4286d49c7f0da1d9cdb90b4fdc15dd3c91e7a22c/metrics-core/src/main/java/io/dropwizard/metrics/Meter.java
// for details.
pub struct Meter {
    pub name: String,
    start_time: RwLock<time::PreciseTime>,
    count: AtomicUsize,
}

impl Meter {
    pub fn new(name: String) -> Meter {
        Meter {
            name: name,
            start_time: RwLock::new(time::PreciseTime::now()),
            count: AtomicUsize::new(0),
        }
    }

    pub fn mark(&self, num: usize) {
        self.count.fetch_add(num, Ordering::Relaxed);
    }

    fn get_rate_micros(&self) -> f64 {
        let cur_time = time::PreciseTime::now();
        let count = self.count.load(Ordering::SeqCst);
        let dur: time::Duration = self.start_time.read().unwrap().to(cur_time);
        let whole_secs = time::Duration::seconds(dur.num_seconds());
        let sub_sec_micros = (dur - whole_secs).num_microseconds().unwrap();
        assert!(sub_sec_micros <= NUM_MICROS_PER_SEC);
        let total_micros = whole_secs.num_seconds() * NUM_MICROS_PER_SEC + sub_sec_micros;
        let rate = count as f64 / total_micros as f64;
        rate
    }

    /// Returns the rate of this meter in
    /// events per second.
    pub fn get_rate_secs(&self) -> f64 {
        self.get_rate_micros() * NUM_MICROS_PER_SEC as f64
    }
}

#[derive(Debug)]
struct MeterStats {
    name: String,
    rate: f64,
    unit: String,
}

impl Metric for Meter {
    fn clear(&self) {
        let mut t = self.start_time.write().unwrap();
        *t = time::PreciseTime::now();
        self.count.store(0, Ordering::SeqCst);
    }

    fn report(&self) -> String {

        let stats = MeterStats {
            name: self.name.clone(),
            rate: self.get_rate_secs(),
            unit: "events per second".to_string(),
        };
        format!("{:?}", stats)
    }
}

#[derive(Debug)]
pub struct HistStats {
    pub name: String,
    pub min: i64,
    pub max: i64,
    pub mean: f64,
    pub std: f64,
    pub p95: f64,
    pub p99: f64,
    pub p50: f64,
}

// This gives me latency distribution, min, mean, max, etc.
pub struct Histogram {
    pub name: String,
    sample: RwLock<ReservoirSampler>,
}

impl Histogram {
    pub fn new(name: String, sample_size: usize) -> Histogram {
        Histogram {
            name: name,
            sample: RwLock::new(ReservoirSampler::new(sample_size)),
        }
    }

    pub fn insert(&self, item: i64) {
        let mut res = self.sample.write().unwrap();
        res.sample(item);
    }

    pub fn stats(&self) -> HistStats {
        let mut snapshot: Vec<i64> = {
            self.sample.read().unwrap().snapshot()
        };
        let sample_size = snapshot.len();
        if sample_size == 0 {
            HistStats {
                name: self.name.clone(),
                min: 0,
                max: 0,
                mean: 0.0,
                std: 0.0,
                p95: 0.0,
                p99: 0.0,
                p50: 0.0,
            }

        } else {

            snapshot.sort();
            let min = snapshot.first().unwrap();
            let max = snapshot.last().unwrap();
            let p99 = Histogram::percentile(&snapshot, 99);
            let p95 = Histogram::percentile(&snapshot, 95);
            let p50 = Histogram::percentile(&snapshot, 50);
            let mean = snapshot.iter().fold(0, |acc, &x| acc + x) as f64 / snapshot.len() as f64;
            let mut var: f64 = snapshot.iter().fold(0.0, |acc, &x| acc + (x as f64 - mean).powi(2));
            var = var / (sample_size - 1) as f64;
            HistStats {
                name: self.name.clone(),
                min: *min,
                max: *max,
                mean: mean,
                std: var.sqrt(),
                p95: p95,
                p99: p99,
                p50: p50,
            }
        }
    }

    fn percentile(snapshot: &Vec<i64>, p: usize) -> f64 {
        assert!(p <= 100, "must supply a percentile between 0 and 100");
        let sample_size = snapshot.len();
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
            (snapshot[per_below] as f64 + snapshot[per_above] as f64) / 2.0_f64
        };
        per
    }
}

impl Metric for Histogram {
    fn clear(&self) {
        let mut res = self.sample.write().unwrap();
        res.clear();
    }

    fn report(&self) -> String {
        format!("{:?}", self.stats())
    }
}



struct ReservoirSampler {
    reservoir: Vec<i64>,
    sample_size: usize,
    n: usize,
}

impl ReservoirSampler {
    pub fn new(sample_size: usize) -> ReservoirSampler {
        ReservoirSampler {
            reservoir: Vec::with_capacity(sample_size),
            sample_size: sample_size,
            n: 0,
        }
    }

    pub fn sample(&mut self, value: i64) {
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

    pub fn snapshot(&self) -> Vec<i64> {
        self.reservoir.clone()
    }
}


pub struct Registry {
    pub name: String,
    counters: Vec<Arc<Counter>>,
    // sum_counters: Vec<SumCounter>,
    ratio_counters: Vec<Arc<RatioCounter>>,
    histograms: Vec<Arc<Histogram>>,
    meters: Vec<Arc<Meter>>,
}

impl Registry {
    pub fn new(name: String) -> Registry {
        Registry {
            name: name,
            counters: Vec::new(),
            ratio_counters: Vec::new(),
            histograms: Vec::new(),
            meters: Vec::new(),
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
        let mut report_string = String::new();
        report_string.push_str(&format!("\n{} Metrics\n", self.name));

        if self.counters.len() > 0 {
            report_string.push_str("\tCounters:\n");
            for x in self.counters.iter() {
                report_string.push_str(&format!("\t\t{}\n", x.report()));
            }
        }

        if self.ratio_counters.len() > 0 {
            report_string.push_str("\tRatios:\n");
            for x in self.ratio_counters.iter() {
                report_string.push_str(&format!("\t\t{}\n", x.report()));
            }
        }

        if self.histograms.len() > 0 {
            report_string.push_str("\tHistograms:\n");
            for x in self.histograms.iter() {
                report_string.push_str(&format!("\t\t{}\n", x.report()));
            }
        }

        if self.meters.len() > 0 {
            report_string.push_str("\tMeters:\n");
            for x in self.meters.iter() {
                report_string.push_str(&format!("\t\t{}\n", x.report()));
            }
        }


        debug!("{}", report_string);
        report_string
    }

    // pub fn report_and_reset(&self) -> String {
    //
    // }

    pub fn reset(&self) {
        for x in self.counters.iter() {
            x.clear();
        }
        for x in self.ratio_counters.iter() {
            x.clear();
        }

        for x in self.histograms.iter() {
            x.clear();
        }

        for x in self.meters.iter() {
            x.clear();
        }
    }
}
