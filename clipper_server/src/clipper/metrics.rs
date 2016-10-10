
use time;
use std::sync::atomic::{AtomicUsize, AtomicIsize, Ordering};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::sync::{RwLock, Arc};
use tsdb::{Tsdb};

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

    pub fn value(&self) -> isize {
        self.count.load(Ordering::SeqCst)
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
    pub unit: String,
    start_time: RwLock<time::PreciseTime>,
    count: AtomicUsize,
}

impl Meter {
    pub fn new(name: String) -> Meter {
        Meter {
            name: name,
            unit: "events per second".to_string(),
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
            unit: self.unit.clone(),
        };
        format!("{:?}", stats)
    }
}

#[derive(Debug)]
pub struct HistStats {
    pub name: String,
    pub size: u64,
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
                size: 0,
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
            let p99 = Histogram::percentile(&snapshot, 0.99);
            let p95 = Histogram::percentile(&snapshot, 0.95);
            let p50 = Histogram::percentile(&snapshot, 0.50);
            let mean = snapshot.iter().fold(0, |acc, &x| acc + x) as f64 / snapshot.len() as f64;
            let var: f64 = if sample_size > 1 {
                snapshot.iter().fold(0.0, |acc, &x| acc + (x as f64 - mean).powi(2)) / (sample_size - 1) as f64
            } else {
                0.0
            };
            HistStats {
                name: self.name.clone(),
                size: sample_size as u64,
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


    /// Compute the percentile rank of `snapshot`. The rank `p` must
    /// be in [0.0, 1.0] (inclusive) and `snapshot` must be sorted.
    ///
    /// Algorithm is the third variant from
    /// [Wikipedia](https://en.wikipedia.org/wiki/Percentile)
    pub fn percentile(snapshot: &Vec<i64>, p: f64) -> f64 {
        assert!(snapshot.len() > 0);
        let sample_size = snapshot.len() as f64;
        assert!(p >= 0.0 && p <= 1.0, "percentile out of bounds");
        let x = if p <= 1.0 / (sample_size + 1.0) {
            // println!("a");
            1.0
        } else if p > 1.0 / (sample_size + 1.0) && p < sample_size / (sample_size + 1.0) {
            // println!("b");
            p * (sample_size + 1.0)
        } else {
            // println!("c");
            sample_size
        };
        let index = x.floor() as usize - 1;
        let v = snapshot[index] as f64;
        let rem = x % 1.0;
        let per = if rem != 0.0 {
            // println!("rem: {}", rem);
            v + rem * (snapshot[index + 1] - snapshot[index]) as f64
        } else {
            v
        };
        per
    }

    // // TODO: this percentile calculation is wrong for numbers less than 100
    // fn percentile(snapshot: &Vec<i64>, p: usize) -> f64 {
    //     assert!(p <= 100, "must supply a percentile between 0 and 100");
    //     let sample_size = snapshot.len();
    //     let per = if sample_size == 1 {
    //         snapshot[0]
    //     } else if sample_size < 100 {
    //         warn!("computing p{} of sample size smaller than 100", p);
    //         snapshot[(sample_size - 1 - (100 - p)) as usize] as f64
    //     } else if (sample_size % 100) == 0 {
    //         let per_index: usize = sample_size * p / 100;
    //         snapshot[per_index as usize] as f64
    //     } else {
    //         let per_index: f64 = (sample_size as f64) * (p as f64 / 100.0);
    //         let per_below = per_index.floor() as usize;
    //         let per_above = per_index.ceil() as usize;
    //         (snapshot[per_below] as f64 + snapshot[per_above] as f64) / 2.0_f64
    //     };
    //     per
    // }
}

// Bucketed multiclass histogram
pub struct BucketHistogram {
    pub name: String,
    buckets: RwLock<HashMap<i32, AtomicUsize>>,
}

impl BucketHistogram {
    pub fn new(name: String) -> BucketHistogram {
        BucketHistogram {
            name: name,
            buckets: RwLock::new(HashMap::new()),
        }
    }

    pub fn incr(&self, bucket: i32) {
        let mut buckets = self.buckets.write().unwrap();
        if !buckets.contains_key(&bucket) {
            buckets.insert(bucket, AtomicUsize::new(0));
        }
        buckets.get_mut(&bucket).unwrap().fetch_add(1, Ordering::Relaxed);
        println!("Incrementing {} {}", bucket,
                 buckets.get(&bucket).unwrap().load(Ordering::Relaxed));
    }

    pub fn stats(&self) -> BucketHistStats {
        let mut retmap = HashMap::new();
        let buckets = self.buckets.read().unwrap();
        for (i, b) in buckets.iter() {
            retmap.insert(i.clone(), b.load(Ordering::Relaxed));
        }

        BucketHistStats {
            name: self.name.clone(),
            buckets: retmap,
        }
    }


}

impl Metric for BucketHistogram {
    fn clear(&self) {
        let mut buckets = self.buckets.write().unwrap();
        buckets.clear();
    }

    fn report(&self) -> String {
        format!("{:?}", self.stats())
    }
}

#[derive(Debug)]
pub struct BucketHistStats {
    name: String,
    buckets: HashMap<i32, usize>,
}

#[cfg(test)]
#[cfg_attr(rustfmt, rustfmt_skip)]
mod tests {
    use super::*;

    #[test]
    fn percentile() {
        let snap = vec![15, 20, 35, 40, 50];
        let p = 0.4;
        let computed_percentile = Histogram::percentile(&snap, p);
        assert!(computed_percentile - 26.0 < 0.000001);
    }

    #[test]
    fn percentile_one_elem() {
        let snap = vec![15];
        let p = 0.4;
        let computed_percentile = Histogram::percentile(&snap, p);
        assert!(computed_percentile - 15.0 < 0.000001);
    }

    #[test]
    fn percentile_one_elem_pzero() {
        let snap = vec![15];
        let p = 0.0;
        let computed_percentile = Histogram::percentile(&snap, p);
        assert!(computed_percentile - 15.0 < 0.000001);
    }

    #[test]
    fn percentile_one_elem_p100() {
        let snap = vec![15];
        let p = 1.0;
        let computed_percentile = Histogram::percentile(&snap, p);
        assert!(computed_percentile - 15.0 < 0.000001);
    }

    #[test]
    #[should_panic]
    fn percentile_zero_elem() {
        let snap = Vec::new();
        let p = 0.5;
        let _ = Histogram::percentile(&snap, p);
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
    db: Tsdb,
    counters: Vec<Arc<Counter>>,
    // sum_counters: Vec<SumCounter>,
    ratio_counters: Vec<Arc<RatioCounter>>,
    histograms: Vec<Arc<Histogram>>,
    meters: Vec<Arc<Meter>>,
    bucket_histograms: Vec<Arc<BucketHistogram>>,
}

impl Registry {
    pub fn new(name: String, db_ip: String, db_port: u16) -> Registry {
        // Create a new time series database to store these metrics
        Registry {
            name: name.clone(),
            db: Tsdb::new(name.clone(), db_ip.to_string(), db_port),
            counters: Vec::new(),
            ratio_counters: Vec::new(),
            histograms: Vec::new(),
            meters: Vec::new(),
            bucket_histograms: Vec::new(),
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

    pub fn create_bucket_hist(&mut self, name: String) -> Arc<BucketHistogram> {
        let bucket_hist = Arc::new(BucketHistogram::new(name));
        self.bucket_histograms.push(bucket_hist.clone());
        bucket_hist
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

        if self.bucket_histograms.len() > 0 {
            report_string.push_str("\tBucket Histograms:\n");
            for x in self.bucket_histograms.iter() {
                report_string.push_str(&format!("\t\t{}\n", x.report()));
            }
        }


        debug!("{}", report_string);
        report_string
    }

    pub fn persist(&self) {
        let mut write = self.db.new_write();
        for x in self.counters.iter() {
            write.append_counter(x);
        }
        for x in self.meters.iter() {
            write.append_meter(x);
        }
        for x in self.ratio_counters.iter() {
            write.append_ratio(x);
        }
        for x in self.histograms.iter() {
            write.append_histogram(x);
        }
        for x in self.bucket_histograms.iter() {
            write.append_bucket_histogram(x);
        }
        write.execute();
    }

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

        for x in self.bucket_histograms.iter() {
            x.clear();
        }
    }
}
