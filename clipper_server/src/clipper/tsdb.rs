use curl::easy::Easy;
use curl::Error;
use std::io::Read;
use url::{Url, form_urlencoded};
use metrics::{Counter, Meter, RatioCounter, Histogram, RealtimeClock};
use std::sync::{Arc};
use time;
use regex::Regex;

const NUM_NANOS_PER_SEC: u64 = 1_000_000_000;

pub struct Tsdb {
	name: String,
	ip: String,
	port: u16,
}

impl Tsdb {
	pub fn new(name: String, ip: String, port: u16) -> Tsdb {
		let re = Regex::new(r"\s").unwrap();
		let db_name = re.replace_all(&name, "-");
		create_influx(&db_name, &ip, port);
		Tsdb {
			name: db_name.clone(),
			ip: ip.clone(),
			port: port,
		}
	}

	pub fn new_write(&self) -> Write {
		Write::new(self)
	}
}

fn create_influx(db_name: &str, ip: &str, port: u16) {
    let url = format!("http://{}:{}/query", ip, port);
	let encoded_body = form_urlencoded::Serializer::new(String::new())
        .append_pair("q", &format!("{} \"{}\"", "CREATE DATABASE", db_name))
        .finish();
    match send_post_request(&url, &encoded_body) {
    	Ok(_) => {},
    	Err(_) => {
    		info!("Failed to create influx database! Is Influx running at {} on port {}?", ip, port);
    	},
    }
}

pub struct Write<'a> {
	db: &'a Tsdb,
	timestamp: u64,
	write_ops: Vec<String>,
}

/// Implements methods for performing a batch write operation of multiple
/// metrics to a specified time series database. "Append" methods add new
/// metrics to be written, and the "execute" method persists the data when
/// the caller has finished adding metrics.
impl <'a> Write<'a> {
	fn new(db: &'a Tsdb) -> Write {
		let sys_time = time::get_time();
		let ts = ((sys_time.sec as u64) * (NUM_NANOS_PER_SEC as u64)) + (sys_time.nsec as u64);
		Write {
			db: db,
			timestamp: ts,
			write_ops: Vec::new(),
		}
	}

	pub fn append_counter(&mut self, counter: &Arc<Counter>) {
		let re = Regex::new(r"\s").unwrap();
		let name = re.replace_all(&counter.name, "_");
		// Construct the arguments of the InfluxSQL write operation that will be executed
		let op = format!("{} value={} {}", name, counter.value(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn append_ratio(&mut self, ratio: &Arc<RatioCounter>) {
		let re = Regex::new(r"\s").unwrap();
		let name = re.replace_all(&ratio.name, "_");
		// Construct the arguments of the InfluxSQL write operation that will be executed
		let op = format!("{} value={} {}", name, ratio.get_ratio(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn append_meter(&mut self, meter: &Arc<Meter<RealtimeClock>>) {
		let re = Regex::new(r"\s").unwrap();
		let unit = format!("units={}", re.replace_all(&meter.unit, "-"));
		let name = re.replace_all(&meter.name, "_");
		// Construct the arguments of the InfluxSQL write operation that will be executed
		let op = 
			format!(
				"{},{} rate={},one_min={},five_min={},fifteen_min={} {}", 
				name, unit, meter.get_rate_secs(), meter.get_one_minute_rate_secs(), 
				meter.get_five_minute_rate_secs(), meter.get_fifteen_minute_rate_secs(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn append_histogram(&mut self, hist: &Arc<Histogram>) {
		let stats = hist.stats();
		let re = Regex::new(r"\s").unwrap();
		let name = re.replace_all(&hist.name, "_");
		// Construct the arguments of the InfluxSQL write operation that will be executed
		let op = 
			format!(
				"{} size={},min={},max={},mean={},std={},p95={},p99={},p50={} {}",
				name, stats.size, stats.min, stats.max, stats.mean, stats.std, stats.p95, stats.p99, stats.p50, self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	/// Persists the metrics appended to this write operation to the associated
	/// time series database (InfluxDB instance). 
	pub fn execute(&mut self) {
		let raw_url = &format!("http://{}:{}/write?db={}", self.db.ip, self.db.port, self.db.name);
		let encoded_url = Url::parse(raw_url).unwrap();	
		// The body of the post request to InfluxDB is a series of InfluxSQL write arguments
		// created via "append" methods, delimited by new lines (one write argument set per line)
		let body = self.write_ops.join("\n");
		match send_post_request(encoded_url.as_str(), &body) {
			Ok(_) => {},
			Err(_) => {
				info!("Failed to write metrics to influx database at {} on port {}", self.db.ip, self.db.port);
			}
		}
	}
}

fn send_post_request(url: &str, body: &str) -> Result<(), Error> {
	let mut data = body.as_bytes();

	let mut request = Easy::new();
	request.url(url).unwrap();
	request.post(true).unwrap();
	request.post_field_size(data.len() as u64).unwrap();

	// Reads data from the CURL request object and 
	// posts it to the destination specified by the request
	let mut transfer = request.transfer();
    transfer.read_function(|buf| {
        Ok(data.read(buf).unwrap_or(0))
    }).unwrap();
    transfer.perform()
}
