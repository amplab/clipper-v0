use curl::easy::Easy;
use std::io::Read;
use url::{Url, form_urlencoded};
use metrics::{Counter, Meter, RatioCounter, Histogram};
use std::sync::{Arc};
use time;
use regex::Regex;

//const BASE_URL: &'static str = "http://localhost:8086";
//const CREATE_DB_BODY: &'static str = "CREATE DATABASE";

const SECONDS_IN_NANOS: u64 = 1000000000;

pub struct Tsdb {
	name: String,
	ip: String,
	port: u16,
}

impl Tsdb {
	pub fn new(name: String, ip: String, port: u16) -> Tsdb {
		create_influx(&name, &ip, port);
		Tsdb {
			name: name.clone(),
			ip: ip.clone(),
			port: port,
		}
	}

	pub fn new_write(&self) -> Write {
		Write::new(self)
	}
}

fn create_influx(name: &str, ip: &str, port: u16) {
    let url = format!("http://{}:{}/query", ip, port);
	let encoded_body = form_urlencoded::Serializer::new(String::new())
        .append_pair("q", &format!("{} \"{}\"", "CREATE DATABASE", name))
        .finish();
    send_post_request(&url, &encoded_body);
}

pub struct Write<'a> {
	db: &'a Tsdb,
	timestamp: String,
	write_ops: Vec<String>,
}

impl <'a> Write<'a> {
	fn new(db: &'a Tsdb) -> Write {
		let sys_time = time::get_time();
		let ts = ((sys_time.sec as u64) * (SECONDS_IN_NANOS as u64)) + (sys_time.nsec as u64);
		Write {
			db: db,
			timestamp: ts.to_string(),
			write_ops: Vec::new(),
		}
	}

	pub fn append_counter(&mut self, counter: &Arc<Counter>) {
		let re = Regex::new(r"\s").unwrap();
		let name = re.replace_all(&counter.name, "_");
		let op = format!("{} value={} {}", name, counter.value(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn append_ratio(&mut self, ratio: &Arc<RatioCounter>) {
		let re = Regex::new(r"\s").unwrap();
		let name = re.replace_all(&ratio.name, "_");
		let op = format!("{} value={} {}", name, ratio.get_ratio(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn append_meter(&mut self, meter: &Arc<Meter>) {
		let re = Regex::new(r"\s").unwrap();
		let unit = format!("units={}", re.replace_all(&meter.unit, "-"));
		let name = re.replace_all(&meter.name, "_");
		let op = format!("{},{} value={} {}", name, unit, meter.get_rate_secs(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn append_histogram(&mut self, hist: &Arc<Histogram>) {
		let stats = hist.stats();
		let re = Regex::new(r"\s").unwrap();
		let name = re.replace_all(&hist.name, "_");
		let op = 
			format!(
				"{} size={},min={},max={},mean={},std={},p95={},p99={},p50={} {}",
				name, stats.size, stats.min, stats.max, stats.mean, stats.std, stats.p95, stats.p99, stats.p50, self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	pub fn execute(&mut self) {
		let raw_url = &format!("http://{}:{}/write?db={}", self.db.ip, self.db.port, self.db.name);
		let encoded_url = Url::parse(raw_url).unwrap();	
		let body = self.write_ops.join("\n");
		send_post_request(encoded_url.as_str(), &body);
	}
}

fn send_post_request(url: &str, body: &str) {
	let mut data = body.as_bytes();

	let mut request = Easy::new();
	request.url(url).unwrap();
	request.post(true).unwrap();
	request.post_field_size(data.len() as u64).unwrap();

	let mut transfer = request.transfer();
    transfer.read_function(|buf| {
        Ok(data.read(buf).unwrap_or(0))
    }).unwrap();
    transfer.perform().unwrap();
}
