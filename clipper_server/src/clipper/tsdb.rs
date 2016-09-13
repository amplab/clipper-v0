use curl::easy::Easy;
use std::io::Read;
use url::{Url, form_urlencoded};
use metrics::{Counter, Meter};
use std::sync::{Arc};
use time;
use regex::Regex;

//const BASE_URL: &'static str = "http://localhost:8086";
//const CREATE_DB_BODY: &'static str = "CREATE DATABASE";

pub fn create(db_name: &str) {
    let url = "http://localhost:8086/query";
	let encoded_body = form_urlencoded::Serializer::new(String::new())
        .append_pair("q", &format!("{} \"{}\"", "CREATE DATABASE", db_name))
        .finish();
    send_post_request(url, &encoded_body);
}

pub struct Write<'a> {
	db_name: &'a str,
	timestamp: String,
	write_ops: Vec<String>,
}

impl <'a> Write<'a> {
	pub fn new(db_name: &str) -> Write {
		Write {
			db_name: db_name,
			timestamp: time::get_time().nsec.to_string(),
			write_ops: Vec::new(),
		}
	}

	pub fn append_counter(&mut self, counter: &Arc<Counter>) {
		let op = format!("{} value={} {}", counter.name, counter.value(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	// pub fn append_ratio(&mut self, ratio: &Arc<RatioCounter>) {

	// }

	pub fn append_meter(&mut self, meter: &Arc<Meter>) {
		let re = Regex::new(r"\s").unwrap();
		let unit = format!("units={}", re.replace_all(&meter.unit, "-"));
		let op = format!("{},{} value={} {}", meter.name, unit, meter.get_rate_secs(), self.timestamp);
		info!("Added influx write op: {}", op);
		self.write_ops.push(op);
	}

	// pub fn append_histogram(&mut self, histogram: &Arc<Histogram>) {

	// }

	pub fn execute(&mut self) {
		let raw_url = &format!("http://localhost:8086/write?db={}", self.db_name);
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
