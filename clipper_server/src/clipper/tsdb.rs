use curl::easy::Easy;
use std::io::Read;
use url::{Url, form_urlencoded};
use metrics::{Counter};
use std::sync::{Arc};
use time;

//const BASE_URL: &'static str = "http://localhost:8086";
//const CREATE_DB_BODY: &'static str = "CREATE DATABASE";

pub fn create(db_name: &str) {
    let url = "http://localhost:8086/query";
	let encoded_body = form_urlencoded::Serializer::new(String::new())
        .append_pair("q", &format!("{} {}", "CREATE DATABASE", db_name))
        .finish();
    send_post_request(url, &encoded_body);
}

pub fn write_counter(db_name: &str, counter: &Arc<Counter>) {
	let raw = &format!("http://localhost:8086/write?db={}", db_name);
	let encoded_url = Url::parse(raw).unwrap();
    let body = format!("{} value={} {}", counter.name, counter.value().to_string(), time::precise_time_ns().to_string());
    send_post_request(encoded_url.as_str(), &body);
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
