
//! Correction Model Table interface and backing implementation.
//!
//! The implementation is a simple wrapper around Redis.
//!
//!


use std::hash::Hash;
use serde;
use serde::ser::Serialize;
use serde::de::Deserialize;
use bincode;
use bincode::serde::SerializeError;
use std::error::Error;
use redis::{self, Commands};
use std::marker::PhantomData;

pub trait CorrectionModelTable<S> where S: Serialize + Deserialize {

    // fn new() -> Self;

    // fn insert(&mut self, uid: u32, state: S);

    fn put(&mut self, uid: u32, state: &S) -> Result<(), String>;

    fn get(&self, uid: u32) -> Option<S>;


}


pub struct RedisCMT<S>
    where S: Serialize + Deserialize
{
    connection: redis::Connection,
    _correction_state_marker: PhantomData<S>,
}

impl<S> RedisCMT<S> where S: Serialize + Deserialize
{
    pub fn new_socket_connection() -> RedisCMT<S> {
        let client = redis::Client::open("unix:///tmp/redis.sock?db=1").unwrap();
        let con = client.get_connection().unwrap();
        RedisCMT {
            connection: con,
            _correction_state_marker: PhantomData,
        }
    }

    pub fn new_tcp_connection() -> RedisCMT<S> {
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let con = client.get_connection().unwrap();
        RedisCMT {
            connection: con,
            _correction_state_marker: PhantomData,
        }
    }
}


impl<S> CorrectionModelTable<S> for RedisCMT<S> where S: Serialize + Deserialize
{
    fn put(&mut self, uid: u32, state: &S) -> Result<(), String> {
        let bytes = try!(bincode::serde::serialize(state, bincode::SizeLimit::Infinite)
                             .map_err(|e| format!("{}", e.description())));
        let _: () = try!(self.connection
                             .set(uid, bytes)
                             .map_err(|e| format!("{}", e.description())));
        Ok(())

    }

    fn get(&self, uid: u32) -> Option<S> {
        let fetch_result: redis::RedisResult<Vec<u8>> = self.connection.get(uid);
        let stored_state: Option<S> = match fetch_result {
            Ok(bytes) => Some(bincode::serde::deserialize(&bytes).unwrap()),
            Err(_) => None,
        };
        stored_state
    }
}
