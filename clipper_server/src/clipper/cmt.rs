
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
use redis::{self, Commands};


// TODO: Trait bounds on S? Serialize and Deserialize?
trait CorrectionModelTable<S> where S: Serialize + Deserialize {

    // fn new() -> Self;

    // fn insert(&mut self, uid: u32, state: S);

    fn put(&mut self, uid: u32, state: &S) -> Result<(), String>;

    fn get(&self, uid: u32) -> Option<S>;


}


//
struct RedisCMT {
    connection: redis::Connection,
}

impl RedisCMT {
    pub fn new_socket() -> RedisCMT {
        let client = redis::Client::open("unix:///tmp/redis.sock?db=1").unwrap();
        let con = client.get_connection().unwrap();
        RedisCMT { connection: con }
    }

    pub fn new_tcp() -> RedisCMT {
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let con = client.get_connection().unwrap();
        RedisCMT { connection: con }
    }
}


impl CorrectionModelTable<S> for RedisCMT where S: Serialize + Deserialize
{
    fn put(&mut self, uid: u32, state: &S) -> Result<(), String> {
        let bytes = try!(bincode::serde::serialize(state, bincode::SizeLimit::Infinite));
        let _: () = try(self.connection.set(uid, bytes));
        Ok(())

    }

    fn get(&self, uid: u32) -> Option<S> {
        let fetch_result: redis::RedisResult<Vec<u8>> = self.connection.get(uid);
        let stored_state: S = match fetch_result {
            Ok(bytes) => Some(bincode::serde::deserialize(&bytes).unwrap()),
            Err(_) => None,
        };
        stored_state
    }
}
