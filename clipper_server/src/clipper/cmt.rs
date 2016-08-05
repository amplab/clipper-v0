
//! Correction Model Table interface and backing implementation.
//!
//! The implementation is a simple wrapper around Redis.
//!
//!


use serde::ser::Serialize;
use serde::de::Deserialize;
use bincode;
use std::error::Error;
use redis::{self, Commands};
use std::marker::PhantomData;
use std::isize;
use server::{Input, Output, VersionedModel};
use std::hash::{Hash, SipHasher, Hasher};
use std::thread;
use std::time::Duration as StdDuration;

pub const REDIS_CMT_DB: u32 = 1;
pub const REDIS_UPDATE_DB: u32 = 2;
pub const REDIS_DEFAULT_PORT: u16 = 6379;
// pub const REDIS_DEFAULT_PORT: u16 = 32775;
pub const DEFAULT_REDIS_SOCKET: &'static str = "/tmp/redis.sock";



pub trait UpdateTable {

    // fn new() -> Self;

    /// Get the most `max_items` most recent updates for a user.
    /// If the list contains less than `max_items` items, all updates
    /// will be returned.
    ///
    /// If `max_items` is less than 0, all updates will be provided.
    fn get_updates(&self, uid: u32, max_items: isize) -> Result<Vec<(Input, Output)>, String>;

    fn add_update(&mut self, uid: u32, item: &Input, label: &Output) -> Result<(), String>;

}

pub struct RedisUpdateTable {
    connection: redis::Connection,
}


fn redis_connect(conn_string: String) -> redis::Connection {
    info!("Trying to connect to Redis");
    loop {
        match redis::Client::open(conn_string.as_str()) {
            Ok(client) => {
                match client.get_connection() {
                    Ok(con) => return con,
                    _ => {
                        info!("Couldn't connect to Redis: {}, sleeping 1 second",
                              conn_string);
                        thread::sleep(StdDuration::from_millis(500));
                    }
                }
            }
            _ => {
                info!("Couldn't connect to Redis: {}, sleeping 1 second",
                      conn_string);
                thread::sleep(StdDuration::from_millis(500));
            }
        }
    }

}

impl RedisUpdateTable {
    pub fn new_socket_connection(socket_file: &str, db: u32) -> RedisUpdateTable {
        // let conn_string = format!("unix:///tmp/redis.sock?db={}", REDIS_UPDATE_DB);
        let conn_string = format!("unix://{}?db={}", socket_file, db);
        info!("RedisUpdateTable connection string {}", conn_string);
        RedisUpdateTable { connection: redis_connect(conn_string) }
    }


    pub fn new_tcp_connection(addr: &str, port: u16, db: u32) -> RedisUpdateTable {
        // let conn_string = format!("redis://127.0.0.1/{}", REDIS_UPDATE_DB);
        let conn_string = format!("redis://{}:{}/{}", addr, port, db);
        info!("RedisUpdateTable connection string {}", conn_string);
        RedisUpdateTable { connection: redis_connect(conn_string) }
        // let client = redis::Client::open(conn_string.as_str()).unwrap();
        // let con = client.get_connection().unwrap();
        // RedisUpdateTable { connection: con }
    }
}

impl UpdateTable for RedisUpdateTable {
    /// Get the most `max_items` most recent updates for a user.
    /// If the list contains less than `max_items` items, all updates
    /// will be returned.
    ///
    /// If `max_items` is less than 0, all updates will be provided.
    fn get_updates(&self, uid: u32, max_items: isize) -> Result<Vec<(Input, Output)>, String> {
        if max_items == 0 {
            return Ok(Vec::new());
        }

        // NOTE: Redis LRANGE command is inclusive on both ends of the range
        let low_idx = 0;
        let high_idx = if max_items < 0 {
            isize::MAX
        } else {
            max_items - 1
        };
        let bytes: Vec<Vec<u8>> = try!(self.connection
                                           .lrange(uid, low_idx, high_idx)
                                           .map_err(|e| format!("{}", e.description())));


        let mut train_data: Vec<(Input, Output)> = Vec::with_capacity(bytes.len());
        for b in bytes {
            let example: (Input, Output) = try!(bincode::serde::deserialize(&b)
                                                    .map_err(|e| format!("{}", e.description())));
            train_data.push(example);
        }
        // let train_data: Vec<(Input, Output)> = bytes.iter()
        //                                             .map(|b| {
        //                                                 try!(bincode::serde::deserialize(b)
        //                                                          .map_err(|e| {
        //                                                              format!("{}", e.description())
        //                                                          }))
        //                                             })
        //                                             .collect::<Vec<_>>();
        Ok(train_data)
    }

    fn add_update(&mut self, uid: u32, item: &Input, label: &Output) -> Result<(), String> {
        let bytes = try!(bincode::serde::serialize(&(item, label), bincode::SizeLimit::Infinite)
                             .map_err(|e| format!("{}", e.description())));
        let _: () = try!(self.connection
                             .lpush(uid, bytes)
                             .map_err(|e| format!("{}", e.description())));
        Ok(())

    }
}


pub trait CorrectionModelTable<S> where S: Serialize + Deserialize {

    // TODO: how to store correction state version? Hash of Vec<VersionedModel>??
    fn put(&mut self,
           uid: u32,
           state: &S,
           versioned_models: &Vec<VersionedModel>)
           -> Result<(), String>;

    fn get(&self, uid: u32, versioned_models: &Vec<VersionedModel>) -> Result<S, String>;

}


pub struct RedisCMT<S>
    where S: Serialize + Deserialize
{
    connection: redis::Connection,
    _correction_state_marker: PhantomData<S>,
}

impl<S> RedisCMT<S> where S: Serialize + Deserialize
{
    pub fn new_socket_connection(socket_file: &str, db: u32) -> RedisCMT<S> {
        // let conn_string = format!("unix:///tmp/redis.sock?db={}", REDIS_CMT_DB);
        let conn_string = format!("unix://{}?db={}", socket_file, db);
        info!("RedisCMT connection string {}", conn_string);
        // RedisUpdateTable { connection: redis_connect(conn_string) }
        // let client = redis::Client::open(conn_string.as_str()).unwrap();
        // let con = client.get_connection().unwrap();
        RedisCMT {
            connection: redis_connect(conn_string),
            _correction_state_marker: PhantomData,
        }
    }

    pub fn new_tcp_connection(addr: &str, port: u16, db: u32) -> RedisCMT<S> {
        // let conn_string = format!("redis://127.0.0.1/{}", REDIS_CMT_DB);
        let conn_string = format!("redis://{}:{}/{}", addr, port, db);
        info!("RedisCMT connection string {}", conn_string);
        // let client = redis::Client::open(conn_string.as_str()).unwrap();
        // let con = client.get_connection().unwrap();
        RedisCMT {
            connection: redis_connect(conn_string),
            _correction_state_marker: PhantomData,
        }
    }
}


impl<S> CorrectionModelTable<S> for RedisCMT<S> where S: Serialize + Deserialize
{
    // TODO: should this be immutable
    fn put(&mut self,
           uid: u32,
           state: &S,
           versioned_models: &Vec<VersionedModel>)
           -> Result<(), String> {
        let bytes = try!(bincode::serde::serialize(state, bincode::SizeLimit::Infinite)
                             .map_err(|e| format!("{}", e.description())));
        let mut s = SipHasher::new();
        versioned_models.hash(&mut s);
        let version_hash = s.finish();
        let _: () = try!(self.connection
                             .hset(uid, version_hash, bytes)
                             .map_err(|e| format!("{}", e.description())));
        Ok(())

    }

    fn get(&self, uid: u32, versioned_models: &Vec<VersionedModel>) -> Result<S, String> {
        debug!("fetching state for uid: {}", uid);
        let mut s = SipHasher::new();
        versioned_models.hash(&mut s);
        let version_hash = s.finish();
        let bytes: Vec<u8> = try!(self.connection
                                      .hget(uid, version_hash)
                                      .map_err(|e| format!("{}", e.description())));
        let stored_state: S = try!(bincode::serde::deserialize(&bytes)
                                       .map_err(|e| format!("{}", e.description())));
        Ok(stored_state)
    }
}


#[cfg(test)]
// #[cfg_attr(rustfmt, rustfmt_skip)]
mod tests {
    // Some test code borrowed from redis-rs (https://github.com/mitsuhiko/redis-rs)
    //
    // Copyright (c) 2013 by Armin Ronacher.
    //
    // Some rights reserved.
    //
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the following conditions are
    // met:
    //
    //     * Redistributions of source code must retain the above copyright
    //       notice, this list of conditions and the following disclaimer.
    //
    //     * Redistributions in binary form must reproduce the above
    //       copyright notice, this list of conditions and the following
    //       disclaimer in the documentation and/or other materials provided
    //       with the distribution.
    //
    //     * The names of the contributors may not be used to endorse or
    //       promote products derived from this software without specific
    //       prior written permission.
    //
    // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    // "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    // LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    // A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    // OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    // SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    // LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    // DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    // THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    use super::*;
    // use redis::{self, Commands};
    use redis;
    use server::{Input, Output, VersionedModel};

    use std::process;
    use std::thread::sleep;
    use std::time::Duration;
    use std::collections::LinkedList;
    use rand::{thread_rng, Rng};

    // pub static SERVER_PORT: u16 = 38991;
    // pub static SERVER_UNIX_PATH: &'static str = "/tmp/clipper-redis-test.sock";

    pub struct RedisServer {
        pub process: process::Child,
        pub port: u16,
    }

    impl RedisServer {
        pub fn new(port: u16) -> RedisServer {
            let mut cmd = process::Command::new("redis-server");
            cmd.stdout(process::Stdio::null())
               .stderr(process::Stdio::null())
               .arg("--port")
               .arg(port.to_string())
               .arg("--bind")
               .arg("127.0.0.1");

            let process = cmd.spawn().unwrap();
            RedisServer {
                process: process,
                port: port,
            }
        }

        #[allow(dead_code)]
        pub fn wait(&mut self) {
            self.process.wait().unwrap();
        }


        pub fn get_client_addr(&self) -> redis::ConnectionAddr {
            redis::ConnectionAddr::Tcp("127.0.0.1".to_string(), self.port)
        }
    }

    impl Drop for RedisServer {
        fn drop(&mut self) {
            let _ = self.process.kill();
            let _ = self.process.wait();
        }
    }

    pub struct TestContext {
        pub server: RedisServer,
        pub client: redis::Client,
    }

    impl TestContext {
        fn new(port: u16, db: u32) -> TestContext {
            let server = RedisServer::new(port);

            let client = redis::Client::open(redis::ConnectionInfo {
                             addr: Box::new(server.get_client_addr()),
                             db: db as i64,
                             passwd: None,
                         })
                             .unwrap();
            let con;

            // try to connect in loop to ensure Redis server is running
            let millisecond = Duration::from_millis(1);
            loop {
                match client.get_connection() {
                    Err(err) => {
                        if err.is_connection_refusal() {
                            sleep(millisecond);
                        } else {
                            panic!("Could not connect: {}", err);
                        }
                    }
                    Ok(x) => {
                        con = x;
                        break;
                    }
                }
            }
            redis::cmd("FLUSHDB").execute(&con);
            // allow some time for flush to finish
            sleep(Duration::from_secs(1));

            TestContext {
                server: server,
                client: client,
            }
        }

        fn connection(&self) -> redis::Connection {
            self.client.get_connection().unwrap()
        }
    }

    fn random_ints(d: usize) -> Vec<i32> {
        let mut rng = thread_rng();
        rng.gen_iter::<i32>().take(d).collect::<Vec<i32>>()
    }

    #[test]
    #[allow(unused_variables)]
    fn cmt_put_get() {
        let db = 0;
        let port = 38890;
        let ctx = TestContext::new(port, db);
        let mut cmt: RedisCMT<Vec<i32>> = RedisCMT::new_tcp_connection("127.0.0.1", port, db);

        let user_id = 33;
        let state = vec![4, 3, 2, 6, 73345, 2312];
        let versioned_models = vec![VersionedModel {
                                        name: "m1".to_string(),
                                        version: Some(0),
                                    },
                                    VersionedModel {
                                        name: "m3".to_string(),
                                        version: Some(1),
                                    }];
        cmt.put(user_id, &state, &versioned_models).unwrap();
        let fetched_state = cmt.get(user_id, &versioned_models).unwrap();
        assert_eq!(state, fetched_state);
    }

    #[test]
    #[allow(unused_variables)]
    fn update_table_add_updates() {
        let db = 0;
        let port = 38891;
        let ctx = TestContext::new(port, db);
        let l = 3;
        let input_vec = vec![Input::Ints {
                                 i: random_ints(l),
                                 length: l as i32,
                             },
                             Input::Ints {
                                 i: random_ints(l),
                                 length: l as i32,
                             }];

        let output_vec: Vec<Output> = vec![3.3, 323.4];
        let uid: u32 = 44;
        // make sure key doesn't already exist
        let con = ctx.connection();
        redis::cmd("DEL").arg(uid).execute(&con);


        let mut update_table = RedisUpdateTable::new_tcp_connection("127.0.0.1", port, db);
        update_table.add_update(uid, &input_vec[0], &output_vec[0]).unwrap();
        let fetched_update: Vec<(Input, Output)> = update_table.get_updates(uid, 5).unwrap();
        assert_eq!(fetched_update.len(), 1);
        assert_eq!(fetched_update[0].0, input_vec[0]);
        assert_eq!(fetched_update[0].1, output_vec[0]);
    }

    #[test]
    #[allow(unused_variables)]
    fn update_table_window_updates() {
        let db = 0;
        let port = 38892;
        let ctx = TestContext::new(port, db);
        let num_inputs = 45;
        let input_len = 33;
        let uid = 21;
        // make sure key doesn't already exist
        let con = ctx.connection();
        redis::cmd("DEL").arg(uid).execute(&con);
        // con.del(uid).unwrap();
        let mut rng = thread_rng();
        let mut local_updates = LinkedList::new();
        let mut update_table = RedisUpdateTable::new_tcp_connection("127.0.0.1", port, db);
        for i in 0..num_inputs {
            let new_input = Input::Ints {
                i: random_ints(input_len),
                length: input_len as i32,
            };
            let output = rng.gen::<f64>();
            update_table.add_update(uid, &new_input, &output).unwrap();
            local_updates.push_front((new_input, output));
        }

        let window_ten_results = update_table.get_updates(uid, 10).unwrap();
        assert_eq!(window_ten_results.len(), 10);
        for x in window_ten_results.iter().zip(local_updates.iter()) {
            assert_eq!(x.0, x.1);
        }
    }



}
