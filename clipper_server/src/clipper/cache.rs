


//! There are two components to the prediction cache: the hashing, covered in
//! the [hashing](hashing.rs) module, and the cache internals, described here.
//!
//!
//!

use std::collections::HashMap;
use hashing::HashKey;


struct CacheListener<V> {
    listener: Box<Fn(V) -> ()>,
}


struct HashEntry<V> {
    key: Option<HashKey>,
    value: Option<V>,
    listeners: Vec<CacheListener<V>>,
}

pub struct Cache<V> {
    data: Arc<Vec<RwLock<HashEntry<V>>>>,
}


impl<V> Cache<V> {
    pub fn new(size: usize) -> Cache<V> {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(HashEntry {
                key: None,
                value: None,
                listeners: Vec::new(),
            })
        }
        Cache { data: data }
    }

    pub fn put(hash: HashKey, value: V) -> Option<V> {}

    pub fn get(hash: HashKey) -> Option<V> {}
}


// TODO: Think about this design a little more. Where do things happen?
// TODO: how to combine two Vs with the same HashKey? UDF
trait PredictionCache<V> {

    fn fetch(&self, k: HashKey) -> Option<V>;

    fn put(&self, k: HashKey, v: V);

}
