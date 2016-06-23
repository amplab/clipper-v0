//! There are two components to the prediction cache: the hashing, covered in
//! the [hashing](hashing.rs) module, and the cache internals, described here.
//!
//!
//!

use std::hash::{Hash, SipHasher, Hasher};
use std::collections::HashMap;
use hashing::{HashKey, HashStrategy, EqualityHasher};
use configuration::{ClipperConf, ModelConf};
use batching;


struct CacheListener<V> {
    /// Return `true` to be dropped from the listener list,
    /// otherwise the listener will be kept in place.
    listener: Box<Fn(&HashKey, V) -> bool>,
}


struct HashEntry<V> {
    pub key: Option<HashKey>,
    pub value: Option<V>,
    pub listeners: Vec<CacheListener<V>>,
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

    fn get_index(&self, hashkey: HashKey) -> usize {
        let mut hasher = SipHasher::new();
        hashkey.hash(&mut hasher);
        (hasher.finish() % self.data.len()) as usize
    }

    pub fn put(&self, hashkey: HashKey, value: V) {
        let index = self.get_index(hashkey);
        let mut entry = self.data[index].write().unwrap();
        entry.key = Some(hashkey);
        entry.value = Some(value);
        entry.listeners = (*entry).listeners.into_iter().filter(|l| !(l)(k, v.clone())).collect();
    }

    pub fn get(&self, hashkey: HashKey) -> Option<V> {
        let index = self.get_index(hashkey);
        let entry = self.data[index].write().unwrap();
        match entry.key {
            Some(k) => {
                if k == hashkey {
                    entry.value.clone()
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Adds a listener to the provided hash key.
    /// If the entry for that key is not empty, the listener will be fired immediately
    /// with the current value.
    pub fn add_listener(&self, hashkey: HashKey, new_listener: CacheListener<V>) {
        let index = self.get_index(hashkey);
        let mut entry = self.data[index].write().unwrap();
        entry.listeners.push(new_listener);
        if let Some(ref k) = entry.key {
            let v = entry.value.as_ref().unwrap();
            entry.listeners = (*entry).listeners.into_iter().filter(|l| !(l)(k, v)).collect();
        }
    }
}


/// Handles higher-level semantics of caching predictions. This includes
/// locality-sensitive hashing and other specialized hash functions, as well
/// as sending cache requests on to the batching layer for model evaluation
/// when needed.
pub trait PredictionCache<V> {

    /// Look up the key in the cache and return immediately
    fn fetch(&self, model: String, input: &Input) -> Option<V>;

    // /// Request that the cache entry be populated.
    // /// If there is already an entry in the cache, nothing happens.
    // /// Otherwise, the prediction will be sent to the batch scheduler for evaluation.
    // fn request(&self, model: String, input: Input);

    /// Insert the key-value pair into the hash, evicting an older entry
    /// if necessary. Called by model batch schedulers
    fn put(&self, model: String, input: &Input, v: V);

    fn add_listener(&self, input: &Input, listener: Fn(V) -> ());

}

pub struct SimplePredictionCache<V, H: HashStrategy> {
    caches: HashMap<String, Cache<V>>,
    // model_batchers: Arc<RwLock<HashMap<String, PredictionBatcher<SimplePredictionCache<V>, V>>>>,
    hash_strategy: H,
}

impl<V, H: HashStrategy> SimplePredictionCache<V, H> {
    /// Creates a new prediction cache for the provided set of models. All
    /// caches will be the same size and use the same hash function.
    pub fn new(model_conf: &Vec<ModelConf>, cache_size: usize) -> PredictionCache {
        let mut caches: HashMap<String, Cache<V>> = HashMap::new();
        for m in model_conf.iter() {
            caches.insert(m.name.clone(), Cache::new(cache_size));
        }

        // hashing might be stateful (e.g. LSH) which is why we create a new instance here
        let hash_strategy = H::new();

        SimplePredictionCache {
            caches: caches,
            hash_strategy: hash_strategy,
        }
    }
}

impl<V> PredictionCache<V> for SimplePredictionCache<V> {
    fn fetch(&self, model: &String, input: &Input) -> Option<V> {
        match self.caches.get(model) {
            Some(c) => {
                let hashkey = self.hash_strategy.hash(input, None);
                c.get(hashkey)
            }
            None => panic!("no cache for model {}", model),
        }
    }

    // fn request(&self, model: String, input: Input) {
    //     if self.fetch(model.clone(), &input).is_none() {
    //         // TODO: schedule for evaluation
    //         unimplemented!();
    //     }
    // }


    fn put(&self, model: &String, input: &Input, v: V) {
        match self.caches.get(model) {
            Some(c) => {
                let hashkey = self.hash_strategy.hash(input, None);
                c.put(hashkey, v);
            }
            None => panic!("no cache for model {}", model),
        };
    }


    fn add_listener(&self, model: &String, input: &Input, listener: Fn(V) -> ()) {
        match self.caches.get(model) {
            Some(c) => {
                let hashkey = self.hash_strategy.hash(input, None);
                let wrapped_listener = Box::new(move |h, v| {
                    if h == hashkey {
                        (listener)(v);
                        true
                    } else {
                        false
                    }
                });
                c.add_listener(hashkey, CacheListener { listener: wrapped_listener });
            }
            None => panic!("no cache for model {}", model),
        };
    }
}


#[cfg(test)]
mod tests {
    use super::*;



}
