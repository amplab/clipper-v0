use std::hash::{Hash, SipHasher, Hasher};
use std::collections::HashMap;
use hashing::{HashKey, HashStrategy};
use configuration::ModelConf;
use std::sync::{RwLock, Arc};
use server::Input;


pub struct CacheListener<V: 'static + Clone> {
    /// Return `true` to be dropped from the listener list,
    /// otherwise the listener will be kept in place.
    listener: Box<Fn(HashKey, V) -> bool + Send + Sync>,
}


struct HashEntry<V: 'static + Clone> {
    pub key: Option<HashKey>,
    pub value: Option<V>,
    pub listeners: Vec<CacheListener<V>>,
}

pub struct Cache<V: 'static + Clone> {
    data: Arc<Vec<RwLock<HashEntry<V>>>>,
}


impl<V: 'static + Clone> Cache<V> {
    pub fn new(size: usize) -> Cache<V> {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(RwLock::new(HashEntry {
                key: None,
                value: None,
                listeners: Vec::new(),
            }))
        }
        Cache { data: Arc::new(data) }
    }

    fn get_index(&self, hashkey: HashKey) -> usize {
        let mut hasher = SipHasher::new();
        hashkey.hash(&mut hasher);
        (hasher.finish() % self.data.len() as u64) as usize
    }

    pub fn put(&self, hashkey: HashKey, value: V) {
        let index = self.get_index(hashkey);
        let mut entry = self.data[index].write().unwrap();
        entry.key = Some(hashkey);
        entry.value = Some(value.clone());
        entry.listeners.retain(|l| !(l.listener)(hashkey, value.clone()));

        // entry.listeners
        //      .into_iter()
        //      .filter(|l| !(l.listener)(hashkey, value.clone()))
        //      .collect();
        // entry.listeners =
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
        if entry.key.is_some() {
            let k = entry.key.clone().unwrap();
            let v = entry.value.clone().unwrap();
            entry.listeners.retain(|l| !(l.listener)(k.clone(), v.clone()));
        }
        // if let Some(ref k) = entry.key {
        //     let v = entry.value.as_ref().unwrap();
        //     entry.listeners.retain(|l| !(l.listener)(k.clone(), v.clone()));
        // entry.listeners = (*entry)
        //                       .listeners
        //                       .into_iter()
        //                       .filter(|l| !(l.listener)(k.clone(), v.clone()))
        //                       .collect();
        // }
    }
}


/// Handles higher-level semantics of caching predictions. This includes
/// locality-sensitive hashing and other specialized hash functions.
pub trait PredictionCache<V: 'static + Clone> {
    /// Look up the key in the cache and return immediately
    fn fetch(&self, model: &String, input: &Input, salt: Option<i32>) -> Option<V>;


    /// Insert the key-value pair into the hash, evicting an older entry
    /// if necessary. Called by model batch schedulers
    fn put(&self, model: String, input: &Input, v: V, salt: Option<i32>);

    fn add_listener(&self,
                    model: &String,
                    input: &Input,
                    salt: Option<i32>,
                    listener: Box<Fn(V) -> () + Send + Sync>);
}

pub struct SimplePredictionCache<V: 'static + Clone, H: HashStrategy + Send + Sync> {
    caches: Arc<RwLock<HashMap<String, Cache<V>>>>,
    hash_strategy: H,
    cache_size: usize,
}

impl<V, H> SimplePredictionCache<V, H>
    where V: Clone,
          H: HashStrategy + Send + Sync
{
    /// Similar to `Vec::new()`, this creates an empty prediction cache.
    pub fn new(cache_size: usize) -> SimplePredictionCache<V, H> {
        let hash_strategy = H::new();

        let caches: HashMap<String, Cache<V>> = HashMap::new();
        SimplePredictionCache {
            caches: Arc::new(RwLock::new(caches)),
            hash_strategy: hash_strategy,
            cache_size: cache_size,
        }

    }

    /// Similar to `Vec::with_capacity(), creates a new prediction cache
    /// for the provided set of models. All caches will be the same
    /// size and use the same hash function.
    pub fn with_models(model_conf: &Vec<ModelConf>,
                       cache_size: usize)
                       -> SimplePredictionCache<V, H> {
        let mut caches: HashMap<String, Cache<V>> = HashMap::new();
        for m in model_conf.iter() {
            caches.insert(m.name.clone(), Cache::new(cache_size));
        }

        // The hash function might be stateful (e.g. LSH) which is why we
        // create a new instance here.
        let hash_strategy = H::new();

        SimplePredictionCache {
            caches: Arc::new(RwLock::new(caches)),
            hash_strategy: hash_strategy,
            cache_size: cache_size,
        }
    }
}

impl<V, H> PredictionCache<V> for SimplePredictionCache<V, H>
    where V: Clone,
          H: HashStrategy + Send + Sync
{
    fn fetch(&self, model: &String, input: &Input, salt: Option<i32>) -> Option<V> {
        let cache_reader = self.caches.read().unwrap();
        match cache_reader.get(model) {
            Some(c) => {
                let hashkey = self.hash_strategy.hash(input, salt);
                c.get(hashkey)
            }
            None => {
                warn!("no cache for model {}", model);
                None
            }
        }
    }

    fn put(&self, model: String, input: &Input, v: V, salt: Option<i32>) {
        // very basic OCC
        let mut maybe_add_cache = false;
        let hashkey = self.hash_strategy.hash(input, salt);
        {
            let cache_reader = self.caches.read().unwrap();
            match cache_reader.get(&model) {
                Some(c) => {
                    c.put(hashkey, v.clone());
                }
                None => {
                    maybe_add_cache = true;
                    // If we haven't seen this model before, assume
                    // it's a new model and create a new cache for it.
                    // let model_cache = Cache::new(cache_size);
                    // info!("Creating a cache for new model {}", model);
                    // let hashkey = self.hash_strategy.hash(input, None);
                    // model_cache.put(hashkey, v);
                    // let cache_writer = caches.write.unwrap();
                    // cache_writer.insert(model.clone(), model_cache);
                }
            };
        }
        if maybe_add_cache {
            let mut cache_writer = self.caches.write().unwrap();
            let mut create_cache = false;
            {
                if let Some(c) = cache_writer.get(&model) {
                    c.put(hashkey, v.clone());
                } else {
                    create_cache = true;
                }
            }
            if create_cache {
                let model_cache = Cache::new(self.cache_size);
                info!("Creating a cache for new model {}", model);
                model_cache.put(hashkey, v);
                cache_writer.insert(model.clone(), model_cache);
            }

            // let model_cache_option = cache_writer.get(&model);
            // // Check if some other call to put() has added the model cache in
            // // the mean time, because we only want to create the model once.
            // match model_cache_option {
            //     Some(c) => {
            //         // let hashkey = self.hash_strategy.hash(input, None);
            //         c.put(hashkey, v);
            //     }
            //     None => {
            //         // If we haven't seen this model before, assume
            //         // it's a new model and create a new cache for it.
            //         let model_cache = Cache::new(self.cache_size);
            //         info!("Creating a cache for new model {}", model);
            //         model_cache.put(hashkey, v);
            //         cache_writer.insert(model.clone(), model_cache);
            //     }
            // };
        }
    }

    fn add_listener(&self,
                    model: &String,
                    input: &Input,
                    salt: Option<i32>,
                    listener: Box<Fn(V) -> () + Send + Sync>) {

        // Simple form of OCC
        let maybe_add_cache;
        let hashkey = self.hash_strategy.hash(input, salt);
        let wrapped_listener = Box::new(move |h, v| {
            if h == hashkey {
                (listener)(v);
                true
            } else {
                false
            }
        });
        {
            // First check to see if a cache for this model exists
            let cache_reader = self.caches.read().unwrap();
            match cache_reader.get(model) {
                Some(_) => {
                    maybe_add_cache = false;
                    // c.add_listener(hashkey, CacheListener { listener: wrapped_listener });
                }
                None => {
                    maybe_add_cache = true;
                }
            };
        }
        if maybe_add_cache {
            // If the model cache didn't exist during the previous check,
            // we lock the cache with a write lock and check again, creating it
            // if it doesn't exist.
            let mut cache_writer = self.caches.write().unwrap();
            let create_cache;
            {
                if let Some(_) = cache_writer.get(model) {
                    create_cache = false;
                } else {
                    create_cache = true;
                }
            }
            if create_cache {
                let model_cache = Cache::new(self.cache_size);
                info!("Creating a cache for new model {}", model);
                model_cache.add_listener(hashkey, CacheListener { listener: wrapped_listener });
                cache_writer.insert(model.clone(), model_cache);
            } else {
                cache_writer.get(model)
                    .unwrap()
                    .add_listener(hashkey, CacheListener { listener: wrapped_listener });
            }
        } else {
            // Model caches are never deleted, so if it existed during the
            // earlier check it is guaranteed to still exist.
            let cache_reader = self.caches.read().unwrap();
            cache_reader.get(model)
                .unwrap()
                .add_listener(hashkey, CacheListener { listener: wrapped_listener });

        }
    }
}


#[cfg(test)]
#[cfg_attr(rustfmt, rustfmt_skip)]
mod tests {
    use super::*;
    use hashing::{HashStrategy, HashKey};
    use server::Input;
    #[allow(unused_imports)]
    use configuration::ModelConf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[derive(Clone)]
    struct IdentityHasher {}

    impl HashStrategy for IdentityHasher {
        fn new() -> IdentityHasher {
            IdentityHasher {}
        }

        #[allow(unused_variables)]
        fn hash(&self, input: &Input, salt: Option<i32>) -> HashKey {
            match input {
                &Input::Ints { ref i, length: l } => {
                    assert!(l == 1);
                    i[0] as u64
                }
                _ => panic!("wrong input type")
            }

        }
    }

    fn create_cache(size: usize) -> SimplePredictionCache<i32, IdentityHasher> {
// let m1 = ModelConf { name: "m1".to_string(), addresses: Vec::new(), num_outputs: 1, version: 0 };
// let m2 = ModelConf { name: "m2".to_string(), addresses: Vec::new(), num_outputs: 1, version: 0 };
// let model_conf = vec![m1, m2];
// let cache: SimplePredictionCache<i32, IdentityHasher> =
//     SimplePredictionCache::with_models(&model_conf, size);
        let cache: SimplePredictionCache<i32, IdentityHasher> =
            SimplePredictionCache::new(size);
        cache
    }

    #[test]
    fn fetch_existing_key() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], length: 1 };
        cache.put("m1".to_string(), &input, 33);
        let result = cache.fetch(&"m1".to_string(), &input);
        assert_eq!(result.unwrap(), 33);
        let res2 = cache.fetch(&"m2".to_string(), &input);
        assert!(res2.is_none());
    }

    #[test]
    fn fetch_missing_key() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], length: 1 };
        let other_input = Input::Ints { i: vec![7], length: 1 };
        cache.put("m1".to_string(), &input, 33);
        let result = cache.fetch(&"m1".to_string(), &other_input);
        assert!(result.is_none());
    }

    #[test]
// #[should_panic]
    fn fetch_nonexistent_model() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], length: 1 };
        cache.put("m7".to_string(), &input, 33);
    }


// Listener tests
    #[test]
    fn fire_listener_immediately() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], length: 1 };
        let listener_fired = Arc::new(AtomicBool::new(false));
        cache.put("m1".to_string(), &input, 33);
        {
            let listener_fired = listener_fired.clone();
            cache.add_listener(&"m1".to_string(), &input, Box::new(move |x| {
                assert_eq!(x, 33);
                listener_fired.store(true, Ordering::SeqCst);
            }));
        }
        assert!(listener_fired.load(Ordering::SeqCst));
    }

    #[test]
    fn fire_listener_later() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], length: 1 };
        let listener_fired = Arc::new(AtomicBool::new(false));
        {
            let listener_fired = listener_fired.clone();
            cache.add_listener(&"m1".to_string(), &input, Box::new(move |x| {
                assert_eq!(x, 33);
                listener_fired.store(true, Ordering::SeqCst);
            }));
        }
        assert!(!listener_fired.load(Ordering::SeqCst));
        cache.put("m1".to_string(), &input, 33);
        assert!(listener_fired.load(Ordering::SeqCst));
    }

    #[test]
    fn dont_fire_listener_on_cache_collision() {
        let cache = create_cache(10);
        let input = Input::Ints { i: vec![3], length: 1 };
        let listener_fired = Arc::new(AtomicBool::new(false));
        {
            let listener_fired = listener_fired.clone();
            cache.add_listener(&"m1".to_string(), &input, Box::new(move |x| {
                assert_eq!(x, 33);
                listener_fired.store(true, Ordering::SeqCst);
            }));
        }
        assert!(!listener_fired.load(Ordering::SeqCst));
        for j in 5..5000 {
            cache.put("m1".to_string(), &Input::Ints { i: vec![j], length: 1 }, 66);
        }
        assert!(!listener_fired.load(Ordering::SeqCst));
        cache.put("m1".to_string(), &input, 33);
        assert!(listener_fired.load(Ordering::SeqCst));
    }

    #[test]
    #[should_panic]
    fn deal_with_versioned_cache() {
// panic!("We don't deal with versioned cache yet");
        unimplemented!();
    }



}
