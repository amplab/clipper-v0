use std::hash::{Hash, SipHasher, Hasher};
use std::collections::HashMap;
use hashing::{HashKey, HashStrategy, EqualityHasher};
use configuration::{ClipperConf, ModelConf};
use std::sync::{mpsc, RwLock, Arc};
use server::Input;
use batching;


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
    fn fetch(&self, model: &String, input: &Input) -> Option<V>;


    /// Insert the key-value pair into the hash, evicting an older entry
    /// if necessary. Called by model batch schedulers
    fn put(&self, model: String, input: &Input, v: V);

    fn add_listener(&self,
                    model: &String,
                    input: &Input,
                    listener: Box<Fn(V) -> () + Send + Sync>);

}

pub struct SimplePredictionCache<V: 'static + Clone, H: HashStrategy + Send + Sync> {
    caches: Arc<HashMap<String, Cache<V>>>,
    // model_batchers: Arc<RwLock<HashMap<String, PredictionBatcher<SimplePredictionCache<V>, V>>>>,
    hash_strategy: H,
}

impl<V, H> SimplePredictionCache<V, H>
    where V: Clone,
          H: HashStrategy + Send + Sync
{
    /// Creates a new prediction cache for the provided set of models. All
    /// caches will be the same size and use the same hash function.
    pub fn new(model_conf: &Vec<ModelConf>, cache_size: usize) -> SimplePredictionCache<V, H> {
        let mut caches: HashMap<String, Cache<V>> = HashMap::new();
        for m in model_conf.iter() {
            caches.insert(m.name.clone(), Cache::new(cache_size));
        }

        // hashing might be stateful (e.g. LSH) which is why we create a new instance here
        let hash_strategy = H::new();

        SimplePredictionCache {
            caches: Arc::new(caches),
            hash_strategy: hash_strategy,
        }
    }
}

impl<V, H> PredictionCache<V> for SimplePredictionCache<V, H>
    where V: Clone,
          H: HashStrategy + Send + Sync
{
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


    fn put(&self, model: String, input: &Input, v: V) {
        match self.caches.get(&model) {
            Some(c) => {
                let hashkey = self.hash_strategy.hash(input, None);
                c.put(hashkey, v);
            }
            None => panic!("no cache for model {}", model),
        };
    }


    fn add_listener(&self,
                    model: &String,
                    input: &Input,
                    listener: Box<Fn(V) -> () + Send + Sync>) {
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
#[cfg_attr(rustfmt, rustfmt_skip)]
mod tests {
    use super::*;
    use hashing::{HashStrategy, HashKey};
    use server::Input;
    use configuration::ModelConf;

    struct IdentityHasher {}

    impl HashStrategy for IdentityHasher {
        pub fn new() -> IdentityHasher {
            IdentityHasher {}
        }

        pub fn hash(&self, input: &Input, salt: Option<i32>) -> HashKey {
            match input {
                &server::Input::Ints { ref i, length: l } => {
                    assert!(l == 1);
                    i[0] as u64
                }
                _ => panic!("wrong input type")
            }

        }
    }

    fn create_cache(size: usize) -> SimplePredictionCache<i32, IdentityHasher> {
        let m1 = ModelConf { name: "m1".to_string(), addresses: Vec::new(), num_outputs: 1 };
        let m2 = ModelConf { name: "m2".to_string(), addresses: Vec::new(), num_outputs: 1 };
        let model_conf = vec![m1, m2];
        let cache: SimplePredictionCache<i32, IdentityHasher> =
            SimplePredictionCache::new(model_conf, size);
        cache
    }

    #[test]
    fn fetch_existing_key() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], l: 1 };
        cache.put("m1".to_string(), input, 33);
        let result = cache.fetch("m1".to_string(), input);
        assert_eq!(result.unwrap(), 33);
        let res2 = cache.fetch("m2".to_string(), input);
        assert!(res2.is_none());
    }

    #[test]
    fn fetch_missing_key() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], l: 1 };
        let other_input = Input::Ints { i: vec![7], l: 1 };
        cache.put("m1".to_string(), input, 33);
        let result = cache.fetch("m1".to_string(), other_input);
        assert!(result.is_none());
    }

    #[test]
    #[should_panic]
    fn fetch_nonexistent_model() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], l: 1 };
        cache.put("m7".to_string(), input, 33);
    }


// Listener tests
    #[test]
    fn fire_listener_immediately() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], l: 1 };
        let mut listener_fired = false;
        cache.put("m1".to_string(), input, 33);
        cache.add_listener("m1".to_string(), input, move |x| {
            assert_eq!(x, 33);
            listener_fired = true;
        });
        assert!(listener_fired);
    }

    #[test]
    fn fire_listener_later() {
        let cache = create_cache(100);
        let input = Input::Ints { i: vec![3], l: 1 };
        let mut listener_fired = false;
        cache.add_listener("m1".to_string(), input, move |x| {
            assert_eq!(x, 33);
            listener_fired = true;
        });
        assert!(!listener_fired);
        cache.put("m1".to_string(), input, 33);
        assert!(listener_fired);
    }

    #[test]
    fn dont_fire_listener_on_cache_collision() {
        let cache = create_cache(10);
        let input = Input::Ints { i: vec![3], l: 1 };
        let mut listener_fired = false;
        cache.add_listener("m1".to_string(), input, move |x| {
            assert_eq!(x, 33);
            listener_fired = true;
        });
        assert!(!listener_fired);
        for j in 5..5000 {
            cache.put("m1".to_string(), Input::Ints { i: vec![j], l: 1 }, 66);
        }
        assert!(!listener_fired);
        cache.put("m1".to_string(), input, 33);
        assert!(listener_fired);
    }
}
