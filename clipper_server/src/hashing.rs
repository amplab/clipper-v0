use std::cmp;
use rand::{thread_rng, Rng};
use server;
use std::hash::{Hash, SipHasher, Hasher};

pub type HashKey = u64;

pub trait FeatureHash {
    fn query_hash(&self, input: &server::Input, salt: Option<i32>) -> HashKey;
}

#[derive(Clone)]
pub struct SimpleHasher;

impl FeatureHash for SimpleHasher {
    fn query_hash(&self, input: &server::Input, salt: Option<i32>) -> HashKey {
        let mut hasher = SipHasher::new();
        match input {
            &server::Input::Floats {f: ref f, length: _ } => {
                // lame way to get around rust's lack of floating point equality. Basically,
                // we need a better hash function.
                let mut int_vec: Vec<i32> = Vec::new();
                for i in f.iter() {
                    let iv = (i * 10000000.0).round() as i32;
                    int_vec.push(iv);
                }
                int_vec.hash(&mut hasher);
                if let Some(s) = salt {
                    s.hash(&mut hasher);
                }
                hasher.finish()
            },
            &server::Input::Ints {i: ref i, length: _ } => {
                i.hash(&mut hasher);
                if let Some(s) = salt {
                    s.hash(&mut hasher);
                }
                hasher.finish()
            },
            &server::Input::Str {s: ref s} => {
                s.hash(&mut hasher);
                if let Some(sa) = salt {
                    sa.hash(&mut hasher);
                }
                hasher.finish()
            },
            &server::Input::Bytes {b: ref b, length: _} => {
                s.hash(&mut hasher);
                if let Some(sa) = salt {
                    sa.hash(&mut hasher);
                }
                hasher.finish()
            }
        }
    }
}


// pub struct LocalitySensitiveHash {
//     hash_size: i32
// }
//
// impl FeatureHash for LocalitySensitiveHash {
//     fn hash(&self, input: &Vec<f64>) -> u64 {
//         
//     }
// }
