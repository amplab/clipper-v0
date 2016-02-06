
use capnp_rpc::{RpcSystem, twoparty, rpc_twoparty_capnp};
use feature_capnp::feature;
use std::sync::{Arc, RwLock, mpsc};

trait FeatureCache {

}

struct FeatureConnection {

  name: String,
  rpc: feature::Client, // whatever this is supposed to be
  rx: mpsc::Receiver, // queue to receive more event requests on,
                      // basically on every event tick the receiver
                      // should drain the queue. Really, this should be
                      // connected to the event loop queue.
                      // This can for sure be done with mio event loops,
                      // let's make sure I can do with it gj

  cache: RwLock<Arc<FeatureCache>>, // shared feature cache for this feature
  
}




fn main() {




}
