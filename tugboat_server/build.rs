extern crate capnpc;

fn main() {
    ::capnpc::compile("schema", &["schema/feature.capnp"]).unwrap();
}
