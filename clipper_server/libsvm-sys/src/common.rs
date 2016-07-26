use libc::{c_int, c_double};
use std::mem;
use std::clone::Clone;

#[repr(C)]
// #[derive(Copy, Debug)]
#[derive(Debug, Copy)]
pub struct Struct_feature_node {
    pub index: c_int,
    pub value: c_double,
}

impl Clone for Struct_feature_node {
    fn clone(&self) -> Self {
        *self
    }
}

impl Default for Struct_feature_node {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}
