extern crate gcc;

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;


fn main() {


    gcc::Config::new()
        .cpp(true)
        .include("libsvm")
        .file("libsvm/svm.cpp")
        .file("libsvm/svm.h")
        .compile("libsvm.a");

}
