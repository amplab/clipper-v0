extern crate gcc;

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;


fn main() {
    let current_dir = env::current_dir().unwrap();
    env::set_current_dir(Path::new("liblinear/blas")).unwrap();
    assert!(Command::new("make")
                .args(&["clean"])
                .status()
                .unwrap()
                .success());

    assert!(Command::new("make")
                .args(&["blas", "OPTFLAGS=-Wall -Wconversion -O3 -fPIC", "CC=gcc"])
                .status()
                .unwrap()
                .success());

    env::set_current_dir(current_dir).unwrap();

    gcc::Config::new()
        .cpp(true)
        .include("liblinear")
        .include("liblinear/blas")
        .file("liblinear/linear.cpp")
        .file("liblinear/linear.h")
        .file("liblinear/tron.cpp")
        .file("liblinear/tron.h")
        .compile("liblinear.a");
}
