#!/usr/bin/env bash

RUST_LOG=info RUST_BACKTRACE=1 CLIPPER_CONF_PATH="cifar10.toml" CLIPPER_BENCH_COMMAND="cifar" ./target/release/clipper-benchmarks

