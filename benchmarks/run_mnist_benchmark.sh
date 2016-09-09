#!/usr/bin/env bash

RUST_LOG=info RUST_BACKTRACE=1 CLIPPER_CONF_PATH="digits_bench.toml" CLIPPER_BENCH_COMMAND="digits" ./target/release/clipper-benchmarks

