#!/usr/bin/env bash

RUST_LOG=info RUST_BACKTRACE=1 CLIPPER_CONF_PATH="inception.toml" CLIPPER_BENCH_COMMAND="imagenet" ./target/release/clipper-benchmarks

