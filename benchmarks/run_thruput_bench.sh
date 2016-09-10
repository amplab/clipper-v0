#!/usr/bin/env bash


RUST_LOG=info RUST_BACKTRACE=1 CLIPPER_CONF_PATH="exp_conf.toml" CLIPPER_BENCH_COMMAND="thruput" ./target/release/clipper-benchmarks
exit 0

