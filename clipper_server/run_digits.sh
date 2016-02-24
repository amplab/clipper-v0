#!/usr/bin/env bash

./target/release/clipper digits --conf=features.toml --mnist=/crankshaw-local/mnist/data/test.data --users=50 --traindata=30 --testdata=10
