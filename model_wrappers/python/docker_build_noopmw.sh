#!/usr/bin/env sh

# first build base image
sudo docker build -t clipper/fast-rpc -f FastRPC_Dockerfile ./
sudo time docker build -t clipper/noop-mw-dev -f NoopMW_Dockerfile ./

