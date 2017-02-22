#!/usr/bin/env sh

# first build base image
sudo docker build -t clipper/fast-rpc -f FastRPC_Dockerfile ./
sudo time docker build -t clipper/sklearn-dt-dev -f SklearnDTMW_Dockerfile ./

