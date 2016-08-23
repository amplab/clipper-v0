#!/usr/bin/env sh
time docker build -f ClipperRestDockerFile -t clipper/rest ./
docker tag clipper/rest dcrankshaw/clipper
docker push dcrankshaw/clipper
