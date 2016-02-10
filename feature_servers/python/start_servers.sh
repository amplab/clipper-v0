#!/usr/bin/env bash

echo "starting sklearn server"
python feature_server.py 127.0.0.1:6001 sklearn sklearn_model/predict_1_svm/predict_1_svm.pkl &
export SK_PID=$!
echo "SK PID: $SK_PID"
python feature_server.py 127.0.0.1:6002 spark spark_model &
export SPARK_PID=$!
echo "SPARK PID: $SPARK_PID"
