#!/usr/bin/env bash

######### PYSPARK SVM ################

killall python
sleep 10
killall python
python rpc_feature_server.py 127.0.0.1 6001 sparksvm spark_models/svm_predict_1 &> /dev/null &
sleep 15
ps aux | grep [p]ython



