#!/usr/bin/env bash

for i in {0..9}
do
  echo "starting sklearn server $i"
  # python feature_server.py 127.0.0.1:600$i spark spark_models/lg_predict_$i &
  python feature_server.py 127.0.0.1:600$i sklearn sklearn_models/predict_"$i"_svm/predict_"$i"_svm.pkl &

done
