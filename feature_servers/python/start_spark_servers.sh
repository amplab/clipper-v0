#!/usr/bin/env bash

for i in {1..9}
do
  echo "starting spark server $i"
  python rpc_feature_server.py 127.0.0.1 600$i sparklr spark_models/lg_predict_$i &
  # python feature_server.py 127.0.0.1:600$i sklearn sklearn_models/predict_"$i"_svm/predict_"$i"_svm.pkl &

 done

echo "starting spark server 10"
python rpc_feature_server.py 127.0.0.1 6010 sparklr spark_models/lg_predict_10 &
