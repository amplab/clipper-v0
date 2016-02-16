#!/usr/bin/env bash

echo "starting sklearn server"
python feature_server.py 127.0.0.1:6001 spark /home/eecs/crankshaw/clipper/feature_servers/python/spark_models/10rf_pred_1 &
export SP1_PID=$!
echo "SP1 PID: $SP1_PID"

python feature_server.py 127.0.0.1:6002 spark /home/eecs/crankshaw/clipper/feature_servers/python/spark_models/100rf_pred_1 &
export SP2_PID=$!
echo "SP2 PID: $SP2_PID"

python feature_server.py 127.0.0.1:6003 spark /home/eecs/crankshaw/clipper/feature_servers/python/spark_models/500rf_pred_1 &
export SP3_PID=$!
echo "SP3 PID: $SP3_PID"



# python feature_server.py 127.0.0.1:6002 sklearn sklearn_model/predict_1_svm/predict_1_svm.pkl &
# export SK2_PID=$!
# echo "SK2 PID: $SK2_PID"
# python feature_server.py 127.0.0.1:6002 spark spark_model &
# export SPARK_PID=$!
# echo "SPARK PID: $SPARK_PID"
