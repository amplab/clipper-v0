from __future__ import print_function
import time
import os, shutil, sys, datetime
# from subprocess import call, Popen, PIPE
import subprocess32 as subprocess
from scipy.io import wavfile
import numpy as np
import json


import rpc_feature_server as rpc
from multiprocessing import Process

CLIPPER_SERVER_BASE = "/crankshaw-local/clipper/clipper_server"

toml_str_baseline = """
users=1000
train_examples=50
test_examples=100
mnist_path="/crankshaw-local/mnist/data/test.data"
num_events=1000000
worker_threads=2
target_qps=%(qps)d
query_batch_size=50
max_features=10
salt_hash=true
feature_batch_size=1
"""

toml_str_batching = """
users=1000
train_examples=50
test_examples=100
mnist_path="/crankshaw-local/mnist/data/test.data"
num_events=1000000
worker_threads=2
target_qps=%(qps)d
query_batch_size=200
max_features=10
salt_hash=true
feature_batch_size=100
"""

toml_str_max_features = """
users=1000
train_examples=50
test_examples=100
mnist_path="/crankshaw-local/mnist/data/test.data"
num_events=1000000
worker_threads=2
target_qps=%(qps)d
query_batch_size=100000
max_features=7
salt_hash=true
feature_batch_size=100
"""

toml_str_no_features = """
users=1000
train_examples=50
test_examples=100
mnist_path="/crankshaw-local/mnist/data/test.data"
num_events=1000000
worker_threads=2
target_qps=%(qps)d
query_batch_size=5000
max_features=0
salt_hash=true
feature_batch_size=100
"""

def start_feature(mp, ip, port):
    # model = rpc.SparkSVMServer(mp)
    p = Process(target=rpc.start_svm_from_mp, args=(mp,ip,port))
    p.start()
    return p # to kill: p.terminate()

def start_features():
    procs = []
    for i in range(1,11):
        mp = ("/crankshaw-local/clipper/feature_servers/"
              "python/spark_models/svm_predict_%d" % i)
        procs.append(start_feature(mp, "127.0.0.1", (6000 + i)))
    done = False
    time.sleep(10)
    # make sure they all started. Sometimes a socket is still bound, so
    # we need to wait a little bit and retry
    while not done:
        for i in range(len(procs)):
            if not procs[i].is_alive():
                print("restarting %d" % (i+1))
                mp = ("/crankshaw-local/clipper/feature_servers/"
                      "python/spark_models/svm_predict_%d" % (i + 1))
                new_p = start_feature(mp, "127.0.0.1", (6000 + i))
                procs[i] = new_p
        time.sleep(10)
        done = all([p.is_alive() for p in procs])
    print("Feature servers started!")
    return procs


def run_exp(toml, qps):
    feature_procs = start_features()
    toml_str = toml % {"qps": qps}
    toml_path = os.path.join(CLIPPER_SERVER_BASE, "anytime.toml")
    with open(toml_path, "w") as toml_file:
        print(toml_str, file=toml_file)

    clipper_cmd = ("/crankshaw-local/clipper/clipper_server/target/release/clipper "
                   "digits --feature-conf=features.toml --bench-conf=anytime.toml")
    clipper_cmd_seq = ["/crankshaw-local/clipper/clipper_server/target/release/clipper",
                       "digits", "--feature-conf=features.toml", "--bench-conf=anytime.toml"]


    clipper_out = None
    clipper_err = None
    clipper_proc = subprocess.Popen(clipper_cmd_seq,
                                          cwd=CLIPPER_SERVER_BASE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
    try:
        # (output, err) = clipper_proc.communicate(timeout=100)
                                              # timeout=100)

        clipper_out, clipper_err = clipper_proc.communicate(timeout=200)
    except subprocess.TimeoutExpired:
        clipper_proc.kill()
        clipper_out, clipper_err = clipper_proc.communicate()

    # print(clipper_out)


    #     clipper_proc.kill()
    #     outs, errs = proc.communicate()
    #
    #
    # time.sleep(100)
    # clipper_proc.terminate()
    # output = clipper_proc.stdout.read()
    # print(output)
    for i in range(len(feature_procs)):
        feature_procs[i].terminate()

    time.sleep(10)
    # print(clipper_out)
    return clipper_out

    # (output, err) = proc.communicate()


    
if __name__=="__main__":

    # first do baseline experiments
    # out_file = os.path.join(CLIPPER_SERVER_BASE, "experiments_RAW/end_to_end_THRUPUT/baseline.txt")
    # with open(out_file, "wa") as results_file:
    #     for q in [1000, 3000, 5000, 7000, 9000, 10000]:
    #         print("\n\nEXPERIMENT RUN QPS: %d" % q)
    #         print("\n\nEXPERIMENT RUN QPS: %d" % q, file=results_file)
    #         out = run_exp(toml_str_baseline, q)
    #         print(out)
    #         print(out, file=results_file)
    #
    # print("FINISHED BASELINE EXPERIMENTS")

    # out_file = os.path.join(CLIPPER_SERVER_BASE, "experiments_RAW/end_to_end_THRUPUT/batching2.txt")
    # with open(out_file, "a") as results_file:
    #     for q in [45000, 48000, 52000]:
    #         print("\n\nEXPERIMENT RUN QPS: %d" % q)
    #         print("\n\nEXPERIMENT RUN QPS: %d" % q, file=results_file)
    #         out = run_exp(toml_str_batching, q)
    #         print(out)
    #         print(out, file=results_file)
    #         results_file.flush()
    #
    # print("FINISHED BATCHING EXPERIMENTS")

    # out_file = os.path.join(CLIPPER_SERVER_BASE, "experiments_RAW/end_to_end_THRUPUT/max_features.txt")
    # with open(out_file, "a") as results_file:
    #     for q in [64000, 80000]: #, 68000, 72000, 76000, 80000]:
    #         print("\n\nEXPERIMENT RUN QPS: %d" % q)
    #         print("\n\nEXPERIMENT RUN QPS: %d" % q, file=results_file)
    #         out = run_exp(toml_str_max_features, q)
    #         print(out)
    #         print(out, file=results_file)
    #         results_file.flush()
    
    out_file = os.path.join(CLIPPER_SERVER_BASE, "experiments_RAW/end_to_end_THRUPUT/no_features.txt")
    with open(out_file, "a") as results_file:
        for q in [20000, 30000, 40000, 50000]: #, 68000, 72000, 76000, 80000]:
            print("\n\nEXPERIMENT RUN QPS: %d" % q)
            print("\n\nEXPERIMENT RUN QPS: %d" % q, file=results_file)
            out = run_exp(toml_str_no_features, q)
            print(out)
            print(out, file=results_file)
            results_file.flush()






# no_op_features = "noop_features.toml"
# spark_svm_feats = "features.toml"



# 

# batch_sizes = [1, 100]

