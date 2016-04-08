from __future__ import print_function
import time
import os, shutil, sys, datetime
# from subprocess import call, Popen, PIPE
import subprocess32 as subprocess
from scipy.io import wavfile
import numpy as np
import json

from sample_feature import TestFeature


import rpc_feature_server as rpc
from multiprocessing import Process

CLIPPER_SERVER_BASE = "/crankshaw-local/clipper/clipper_server"


def start_feature(ip, port):
    # mp = "sklearn_models/predict_1_svm/predict_1_svm.pkl"
    # p = Process(target=rpc.start_sksvm_from_mp, args=(mp, ip,port))
    mp = "spark_models/lg_predict_1"
    p = Process(target=rpc.start_sparklr_from_mp, args=(mp, ip,port))
    p.start()
    return p # to kill: p.terminate()

def start_features():
    procs = {}
    i = 1
    procs["127.0.0.1", (6000 + i)] = start_feature("127.0.0.1", (6000 + i))
    # procs["127.0.0.1", (7000 + i)] = start_feature("127.0.0.1", (7000 + i))
    # procs["127.0.0.1", (8000 + i)] = start_feature("127.0.0.1", (8000 + i))
    done = False
    time.sleep(5)
    # make sure they all started. Sometimes a socket is still bound, so
    # we need to wait a little bit and retry

    while not done:
        for ip in procs.keys():
        # for i in range(len(procs)):
            if not procs[ip].is_alive():
                print("restarting %s" % (ip))
                new_p = start_feature(ip)
                procs[ip] = new_p
        time.sleep(10)
        done = all([procs[p].is_alive() for p in procs.keys()])
    print("Feature servers started!")
    return procs


def run_exp(batch_size):
    feature_procs = start_features()

    clipper_cmd_seq = ["/crankshaw-local/clipper/clipper_server/target/release/clipper",
                       "featurelats", "%d" % batch_size]


    clipper_out = None
    clipper_err = None
    my_env = os.environ.copy()
    my_env["RUST_LOG"] = "info"
    my_env["RUST_BACKTRACE"] = "1"
    clipper_proc = subprocess.Popen(clipper_cmd_seq,
                                          cwd=CLIPPER_SERVER_BASE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          env=my_env)
    try:
        clipper_out, clipper_err = clipper_proc.communicate(timeout=150)
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
    for i in feature_procs:
        feature_procs[i].terminate()

    time.sleep(10)
    print(clipper_out)
    print("STDERR:")
    print(clipper_err)
    return clipper_err

    # (output, err) = proc.communicate()


    
if __name__=="__main__":

    # batch = 1
    # out = run_exp(batch)
    # print(out)
    # exit(0)

    # first do baseline experiments
    out_file = os.path.join(CLIPPER_SERVER_BASE, "experiments_RAW/faas_benchmarks/spark_lr.txt")
    with open(out_file, "a") as results_file:
        for batch in [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400]:
        # for batch in range(550, 1001, 50):
            print("\n\nEXPERIMENT RUN BATCH SIZE: %d" % batch)
            print("\n\nEXPERIMENT RUN BATCH SIZE: %d" % batch, file=results_file)
            out = run_exp(batch)
            print(out)
            print(out, file=results_file)
            results_file.flush()

    print("FINISHED")
    #
