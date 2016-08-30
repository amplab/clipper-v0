from __future__ import print_function
import sys
import yaml
import toml
import time
import os
import json
# import shutil
import subprocess32 as subprocess

cur_model_core_num = 0
MAX_CORES = 47
isolated_cores = True
experiment_name = "with_isolation_6_replicas_%s" % str(time.strftime("%y%m%d-%H%M%S"))
benchmarking_logs = "benchmarking_logs/container-isolation"
CLIPPER_ROOT = os.path.abspath("..")

def reserve_cores(num_cores):
    global cur_model_core_num
    s = "%d-%d" % (cur_model_core_num, cur_model_core_num + num_cores - 1)
    cur_model_core_num += num_cores
    if cur_model_core_num >= MAX_CORES:
        print("WARNING: Trying to reserve more than %d cores: %d" % (MAX_CORES, cur_model_core_num))
        sys.exit(1)
    return s

def overlap_reserve_cores():
    s = "%d-%d" % (cur_model_core_num, MAX_CORES)
    return s


clipper_conf_dict = {
        "name" : experiment_name,
        "slo_micros" : 20000,
        "num_message_encodes" : 1,
        "correction_policy" : "logistic_regression",
        "use_lsh" : False,
        "input_type" : "float",
        "input_length" : 784,
        "window_size" : -1,
        "redis_ip" : "redis",
        "redis_port" : 6379,
        "results_path" : "/tmp/benchmarking_logs",
        "num_predict_workers" : 10,
        "num_update_workers" : 1,
        "cache_size" : 49999,
        "mnist_path" : "/mnist_data/test.data",
        "num_benchmark_requests" : 5000000,
        "target_qps" : 100000,
        "bench_batch_size" : 10000,
        "salt_cache" : True,
        # "batching": { "strategy": "aimd" },
        # "batching": { "strategy": "static", "batch_size": 1 },
        "batching": { "strategy": "learned", "sample_size": 500, "opt_addr": "quantilereg:7777"},
        "models": []
        }



dc_dict = {
        "version": "2",
        "services": {
            "redis": {"image": "redis:alpine", "cpuset": reserve_cores(1)},
            "quantilereg": {"image": "clipper/quantile-reg", "cpuset": reserve_cores(1)},
            "clipper": {"image": "cl-dev-digits",
                "cpuset": reserve_cores(10),
                "depends_on": ["redis", "quantilereg"],
                "volumes": [
                    "${MNIST_PATH}:/mnist_data:ro",
                    "${CLIPPER_ROOT}/digits_bench.toml:/tmp/digits_bench.toml:ro",
                    "${CLIPPER_ROOT}/%s:/tmp/benchmarking_logs" % benchmarking_logs
                    ],
                }
            }
        }




def add_model(name_base, image, mp, container_mp, num_replicas=1):
    model_names = [name_base + "_r%d" % i for i in range(num_replicas)]
    model_addrs = ["%s:6001" % n for n in model_names]
    clipper_model_def = {
            "name": name_base,
            "addresses": model_addrs,
            "num_outputs": 1,
            "version": 1
    }
    dc_entries = {}
    for n in model_names:
        if isolated_cores:
            core_res = reserve_cores(1)
        else:
            core_res = overlap_reserve_cores()

        dc_entries[n] = {
                "image": image,
                "volumes": ["%s:/model:ro" % mp],
                # "environment": ["CLIPPER_MODEL_PATH=%s" % container_mp],
                "cpuset": core_res,
                }

    # dc_entries should be added to the depends_on list

    global clipper_conf_dict
    global dc_dict
    clipper_conf_dict["models"].append(clipper_model_def)
    dc_dict["services"]["clipper"]["depends_on"].extend(model_names)
    dc_dict["services"].update(dc_entries)
    # return (clipper_model_def, model_names, dc_entries)

def add_sklearn_rf(depth, num_replicas=1):
    name_base = "rf_d%d" % depth
    image = "clipper/sklearn-mw-dev"
    mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/50rf_pred3_depth%d/" % depth
    container_mp =  "/model"
    add_model(name_base, image, mp, container_mp, num_replicas)

def add_sklearn_log_regression(num_replicas=1):
    name_base = "logistic_reg"
    image = "clipper/sklearn-mw-dev"
    mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/log_regression_pred3/",
    container_mp =  "/model"
    add_model(name_base, image, mp, container_mp, num_replicas)

def add_sklearn_linear_svm(num_replicas=1):
    name_base = "linear_svm"
    image = "clipper/sklearn-mw-dev"
    mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/linearsvm_pred3/",
    container_mp =  "/model"
    add_model(name_base, image, mp, container_mp, num_replicas)

def add_sklearn_kernel_svm(num_replicas=1):
    name_base = "kernel_svm"
    image = "clipper/sklearn-mw-dev"
    mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/svm_pred3/",
    container_mp = "/model"
    add_model(name_base, image, mp, container_mp, num_replicas)

def add_noop(num_replicas=1):
    name_base = "noop"
    image = "clipper/noop-mw"
    # these values don't matter
    model_path = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/linearsvm_pred3/",
    container_mp = "/model"


def add_spark_svm(num_replicas=1):
    name_base = "spark_svm"
    image = "clipper/spark-mw-dev"
    mp = "${CLIPPER_ROOT}/model_wrappers/python/spark_models/svm_predict_3",
    container_mp ="/model"
    add_model(name_base, image, mp, container_mp, num_replicas)


def do_docker_run():
    """
    alias dcstop="sudo docker-compose stop"
    alias dcup="sudo docker-compose up clipper"
    """

    docker_compose_up = ["sudo", "docker-compose", "up", "clipper"]
    docker_compose_stop = ["sudo", "docker-compose", "stop"]
    print("Starting experiment: %s" % experiment_name)
    clipper_err_code = subprocess.call(docker_compose_up)
    if clipper_err_code > 0:
        print("WARNING: Clipper benchmark terminated with non-zero error code: %d" % clipper_err_code)
    print("Shutting down containers")
    dcstop_err_code = subprocess.call(docker_compose_stop)
    if dcstop_err_code > 0:
        print("WARNING: Docker container shutdown terminated with non-zero error code: %d" % dcstop_err_code)


def gen_configs():
    ## SKLEARN RF
    global cur_model_core_num
    num_reps = 6
    add_spark_svm(num_replicas=num_reps)
    add_sklearn_log_regression(num_replicas=num_reps)
    add_sklearn_rf(depth=16, num_replicas=num_reps)
    add_sklearn_linear_svm(num_replicas=num_reps)
    add_sklearn_rf(depth=8, num_replicas=num_reps)
    # add_sklearn_kernel_svm(num_replicas=num_reps)

    print("CORES USED: %d" % cur_model_core_num)

    # with open("../digits_bench_TEST.toml", 'w') as f:
    with open("../digits_bench.toml", 'w') as f:
        toml.dump(clipper_conf_dict, f)

    with open("docker-compose.yml", 'w') as f:
        yaml.dump(dc_dict, f)

    with open(os.path.join(CLIPPER_ROOT, benchmarking_logs, "%s_config.json" % experiment_name), "w") as f:
        json.dump({"clipper_conf": clipper_conf_dict, "docker_compose_conf": dc_dict}, f, indent=4)

    do_docker_run()

    print("Finished running. Saving Clipper log")

    with open(os.path.join(CLIPPER_ROOT, benchmarking_logs, "%s_logs.txt" % experiment_name), "w") as f:
        subprocess.call(["sudo", "docker", "logs", "experimentsbin_clipper_1"], stdout=f, stderr=subprocess.STDOUT, universal_newlines=True)

    cur_model_core_num = 0


if __name__=='__main__':
    gen_configs()







# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
