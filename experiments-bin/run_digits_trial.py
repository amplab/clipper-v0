from __future__ import print_function
import sys
import yaml
import toml
import time
import os
import json
# import shutil
import subprocess32 as subprocess
from fabric.api import *
from fabric.colors import green as _green, yellow as _yellow
from fabric.contrib.console import confirm
from fabric.contrib.files import append
from datetime import datetime


class DigitsBenchmarker:

    def __init__(self,
                 experiment_name,
                 log_dest,
                 target_qps=10000,
                 bench_batch_size=100,
                 num_requests=3000000,
                 # message_size=200000,
                 # batch_size=100,
                 cache_hit_rate = 0.0,
                 window_size = -1,
                 send_updates = False,
                 salt_cache=True,
                 track_blocking_latency=True,
                 batch_strategy = { "strategy": "aimd" },
                 isolate_cores=True,
                 slo_micros=20000):
        self.remote_node = "192.168.142.68"
        env.key_filename = "~/.ssh/c70.millenium"
        env.user = "crankshaw"
        env.host_string = self.remote_node
        self.cur_model_core_num = 0
        self.remote_cur_model_core_num = 0
        self.remote_port_start = 7001
        self.MAX_CORES = 47
        # self.NUM_REPS = NUM_REPS
        self.isolated_cores = isolate_cores
        # self.experiment_name = "DEBUG_sklearn_logreg_%d_replicas_%s" % (NUM_REPS, str(time.strftime("%y%m%d-%H%M%S")))
        self.experiment_name = "%s_%s" % (experiment_name, str(time.strftime("%y%m%d-%H%M%S")))
        # self.benchmarking_logs = "benchmarking_logs/replica-scaling"
        self.benchmarking_logs = log_dest
        self.CLIPPER_ROOT = os.path.abspath("..")

        self.clipper_conf_dict = {
                "name" : self.experiment_name,
                "slo_micros" : slo_micros,
                "num_message_encodes" : 1,
                "correction_policy" : "logistic_regression",
                "use_lsh" : False,
                "track_blocking_latency" : track_blocking_latency,
                "input_type" : "float",
                "input_length" : 784,
                "window_size" : window_size,
                "redis_ip" : "redis",
                "redis_port" : 6379,
                "results_path" : "/tmp/benchmarking_logs",
                "num_predict_workers" : 20,
                "num_update_workers" : 1,
                "cache_size" : 10000000,
                "mnist_path" : "/mnist_data/test.data",
                # "num_benchmark_requests" : 10000000,
                "num_benchmark_requests" : num_requests,
                # "target_qps" : 10000*self.NUM_REPS,
                "target_qps" : target_qps,
                # "bench_batch_size" : 150*self.NUM_REPS,
                "bench_batch_size" : bench_batch_size,
                "report_interval_secs" : 20,
                # "salt_cache" : False,
                "salt_cache" : salt_cache,
                "salt_update_cache" : salt_cache,
                "send_updates": send_updates,
                "load_generator": "uniform",
                "request_generator": "balanced",
                # "request_generator": "cached_updates",
                # "request_generator": "cache_hits",
                "wait_to_end": False,
                "batching": batch_strategy,
                "cache_hit_rate": cache_hit_rate,
                # "batching": { "strategy": "aimd" },
                # "batching": { "strategy": "static", "batch_size": batch_size },
                # "batching": { "strategy": "learned", "sample_size": 500, "opt_addr": "quantilereg:7777"},
                "models": []
                }

        self.remote_dc_dict = {
                "version": "2",
                "services": { }
                }

        self.dc_dict = {
                "version": "2",
                "services": {
                    "redis": {"image": "redis:alpine", "cpuset": self.reserve_cores(1)},
                    # "quantilereg": {"image": "clipper/quantile-reg", "cpuset": self.reserve_cores(1)},
                    "clipper": {"image": "cl-dev-digits",
                        "cpuset": self.reserve_cores(28),
                        # "depends_on": ["redis", "quantilereg"],
                        "depends_on": ["redis",],
                        "environment": {
                            "CLIPPER_BENCH_COMMAND": "digits",
                            "CLIPPER_CONF_PATH":"/tmp/exp_conf.toml",
                            },
                        "volumes": [
                            "${MNIST_PATH}:/mnist_data:ro",
                            "${CLIPPER_ROOT}/exp_conf.toml:/tmp/exp_conf.toml:ro",
                            "${CLIPPER_ROOT}/%s:/tmp/benchmarking_logs" % self.benchmarking_logs
                            ],
                        }
                    }
                }

    def reserve_cores(self, num_cores):
        # global cur_model_core_num
        s = "%d-%d" % (self.cur_model_core_num, self.cur_model_core_num + num_cores - 1)
        self.cur_model_core_num += num_cores
        if self.cur_model_core_num > self.MAX_CORES:
            print("WARNING: Trying to reserve more than %d cores: %d" % (self.MAX_CORES, self.cur_model_core_num))
            sys.exit(1)
        return s

    def overlap_reserve_cores(self):
        s = "%d-%d" % (self.cur_model_core_num, self.MAX_CORES)
        return s

    def remote_reserve_cores(self, num_cores):
        s = "%d-%d" % (self.remote_cur_model_core_num, self.remote_cur_model_core_num + num_cores - 1)
        self.remote_cur_model_core_num += num_cores
        if self.remote_cur_model_core_num >= self.MAX_CORES:
            print("WARNING: Trying to reserve more than %d cores: %d" % (self.MAX_CORES, self.remote_cur_model_core_num))
            sys.exit(1)
        return s

    def remote_overlap_reserve_cores(self):
        s = "%d-%d" % (self.remote_cur_model_core_num, self.MAX_CORES)
        return s


    def add_remote_reps(self, name_base, image, mp, container_mp, rep_start, num_replicas=1):
        model_names = [name_base + "_r%d" % i for i in range(rep_start, rep_start + num_replicas)]
        model_addrs = ["%s:%d" % (self.remote_node, self.remote_port_start + i) for i in range(num_replicas)]

        pnum = self.remote_port_start
        self.remote_port_start += num_replicas

        dc_entries = {}
        for n in model_names:
            if self.isolated_cores:
                core_res = self.remote_reserve_cores(1)
            else:
                core_res = self.remote_overlap_reserve_cores()

            dc_entries[n] = {
                    "image": image,
                    "volumes": ["%s:/model:ro" % mp],
                    # "environment": ["CLIPPER_MODEL_PATH=%s" % container_mp],
                    "cpuset": core_res,
                    "ports": ["%d:6001" % pnum],
                    }
            pnum += 1

        # self.clipper_conf_dict["models"].append(clipper_model_def)
        self.remote_dc_dict["services"].update(dc_entries)
        # self.dc_dict["services"]["clipper"]["depends_on"].extend(model_names)
        # self.dc_dict["services"].update(dc_entries)
        return model_addrs



    def add_model(self, name_base, image, mp, container_mp, num_replicas, remote_replicas=0, wait_time_nanos=None):
        model_names = [name_base + "_r%d" % i for i in range(num_replicas)]
        model_addrs = ["%s:6001" % n for n in model_names]
        remote_model_addrs = []
        if remote_replicas > 0:
            remote_model_addrs = self.add_remote_reps("%s_remote" % name_base, image, mp, container_mp, num_replicas, remote_replicas)
        clipper_model_def = {
                "name": name_base,
                "addresses": model_addrs + remote_model_addrs,
                "num_outputs": 1,
                "version": 1
        }
        if wait_time_nanos is not None:
            clipper_model_def["wait_time_nanos"] = wait_time_nanos
        dc_entries = {}
        for n in model_names:
            if self.isolated_cores:
                core_res = self.reserve_cores(1)
            else:
                core_res = self.overlap_reserve_cores()

            dc_entries[n] = {
                    "image": image,
                    "volumes": ["%s:/model:ro" % mp],
                    # "environment": ["CLIPPER_MODEL_PATH=%s" % container_mp],
                    "cpuset": core_res,
                    }

        # dc_entries should be added to the depends_on list

        # global clipper_conf_dict
        # global dc_dict
        self.clipper_conf_dict["models"].append(clipper_model_def)
        self.dc_dict["services"]["clipper"]["depends_on"].extend(model_names)
        self.dc_dict["services"].update(dc_entries)
    # return (clipper_model_def, model_names, dc_entries)

    def add_sklearn_rf(self, depth, name_base="sklearn_rf", num_replicas=1):
        name_base = "%s_d%d" % (name_base, depth)
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/50rf_pred3_depth%d/" % depth
        container_mp =  "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas, wait_time_nanos=4*1000*1000)

    def add_sklearn_log_regression(self, local_replicas=1, remote_replicas=0):
        name_base = "logistic_reg"
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/log_regression_pred3/",
        container_mp =  "/model"
        self.add_model(name_base, image, mp, container_mp, local_replicas, remote_replicas=remote_replicas, wait_time_nanos=5*1000*1000)


    def add_sklearn_linear_svm(self, num_replicas=1, wait_time_nanos=1*1000*1000):
        name_base = "linear_svm"
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/linearsvm_pred3/",
        container_mp =  "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas, wait_time_nanos=wait_time_nanos)

    def add_sklearn_kernel_svm(self, num_replicas=1):
        name_base = "kernel_svm"
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/svm_pred3/",
        container_mp = "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    def add_noop(self, num_replicas=1):
        name_base = "noop"
        image = "clipper/noop-mw-dev"
        # these values don't matter
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/linearsvm_pred3/",
        container_mp = "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)


    def add_spark_svm(self, name_base="spark_svm", local_replicas=1, remote_replicas=0, wait_time_nanos=1*1000*1000):
        image = "clipper/spark-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/spark_models/svm_predict_3",
        container_mp ="/model"
        self.add_model(name_base, image, mp, container_mp, local_replicas, remote_replicas=remote_replicas, wait_time_nanos=wait_time_nanos)

    def add_spark_rf(self, num_replicas=1):
        name_base = "spark_svm"
        image = "clipper/spark-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/spark_models/50rf_pred_3_depth_4",
        container_mp ="/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    def run_clipper(self):

        print("CORES USED: %d" % self.cur_model_core_num)

        # with open("../digits_bench_TEST.toml", 'w') as f:
        with open("../exp_conf.toml", 'w') as f:
            toml.dump(self.clipper_conf_dict, f)

        with open("docker-compose.yml", 'w') as f:
            yaml.dump(self.dc_dict, f)

        with open(os.path.join(self.CLIPPER_ROOT, self.benchmarking_logs, "%s_config.json" % self.experiment_name), "w") as f:
            json.dump({"clipper_conf": self.clipper_conf_dict, "docker_compose_conf": self.dc_dict}, f, indent=4)
        
        if len(self.remote_dc_dict["services"]) > 0:
            self.remote_dc_dir = "/data/crankshaw/remote-replica-tmp"
            with cd(self.remote_dc_dir):
                with open("/tmp/docker-compose.yml", "w") as f:
                    yaml.dump(self.remote_dc_dict, f)
                    put("/tmp/docker-compose.yml", "docker-compose.yml")

        self.run_with_docker()

        print("Finished running. Saving Clipper log")

        with open(os.path.join(self.CLIPPER_ROOT, self.benchmarking_logs, "%s_logs.txt" % self.experiment_name), "w") as f:
            subprocess.call(["sudo", "docker", "logs", "experimentsbin_clipper_1"], stdout=f, stderr=subprocess.STDOUT, universal_newlines=True)


    def run_with_docker(self):
        """
        alias dcstop="sudo docker-compose stop"
        alias dcup="sudo docker-compose up clipper"
        """

        # start remote model wrappers
        if len(self.remote_dc_dict["services"]) > 0:
            # remote_dc_dir = "/data/crankshaw/clipper/experiments-bin"
            with cd(self.remote_dc_dir):
                sudo("docker-compose up -d")
                # for sname in self.remote_dc_dict["services"]:
                    # sudo("docker-compose up %s" % sname)

        docker_compose_up = ["sudo", "docker-compose", "up", "clipper"]
        docker_compose_stop = ["sudo", "docker-compose", "stop"]
        print("Starting experiment: %s" % self.experiment_name)
        clipper_err_code = subprocess.call(docker_compose_up)
        if clipper_err_code > 0:
            print("WARNING: Clipper benchmark terminated with non-zero error code: %d" % clipper_err_code)
        print("Shutting down containers")
        dcstop_err_code = subprocess.call(docker_compose_stop)
        if dcstop_err_code > 0:
            print("WARNING: Docker container shutdown terminated with non-zero error code: %d" % dcstop_err_code)


        # Shutdown remote containers if they exist
        if len(self.remote_dc_dict["services"]) > 0:
            # remote_dc_dir = "/data/crankshaw/clipper/experiments-bin"
            with cd(self.remote_dc_dir):
                sudo("docker-compose stop")



def straggler_mitigation_exp():
    bs = { "strategy": "aimd" }
    for trial in range(1,6):
        trial_start = datetime.now()
        for ensemble_size in [1,] + range(2,21,2):
            print("STARTING EXPERIMENT: STRAGGLER MITIGATION WITH ENSEMBLE SIZE: %d TRIAL %d" % (ensemble_size, trial))
            time.sleep(5)
            num_reqs = 2000000
            num_reps = 1
            debug = ""
            # debug = "DEBUG_"
            exp_name = "%sensemble_size_%d_trial_%d" % (debug, ensemble_size, trial)
            log_dest = "experiments_logs/straggler_mitigation"
            benchmarker = DigitsBenchmarker(exp_name,
                                            log_dest,
                                            target_qps=15000,
                                            num_requests=num_reqs,
                                            send_updates=False,
                                            batch_strategy=bs,
                                            salt_cache=True,
                                            track_blocking_latency=True,
                                            )
            for comp_num in range(ensemble_size):
                benchmarker.add_sklearn_rf(depth=16, name_base="sklearn_rf_comp_%d" % comp_num, num_replicas=num_reps)
            benchmarker.run_clipper()
        trial_end = datetime.now()
        with open("exp_status.log", "a") as f:
            print("TRIAL %d COMPLETED IN %f SECONDS" % (trial, (trial_end-trial_start).total_seconds()))
            print("TRIAL %d COMPLETED IN %f SECONDS" % (trial, (trial_end-trial_start).total_seconds()), file=f)

def replica_scaling_exp():
    # bs = { "strategy": "aimd" }
    bs = { "strategy": "static", "batch_size": 400 }
    # max_local_reps = 8
    
    # max_local_reps = 2
    for reps in range(8,17) + range(1,8):
        remote_reps = max(0, reps - max_local_reps)
        local_reps = reps - remote_reps
        print("STARTING EXPERIMENT: %d LOCAL AND %d REMOTE REPLICAS" % (local_reps, remote_reps))
        time.sleep(5)
        num_reqs = 2000000*reps
        debug = ""
        # debug = "DEBUG_"
        exp_name = "%s%d_local_reps_%d_remote_reps" % (debug, local_reps, remote_reps)
        log_dest = "experiments_logs/replica_scaling_fast_network"
        benchmarker = DigitsBenchmarker(exp_name,
                                        log_dest,
                                        target_qps=100000*reps,
                                        num_requests=num_reqs,
                                        send_updates=False,
                                        batch_strategy=bs,
                                        salt_cache=True,
                                        track_blocking_latency=True,
                                        )
        benchmarker.add_spark_svm(name_base="spark_svm", local_replicas=local_reps, remote_replicas=remote_reps)
        benchmarker.run_clipper()

def resource_isolation_exp():
    bs = { "strategy": "aimd" }
    # max_local_reps = 8
    
    # max_local_reps = 2
    for reps in range(1, 13):
        for isolate_cores in [True, False]:
            # remote_reps = max(0, reps - max_local_reps)
            local_reps = reps
            print("STARTING EXPERIMENT: %d REPLICAS, ISOLATED_CORES: %s" % (local_reps, str(isolate_cores).upper()))
            time.sleep(5)
            num_reqs = 2000000*reps
            debug = ""
            iso_str = "OFF"
            if isolate_cores:
                iso_str = "ON"

            exp_name = "%s%d_reps_isolation_%s" % (debug, local_reps, iso_str)
            log_dest = "experiments_logs/resource_isolation"
            benchmarker = DigitsBenchmarker(exp_name,
                                            log_dest,
                                            target_qps=100000*reps,
                                            num_requests=num_reqs,
                                            send_updates=False,
                                            batch_strategy=bs,
                                            salt_cache=True,
                                            track_blocking_latency=True,
                                            isolate_cores=isolate_cores)
            benchmarker.add_spark_svm(name_base="spark_svm", local_replicas=local_reps)
            benchmarker.run_clipper()



if __name__=='__main__':
    replica_scaling_exp()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
