from __future__ import print_function, with_statement
from fabric.api import *
from fabric.colors import green as _green, yellow as _yellow
from fabric.contrib.console import confirm
from fabric.contrib.files import append
from StringIO import StringIO
import sys
import os
import requests
import json
import gevent
import traceback
import toml
import subprocess32 as subprocess

MODEL_REPO = "/tmp/clipper-models"
DOCKER_NW = "clipper_nw"
CLIPPER_METADATA_FILE = "clipper_metadata.json"

aws_cli_config="""
[default]
region = us-east-1
aws_access_key_id = {access_key}
aws_secret_access_key = {secret_key}
"""


class Cluster:
    def __init__(self, host, user, key_path):
        # global env
        env.key_filename = key_path
        env.user = user
        print(env.user)
        env.host_string = host
        self.host = host
        print(env.hosts)
        # Make sure docker is running on cluster
        sudo("docker ps")
        nw_create_command = "docker network create --driver bridge {nw}".format(nw=DOCKER_NW)
        sudo(nw_create_command, warn_only=True)
        run("mkdir -p {model_repo}".format(model_repo=MODEL_REPO))
        self.clipper_up = False




    def add_replicas(self, name, version, num_replicas=1):
        vol = "{model_repo}/{name}/{version}".format(model_repo=MODEL_REPO, name=name, version=version)
        present = run("stat {vol}".format(vol), warn_only=True)
        if present.return_code == 0:
            # Look up image name
            fd = StringIO()
            get(os.path.join(vol, CLIPPER_METADATA_FILE), fd)
            metadata = json.loads(fd.getvalue())
            image_name = metadata["image_name"]
            
            # find the max current replica num
            docker_ps_out = sudo("docker ps")
            rs = []
            nv = "{name}_{version}".format(name=name, version=version)
            for line in docker_ps_out.split("\n"):
                line_name = line.split()[-1]
                if nv in line_name:
                    rep_num = int(line_name.split("_")[-1].lstrip("r"))
                    rs.append(rep_num)
            next_replica_num = max(rs) + 1
            addrs = []
            # Add new replicas
            for r in range(next_replica_num, next_replica_num + num_replicas):
                container_name = "%s_v%d_r%d" % (name, version, r)
                run_mw_command = "docker run -d --network={nw} --name {name} -v {vol}:/model:ro {image}".format(
                        name=container_name,
                        vol=os.path.join(vol, os.path.basename(data_path)),
                        nw=DOCKER_NW, image=image_name)
                sudo("docker stop {name}".format(name=container_name), warn_only=True)
                sudo("docker rm {name}".format(name=container_name), warn_only=True)
                sudo(run_mw_command)
                addrs.append("{cn}:6001".format(cn=container_name))

            if self.clipper_up:
                new_replica_data = {
                        "name": name,
                        "version": version,
                        "addrs": addrs
                        }
                self.inform_clipper_new_replica(new_model_data)
        else:
            print("{model} version {version} not found!".format(model=name, version=version))

    def add_local_model(self, name, image_id, container_name, data_path, replicas=1):
        subprocess.call("docker save -o /tmp/{cn}.tar {image_id}".format(
            cn=container_name,
            image_id=image_id))
        tar_loc = "/tmp/{cn}.tar".format(cn=container_name)
        put(tar_loc, tar_loc)
        sudo("docker load -i {loc}".format(loc=tar_loc))
        sudo("docker tag {image_id} {cn}".format(image_id = image_id, cn=cn))
        self.add_model(name, container_name, data_path, replicas=replicas)


    def add_sklearn_model(self, name, data_path, replicas=1):
        image_name = "dcrankshaw/clipper-sklearn-mw"
        self.add_model(name, image_name, data_path, replicas=replicas)

    def add_pyspark_model(self, name, data_path, replicas=1):
        image_name = "dcrankshaw/clipper-spark-mw"
        self.add_model(name, image_name, data_path, replicas=replicas)


    def add_model(self, name, image_name, data_path, replicas=1):
        version = 1
        vol = "{model_repo}/{name}/{version}".format(model_repo=MODEL_REPO, name=name, version=version)
        print(vol)
        run("mkdir -p {vol}".format(vol=vol))

        with cd(vol):
            append(CLIPPER_METADATA_FILE, json.dumps({"image_name": image_name}))
            if data_path.startswith("s3://"):
                aws_cli_installed = run("dpkg-query -Wf'${db:Status-abbrev}' awscli 2>/dev/null | grep -q '^i'", warn_only=True).return_code
                if not aws_cli_installed:
                    sudo("apt-get install awscli")
                if sudo("stat ~/.aws/config", warn_only=True).return_code != 0:
                    run("mkdir -p ~/.aws")
                    append("~/.aws/config", aws_cli_config.format(
                        access_key=os.environ["AWS_ACCESS_KEY_ID"],
                        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"]))
                run("aws s3 cp s3://clipperdbdemo/dbcloud/ . --recursive")
            else:
                put(data_path, ".")
        addrs = []
        for r in range(replicas):
            container_name = "%s_v%d_r%d" % (name, version, r)
            run_mw_command = "docker run -d --network={nw} --name {name} -v {vol}:/model:ro {image}".format(
                    name=container_name,
                    vol=os.path.join(vol, os.path.basename(data_path)),
                    nw=DOCKER_NW, image=image_name)
            sudo("docker stop {name}".format(name=container_name), warn_only=True)
            sudo("docker rm {name}".format(name=container_name), warn_only=True)
            sudo(run_mw_command)
            addrs.append("{cn}:6001".format(cn=container_name))
        if self.clipper_up:
            new_model_data = {
                    "name": name,
                    "version": version,
                    "addrs": addrs
                    }
            self.inform_clipper_new_model(new_model_data)

    def inform_clipper_new_model(self, new_model_data):
        url = "http://%s:1337/addmodel" % self.host
        req_json = json.dumps(new_model_data)
        headers = {'Content-type': 'application/json'}
        r = requests.post(url, headers=headers, data=req_json)

    def inform_clipper_new_replica():
        url = "http://%s:1337/addreplica" % self.host
        req_json = json.dumps(new_model_data)
        headers = {'Content-type': 'application/json'}
        r = requests.post(url, headers=headers, data=req_json)
        
    def start_clipper(self, config=None):
        conf_loc = "/home/ubuntu/conf.toml"
        if config is not None:
            put(config, "~/conf.toml")
        else:
            # Use default config
            clipper_conf_dict = {
                    "name" : "clipper-demo",
                    "slo_micros" : 20000,
                    "correction_policy" : "logistic_regression",
                    "use_lsh" : False,
                    "input_type" : "float",
                    "input_length" : 784,
                    "window_size" : -1,
                    "redis_ip" : "redis-clipper",
                    "redis_port" : 6379,
                    "num_predict_workers" : 1,
                    "num_update_workers" : 1,
                    "cache_size" : 49999,
                    "batching": { "strategy": "learned", "sample_size": 1000},
                    "models": []
                    }
            run("rm ~/conf.toml", warn_only=True)
            append("~/conf.toml", toml.dumps(clipper_conf_dict))
        sudo("docker run -d --network={nw} -p 6379:6379 "
                "--cpuset-cpus=\"0\" --name redis-clipper redis:alpine".format(nw=DOCKER_NW))

        sudo("docker run -d --network={nw} -p 1337:1337 "
                "--cpuset-cpus=\"{min_core}-{max_core}\" --name clipper "
                "-v ~/conf.toml:/tmp/conf.toml dcrankshaw/clipper".format(nw=DOCKER_NW, min_core=1, max_core=4))
        self.clipper_up = True

    def get_metrics(self):
        # for h in self.hosts:
        url = "http://%s:1337/metrics" % self.host
        r = requests.get(url)
        print(json.dumps(r.json(), indent=4))




    def stop_all(self):
        sudo("docker stop $(docker ps -a -q)", warn_only=True)
        sudo("docker rm $(docker ps -a -q)", warn_only=True)
        self.clipper_up = False


def test():

    pass
    return installed

#
# if __name__=="__main__":
#     # session = Session("ubuntu", os.path.expanduser("~/.ssh/aws_rsa"))
#     # session = Session("crankshaw", os.path.expanduser("~/.ssh/c70.millenium"))
#     # cluster = Cluster(['c70.millennium.berkeley.edu'], session)
#     image_source_host = "ec2-54-163-142-10.compute-1.amazonaws.com"
#     from_image_host = "ec2-52-90-82-153.compute-1.amazonaws.com"
#
#     cluster = Cluster([from_image_host], "ubuntu", os.path.expanduser("~/.ssh/aws_rsa"))
#     cluster.stop_all()
#     # cluster.start_clipper("/Users/crankshaw/clipper/clipper_server/conf/test.toml")
#




























#
#
#
# """
# Maybe we should define the abstractions in a Clipper cluster.
#
# What does this look like in local mode????
# """
#
# fake_authenticator = { 'root': 'password' }
#
# def authenticate(user_id, key):
#     print("WARNING, THIS IS FAKE AUTHENTICATION. DO NOT USE!!!!!!!")
#     if fake_authenticator.get(user_id) == key:
#         True
#     else:
#         False
#
# class ClipperManager:
#     """
#         Object to manage Clipper models and versions.
#
#         Questions:
#         + How do rolling updates work?
#             + What is the update granularity? Node, traffic percentage, something else?
#
#
#     """
#
#     def __init__(user_id):
#         """
#             Once we figure out cluster connection, this class will use the same
#             connection mechanism as `ClipperClusterManager`.
#         """
#         pass
#
#
#
#     def add_model(self, name, model_wrapper, model_data, rollout_strategy):
#         """
#             Add a new model to an existing model wrapper.
#
#             When the model wrapper docker container is launched, the special
#             environment variable `CLIPPER_MODEL_DATA` is set to the path
#             where the `model_data` package was unzipped.
#
#             Args:
#                 name(str): unique identifier for this model.                
#                 model_wrapper(str): path to Docker container
#                 model_data(str): path to file or root of a directory containing
#                                  model data (e.g. parameters). If the path contains a zipfile
#                                  or tarball the file will be shipped as is, otherwise a tarball
#                                  will be created. Either way, the file will be unpackaged at the destination.
#
#
#             Raises:
#                 ValueError: If the provided name conflicts with an existing model name.
#
#         """
#         pass
#
#     def update_model(self, name, model_data, rollout_strategy):
#         """
#             Update to a new version of an existing model.
#
#             It will launch a new replica of the model wrapper
#             (according to the update strategy) with CLIPPER_MODEL_DATA
#             pointing to the new data.
#
#             Question:
#                 How to specify partial updates?
#         """
#         pass
#
#     def rollback_model(self, name, version, rollout_strategy):
#         """
#             Rollback the named model to the specified version.
#             
#
#             Raises:
#                 ValueError: If the supplied version of the model does not exist.
#         """
#
#         pass
#
#     def replicate_model(self, name, num_replicas, nodes):
#         """
#             Replicate the specified model wrapper `num_replica` times on each of
#             the specified nodes.
#         """
#         pass
#
#
#     def alias(self, name, version, new_name, rollout_strategy):
#         """
#             Create a replica of an existing model wrapper with a new
#             name and treat it as a new model. From now on the two models
#             will be considered independent, and version updates to one will
#             not affect the other.
#         """
#         pass
#
#     def inspect_model(self, name):
#         """
#             Get diagnostic information about a model. This includes both
#             history of the model (who last modified it, when it was last updated)
#             but also performance (e.g. latency and throughput, cache hit rates,
#             how often it misses the latency SLO, etc.)
#         """
#         pass
#
#     def set_model_permissions(self, name, permissions):
#         """
#             Let's look into Etcd permissions for this.
#         """
#         pass
#
#
#
# class ClipperClusterManager:
#     """
#         All actions through the API are associated with a specific user
#         to track provenance and add permissions.
#
#
#
#
#         How does cluster membership work?
#         How does cluster deployment work?
#         Atomicity, Isolation, Idempotence
#
#         Proposal:
#             Let's use etcd for cluster membership.
#             Does that mean the cluster manager connects directly to Etcd
#             to make changes, or does it connect to one of the Clipper instances
#             which then propagates the changes to Etcd? How do we want to manage shipping
#             models which could be 100s of MB?
#
#     """
#
#     def __init__(user_id):
#         pass
#
#
#     
#     def start(num_instances, nodes):
#         """
#             Start a new Clipper cluster on the provided list of machines.
#             
#             Questions:
#             How to access these machines? SSH? Expect a cluster manager?
#
#         """
#         pass
#
#     def connect(address):
#         """
#             Connect to a running Clipper cluster.
#
#             Questions:
#             + How is cluster membership handled? Etcd?
#             
#         """
#         pass
#
#     def shutdown(self):
#         """
#             Shutdown the connected cluster
#
#         """
#         pass
#
#     def get_metrics(self):
#         self.get_system_metrics()
#         self.get_model_metrics()
#
#     def get_system_metrics(self):
#         """
#             Get physical performance metrics (latency, throughput, cache hits, perf, etc.)
#         """
#         pass
#
#     def get_model_metrics(self):
#         """
#             Get model performance metrics
#         """
#         pass
#
#
#
#
#
#
#
#
#
















