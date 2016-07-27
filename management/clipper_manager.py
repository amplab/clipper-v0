
"""
Maybe we should define the abstractions in a Clipper cluster.

What does this look like in local mode????
"""

fake_authenticator = { 'root': 'password' }

def authenticate(user_id, key):
    print("WARNING, THIS IS FAKE AUTHENTICATION. DO NOT USE!!!!!!!")
    if fake_authenticator.get(user_id) == key:
        True
    else:
        False

class ClipperManager:
    """
        Object to manage Clipper models and versions.

        Questions:
        + How do rolling updates work?
            + What is the update granularity? Node, traffic percentage, something else?


    """

    def __init__(user_id):
        """
            Once we figure out cluster connection, this class will use the same
            connection mechanism as `ClipperClusterManager`.
        """
        pass



    def add_model(self, name, model_wrapper, model_data, rollout_strategy):
        """
            Add a new model to an existing model wrapper.

            When the model wrapper docker container is launched, the special
            environment variable `CLIPPER_MODEL_DATA` is set to the path
            where the `model_data` package was unzipped.

            Args:
                name(str): unique identifier for this model.                
                model_wrapper(str): path to Docker container
                model_data(str): path to file or root of a directory containing
                                 model data (e.g. parameters). If the path contains a zipfile
                                 or tarball the file will be shipped as is, otherwise a tarball
                                 will be created. Either way, the file will be unpackaged at the destination.


            Raises:
                ValueError: If the provided name conflicts with an existing model name.

        """
        pass

    def update_model(self, name, model_data, rollout_strategy):
        """
            Update to a new version of an existing model.

            It will launch a new replica of the model wrapper
            (according to the update strategy) with CLIPPER_MODEL_DATA
            pointing to the new data.

            Question:
                How to specify partial updates?
        """
        pass

    def rollback_model(self, name, version, rollout_strategy):
        """
            Rollback the named model to the specified version.
            

            Raises:
                ValueError: If the supplied version of the model does not exist.
        """

        pass

    def replicate_model(self, name, num_replicas, nodes):
        """
            Replicate the specified model wrapper `num_replica` times on each of
            the specified nodes.
        """
        pass


    def alias(self, name, version, new_name, rollout_strategy):
        """
            Create a replica of an existing model wrapper with a new
            name and treat it as a new model. From now on the two models
            will be considered independent, and version updates to one will
            not affect the other.
        """
        pass

    def inspect_model(self, name):
        """
            Get diagnostic information about a model. This includes both
            history of the model (who last modified it, when it was last updated)
            but also performance (e.g. latency and throughput, cache hit rates,
            how often it misses the latency SLO, etc.)
        """

    def set_model_permissions(self, name, permissions):
        """
            Let's look into Etcd permissions for this.
        """



class ClipperClusterManager:
    """
        All actions through the API are associated with a specific user
        to track provenance and add permissions.




        How does cluster membership work?
        How does cluster deployment work?
        Atomicity, Isolation, Idempotence

        Proposal:
            Let's use etcd for cluster membership.
            Does that mean the cluster manager connects directly to Etcd
            to make changes, or does it connect to one of the Clipper instances
            which then propagates the changes to Etcd? How do we want to manage shipping
            models which could be 100s of MB?

    """

    def __init__(user_id):
        pass


    
    def start(num_instances, nodes):
        """
            Start a new Clipper cluster on the provided list of machines.
            
            Questions:
            How to access these machines? SSH? Expect a cluster manager?

        """
        pass

    def connect(address):
        """
            Connect to a running Clipper cluster.

            Questions:
            + How is cluster membership handled? Etcd?
        """

    def shutdown(self):
        """
            Shutdown the connected cluster

        """

    def get_metrics(self):
        self.get_system_metrics()
        self.get_model_metrics()

    def get_system_metrics(self):
        """
            Get physical performance metrics (latency, throughput, cache hits, perf, etc.)
        """

    def get_model_metrics(self):
        """
            Get model performance metrics
        """



























