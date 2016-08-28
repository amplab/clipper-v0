FROM continuumio/anaconda:latest

MAINTAINER Dan Crankshaw <dscrankshaw@gmail.com>

# COPY spark_docker_deps/spark-1.6.2-bin-hadoop2.6.tgz /tmp/
COPY optimization-server.py /tmp/

EXPOSE 6001

CMD ["python", "/tmp/optimization-server.py"]

