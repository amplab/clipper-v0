FROM alpine:3.4

MAINTAINER Dan Crankshaw <dscrankshaw@gmail.com>


RUN echo "@testing http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories \
&& apk update \
&& apk add --no-cache --virtual .build_deps git python \
python-dev gfortran py-pip build-base  \
&& apk add --no-cache py-numpy@testing \
# && apk search openblas@testing \
# openblas@testing py-numpy@testing \
# && pip install numpy
# && rm -rf /tmp/build \
&& apk del .build_deps

COPY mw /tmp/

EXPOSE 6001

CMD ["python", "/tmp/test_new_rpc.py"]



