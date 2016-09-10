FROM gcc:5.2
COPY . /usr/src/noop-mw
WORKDIR /usr/src/noop-mw
RUN cd /usr/src/noop-mw && make noop

CMD ["/usr/src/noop-mw/noop"]
