FROM scorpil/rust:stable

# COPY clipper_gitrepo_for_docker /root/.ssh/id_rsa

COPY clipper_server /rust/clipper/clipper_server
COPY benchmarks /rust/clipper/benchmarks

RUN apt-get update \
# && echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config \
&& apt-get install libssl-dev git g++ -qqy --no-install-recommends \
# && ssh-keyscan -H github.com >> ~/.ssh/known_hosts \
# && git clone --recursive git@github.com:amplab/clipper.git \
&& cd clipper/benchmarks \
&& cargo build \
&& rm -rf /var/lib/apt/lists/*

ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
# EXPOSE 1337

# COPY test.toml /rust/clipper/clipper_server/conf/test.toml
# ENTRYPOINT ["/rust/clipper/clipper_server/target/debug/clipper-rest", "start"]
# CMD ["--conf=/rust/clipper/clipper_server/conf/test.toml"]

COPY digits_bench.toml /rust/clipper/benchmarks/digits_bench.toml
ENTRYPOINT ["/rust/clipper/benchmarks/target/debug/clipper-benchmarks", "digits"]
CMD ["--conf=/rust/clipper/benchmarks/digits_bench.toml"]

