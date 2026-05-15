ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake git libssl-dev && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

ARG LLAMA_CPP_VERSION=master
RUN git clone --depth 1 --branch ${LLAMA_CPP_VERSION} \
    https://github.com/ggml-org/llama.cpp.git /build/llama.cpp

WORKDIR /build/llama.cpp

# T4 (sm_75), A100 (sm_80), L4 (sm_89), H100 (sm_90)
ARG CUDA_DOCKER_ARCH="75;80;89;90"

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    cp build/bin/llama-server /app/

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/lib/ /usr/local/lib/
COPY --from=builder /app/llama-server /usr/local/bin/llama-server
RUN ldconfig

RUN mkdir -p /models

EXPOSE 8080

ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--gpu-layers", "all", "--jinja", "--metrics"]
