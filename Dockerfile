FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

LABEL org.label-schema.name="happyray-dev"

RUN apt-get update && apt-get install -y --no-install-recommends \
    xauth \
    gcc-11 \
    g++-11 \
    libtbb-dev \
    libpng-dev \
    libsdl2-dev \
    pkg-config \
    python3-pip \
    && python3 -m pip install --no-cache-dir cmake==3.31.6 \
    && rm -rf /var/lib/apt/lists/*

RUN touch /root/.Xauthority

# WORKDIR /app
# COPY . .

# RUN cmake -S . -B build \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
#     -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
#     -DHAPPYRAY_THRUST_BACKEND=CUDA \
#     && cmake --build build --target happyray --parallel

# ENTRYPOINT ["/app/build/src/app/happyray"]
# CMD ["assets/ogre.obj"]