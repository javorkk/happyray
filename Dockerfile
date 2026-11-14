# Toolchain follows VFX Reference Platform CY2026 where it matters for ABI:
#   gcc 14.2 (Ubuntu 24.04 ships 14.2), oneTBB 2022.x, C++20-capable compiler.
# CUDA 12.9 officially supports gcc up to 14 as nvcc host compiler.
# OpenUSD has no CY2026 release yet (26.08 still targets CY2024 and builds as C++17); we build the
# latest release against the CY2026 toolchain above, core libraries only (no Python, no imaging).
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG ONETBB_VERSION=2022.3.0
ARG OPENUSD_VERSION=26.08
# OpenUSD's PEGTL-based parsers need ~2-3 GB per compiler process (16 jobs ran out of 32 GB).
# Under WSL2, gcc 14 also hits occasional non-reproducible cc1plus segfaults on heavy parallel
# builds, so each build is retried up to 3 times (make resumes where it stopped).
ARG BUILD_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=/usr/bin/gcc-14
ENV CXX=/usr/bin/g++-14
ENV CUDAHOSTCXX=/usr/bin/g++-14

LABEL org.label-schema.name="happyray-dev"

RUN apt-get update && apt-get install -y --no-install-recommends \
    xauth \
    gcc-14 \
    g++-14 \
    make \
    ca-certificates \
    curl \
    libpng-dev \
    libsdl2-dev \
    pkg-config \
    python3 \
    python3-pip \
    && python3 -m pip install --no-cache-dir --break-system-packages cmake==3.31.6 \
    && rm -rf /var/lib/apt/lists/*

# oneTBB 2022.x (CY2026). Built from source because Ubuntu 24.04's libtbb-dev is 2021.11.
# Installed to /usr/local so both OpenUSD and HappyRay's find_package(TBB CONFIG) pick it up.
RUN mkdir -p /tmp/src && cd /tmp/src \
    && curl -fsSL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${ONETBB_VERSION}.tar.gz | tar xz \
    && cmake -S oneTBB-${ONETBB_VERSION} -B build-tbb \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DTBB_TEST=OFF \
        -DTBB_EXAMPLES=OFF \
        -DTBB_STRICT=OFF \
    && (for i in 1 2 3; do cmake --build build-tbb --parallel ${BUILD_JOBS} && break; [ $i -lt 3 ] || exit 1; done) \
    && cmake --install build-tbb \
    && ldconfig \
    && rm -rf /tmp/src

# OpenUSD core (usd, usdGeom, sdf, ...). No Python bindings, so no Boost/Python dependency;
# imaging (Hydra/Storm) is off, so no OpenSubdiv/OpenGL either.
RUN mkdir -p /tmp/src && cd /tmp/src \
    && curl -fsSL https://github.com/PixarAnimationStudios/OpenUSD/archive/refs/tags/v${OPENUSD_VERSION}.tar.gz | tar xz \
    && cmake -S OpenUSD-${OPENUSD_VERSION} -B build-usd \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/openusd \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -DPXR_ENABLE_PYTHON_SUPPORT=OFF \
        -DPXR_BUILD_IMAGING=OFF \
        -DPXR_BUILD_USD_IMAGING=OFF \
        -DPXR_ENABLE_GL_SUPPORT=OFF \
        -DPXR_BUILD_USD_TOOLS=OFF \
        -DPXR_BUILD_USDVIEW=OFF \
        -DPXR_BUILD_EXEC=OFF \
        -DPXR_BUILD_USD_VALIDATION=OFF \
        -DPXR_BUILD_TESTS=OFF \
        -DPXR_BUILD_EXAMPLES=OFF \
        -DPXR_BUILD_TUTORIALS=OFF \
        -DPXR_BUILD_HTML_DOCUMENTATION=OFF \
    && (for i in 1 2 3; do cmake --build build-usd --parallel ${BUILD_JOBS} && break; [ $i -lt 3 ] || exit 1; done) \
    && cmake --install build-usd \
    && echo /opt/openusd/lib > /etc/ld.so.conf.d/openusd.conf \
    && ldconfig \
    && rm -rf /tmp/src

# find_package(pxr CONFIG) in src/app/CMakeLists.txt searches CMAKE_PREFIX_PATH.
ENV CMAKE_PREFIX_PATH=/opt/openusd

RUN touch /root/.Xauthority

# WORKDIR /app
# COPY . .

# RUN cmake -S . -B build \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DHAPPYRAY_THRUST_BACKEND=CUDA \
#     -DHAPPYRAY_ENABLE_USD=ON \
#     && cmake --build build --target happyray --parallel

# ENTRYPOINT ["/app/build/src/app/happyray"]
# CMD ["assets/ogre.obj"]
