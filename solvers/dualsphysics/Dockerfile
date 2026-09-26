# Compiles the genuine DualSPHysics v5.4 CPU (OpenMP) solver from the source tree shipped
# in this repository and bundles the official Linux pre/post-processing tools
# (GenCase, PartVTK, MeasureTool, IsoSurface). Build context = DualSPHysics-master.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential make && rm -rf /var/lib/apt/lists/*
COPY src /dsph/src
COPY bin/linux /dsph/bin/linux
WORKDIR /dsph/src/source
RUN sed -i 's/^USE_NATIVE_CPU_OPTIMIZATIONS=NO/USE_NATIVE_CPU_OPTIMIZATIONS=YES/' Makefile_cpu \
 && make -f Makefile_cpu -j6 \
 && chmod +x /dsph/bin/linux/*_linux64
ENV LD_LIBRARY_PATH=/dsph/bin/linux
ENV PATH=/dsph/bin/linux:$PATH
RUN ls -la /dsph/bin/linux
WORKDIR /case
