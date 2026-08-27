# syntax=docker/dockerfile:1

# Build from a staged context (see stage-build-context.sh):
#
#   STAGE=$(./stage-build-context.sh /path/to/noble-ml-core)
#   docker build -t ml-core:latest "$STAGE"
#   az acr build --registry <acr> --image ml-core:latest "$STAGE"
#
# Or with Docker BuildKit additional context (local docker build only):
#   docker build -t ml-core:latest \
#     --build-context noble-ml-core=/path/to/noble-ml-core .

FROM mcr.microsoft.com/azurelinux/base/python:3.12

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

RUN tdnf -y install ca-certificates git gcc gcc-c++ cmake python3-devel && \
    update-ca-trust && \
    tdnf clean all

RUN pip3 install --no-cache-dir uv && \
    git config --global --add safe.directory '*'

ARG INFERENCE_BACKEND=cpu
ARG EXTRAS=models-forest,models-torch,training,chemical-featurizers-all

# Staged context (az acr build / stage-build-context.sh)
COPY noble-ml-core/ /src/noble-ml-core/

WORKDIR /src/noble-ml-core

RUN FINAL_EXTRAS="${EXTRAS},${INFERENCE_BACKEND}" && \
    uv pip install --prefix=/opt/noble-ml --no-cache \
      --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      ".[${FINAL_EXTRAS}]" && \
    rm -rf /src/noble-ml-core /root/.cache && \
    tdnf -y remove git gcc gcc-c++ cmake python3-devel && \
    tdnf clean all

ENV PATH="/opt/noble-ml/bin:${PATH}"
ENV PYTHONPATH="/opt/noble-ml/lib/python3.12/site-packages:/app"

RUN mkdir -p /workdir /input /output /tmp/matplotlib /app && \
    chmod 755 /workdir /input /output

COPY ml_core_utils.py /app/ml_core_utils.py
COPY entrypoint.py /app/entrypoint.py
RUN chmod +x /app/entrypoint.py

WORKDIR /workdir

CMD ["/bin/bash"]
