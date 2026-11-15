FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS dependencies

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_ROOT=/input \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends wine wine64 xvfb util-linux \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/program/solution
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 \
    && python -m pip install --no-cache-dir -r requirements.txt
# Official 2026-11-15 runtime: add 32-bit Wine while reusing Python dependency layers.
RUN dpkg --add-architecture i386 \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends wine32:i386 \
    && rm -rf /var/lib/apt/lists/*
FROM dependencies AS solution
COPY . ./
RUN install -m 0755 runtime/wine64-loader.sh /usr/local/bin/wine \
    && install -m 0755 runtime/wine64-loader.sh /usr/local/bin/wine-stable \
    && python -m pip install --no-cache-dir -r environment.lock.txt \
    && python -m pip check \
    && chmod +x run.sh run_client.sh \
    && test -f data/official/serve.exe && command -v setsid \
    && python -m unittest discover -s tests -v
ENTRYPOINT ["/app/program/solution/run.sh"]
CMD ["--task", "all", "--test", "--out-dir", "/output"]
