ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/kontakt" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/bertsky/detectron2" \
    org.label-schema.build-date=$BUILD_DATE

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8

WORKDIR /build/ocrd_detectron2
COPY setup.py .
COPY ocrd_detectron2/ocrd-tool.json .
COPY README.md .
COPY requirements.txt .
COPY requirements-test.txt .
COPY ocrd_detectron2 ./ocrd_detectron2
COPY Makefile .
RUN apt-get install -y --no-install-recommends g++ && \
    make deps && \
    make install && \
    rm -rf /build/ocrd_detectron2 && \
    apt-get -y remove --auto-remove g++

WORKDIR /data
VOLUME /data
