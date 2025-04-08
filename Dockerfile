ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/en/contact" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/bertsky/ocrd_detectron2" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_detectron2" \
    org.opencontainers.image.description="OCR-D wrapper for detectron2 based segmentation models" \
    org.opencontainers.image.source="https://github.com/bertsky/ocrd_detectron2" \
    org.opencontainers.image.documentation="https://github.com/bertsky/ocrd_detectron2/blob/${VCS_REF}/README.md" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=ocrd/core-cuda-torch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# avoid HOME/.local/share (hard to predict USER here)
# so let XDG_DATA_HOME coincide with fixed system location
# (can still be overridden by derived stages)
ENV XDG_DATA_HOME /usr/local/share
# avoid the need for an extra volume for persistent resource user db
# (i.e. XDG_CONFIG_HOME/ocrd/resources.yml)
ENV XDG_CONFIG_HOME /usr/local/share/ocrd-resources

WORKDIR /build/ocrd_detectron2

COPY . .
COPY ocrd-tool.json .
# prepackage ocrd-tool.json as ocrd-all-tool.json
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json
# install everything and reduce image size
RUN apt-get install -y --no-install-recommends g++ && \
    make deps && \
    make install && \
    rm -rf /build/ocrd_detectron2 && \
    apt-get -y remove --auto-remove g++

WORKDIR /data
VOLUME /data
