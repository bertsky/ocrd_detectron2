PYTHON = python3
PIP = pip3
PYTHONIOENCODING=utf8
SHELL = /bin/bash

# Docker container tag
DOCKER_TAG = 'bertsky/ocrd_detectron2'

help:
	@echo
	@echo "  Targets"
	@echo
	@echo "    deps      Install only Python dependencies via pip"
	@echo "    install   Install full Python package via pip"
	@echo "    deps-test Install Python dependencies for tests via pip and models via resmgr"
	@echo "    test      Run regression tests"
	@echo "    build     Build Python package as source and wheel distribution"
	@echo "    clean     Remove symlinks in test/assets"
	@echo "    docker    Build Docker image"
	@echo
	@echo "  Variables"
	@echo "    PYTHON"
	@echo "    CUDA_VERSION  override detection of CUDA runtime version (e.g. '11.3' or 'CPU')"
	@echo "    DOCKER_TAG    Docker image tag of result for the docker target"

# Install Python deps via pip
# There is no prebuilt for detectron2 on PyPI, and the public wheels depend on CUDA and Torch version.
# See https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#install-pre-built-detectron2
# and https://github.com/facebookresearch/detectron2/issues/969
# While there is a web site which lists them, which works with `pip install -f`, this unfortunately cannot
# be encapsulated via setuptools, see https://github.com/pypa/pip/issues/5898
# and https://stackoverflow.com/questions/3472430/how-can-i-make-setuptools-install-a-package-thats-not-on-pypi
# and https://github.com/pypa/pip/issues/4187
# Detectron2 requires Torch >=1.10 and <1.11, which is quite out of date now.
# Also, the prebuilt versions on https://dl.fbaipublicfiles.com/detectron2/wheels/*/torch1.10/index.html
# are only available for CUDA 10.1, 10.2, 11.1, 11.3 or CPU.
# Moreoever, even Torch >=1.10 and <1.11 is not available on https://download.pytorch.org/whl/torch/
# except for a narrow few CUDA versions.
# To make matters worse, Detectron2 setup fails specifying Torch as build-time and run-time dependency:
# https://github.com/facebookresearch/detectron2/issues/4472
# Therefore, source build of Detectron2 fails unless Torch is already installed before _and_ using
# pip install --no-build-isolation.
# Finally, due to https://github.com/pypa/pip/issues/4321, we cannot even mix -f links and pkgindex (for Pytorch versions)
# because pip will (more or less) randomly pick the one or the other.
# Detectron2 must always have the same version of Torch at runtime which it was compiled against.
deps:
	@$(PIP) install -r <(sed "/torch/d;/detectron2/d" requirements.txt)
	@if test -n "$$CUDA_VERSION"; then :; \
	elif test -s /usr/local/cuda/version.txt; then \
		CUDA_VERSION=$$(sed 's/^.* //;s/\([0-9]\+[.][0-9]\).*/\1/' /usr/local/cuda/version.txt); \
	elif command -v nvcc &>/dev/null; then \
		CUDA_VERSION=$$(nvcc --version | sed -n '/^Cuda/{s/.* release //;s/,.*//;p;}'); \
	elif command -v nvidia-smi &>/dev/null; then \
		CUDA_VERSION=$$(nvidia-smi | sed -n '/CUDA Version/{s/.*CUDA Version: //;s/ .*//;p;}'); \
	elif command -v pkg-config &>/dev/null; then \
		CUDA_VERSION=$$(pkg-config --list-all | sed -n '/^cudart/{s/cudart-//;s/ .*//;p;q;}'); \
	fi && \
	if test -z "$$CUDA_VERSION"; then \
		echo "Cannot find CUDA runtime library, assuming CPU-only"; CUDA_VERSION=CPU; \
	fi && echo "Detected CUDA version: $$CUDA_VERSION" && \
	if test "$$CUDA_VERSION" = CPU; then CUDA=cpu; \
	else IFS=. CUDA=($$CUDA_VERSION) && CUDA=cu$${CUDA[0]}$${CUDA[1]}; \
	fi && \
	$(PIP) install -i "https://download.pytorch.org/whl/$$CUDA" \
	-r <(sed -n "/torch/p" requirements.txt) && \
	$(PIP) install --no-build-isolation -f "https://dl.fbaipublicfiles.com/detectron2/wheels/$$CUDA/torch1.10/index.html" \
	"git+https://github.com/facebookresearch/detectron2@v0.6#egg=detectron2"
	-wget --quiet -O- https://github.com/facebookresearch/detectron2/commit/ed294fabf043eadbb33948d5c39f8f0361829a4f.patch | patch -d `$(PYTHON) -c "import detectron2; print(detectron2.__path__[0])"` -p2

# Install Python package via pip
install: deps
	$(PIP) install .

# Install testing python deps via pip
deps-test: models-test
	$(PIP) install -r requirements-test.txt

build:
	$(PIP) install build
	$(PYTHON) -m build .

# Clone OCR-D/assets to ./repo/assets
repo/assets:
	@mkdir -p $(@D)
	git clone https://github.com/OCR-D/assets $@

# Setup test data
test/assets: repo/assets
	@mkdir -p $@
	cp -r -t $@ repo/assets/data/*

# Remove test data copies and intermediate results
clean:
	-$(RM) -r test/assets

# Build docker image
docker:
	docker build \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

#MODELDIR := $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/ocrd-resources/ocrd-detectron2-segment

TESTMODEL := TableBank_X152_Psarpei
TESTMODEL += DocBank_X101
TESTMODEL += Jambo-sudo_X101
TESTMODEL += PRImALayout_R50

TESTBED := gutachten
TESTBED += column-samples

models-test: $(TESTMODEL:=.yaml)
models-test: $(TESTMODEL:=.pth)

%.yaml:
	ocrd resmgr download ocrd-detectron2-segment $@
%.pth:
	ocrd resmgr download ocrd-detectron2-segment $@

test: $(patsubst %,test/assets/%/data/test-result,$(TESTBED))
	@cat $^

count-regions := python -c "import sys; from ocrd_models.ocrd_page import parse; print('%s: %d' % (sys.argv[1], len(parse(sys.argv[1], silence=True).get_Page().get_AllRegions())))"

%/test-result: test/assets
	for MODEL in $(TESTMODEL); do \
		$(MAKE) MODEL=$$MODEL $*/OCR-D-SEG-$$MODEL; \
	done
	@shopt -s nullglob; { for file in $(TESTMODEL:%=$*/OCR-D-SEG-%/*.xml); do \
		$(count-regions) $$file; \
	done; } > $@

%/OCR-D-BIN: 
	cd $(@D) && ocrd-skimage-binarize -I `grp=(*IMG); basename $$grp` -O $(@F)

# workaround for OCR-D/core#930:
%/OCR-D-SEG-$(MODEL): PRESET = $(shell ocrd-detectron2-segment -D)/presets_$(MODEL).json

%/OCR-D-SEG-$(MODEL): %/OCR-D-BIN
	cd $(@D) && ocrd-detectron2-segment -I $(<F) -O $(@F) -P debug_img instance_colors_only -P postprocessing only-nms -P min_confidence 0.3 -p $(PRESET)

# make cannot delete directories, so keep them
.PRECIOUS .SECONDARY: %/OCR-D-BIN %/OCR-D-SEG-$(MODEL)

.PHONY: help deps install build deps-test models-test test clean docker
