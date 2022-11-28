PYTHON = python3
PIP = pip3
PYTHONIOENCODING=utf8
SHELL = /bin/bash

help:
	@echo
	@echo "  Targets"
	@echo
	@echo "    deps    Install only Python deps via pip"
	@echo "    install Install full Python package via pip"
	@echo
	@echo "  Variables"
	@echo "    PYTHON"
	@echo "    CUDA_VERSION  override detection of CUDA runtime version (e.g. '11.3' or 'CPU')"

# Install Python deps via pip
# There is no prebuilt for detectron2 on PyPI, and the wheels depend on CUDA and Torch version.
# See https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#install-pre-built-detectron2
# and https://github.com/facebookresearch/detectron2/issues/969
# While there is a web site which lists them, which works with `pip -f`, this unfortunately cannot
# be encapsulated via setuptools, see https://github.com/pypa/pip/issues/5898
# and https://stackoverflow.com/questions/3472430/how-can-i-make-setuptools-install-a-package-thats-not-on-pypi
# and https://github.com/pypa/pip/issues/4187
# Detectron2 requires Torch >=1.10 and <1.11, which is quite out of date now.
# Also, the prebuilt versions on https://dl.fbaipublicfiles.com/detectron2/wheels/*/torch1.10/index.html
# are only available for CUDA 10.1, 10.2, 11.1, 11.3 or CPU.
# Moreoever, even Torch >=1.10 and <1.11 is not available on https://download.pytorch.org/whl/torch/
# except for a narrow few CUDA versions.
# To make matters worse, source build of Detectron2 fails unless Torch is already installed before:
# https://github.com/facebookresearch/detectron2/issues/4472
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
	$(PIP) install -f "https://dl.fbaipublicfiles.com/detectron2/wheels/$$CUDA/torch1.10/index.html" \
	"git+https://github.com/facebookresearch/detectron2@v0.6#egg=detectron2==0.6"

# Install Python package via pip
install: deps
	$(PIP) install .

.PHONY: help deps install
