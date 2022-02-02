PYTHON = python3
PIP = pip3
PYTHONIOENCODING=utf8

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
deps:
	if test -n "$$CUDA_VERSION"; then :; \
	elif test -s /usr/local/cuda/version.txt; then \
		CUDA_VERSION=$$(sed 's/^.* //;s/\([0-9]\+[.][0-9]\).*/\1/' /usr/local/cuda/version.txt); \
	elif command -v nvcc &>/dev/null; then \
		CUDA_VERSION=$$(nvcc --version | sed -n '/^Cuda/{s/.* release //;s/,.*//;p}'); \
	elif command -v nvidia-smi &>/dev/null; then \
		CUDA_VERSION=$$(nvidia-smi --version | sed -n '/CUDA Version/{s/.*CUDA Version: //;s/ .*//;p}'); \
	elif command -v pkg-config &>/dev/null; then \
		CUDA_VERSION=$$(pkg-config --list-all | sed -n '/^cudart/{s/cudart-//;s/ .*//;p;q}'); \
	else \
		echo >&2 "Cannot find CUDA runtime library, assuming CPU-only"; CUDA_VERSION=CPU; \
	fi && echo "Detected CUDA version $$CUDA_VERSION" && \
	if test "$$CUDA_VERSION" = CPU; then CUDA=cpu; else CUDA=cu$${CUDA_VERSION//.}; fi && \
	$(PIP) install -r requirements.txt \
	-f 'https://dl.fbaipublicfiles.com/detectron2/wheels/$$CUDA/torch1.10/index.html' 

# Install Python package via pip
install: deps
	$(PIP) install .

.PHONY: help deps install
