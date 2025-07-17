SHELL := /bin/bash

.PHONY: help
help: ## Displays help information about available make commands
	@if command -v uv &> /dev/null; then \
		uv run --no-project python -c "import re; \
		[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open('$(MAKEFILE_LIST)').read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"; \
	else \
		echo "Available commands include: help, compile, test..."; \
		echo "Run 'make install-uv' to install uv."; \
		echo "Then run 'make help' again to see docs on available commands."; \
	fi

.PHONY: check-system-dep
check-system-dep: ## Checks if system dependencies are installed (nvcc) based on OS
	@if ! command -v nvcc &> /dev/null; then \
		echo "nvcc not found. Please install CUDA and set up your environment."; \
		exit 1; \
	fi
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found.  Please install uv with 'make install-uv'"; \
		exit 1; \
	fi

.PHONY: install-uv
install-uv: ## Installs uv package manger
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/0.6.10/install.sh | sh; \
		echo "please follow the uv suggestions to update or restart your shell environment"; \
	fi
	@uv_version=$$(uv --version | awk '{print $$2}'); \
	if [ "$$(printf '%s\n' "0.6.10" "$$uv_version" | sort -V | head -n1)" != "0.6.10" ]; then \
		echo "Updating uv to latest version..."; \
		uv self update 0.6.10; \
	fi

.PHONY: install-python-dep
install-python-dep: ## Installs the Python dependencies
	uv sync

.PHONY: compile
compile: check-system-dep ## Compiles the CUDA extension with nanobind
	@echo "Compiling CUDA extension with nanobind..."
	@echo "If you get an scikit-build-core error, you may need to 'uv cache clean' and 'trash build/'"
	uv pip install scikit-build-core nanobind ninja cmake
	uv sync --group build
	# Not sure if the pip command is also needed
	uv pip install -ve . --no-build-isolation

.PHONY: check
check: ## Checks the code
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Type checking: Running ty"
	@uv run ty check
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a

.PHONY: test
test: ## Runs Python tests
	@echo "🚀 Running tests"
	uv run --group test --group array --group compare pytest tests -v -s --benchmark-disable --save-output

.PHONY: benchmark
benchmark: ## Runs benchmarking comparisons
	@echo "🚀 Running benchmarking comparisons"
	uv run --group test --group array --group compare pytest tests -v -s --benchmark-only --benchmark-histogram --benchmark-autosave --benchmark-save-data

.PHONY: profile
profile: ## Runs Python test with simple profiling. Recommend using Nsight Compute or Nsight Systems for more detailed profiling.
	@echo "Building with CUDA_PROFILE"
	uv pip install --no-build-isolation -ve . -Ccmake.define.CMAKE_CUDA_FLAGS_INIT="-DCUDA_PROFILE"
	@echo "🚀 Running tests"
	uv run --group profile pyinstrument --timeline -m pytest tests/test_beamform.py -v -s --tile-total-frames 200

.PHONY: docs
docs: ## Build the documentation
	@echo "🚀 Building documentation"
	uv run $(MAKE) -C docs html;

.PHONY: docs-open
docs-open: docs ## Build and open the documentation
	@if [ -n "$$SSH_CONNECTION" ]; then \
		@echo "🚀 Serving documentation"; \
		uv run python -m http.server --directory docs/_build/html 8000; \
	else \
		@echo "🚀 Opening documentation in browser"; \
		uv run python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/docs/_build/html/index.html')"; \
	fi

.PHONY: wheel
wheel: ## Builds a wheel
	uv build

.PHONY: clean
clean: ## Cleans build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf dist/ build/ mach.*.so

.DEFAULT_GOAL := help
