# Makefile for Real-Time Object Detection Pipeline
# Provides convenient commands for common tasks

.PHONY: help install install-dev setup check test benchmark clean lint format

# Default target
help:
	@echo "Real-Time Object Detection Pipeline - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make install      - Install system dependencies and Python packages"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make setup        - Run system setup scripts"
	@echo ""
	@echo "Development:"
	@echo "  make check        - Run system health check"
	@echo "  make test         - Run test suite"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with black"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Clean temporary files and caches"
	@echo "  make clean-all    - Clean everything including models"
	@echo ""
	@echo "Runtime:"
	@echo "  make run          - Run standard detection pipeline"
	@echo "  make run-web      - Run web viewer"
	@echo "  make run-perf     - Run high-performance mode"

# Installation targets
install:
	@echo "Installing system dependencies and Python packages..."
	./scripts/setup.sh
	@echo "Installation complete. To install RealSense SDK, run: make setup-realsense"

install-dev: install
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"

setup:
	@echo "Running system setup scripts..."
	./scripts/setup.sh
	@echo "Basic setup complete. For RealSense SDK: make setup-realsense"

setup-realsense:
	@echo "Installing Intel RealSense SDK (this takes 30-60 minutes)..."
	./scripts/setup_realsense.sh
	./scripts/fix_realsense_venv.sh
	@echo "RealSense SDK installation complete."

# Development targets
check:
	@echo "Running system health check..."
	./scripts/system_check.sh

test:
	@echo "Running test suite..."
	python -m pytest tests/ -v

test-fast:
	@echo "Running fast tests only..."
	python -m pytest tests/ -v -m "not slow"

lint:
	@echo "Running code linting..."
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting code..."
	black src/ tests/
	@echo "Code formatting complete."

type-check:
	@echo "Running type checking..."
	mypy src/

# Cleanup targets
clean: ## Clean temporary files and caches
	@echo "🧹 Running cleanup script..."
	./scripts/cleanup.sh

clean-all: clean ## Clean everything including models and virtual environment
	@echo "🗑️  Deep cleaning all generated files including models..."
	rm -rf models/*.pt models/*.onnx models/*.param models/*.bin
	rm -rf venv/
	@echo "Deep cleanup complete."

# Runtime targets
run:
	@echo "Starting standard detection pipeline..."
	python3 src/main.py --width 424 --height 240

run-web:
	@echo "Starting web viewer at http://localhost:5000..."
	python3 src/web_viewer.py --width 640 --height 480

run-perf:
	@echo "Starting high-performance mode..."
	python3 src/high_performance_viewer.py

run-headless:
	@echo "Starting headless mode..."
	python3 src/main.py --headless --width 424 --height 240

# Environment setup
env:
	@echo "Setting up environment variables..."
	@echo "export LD_LIBRARY_PATH=/usr/local/lib:\$$LD_LIBRARY_PATH"
	@echo "source venv/bin/activate"
	@echo ""
	@echo "Add the following to your ~/.bashrc for persistence:"
	@echo "echo 'export LD_LIBRARY_PATH=/usr/local/lib:\$$LD_LIBRARY_PATH' >> ~/.bashrc"

# Status and info
status:
	@echo "Project Status:"
	@echo "==============="
	@./scripts/system_check.sh | head -20

info:
	@echo "Real-Time Object Detection Pipeline"
	@echo "==================================="
	@echo "Python: $$(python3 --version)"
	@echo "Working Directory: $$(pwd)"
	@echo "Virtual Environment: $$(which python3)"
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Last Modified: $$(git log -1 --format=%cd --date=short 2>/dev/null || echo 'Unknown')"

# Docker targets (for future containerization)
docker-build:
	@echo "Building Docker image..."
	docker build -t rt-object-detection .

docker-run:
	@echo "Running in Docker container..."
	docker run --rm -it --device=/dev/bus/usb rt-object-detection

# Documentation targets
docs:
	@echo "Building documentation..."
	@echo "Documentation is in markdown format in docs/ directory"
	@echo "Main files:"
	@echo "  - docs/installation_guide.md"
	@echo "  - docs/performance_optimization.md"
	@echo "  - docs/troubleshooting.md"

# Configuration management
config-list:
	@echo "Available configurations:"
	@ls -la config/*.yaml

config-validate:
	@echo "Validating configuration files..."
	@for file in config/*.yaml; do \
		echo "Checking $$file..."; \
		python3 -c "import yaml; yaml.safe_load(open('$$file'))" && echo "✓ Valid" || echo "✗ Invalid"; \
	done

# Model management
.PHONY: download-models benchmark-models compare-models

download-models: ## Download all YOLO models for comparison
	@echo "📥 Downloading YOLO models..."
	python scripts/download_models.py

benchmark-models: ## Run comprehensive multi-model benchmark
	@echo "🔥 Running multi-model benchmark..."
	./scripts/multi_model_benchmark.sh

compare-models: ## Quick model comparison in Jupyter
	@echo "📊 Opening model comparison notebook..."
	jupyter notebook notebooks/model_fine_tuning_export.ipynb

# Performance testing
.PHONY: benchmark test-fps system-check

benchmark: ## Run performance benchmark
	@echo "⚡ Running performance benchmark..."
	./scripts/benchmark.sh

system-check: ## Check system compatibility and performance
	@echo "🔍 Checking system compatibility..."
	./scripts/system_check.sh

# System optimization
optimize:
	@echo "Applying system optimizations..."
	@echo "Setting CPU governor to performance..."
	sudo cpufreq-set -g performance || echo "Could not set CPU governor (may need admin rights)"
	@echo "Checking temperature..."
	vcgencmd measure_temp
	@echo "System optimization complete."
