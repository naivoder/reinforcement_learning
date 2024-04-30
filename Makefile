# Define the shell used by make
SHELL := /bin/bash

# Install dependencies
install:
	@echo "Installing dependencies from the root requirements.txt"
	@pip install --upgrade pip
	@pip install -r requirements.txt

# Format all Python files using black
format:
	@find . -mindepth 1 -maxdepth 2 -type f -name '*.py' -exec black {} +

# Lint all Python files in the project
lint:
	@find . -mindepth 1 -maxdepth 2 -type f -name '*.py' -exec pylint --disable=R,C {} + || true

# Run pytest in subdirectories that contain tests
test:
	@for dir in $$(find . -mindepth 1 -maxdepth 1 -type d); do \
		if [ -f $$dir/test_*.py ] || [ -f $$dir/*_test.py ]; then \
			echo "Running tests in $$dir"; \
			(cd $$dir && python -m pytest -vv --cov=.); \
		fi \
	done
