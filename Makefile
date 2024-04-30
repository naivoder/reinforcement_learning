# Define the shell used by make
SHELL := /bin/bash

# Install command to handle multiple requirements.txt in subdirectories
install:
    @for dir in $$(find . -mindepth 1 -maxdepth 1 -type d); do \
        if [ -f $$dir/requirements.txt ]; then \
            echo "Installing dependencies in $$dir"; \
            pip install --upgrade pip && \
            pip install -r $$dir/requirements.txt; \
        fi \
    done

# Format all Python files using black
format:
    @find . -mindepth 1 -maxdepth 2 -type f -name '*.py' -exec black {} +

# Lint all Python files in the project
lint:
    @find . -mindepth 1 -maxdepth 2 -type f -name '*.py' -exec pylint --disable=R,C {} +

# Run pytest in subdirectories that contain tests
test:
    @for dir in $$(find . -mindepth 1 -maxdepth 1 -type d); do \
        if [ -f $$dir/test_*.py ] || [ -f $$dir/*_test.py ]; then \
            echo "Running tests in $$dir"; \
            python -m pytest -vv --cov=$$dir $$dir; \
        fi \
    done
