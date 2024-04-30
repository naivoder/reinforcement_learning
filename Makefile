# Define the shell used by make
SHELL := /bin/bash

# Define the subdirectories automatically
SUBDIRS = $(shell find . -mindepth 1 -maxdepth 1 -type d)

# Define phony targets
.PHONY: all $(SUBDIRS)

# Master target to run all steps for each subdirectory
all: $(SUBDIRS)

# Rule to process each subdirectory
$(SUBDIRS):
	@echo "Processing $$@..."
	@if [ -f $$@/requirements.txt ]; then \
		echo "Installing dependencies in $$@"; \
		(cd $$@ && pip install --upgrade pip && pip install -r requirements.txt); \
	fi
	@if [ -f $$@/test_*.py ] || [ -f $$@/*_test.py ]; then \
		echo "Linting and testing $$@"; \
		(find $$@ -type f -name '*.py' -exec pylint --disable=R,C {} +); \
		(cd $$@ && python -m pytest -vv --cov=.); \
	else \
		echo "No tests to run in $$@"; \
		(find $$@ -type f -name '*.py' -exec pylint --disable=R,C {} +); \
	fi


