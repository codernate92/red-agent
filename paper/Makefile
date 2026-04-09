SHELL := /bin/bash

PDF := main.pdf
BUILD_DIR := build
DIST_DIR := dist
ARXIV_BUNDLE := $(DIST_DIR)/red-agent-arxiv-source.tar.gz

.PHONY: pdf arxiv clean

pdf:
	tectonic --outdir . main.tex

arxiv:
	./scripts/create_arxiv_bundle.sh

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
