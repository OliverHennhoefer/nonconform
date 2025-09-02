# Minimal makefile for MkDocs documentation
#

BUILDDIR = site

# Put it first so that "make" without argument builds HTML documentation
docs:
	cd "$(shell pwd)" && uv run mkdocs build

serve:
	cd "$(shell pwd)" && uv run mkdocs serve

clean:
	rm -rf $(BUILDDIR)

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  docs      to build the HTML documentation"
	@echo "  serve     to serve documentation locally with live reload"
	@echo "  clean     to remove build artifacts"

.PHONY: help docs serve clean
