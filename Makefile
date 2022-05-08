RUN := poetry run
DIR := $(shell pwd)
VENV := .venv

FILES?=$(DIR)


all: help

help:
	@echo "Commands:"
	@echo "  \033[00;32minstall\033[0m - setup environment."
	@echo "  \033[00;32mformat\033[0m  - format code with Black. Override \033[00;33mFILES\033[0m variable to format certain file or files."
	@echo "  \033[00;32mlint\033[0m    - run linting in the code base. Override \033[00;33mFILES\033[0m variable to lint certain file or files."
	@echo "  \033[00;32mclean\033[0m   - remove all python artifacts."

install:
	@poetry config virtualenvs.in-project true
	@poetry install
	@echo "[ \033[00;32mPoetry setup completed. You are good to go!\033[0m ]"

format:
	@echo "[ \033[00;33mFormat with Catalyst\033[0m ]"
	$(RUN) catalyst-make-codestyle -l 100 $(FILES)

lint:
	@echo "[ \033[00;33mCatalyst linter\033[0m ]"
	$(RUN) catalyst-check-codestyle -l 100 $(FILES)

clean:
	find . -name "*.pyc" -exec rm -f {} +
	find . -name "*.pyo" -exec rm -f {} +
	find . -name "*~" -exec rm -f {} +
	find . -name "__pycache__" -exec rm -fr {} +
