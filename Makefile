# vars
ROOT:=./

NOTEBOOK-PORT=5050

VENV_BIN_DIR:="venv/bin"
PIP:="$(VENV_BIN_DIR)/pip"
VIRTUALENV:=$(shell which virtualenv)

# PHONY

.PHONY: clean prune-venv venv start-notebook commit

# functions

define create-venv
virtualenv venv -p python3
endef

# commands

prune-venv:

clean:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@find . -type d -name .pytest_cache -delete
	@rm -rf venv

venv:
	$(create-venv)
	@$(PIP) install --no-cache-dir -r $(REQUIREMENTS) | grep -v 'already satisfied' || true

start-notebook-nb:
	jupyter notebook --no-browser --port $(NOTEBOOK-PORT)

start-notebook-wb:
	jupyter notebook --port $(NOTEBOOK-PORT)

howami-git:
	@git config --global user.email "victor.bona@hotmail.com"
	@git config --global user.name "vicotrbb"

commit: howami-git
	@git-pull
	@git add .
	@git commit -m "Commit made by makefile command"
	@git push origin master
