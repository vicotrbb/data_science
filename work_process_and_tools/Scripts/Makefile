# VARIABLES
ROOT:=./
VENV_BIN_DIR:="venv/bin"

PIP:="$(VENV_BIN_DIR)/pip"
TEST:="$(VENV_BIN_DIR)/pytest"
LOCAL:="$(VENV_BIN_DIR)/gunicorn"

VIRTUALENV:=$(shell which virtualenv)
REQUIREMENTS:="requirements.txt"

APP_NAME=
DOCKER_IMAGE_REMOTE=vicotrbb/

# PHONY

.PHONY: help stack clean venv test run docker-run docker-build up

# UTILS

help:
	@echo "#####################--HELP--#####################"

stack:
	@echo "#####################--STACK--#####################"

clean:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@find . -type d -name .pytest_cache -delete
	@rm -rf venv

# DEVELOPMENT

define create-venv
virtualenv venv -p python3
endef

venv:
	$(create-venv)
	@$(PIP) install --no-cache-dir -r $(REQUIREMENTS) | grep -v 'already satisfied' || true

run: venv
	@$(LOCAL) --workers 2 --log-level info --timeout 90 --bind 0.0.0.0:5000 app:app

test: venv 
	@$(TEST) -v

freeze:
	@$(PIP) freeze > requirements.txt

docker-run-remote:
	docker run -it --rm --publish 5000:5000 $(DOCKER_IMAGE_REMOTE)

docker-run:
	docker run --name $(APP_NAME) -p 5000:5000 $(APP_NAME)

docker-build:
	docker build --tag $(APP_NAME) .

up: docker-build docker-run
