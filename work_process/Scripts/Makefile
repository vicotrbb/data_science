# VARIABLES
ROOT:=./
VENV_BIN_DIR:="venv/bin"

PIP:="$(VENV_BIN_DIR)/pip"
TEST:="$(VENV_BIN_DIR)/pytest"
LOCAL:="$(VENV_BIN_DIR)/gunicorn"

VIRTUALENV:=$(shell which virtualenv)
REQUIREMENTS:="requirements.txt"

APP_NAME=serasa-victorbona
DOCKER_IMAGE_REMOTE=vicotrbb/serasa-victorbona

# PHONY

.PHONY: help stack clean venv test run docker-run docker-build up

# UTILS

help:
	@echo "#####################--HELP--#####################"
	@echo "Info: Commands run-local and test execute setup as well"
	@echo "Info: When running, the project will be deployed at 5000 port"
	@echo "Info: Documentation can be found at http://localhost:5000/"
	@echo "Warning: This project needs virtualenv as prerequisite"
	@echo "Warning: As example the API was setup with APIKEY: c3bad652-c011-4d59-9038-d4dbbe056cde"
	@echo "#####################--COMMANDS--#####################"
	@echo "clean -> Cleans project temp files(aka cache)"
	@echo "stack -> Print project basic tech stack"
	@echo "venv -> Setups project virtual env"
	@echo "run -> Runs the project with gunicorn locally"
	@echo "test -> Runs pytest on the project"
	@echo "docker-run-remote -> Runs the docker image from local from registry"
	@echo "docker-run -> Runs the docker image from local"
	@echo "docker-build -> Builds docker image from application"
	@echo "up -> Builds and run application from docker"

stack:
	@echo "#####################--STACK--#####################"
	@echo "flask-restx -> Web framework, a REST alternative from flask"
	@echo "gunicorn -> HTTP server"
	@echo "Elasticserach -> Database"
	@echo "Docker -> Conteinerization"
	@echo "pytest -> Unitary test suit"

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
