.PHONY: all install-poetry install-pip format

all: install-poetry

POETRY-EXECUTOR=poetry run

# Install Python dependencies using Poetry
install-poetry:
	poetry install --no-root -v

# Install Python dependencies using pip
install-pip:
	pip install -r requirements.txt

format:
	$(POETRY-EXECUTOR) ruff format .
