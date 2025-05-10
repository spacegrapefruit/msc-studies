.PHONY: all install-poetry install-pip format

all: install-poetry main

POETRY-EXECUTOR=poetry run

# Install Python dependencies using Poetry
install-poetry:
	poetry install --no-root -v

# Install Python dependencies using pip
install-pip:
	pip install -r requirements.txt

# Run the main script
main:
	$(POETRY-EXECUTOR) python python/main.py

format:
	$(POETRY-EXECUTOR) ruff format .
