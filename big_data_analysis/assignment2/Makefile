all: lint install #test

POETRY-EXECUTOR=poetry run

# Check formatting with ruff https://docs.astral.sh/ruff/
lint:
	$(POETRY-EXECUTOR) ruff check . --exit-non-zero-on-fix
	$(POETRY-EXECUTOR) ruff format . --check

# Run all tests
test:
	$(POETRY-EXECUTOR) pytest --disable-warnings -v

# Fix formatting
format:
	$(POETRY-EXECUTOR) ruff format .
	$(POETRY-EXECUTOR) ruff check . --fix

install:
	poetry install --no-root
