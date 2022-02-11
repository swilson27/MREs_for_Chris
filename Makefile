.PHONY: fmt
fmt:
	isort . \
	&& black .

.PHONY: lint
lint:
	black --check .
	isort --check .
	flake8 .

.PHONY: lint-types
lint-types: lint
	mypy .
