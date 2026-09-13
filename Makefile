.DEFAULT_GOAL := help
IS_CI:=$(CI)

# disable copying some_script.sh to some_script
%: %.sh

help:                       ## show available options
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: sync
sync:                       ## uv sync --extra dev --extra test --extra notebooks
	uv sync --extra dev --extra test --extra notebooks

.PHONY: format-and-lint-code
format-and-lint-code:       ## format and lint code with ruff
	uv run ruff format foundations_of_probability_and_statistics
	uv run ruff format tests
	uv run ruff check foundations_of_probability_and_statistics
	uv run ruff check tests

.PHONY : test
test:                       ## run tests
	uv run pytest -vv -s --cov=foundations_of_probability_and_statistics --cov-report=term-missing
#	uv run pytest -vv -s --cov=foundations_of_probability_and_statistics --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=pytest-report.xml

.PHONY: clean
clean:                      ## clean all
	./clean_all.sh
