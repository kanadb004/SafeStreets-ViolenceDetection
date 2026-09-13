.PHONY: setup test lint verify

setup:
	conda create -n safestreets --clone tf_env
	pip install -e ".[dev]"

lint:
	ruff check safestreets tests

test:
	pytest -m "not slow and not needs_data" -q

verify:
	ruff check . && pytest -m phase$(PHASE) -q
ifeq ($(PHASE),0)
	test -f docs/decisions/ADR-001-runtime-stack.md
endif
