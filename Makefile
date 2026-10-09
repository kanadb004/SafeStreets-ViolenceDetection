.PHONY: setup test lint verify verify-all notebooks

PHASES_DONE := phase0 or phase1 or phase2 or phase3

setup:
	conda create -n safestreets --clone tf_env
	pip install -e ".[dev]"

lint:
	ruff check safestreets tests

test:
	pytest -m "not slow and not needs_data" -q

# Cumulative marker expression for sprint N: all done phases plus sprints 1..N.
# A sprint may not break an earlier gate.
SPRINT_EXPR := $(PHASES_DONE)$(foreach n,$(shell seq 1 $(if $(SPRINT),$(SPRINT),0)), or sprint$(n))

verify:
ifdef SPRINT
	ruff check .
	pytest -m "$(SPRINT_EXPR)" -q
else
	ruff check .
	pytest -m phase$(PHASE) -q
ifeq ($(PHASE),0)
	test -f docs/decisions/ADR-001-runtime-stack.md
endif
endif

verify-all:
	ruff check .
	pytest -q

# Execute every notebook in place so committed outputs are never stale.
notebooks:
	@for nb in notebooks/*.ipynb; do \
	  echo "executing $$nb"; \
	  jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=600 "$$nb" || exit 1; \
	done
