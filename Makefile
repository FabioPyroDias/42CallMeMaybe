export UV_PROJECT_ENVIRONMENT = call-me-maybe

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

RM = rm -rf

install:
	uv sync

run:

clean:

lint:

lint-strict:

destroy: clean
	$(RM) call-me-maybe