FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.10 AS builder

COPY poetry.lock /src/app-root/
COPY pyproject.toml /src/app-root/
COPY src/ /src/app-root/src/
COPY README.md /src/app-root/

RUN cd /src/app-root \
    # we need to build the wheel in order to install the binary python  \
    # package that uses click to parse the command arguments
    && poetry build --format wheel \
    && poetry export --without-hashes -o requirements.txt \
    && poetry export --without-hashes --dev -o requirements_dev.txt

FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.10:latest-slim AS base

COPY --from=builder /src/app-root/dist/*.whl /src/app-root/
COPY --from=builder /src/app-root/requirements.txt /src/app-root/

RUN pip install -r /src/app-root/requirements.txt \
    && pip install /src/app-root/*.whl --no-deps \
    && rm /src/app-root/*.whl

WORKDIR /src/app-root

FROM base AS runner

RUN mkdir /src/app-root/data /src/app-root/ouput

ENTRYPOINT ["pyflexplot"]

FROM base AS tester

COPY --from=builder /src/app-root/requirements_dev.txt /src/app-root/requirements_dev.txt
RUN pip install -r /src/app-root/requirements_dev.txt

COPY tests /src/app-root/tests
COPY pyproject.toml test_ci.sh /src/app-root/

CMD ["/bin/bash", "-c", "source /src/app-root/test_ci.sh && run_ci_tools"]

FROM tester AS documenter

COPY doc /src/app-root/doc
COPY CONTRIBUTING.md HISTORY.md README.md /src/app-root/

CMD ["sphinx-build", "doc", "doc/_build"]
