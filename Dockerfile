FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.10 AS builder
ARG VERSION
LABEL ch.meteoswiss.project=pyflexplot-${VERSION}

COPY poetry.lock /src/app-root/
COPY pyproject.toml /src/app-root/
COPY src/ /src/app-root/src/
COPY README.md /src/app-root/

RUN cd /src/app-root \
    # we need to build the wheel in order to install the binary python  \
    # package that uses click to parse the command arguments
    && poetry build --format wheel \
    && poetry export -o requirements.txt \
    && poetry export --dev -o requirements_dev.txt

FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.10:latest-slim AS base
ARG VERSION
LABEL ch.meteoswiss.project=pyflexplot-${VERSION}

COPY --from=builder /src/app-root/dist/*.whl /src/app-root/
COPY --from=builder /src/app-root/requirements.txt /src/app-root/

RUN pip install -r /src/app-root/requirements.txt \
    && pip install /src/app-root/*.whl --no-deps \
    && rm /src/app-root/*.whl

WORKDIR /src/app-root

FROM base AS runner
ARG VERSION
LABEL ch.meteoswiss.project=pyflexplot-${VERSION}

RUN mkdir /src/app-root/data /src/app-root/output

ENTRYPOINT ["pyflexplot"]

FROM base AS tester
ARG VERSION
LABEL ch.meteoswiss.project=pyflexplot-${VERSION}

COPY --from=builder /src/app-root/requirements_dev.txt /src/app-root/requirements_dev.txt
RUN pip install -r /src/app-root/requirements_dev.txt

COPY tests /src/app-root/tests
