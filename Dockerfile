FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.13:latest-poetry2 AS builder
ARG VERSION=0.0.0

LABEL ch.meteoswiss.project=pyflexplot-${VERSION}

COPY poetry.lock pyproject.toml README.md /src/app-root/
COPY src/ /src/app-root/src/

WORKDIR /src/app-root

# mchbuild's semantic version uses a SemVer '-' pre-release separator (e.g. 9.9.9-main), which is not valid PEP 440.
# Convert it to a PEP 440 local version label instead and set it for the poetry build command
RUN poetry version "$(echo ${VERSION} | sed -E 's/-/+/; s/-/./g')" \
    # The wheel is required to install the python package that uses click to parse the command arguments
    && poetry build --format wheel \
    && poetry export -o requirements.txt \
    && poetry export --with dev -o requirements_dev.txt

FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.13:latest-slim AS base
ARG VERSION
ARG BUILD_ID
LABEL ch.meteoswiss.project=pyflexplot-${VERSION}

ENV VERSION=$VERSION
ENV BUILD_ID=$BUILD_ID

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
