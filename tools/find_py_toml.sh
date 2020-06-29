#!/bin/bash

dirs=(${@})
[ ${#dirs[@]} -eq 0 ] && dirs=(src tests)

\find ${dirs[@]} -type f -name '*.py' -o -type f -name '*.toml'
