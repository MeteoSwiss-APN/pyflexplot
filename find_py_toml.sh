#!/bin/bash

dirs=(${@})
[ ${#dirs[@]} -eq 0 ] && dirs=(src tests)

\find ${dirs[@]} -name '*.py' -o -name '*.toml'
