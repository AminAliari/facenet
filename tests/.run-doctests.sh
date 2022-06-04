#!/bin/bash
set -e -u -o pipefail

git ls-files facenet | grep -e "\.py$" | xargs pytest --doctest-modules