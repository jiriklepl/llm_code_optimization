#!/bin/bash

set -uo pipefail

git submodule update --init --recursive

. functions

patch
