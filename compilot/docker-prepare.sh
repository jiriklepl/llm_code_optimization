#!/bin/bash

set -euo pipefail

docker build -t tiramisu_baseline -f Dockerfile.compilot .
