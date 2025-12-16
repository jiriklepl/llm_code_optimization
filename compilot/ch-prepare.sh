#!/bin/bash

set -euo pipefail

if ! [ -d ./copied-tiramisu ] || ! [ -d ./imgdir ]; then
	rm -rf ./copied-tiramisu ./imgdir
	ch-image build -f Dockerfile.compilot .
	ch-convert -i ch-image -o dir compilot imgdir
	cp -r ./imgdir/tiramisu ./copied-tiramisu~temp
	mv ./copied-tiramisu~temp ./copied-tiramisu
fi
