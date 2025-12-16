#!/bin/bash

set -euo pipefail

sources=$(find datamining linear-algebra medley stencils -type f -name "*.cpp")

for src in $sources; do
	filename_no_ext=$(basename "$src" .cpp)
	sed 's/\(\bcast<\w\+>\)\s*(\(\s*[0-9]*\.[0-9e+-]*\s*\))/\1(Expr(\2))/g;s/#\s*include\s*"'"$filename_no_ext"'.h"/#include "'"$filename_no_ext"'.hpp"/g' -i "$src"
done
