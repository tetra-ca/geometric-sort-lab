#!/bin/sh
# Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
# SPDX-License-Identifier: Apache-2.0

# run.sh — build, prove correct, then measure. Records the machine alongside the numbers,
# because every number below is a property of this machine as much as of the algorithm.
set -e
cd "$(dirname "$0")"

make
./bin/verify

mkdir -p results
{
  echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname:    $(uname -srm)"
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "cpu:      $(sysctl -n machdep.cpu.brand_string)"
    echo "cores:    $(sysctl -n hw.physicalcpu) physical / $(sysctl -n hw.logicalcpu) logical"
    echo "l1d:      $(sysctl -n hw.perflevel0.l1dcachesize 2>/dev/null || sysctl -n hw.l1dcachesize)"
    echo "l2:       $(sysctl -n hw.perflevel0.l2cachesize 2>/dev/null || sysctl -n hw.l2cachesize)"
    echo "line:     $(sysctl -n hw.cachelinesize)"
    echo "page:     $(sysctl -n hw.pagesize)"
    echo "memory:   $(sysctl -n hw.memsize)"
  else
    echo "cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "cores:    $(nproc)"
    echo "memory:   $(grep MemTotal /proc/meminfo | awk '{print $2 * 1024}')"
    [ -r /sys/devices/system/cpu/cpu0/cache/index0/size ] &&
      echo "l1d:      $(cat /sys/devices/system/cpu/cpu0/cache/index0/size)"
  fi
  echo "compiler: $(${CXX:-c++} --version | head -1)"
} | tee results/machine.txt

./bin/bench all | tee results/all.txt

# Hardware counters, where the OS exposes them. macOS does not without a signed tool,
# so the branch-miss argument there rests on the wall-clock shape alone.
if command -v perf >/dev/null 2>&1; then
  perf stat -e branch-misses,branches,cache-misses,cache-references,instructions,cycles \
    ./bin/bench scale 2>results/perf-scale.txt >/dev/null
  echo "wrote results/perf-scale.txt"
fi
