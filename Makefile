# Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
# SPDX-License-Identifier: Apache-2.0

# sortlab — self-contained. No dependencies beyond a C++17 compiler.
#
#   make            build bench + verify
#   make verify     run the correctness oracle
#   make bench      run every sweep
#   make scale      run one sweep (scale entropy lanes payload presort smalln topk simd)

CXX      ?= c++
STD      := -std=c++17
WARN     := -Wall -Wextra -Wno-unused-parameter
OPT      := -O3 -fno-omit-frame-pointer -DNDEBUG

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  ARCH := -mcpu=native
else ifeq ($(UNAME_M),aarch64)
  ARCH := -mcpu=native
else
  # amd64: AVX2 is the baseline the vector path targets; -march=native picks up more.
  ARCH := -march=native -mavx2
endif

CXXFLAGS := $(STD) $(WARN) $(OPT) $(ARCH)
SRC      := src
BIN      := bin

.PHONY: all bench verify clean scale entropy lanes payload presort smalln topk simd

all: $(BIN)/bench $(BIN)/verify

$(BIN):
	mkdir -p $(BIN) results

$(BIN)/bench: $(SRC)/bench.cpp $(SRC)/common.cpp $(SRC)/*.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)/bench.cpp $(SRC)/common.cpp

$(BIN)/verify: $(SRC)/verify.cpp $(SRC)/common.cpp $(SRC)/*.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)/verify.cpp $(SRC)/common.cpp

verify: $(BIN)/verify
	./$(BIN)/verify

bench: $(BIN)/bench verify
	./$(BIN)/bench all | tee results/all.txt

scale entropy lanes payload presort smalln topk simd: $(BIN)/bench
	./$(BIN)/bench $@ | tee results/$@.txt

clean:
	rm -rf $(BIN) results
