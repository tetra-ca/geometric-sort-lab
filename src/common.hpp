// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

// common.hpp — timing, deterministic RNG, allocation accounting, table/CSV output.
// Self-contained: nothing here depends on anything outside sortlab.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------- allocation
// Global operator new/delete are overridden in common.cpp so every contender is
// charged for the memory it touches, including the temporaries std::stable_sort
// allocates internally. Counters are read as a delta around the timed region.
extern uint64_t g_alloc_bytes;
extern uint64_t g_alloc_count;

// ------------------------------------------------------------------- timing
using clk = std::chrono::steady_clock;

static inline double now_ns() {
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clk::now().time_since_epoch())
      .count();
}

// ---------------------------------------------------------------------- rng
// splitmix64: deterministic, seed-addressable, no state sharing between sweeps.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567) {}
  inline uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  inline uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
};

// ------------------------------------------------------------------- result
struct Result {
  std::string bench;   // sweep name
  std::string cont;    // contender
  std::string config;  // free-form config key (entropy/order/layout/...)
  uint64_t n = 0;
  uint32_t axes = 1;
  uint32_t payload = 0;
  double ns_op = 0;     // wall per operation
  double ns_elem = 0;   // wall per element
  double melem_s = 0;   // throughput
  uint64_t bytes = 0;   // heap bytes charged per op
  uint64_t allocs = 0;  // heap allocations per op
  double passes = 0;    // radix byte-passes actually executed (0 for comparators)
  uint64_t reps = 0;
};

extern std::vector<Result> g_results;

// The clock on this class of machine ticks in tens of nanoseconds, which is longer
// than a small sort takes. A batch of independent operations is timed as one region
// and divided, so an 8-element sort is measured rather than rounded to a tick.
inline uint32_t batch_for(uint64_t n) {
  uint64_t b = 200000 / (n ? n : 1);
  return (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(512, b));
}

// timeit runs `prep` (untimed) then `run(i)` for i in [0,batch) (timed), repeating
// until the sample is long enough to be trustworthy, and keeps the MINIMUM — the run
// least disturbed by the OS. Allocation counters come from the first timed batch.
template <class Prep, class Run>
Result timeit(const char* bench, const char* cont, const std::string& config, uint64_t n,
              uint32_t batch, Prep prep, Run run) {
  Result r;
  r.bench = bench;
  r.cont = cont;
  r.config = config;
  r.n = n;

  prep();
  double t0 = now_ns();
  for (uint32_t i = 0; i < batch; i++) run(i);
  double warm = std::max(now_ns() - t0, 1.0);

  // Target ~60ms of measurement, clamped to [3, 2000] repetitions of the batch.
  uint64_t reps = (uint64_t)std::max(3.0, std::min(2000.0, 6.0e7 / warm));
  double best = 1e300;
  uint64_t b0 = 0, c0 = 0, b1 = 0, c1 = 0;
  for (uint64_t rep = 0; rep < reps; rep++) {
    prep();
    uint64_t ab = g_alloc_bytes, ac = g_alloc_count;
    double s = now_ns();
    for (uint32_t i = 0; i < batch; i++) run(i);
    double e = now_ns() - s;
    if (rep == 0) {
      b0 = ab;
      c0 = ac;
      b1 = g_alloc_bytes;
      c1 = g_alloc_count;
    }
    best = std::min(best, e);
  }
  double per_op = best / (double)batch;
  r.reps = reps * batch;
  r.ns_op = per_op;
  r.ns_elem = n ? per_op / (double)n : 0;
  r.melem_s = per_op > 0 ? (double)n * 1000.0 / per_op : 0;
  r.bytes = (b1 - b0) / batch;
  r.allocs = (c1 - c0) / batch;
  return r;
}

void emit(const Result& r);                 // record + print one table row
void table_header(const char* title);       // print a table banner
void write_csv(const char* path);           // dump every recorded result
