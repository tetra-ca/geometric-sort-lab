// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

#include <cstdlib>
#include <new>

uint64_t g_alloc_bytes = 0;
uint64_t g_alloc_count = 0;
std::vector<Result> g_results;

void* operator new(std::size_t sz) {
  g_alloc_bytes += sz;
  g_alloc_count++;
  void* p = std::malloc(sz ? sz : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t sz) { return operator new(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

static bool g_header_done = false;

void table_header(const char* title) {
  std::printf("\n=== %s\n", title);
  std::printf("%-22s %-26s %10s %12s %12s %12s %10s %8s %7s\n", "contender", "config", "n",
              "ns/op", "ns/elem", "Melem/s", "B/op", "allocs", "passes");
  std::printf("%s\n", std::string(126, '-').c_str());
  g_header_done = true;
}

static std::string human_n(uint64_t n) {
  char b[32];
  if (n >= 1000000 && n % 1000000 == 0)
    std::snprintf(b, sizeof b, "%lluM", (unsigned long long)(n / 1000000));
  else if (n >= 1000 && n % 1000 == 0)
    std::snprintf(b, sizeof b, "%lluK", (unsigned long long)(n / 1000));
  else
    std::snprintf(b, sizeof b, "%llu", (unsigned long long)n);
  return b;
}

void emit(const Result& r) {
  g_results.push_back(r);
  if (!g_header_done) table_header("results");
  std::printf("%-22s %-26s %10s %12.0f %12.3f %12.1f %10llu %8llu %7.1f\n", r.cont.c_str(),
              r.config.c_str(), human_n(r.n).c_str(), r.ns_op, r.ns_elem, r.melem_s,
              (unsigned long long)r.bytes, (unsigned long long)r.allocs, r.passes);
  std::fflush(stdout);
}

void write_csv(const char* path) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::fprintf(stderr, "sortlab: cannot write %s\n", path);
    return;
  }
  std::fprintf(f, "bench,contender,config,n,axes,payload,ns_op,ns_elem,melem_s,bytes,allocs,passes,reps\n");
  for (const Result& r : g_results) {
    std::fprintf(f, "%s,%s,%s,%llu,%u,%u,%.1f,%.4f,%.2f,%llu,%llu,%.1f,%llu\n", r.bench.c_str(),
                 r.cont.c_str(), r.config.c_str(), (unsigned long long)r.n, r.axes, r.payload,
                 r.ns_op, r.ns_elem, r.melem_s, (unsigned long long)r.bytes,
                 (unsigned long long)r.allocs, r.passes, (unsigned long long)r.reps);
  }
  std::fclose(f);
  std::printf("\nwrote %s (%zu rows)\n", path, g_results.size());
}
