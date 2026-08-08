// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

// simd.hpp — one vector abstraction, three back ends: NEON (aarch64), AVX2 (amd64),
// and a portable scalar fallback. Every routine below has a scalar twin in sorts.hpp
// so the SIMD claim is always measured against the same algorithm, not a strawman.
//
// The vector unit is asked to do exactly two things a sort needs:
//   1. reduce  — OR/AND over a lane, which is how the varying bytes are discovered
//   2. compare-exchange — the primitive of a sorting network, which is how the
//      small-n floor is beaten without a single unpredictable branch
// It is NOT asked to histogram or to gather: neither has a vector form on these
// machines, and pretending otherwise is where SIMD sort claims usually go wrong.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SORTLAB_NEON 1
#include <arm_neon.h>
#elif defined(__AVX2__)
#define SORTLAB_AVX2 1
#include <immintrin.h>
#endif

// ------------------------------------------------------------------ vec u64
#if defined(SORTLAB_NEON)

using vecu64 = uint64x2_t;
constexpr int kVecLanes = 2;
inline const char* simd_isa() { return "neon"; }

inline vecu64 vload(const uint64_t* p) { return vld1q_u64(p); }
inline void vstore(uint64_t* p, vecu64 v) { vst1q_u64(p, v); }
inline vecu64 vor(vecu64 a, vecu64 b) { return vorrq_u64(a, b); }
inline vecu64 vand(vecu64 a, vecu64 b) { return vandq_u64(a, b); }
inline vecu64 vxor(vecu64 a, vecu64 b) { return veorq_u64(a, b); }
inline vecu64 vdup(uint64_t x) { return vdupq_n_u64(x); }
inline vecu64 vmin_u64(vecu64 a, vecu64 b) { return vbslq_u64(vcgtq_u64(a, b), b, a); }
inline vecu64 vmax_u64(vecu64 a, vecu64 b) { return vbslq_u64(vcgtq_u64(a, b), a, b); }
inline uint64_t vreduce_or(vecu64 v) { return vgetq_lane_u64(v, 0) | vgetq_lane_u64(v, 1); }
inline uint64_t vreduce_and(vecu64 v) { return vgetq_lane_u64(v, 0) & vgetq_lane_u64(v, 1); }

#elif defined(SORTLAB_AVX2)

using vecu64 = __m256i;
constexpr int kVecLanes = 4;
inline const char* simd_isa() { return "avx2"; }

inline vecu64 vload(const uint64_t* p) { return _mm256_loadu_si256((const __m256i*)p); }
inline void vstore(uint64_t* p, vecu64 v) { _mm256_storeu_si256((__m256i*)p, v); }
inline vecu64 vor(vecu64 a, vecu64 b) { return _mm256_or_si256(a, b); }
inline vecu64 vand(vecu64 a, vecu64 b) { return _mm256_and_si256(a, b); }
inline vecu64 vxor(vecu64 a, vecu64 b) { return _mm256_xor_si256(a, b); }
inline vecu64 vdup(uint64_t x) { return _mm256_set1_epi64x((long long)x); }

// AVX2 compares epi64 as SIGNED, so both operands are biased by 2^63 first: one xor
// turns an unsigned compare into the signed one the machine has.
inline vecu64 vgt_u64(vecu64 a, vecu64 b) {
  const vecu64 bias = _mm256_set1_epi64x((long long)0x8000000000000000ull);
  return _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
}
inline vecu64 vmin_u64(vecu64 a, vecu64 b) { return _mm256_blendv_epi8(a, b, vgt_u64(a, b)); }
inline vecu64 vmax_u64(vecu64 a, vecu64 b) { return _mm256_blendv_epi8(b, a, vgt_u64(a, b)); }
inline uint64_t vreduce_or(vecu64 v) {
  uint64_t t[4];
  vstore(t, v);
  return t[0] | t[1] | t[2] | t[3];
}
inline uint64_t vreduce_and(vecu64 v) {
  uint64_t t[4];
  vstore(t, v);
  return t[0] & t[1] & t[2] & t[3];
}

#else

struct vecu64 {
  uint64_t x[2];
};
constexpr int kVecLanes = 2;
inline const char* simd_isa() { return "scalar"; }

inline vecu64 vload(const uint64_t* p) { return vecu64{{p[0], p[1]}}; }
inline void vstore(uint64_t* p, vecu64 v) { p[0] = v.x[0]; p[1] = v.x[1]; }
inline vecu64 vor(vecu64 a, vecu64 b) { return vecu64{{a.x[0] | b.x[0], a.x[1] | b.x[1]}}; }
inline vecu64 vand(vecu64 a, vecu64 b) { return vecu64{{a.x[0] & b.x[0], a.x[1] & b.x[1]}}; }
inline vecu64 vxor(vecu64 a, vecu64 b) { return vecu64{{a.x[0] ^ b.x[0], a.x[1] ^ b.x[1]}}; }
inline vecu64 vdup(uint64_t x) { return vecu64{{x, x}}; }
inline vecu64 vmin_u64(vecu64 a, vecu64 b) {
  return vecu64{{a.x[0] < b.x[0] ? a.x[0] : b.x[0], a.x[1] < b.x[1] ? a.x[1] : b.x[1]}};
}
inline vecu64 vmax_u64(vecu64 a, vecu64 b) {
  return vecu64{{a.x[0] > b.x[0] ? a.x[0] : b.x[0], a.x[1] > b.x[1] ? a.x[1] : b.x[1]}};
}
inline uint64_t vreduce_or(vecu64 v) { return v.x[0] | v.x[1]; }
inline uint64_t vreduce_and(vecu64 v) { return v.x[0] & v.x[1]; }

#endif

// ------------------------------------------------------------ vary reduce
// vary = (OR of every key) XOR (AND of every key): the bits that are not constant
// across the lane. A byte of the lane is uniform exactly when its vary byte is zero,
// and a uniform byte is a pass that can be skipped before it is ever run. DESC is an
// XOR by a constant, which commutes with both reduces and cancels in the difference,
// so this is read off the raw lane.
struct VaryReduce {
  uint64_t any = 0;
  uint64_t all = ~0ull;
  uint64_t vary() const { return any ^ all; }
};

inline VaryReduce vary_scalar(const uint64_t* p, size_t n, size_t stride) {
  VaryReduce v;
  for (size_t i = 0; i < n; i++) {
    uint64_t k = p[i * stride];
    v.any |= k;
    v.all &= k;
  }
  return v;
}

// The same loop with auto-vectorisation switched off. It exists to keep the comparison
// honest: the interesting question is vector vs scalar ISA, and a compiler that already
// vectorises the plain loop would otherwise make the intrinsics look like the whole win.
#if defined(__clang__)
#define SORTLAB_NOVEC _Pragma("clang loop vectorize(disable) interleave(disable) unroll(disable)")
#elif defined(__GNUC__)
#define SORTLAB_NOVEC _Pragma("GCC unroll 1")
#else
#define SORTLAB_NOVEC
#endif

inline VaryReduce vary_novec(const uint64_t* p, size_t n, size_t stride) {
  VaryReduce v;
  SORTLAB_NOVEC
  for (size_t i = 0; i < n; i++) {
    uint64_t k = p[i * stride];
    v.any |= k;
    v.all &= k;
  }
  return v;
}

// Only a stride-1 lane can be reduced by the vector unit: a strided lane has no
// contiguous vector to load, which is the layout question stated as an instruction.
// Four accumulators, not one. A single accumulator chains one OR per load and the loop
// runs at the latency of that chain rather than at the width of the machine; four
// independent chains keep the vector pipes full. This is the difference between using
// the vector registers and using the vector unit.
inline VaryReduce vary_simd(const uint64_t* p, size_t n, size_t stride) {
  if (stride != 1) return vary_scalar(p, n, stride);
  VaryReduce v;
  const size_t step = (size_t)kVecLanes * 4;
  vecu64 any0 = vdup(0), any1 = vdup(0), any2 = vdup(0), any3 = vdup(0);
  vecu64 all0 = vdup(~0ull), all1 = vdup(~0ull), all2 = vdup(~0ull), all3 = vdup(~0ull);
  size_t i = 0;
  for (; i + step <= n; i += step) {
    vecu64 k0 = vload(p + i);
    vecu64 k1 = vload(p + i + kVecLanes);
    vecu64 k2 = vload(p + i + 2 * kVecLanes);
    vecu64 k3 = vload(p + i + 3 * kVecLanes);
    any0 = vor(any0, k0);
    any1 = vor(any1, k1);
    any2 = vor(any2, k2);
    any3 = vor(any3, k3);
    all0 = vand(all0, k0);
    all1 = vand(all1, k1);
    all2 = vand(all2, k2);
    all3 = vand(all3, k3);
  }
  v.any = vreduce_or(vor(vor(any0, any1), vor(any2, any3)));
  v.all = vreduce_and(vand(vand(all0, all1), vand(all2, all3)));
  for (; i < n; i++) {
    v.any |= p[i];
    v.all &= p[i];
  }
  return v;
}

// --------------------------------------------------------- bitonic network
// A sorting network has no data-dependent branch at all: the compare-exchanges are
// fixed by n, so the machine never mispredicts and never stalls on a key it has not
// read yet. That is the whole reason it beats a comparator below the radix floor.
//
// Strides at or above the vector width are done in vector registers. Strides below it
// need an intra-register shuffle whose direction alternates inside the vector, so they
// run scalar — that tail is part of the measurement, not hidden by it.
inline void ce_scalar(uint64_t& a, uint64_t& b, bool ascending) {
  uint64_t lo = a < b ? a : b;
  uint64_t hi = a < b ? b : a;
  a = ascending ? lo : hi;
  b = ascending ? hi : lo;
}

inline void ce_block(uint64_t* a, uint64_t* b, size_t count, bool ascending) {
  size_t i = 0;
  for (; i + kVecLanes <= count; i += kVecLanes) {
    vecu64 x = vload(a + i), y = vload(b + i);
    vecu64 lo = vmin_u64(x, y), hi = vmax_u64(x, y);
    vstore(a + i, ascending ? lo : hi);
    vstore(b + i, ascending ? hi : lo);
  }
  for (; i < count; i++) ce_scalar(a[i], b[i], ascending);
}

// n must be a power of two. Ascending overall.
inline void bitonic_sort_simd(uint64_t* a, size_t n) {
  for (size_t size = 2; size <= n; size <<= 1) {
    for (size_t stride = size >> 1; stride > 0; stride >>= 1) {
      if (stride >= 2) {
        for (size_t base = 0; base < n; base += stride << 1) {
          bool ascending = (base & size) == 0;
          ce_block(a + base, a + base + stride, stride, ascending);
        }
      } else {
        for (size_t i = 0; i < n; i += 2) {
          ce_scalar(a[i], a[i + 1], (i & size) == 0);
        }
      }
    }
  }
}

inline void bitonic_sort_scalar(uint64_t* a, size_t n) {
  for (size_t size = 2; size <= n; size <<= 1) {
    for (size_t stride = size >> 1; stride > 0; stride >>= 1) {
      for (size_t i = 0; i < n; i++) {
        size_t j = i ^ stride;
        if (j > i) ce_scalar(a[i], a[j], (i & size) == 0);
      }
    }
  }
}

// ----------------------------------------------------------------- encode
// Signed and floating axes become order-preserving unsigned coordinates by a single
// bit fold, and DESC is one more XOR. The whole encode is width-independent lane
// arithmetic, so it is the part of the pipeline the vector unit does own outright.
inline void encode_i64_scalar(const int64_t* src, uint64_t* dst, size_t n) {
  SORTLAB_NOVEC
  for (size_t i = 0; i < n; i++) dst[i] = (uint64_t)src[i] ^ (1ull << 63);
}

inline void encode_i64_simd(const int64_t* src, uint64_t* dst, size_t n) {
  const vecu64 flip = vdup(1ull << 63);
  size_t i = 0;
  for (; i + kVecLanes <= n; i += kVecLanes) {
    vstore(dst + i, vxor(vload((const uint64_t*)src + i), flip));
  }
  for (; i < n; i++) dst[i] = (uint64_t)src[i] ^ (1ull << 63);
}
