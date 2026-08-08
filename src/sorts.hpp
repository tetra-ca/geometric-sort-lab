// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

// sorts.hpp — the contenders.
//
// Every contender has the same contract: given n rows of keys, produce a permutation
// of row indices. The payload never moves during the sort; it is gathered once, later,
// and that gather is timed on its own so the two wins can be told apart.
//
//   comparator family — order is decided by asking "is this row before that one?"
//   radix family      — order is decided by counting digits; nothing is ever compared
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "gen.hpp"
#include "simd.hpp"

// ============================================================ comparator family

inline void perm_std_sort(const Keys& K, const uint8_t* desc, uint32_t* out) {
  std::iota(out, out + K.n, 0u);
  std::sort(out, out + K.n, RowLess{&K, desc});
}

inline void perm_std_stable(const Keys& K, const uint8_t* desc, uint32_t* out) {
  std::iota(out, out + K.n, 0u);
  std::stable_sort(out, out + K.n, RowLess{&K, desc});
}

inline void perm_insertion(const Keys& K, const uint8_t* desc, uint32_t* out) {
  RowLess less{&K, desc};
  std::iota(out, out + K.n, 0u);
  for (uint32_t i = 1; i < K.n; i++) {
    uint32_t v = out[i];
    uint32_t j = i;
    while (j > 0 && less(v, out[j - 1])) {
      out[j] = out[j - 1];
      j--;
    }
    out[j] = v;
  }
}

inline void perm_partial_sort(const Keys& K, const uint8_t* desc, uint32_t* out, uint32_t k) {
  std::iota(out, out + K.n, 0u);
  std::partial_sort(out, out + k, out + K.n, RowLess{&K, desc});
}

// The monomorphic control: no permutation, no indirection, no payload. This is the
// floor a comparator sort can possibly reach, and it is the honest scalar baseline.
inline void sort_keys_direct(uint64_t* k, size_t n) { std::sort(k, k + n); }

// ================================================================ radix family

struct RadixStats {
  uint32_t passes = 0;   // byte passes actually scattered
  uint32_t skipped = 0;  // byte passes proven uniform and never run
};

// A digit is one byte of one lane. The order of this list, most significant first, is
// the entire specification of the sort — everything below just executes it backwards.
struct Digit {
  uint32_t axis;
  uint32_t shift;
};

// DESC is carried as a per-lane XOR instead of a rewritten key matrix: ^k == k ^ ~0,
// so the inversion commutes with digit extraction and is applied at read.
inline std::vector<uint64_t> desc_masks(const Keys& K, const uint8_t* desc) {
  std::vector<uint64_t> m(K.nAxes, 0);
  for (uint32_t a = 0; a < K.nAxes; a++) m[a] = desc[a] ? ~0ull : 0ull;
  return m;
}

inline std::vector<uint64_t> vary_masks(const Keys& K, bool use_simd) {
  std::vector<uint64_t> v(K.nAxes, ~0ull);
  for (uint32_t a = 0; a < K.nAxes; a++) {
    Lane L = K.lane(a);
    VaryReduce r = use_simd ? vary_simd(L.base, K.n, L.stride) : vary_scalar(L.base, K.n, L.stride);
    v[a] = r.vary();
  }
  return v;
}

// Least significant digit last in the returned list.
inline std::vector<Digit> digit_plan(const Keys& K, const std::vector<uint64_t>& vary,
                                     RadixStats* st) {
  std::vector<Digit> plan;
  for (uint32_t a = 0; a < K.nAxes; a++) {
    for (int b = 7; b >= 0; b--) {
      uint32_t shift = (uint32_t)(8 * b);
      if (((vary[a] >> shift) & 0xFF) == 0) {
        if (st) st->skipped++;
        continue;
      }
      plan.push_back(Digit{a, shift});
    }
  }
  if (st) st->passes = (uint32_t)plan.size();
  return plan;
}

inline void prefix_sum(uint32_t* count) {
  uint32_t sum = 0;
  for (int i = 0; i < 256; i++) {
    uint32_t c = count[i];
    count[i] = sum;
    sum += c;
  }
}

// Every buffer a radix pass needs, owned by the caller. The engine pools these; a
// benchmark that allocates them per call is measuring malloc below a few hundred rows,
// not measuring a sort. Both forms are reported, because the difference IS the floor.
class RadixScratch {
 public:
  void ensure(uint32_t n, uint32_t nAxes) {
    if (order.size() < n) {
      order.resize(n);
      cur.resize(n);
      nxt.resize(n);
      digits.resize(n);
    }
    if (mask.size() < nAxes) {
      mask.resize(nAxes);
      vary.resize(nAxes);
    }
    plan.clear();
  }
  void ensure_topk(uint32_t n) {
    if (idx.size() < n) {
      idx.resize(n);
      buf.resize(n);
    }
  }
  std::vector<uint32_t> order;
  std::vector<uint64_t> cur, nxt;
  std::vector<uint8_t> digits;
  std::vector<uint64_t> mask, vary;
  std::vector<Digit> plan;
  std::vector<uint32_t> idx, buf, window;
};

inline void fill_masks(const Keys& K, const uint8_t* desc, bool use_simd, RadixScratch& s) {
  for (uint32_t a = 0; a < K.nAxes; a++) {
    s.mask[a] = desc[a] ? ~0ull : 0ull;
    Lane L = K.lane(a);
    VaryReduce r = use_simd ? vary_simd(L.base, K.n, L.stride) : vary_scalar(L.base, K.n, L.stride);
    s.vary[a] = r.vary();
  }
}

inline void fill_plan(const Keys& K, RadixScratch& s, RadixStats* st) {
  for (uint32_t a = 0; a < K.nAxes; a++) {
    for (int b = 7; b >= 0; b--) {
      uint32_t shift = (uint32_t)(8 * b);
      if (((s.vary[a] >> shift) & 0xFF) == 0) {
        if (st) st->skipped++;
        continue;
      }
      s.plan.push_back(Digit{a, shift});
    }
  }
  if (st) st->passes = (uint32_t)s.plan.size();
}

// ---------------------------------------------------------------------------
// perm_radix_gather — the key matrix stays put and each pass reads it through the
// current order. That read is a data-dependent address: the row is known only after
// the previous index lands, so past cache every touch is a miss the front cannot run
// ahead of. Fewest bytes moved, broken stride.
inline void perm_radix_gather(const Keys& K, const uint8_t* desc, uint32_t* out, RadixStats* st,
                              bool use_simd, RadixScratch& s) {
  uint32_t n = K.n;
  std::iota(out, out + n, 0u);
  if (n < 2) return;
  s.ensure(n, K.nAxes);
  fill_masks(K, desc, use_simd, s);
  fill_plan(K, s, st);

  uint32_t* order = out;
  uint32_t* next = s.order.data();

  for (size_t p = s.plan.size(); p-- > 0;) {
    Lane L = K.lane(s.plan[p].axis);
    uint64_t mk = s.mask[s.plan[p].axis];
    uint32_t shift = s.plan[p].shift;
    uint32_t count[256] = {0};

    for (uint32_t i = 0; i < n; i++) {
      uint8_t d = (uint8_t)((L[order[i]] ^ mk) >> shift);
      s.digits[i] = d;
      count[d]++;
    }
    prefix_sum(count);
    for (uint32_t i = 0; i < n; i++) next[count[s.digits[i]]++] = order[i];
    std::swap(order, next);
  }
  if (order != out) std::memcpy(out, order, (size_t)n * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// perm_radix_carry — the lane is materialised into order sequence ONCE per axis, then
// every byte pass of that axis reads it at stride 1. One gather per axis instead of one
// per byte pass; the key rides along with the permutation so the front always knows the
// next address. More bytes moved, unbroken stride.
inline void perm_radix_carry(const Keys& K, const uint8_t* desc, uint32_t* out, RadixStats* st,
                             bool use_simd, RadixScratch& s) {
  uint32_t n = K.n;
  std::iota(out, out + n, 0u);
  if (n < 2) return;
  s.ensure(n, K.nAxes);
  fill_masks(K, desc, use_simd, s);
  fill_plan(K, s, st);

  uint32_t* order = out;
  uint32_t* next = s.order.data();
  uint64_t* cur = s.cur.data();
  uint64_t* nxt = s.nxt.data();

  for (size_t p = s.plan.size(); p-- > 0;) {
    uint32_t axis = s.plan[p].axis;
    bool axis_start = (p + 1 == s.plan.size()) || s.plan[p + 1].axis != axis;
    if (axis_start) {  // the one gather this axis pays for
      Lane L = K.lane(axis);
      uint64_t mk = s.mask[axis];
      for (uint32_t i = 0; i < n; i++) cur[i] = L[order[i]] ^ mk;
    }
    uint32_t shift = s.plan[p].shift;
    uint32_t count[256] = {0};
    for (uint32_t i = 0; i < n; i++) count[(cur[i] >> shift) & 0xFF]++;
    prefix_sum(count);
    for (uint32_t i = 0; i < n; i++) {
      uint32_t slot = count[(cur[i] >> shift) & 0xFF]++;
      next[slot] = order[i];
      nxt[slot] = cur[i];
    }
    std::swap(order, next);
    std::swap(cur, nxt);
  }
  if (order != out) std::memcpy(out, order, (size_t)n * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// perm_radix_prehist — a histogram counts a multiset, and a permutation does not change
// a multiset. Every byte histogram of an axis is therefore knowable from one sequential
// sweep, before any scatter runs. The counting half of every pass after the first
// disappears; only the scatters remain.
inline void perm_radix_prehist(const Keys& K, const uint8_t* desc, uint32_t* out, RadixStats* st,
                               bool use_simd, RadixScratch& s) {
  uint32_t n = K.n;
  std::iota(out, out + n, 0u);
  if (n < 2) return;
  s.ensure(n, K.nAxes);
  fill_masks(K, desc, use_simd, s);
  fill_plan(K, s, st);

  uint32_t* order = out;
  uint32_t* next = s.order.data();
  uint64_t* cur = s.cur.data();
  uint64_t* nxt = s.nxt.data();
  uint32_t hist[8][256];

  for (size_t p = s.plan.size(); p-- > 0;) {
    uint32_t axis = s.plan[p].axis;
    bool axis_start = (p + 1 == s.plan.size()) || s.plan[p + 1].axis != axis;
    if (axis_start) {
      // Only the bytes this axis will actually scatter are counted; a uniform byte was
      // already proven not to order anything and counting it would be work for nothing.
      uint8_t bytes[8];
      int nbytes = 0;
      for (size_t q = p + 1; q-- > 0 && s.plan[q].axis == axis;) bytes[nbytes++] = (uint8_t)(s.plan[q].shift >> 3);
      Lane L = K.lane(axis);
      uint64_t mk = s.mask[axis];
      std::memset(hist, 0, sizeof hist);
      for (uint32_t i = 0; i < n; i++) {  // gather once, count every planned byte, one sweep
        uint64_t k = L[order[i]] ^ mk;
        cur[i] = k;
        for (int b = 0; b < nbytes; b++) hist[bytes[b]][(k >> (8 * bytes[b])) & 0xFF]++;
      }
      for (int b = 0; b < nbytes; b++) prefix_sum(hist[bytes[b]]);
    }
    uint32_t shift = s.plan[p].shift;
    uint32_t* count = hist[shift >> 3];
    for (uint32_t i = 0; i < n; i++) {
      uint32_t slot = count[(cur[i] >> shift) & 0xFF]++;
      next[slot] = order[i];
      nxt[slot] = cur[i];
    }
    std::swap(order, next);
    std::swap(cur, nxt);
  }
  if (order != out) std::memcpy(out, order, (size_t)n * sizeof(uint32_t));
}

// The same three, each allocating its own buffers: what a caller pays when nothing is
// pooled. Below a few hundred rows this difference is the entire measurement.
inline void perm_radix_gather(const Keys& K, const uint8_t* desc, uint32_t* out, RadixStats* st,
                              bool use_simd) {
  RadixScratch s;
  perm_radix_gather(K, desc, out, st, use_simd, s);
}
inline void perm_radix_carry(const Keys& K, const uint8_t* desc, uint32_t* out, RadixStats* st,
                             bool use_simd) {
  RadixScratch s;
  perm_radix_carry(K, desc, out, st, use_simd, s);
}
inline void perm_radix_prehist(const Keys& K, const uint8_t* desc, uint32_t* out, RadixStats* st,
                               bool use_simd) {
  RadixScratch s;
  perm_radix_prehist(K, desc, out, st, use_simd, s);
}

// ---------------------------------------------------------------------------
// perm_radix_msd_topk — LIMIT k is not a sort. The first digit level buckets all n once;
// every level below it touches only the rows still contending for the window. Cost is
// O(n) plus O(k·depth), and it is the shape that makes the whole comparison non-linear.
class MsdTopK {
 public:
  MsdTopK(const Keys& K, const std::vector<uint64_t>& mask, const std::vector<Digit>& plan,
          uint32_t k, RadixStats* st)
      : K_(K), mask_(mask), plan_(plan), k_(k), st_(st) {}

  void run(uint32_t* idx, uint32_t n, uint32_t* buf, std::vector<uint32_t>& out) {
    descend(idx, n, buf, 0, out);
  }

 private:
  void descend(uint32_t* idx, uint32_t n, uint32_t* buf, size_t level,
               std::vector<uint32_t>& out) {
    if (out.size() >= k_ || n == 0) return;
    if (level >= plan_.size() || n == 1) {
      for (uint32_t i = 0; i < n && out.size() < k_; i++) out.push_back(idx[i]);
      return;
    }
    Lane L = K_.lane(plan_[level].axis);
    uint64_t mk = mask_[plan_[level].axis];
    uint32_t shift = plan_[level].shift;

    uint32_t count[256] = {0};
    for (uint32_t i = 0; i < n; i++) count[(uint8_t)((L[idx[i]] ^ mk) >> shift)]++;
    uint32_t start[257];
    uint32_t sum = 0;
    for (int b = 0; b < 256; b++) {
      start[b] = sum;
      sum += count[b];
    }
    start[256] = sum;
    uint32_t cursor[256];
    std::memcpy(cursor, start, sizeof cursor);
    for (uint32_t i = 0; i < n; i++) buf[cursor[(uint8_t)((L[idx[i]] ^ mk) >> shift)]++] = idx[i];
    if (st_) st_->passes++;

    for (int b = 0; b < 256 && out.size() < k_; b++) {
      uint32_t len = start[b + 1] - start[b];
      if (len == 0) continue;
      if (out.size() + len <= k_ && len == 1) {  // whole bucket fits and cannot split
        out.push_back(buf[start[b]]);
        continue;
      }
      std::memcpy(idx + start[b], buf + start[b], (size_t)len * sizeof(uint32_t));
      descend(idx + start[b], len, buf + start[b], level + 1, out);
    }
  }

  const Keys& K_;
  const std::vector<uint64_t>& mask_;
  const std::vector<Digit>& plan_;
  uint32_t k_;
  RadixStats* st_;
};

inline void perm_radix_topk(const Keys& K, const uint8_t* desc, uint32_t* out, uint32_t k,
                            RadixStats* st, bool use_simd, RadixScratch& s) {
  uint32_t n = K.n;
  s.ensure(n, K.nAxes);
  s.ensure_topk(n);
  fill_masks(K, desc, use_simd, s);
  fill_plan(K, s, nullptr);
  std::iota(s.idx.begin(), s.idx.begin() + n, 0u);
  s.window.clear();
  s.window.reserve(k);
  MsdTopK(K, s.mask, s.plan, k, st).run(s.idx.data(), n, s.buf.data(), s.window);
  std::memcpy(out, s.window.data(), s.window.size() * sizeof(uint32_t));
}

inline void perm_radix_topk(const Keys& K, const uint8_t* desc, uint32_t* out, uint32_t k,
                            RadixStats* st, bool use_simd) {
  RadixScratch s;
  perm_radix_topk(K, desc, out, k, st, use_simd, s);
}

// ==================================================================== payload

// The permutation applied. This is the only time payload bytes move.
inline void gather_payload(uint8_t* dst, const uint8_t* src, const uint32_t* perm, uint32_t n,
                           uint32_t width) {
  for (uint32_t i = 0; i < n; i++) {
    std::memcpy(dst + (size_t)i * width, src + (size_t)perm[i] * width, width);
  }
}

// The alternative a comparator sort takes when it sorts records in place: every swap
// carries the whole row. Payload width multiplies the O(n log n) movement directly.
template <int W>
struct Row {
  uint64_t key;
  uint8_t pad[W - 8];
};

template <int W>
inline void sort_rows_inplace(Row<W>* rows, size_t n) {
  std::sort(rows, rows + n, [](const Row<W>& a, const Row<W>& b) { return a.key < b.key; });
}
