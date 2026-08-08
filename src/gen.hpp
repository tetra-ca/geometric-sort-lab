// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

// gen.hpp — the input space.
//
// A key is an order-preserving uint64 coordinate. A row is nAxes of them. Two
// layouts exist because a byte pass reads one lane, and the layout decides whether
// that read is a stride-1 wavefront or a strided one.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "common.hpp"

enum class Layout { RowMajor, LaneMajor };  // keys[r*nAxes+a] vs keys[a*n+r]
enum class Entropy { Bits64, Bits32, Bits24, Bits8, Equal };
enum class Order { Random, Sorted, Reverse, Nearly };  // Nearly = 99% sorted

inline const char* name(Entropy e) {
  switch (e) {
    case Entropy::Bits64: return "e64";
    case Entropy::Bits32: return "e32";
    case Entropy::Bits24: return "e24";
    case Entropy::Bits8:  return "e8";
    default:              return "eq";
  }
}

inline const char* name(Order o) {
  switch (o) {
    case Order::Random:  return "random";
    case Order::Sorted:  return "sorted";
    case Order::Reverse: return "reverse";
    default:             return "99pct";
  }
}

inline const char* name(Layout l) {
  return l == Layout::RowMajor ? "row-major" : "lane-major";
}

inline uint64_t entropy_mask(Entropy e) {
  switch (e) {
    case Entropy::Bits64: return ~0ull;
    case Entropy::Bits32: return 0xFFFFFFFFull;
    case Entropy::Bits24: return 0xFFFFFFull;
    case Entropy::Bits8:  return 0xFFull;
    default:              return 0;
  }
}

// A lane is a base pointer and a stride. RowMajor gives stride nAxes, LaneMajor
// stride 1 — one loop reads both, and the only difference the machine sees is how
// far the front jumps between consecutive rows.
struct Lane {
  const uint64_t* base;
  size_t stride;
  uint64_t operator[](size_t row) const { return base[row * stride]; }
};

class Keys {
 public:
  std::vector<uint64_t> k;
  uint32_t n = 0;
  uint32_t nAxes = 1;
  Layout layout = Layout::RowMajor;

  Lane lane(uint32_t a) const {
    if (layout == Layout::RowMajor) return Lane{k.data() + a, nAxes};
    return Lane{k.data() + (size_t)a * n, 1};
  }
  uint64_t at(uint32_t row, uint32_t a) const { return lane(a)[row]; }
  void put(uint32_t row, uint32_t a, uint64_t v) { k[offset(row, a)] = v; }

 private:
  size_t offset(uint32_t row, uint32_t a) const {
    if (layout == Layout::RowMajor) return (size_t)row * nAxes + a;
    return (size_t)a * n + row;
  }
};

// Row-lex comparison under per-lane desc flags: the reference order, and the
// comparator the scalar contenders pay for on every single compare.
struct RowLess {
  const Keys* K;
  const uint8_t* desc;
  bool operator()(uint32_t x, uint32_t y) const {
    for (uint32_t a = 0; a < K->nAxes; a++) {
      uint64_t xa = K->at(x, a), ya = K->at(y, a);
      if (xa != ya) return desc[a] ? xa > ya : xa < ya;
    }
    return false;
  }
};

// A leading axis is deliberately coarse: a real ORDER BY a, b has a low-cardinality
// primary and a fine tiebreak, and that ratio decides how much work later lanes do.
inline uint64_t lane_mask(Entropy e, uint32_t axis, uint32_t nAxes) {
  uint64_t m = entropy_mask(e);
  bool is_tiebreak = (axis + 1 == nAxes);
  return is_tiebreak ? m : (m & 0xFFull);
}

inline void fill_random(Keys& K, Entropy e, Rng& rng) {
  for (uint32_t a = 0; a < K.nAxes; a++) {
    uint64_t m = lane_mask(e, a, K.nAxes);
    for (uint32_t r = 0; r < K.n; r++) {
      K.put(r, a, e == Entropy::Equal ? 0x5A5Aull : rng.next() & m);
    }
  }
}

inline std::vector<uint32_t> ascending_permutation(const Keys& K) {
  std::vector<uint32_t> idx(K.n);
  std::iota(idx.begin(), idx.end(), 0u);
  std::vector<uint8_t> asc(K.nAxes, 0);
  std::stable_sort(idx.begin(), idx.end(), RowLess{&K, asc.data()});
  return idx;
}

inline Keys apply_permutation(const Keys& K, const std::vector<uint32_t>& idx) {
  Keys out = K;
  for (uint32_t r = 0; r < K.n; r++) {
    for (uint32_t a = 0; a < K.nAxes; a++) out.put(r, a, K.at(idx[r], a));
  }
  return out;
}

inline void swap_rows(Keys& K, uint32_t x, uint32_t y) {
  for (uint32_t a = 0; a < K.nAxes; a++) {
    uint64_t t = K.at(x, a);
    K.put(x, a, K.at(y, a));
    K.put(y, a, t);
  }
}

inline void displace(Keys& K, uint32_t count, Rng& rng) {
  for (uint32_t i = 0; i < count; i++) swap_rows(K, rng.below(K.n), rng.below(K.n));
}

inline Keys gen_keys(uint32_t n, uint32_t nAxes, Layout layout, Entropy e, Order o,
                     uint64_t seed) {
  Keys K;
  K.n = n;
  K.nAxes = nAxes;
  K.layout = layout;
  K.k.assign((size_t)n * nAxes, 0);

  Rng rng(seed);
  fill_random(K, e, rng);
  if (o == Order::Random) return K;

  std::vector<uint32_t> idx = ascending_permutation(K);
  if (o == Order::Reverse) std::reverse(idx.begin(), idx.end());
  Keys S = apply_permutation(K, idx);
  if (o == Order::Nearly) displace(S, std::max(1u, n / 100), rng);
  return S;
}

// Payload rows: what a permutation is eventually applied to. Width 4 is a bare node
// id, 32 a small tuple, 128 a materialised result row.
inline std::vector<uint8_t> gen_payload(uint32_t n, uint32_t width, uint64_t seed) {
  std::vector<uint8_t> p((size_t)n * width);
  Rng rng(seed);
  for (size_t i = 0; i < p.size(); i += 8) {
    uint64_t v = rng.next();
    std::memcpy(&p[i], &v, std::min<size_t>(8, p.size() - i));
  }
  return p;
}
