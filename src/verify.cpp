// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

// verify.cpp — the oracle.
//
// Two properties, checked separately, because they fail separately:
//   ordered  — the emitted sequence is non-decreasing under the row comparator
//   stable   — rows equal on every lane come out in input order
// A radix is stable by construction; std::sort is not, which is why the ordered check
// is the one both must pass and the stable check is asserted only where it is claimed.
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "common.hpp"
#include "gen.hpp"
#include "sorts.hpp"

static int g_failures = 0;

static void fail(const std::string& what, const std::string& why) {
  std::printf("FAIL  %-28s %s\n", what.c_str(), why.c_str());
  g_failures++;
}

static bool is_permutation(const std::vector<uint32_t>& p, uint32_t n) {
  std::vector<uint8_t> seen(n, 0);
  if (p.size() != n) return false;
  for (uint32_t v : p) {
    if (v >= n || seen[v]) return false;
    seen[v] = 1;
  }
  return true;
}

static bool is_ordered(const Keys& K, const uint8_t* desc, const std::vector<uint32_t>& p) {
  RowLess less{&K, desc};
  for (size_t i = 1; i < p.size(); i++) {
    if (less(p[i], p[i - 1])) return false;
  }
  return true;
}

static std::vector<uint32_t> reference(const Keys& K, const uint8_t* desc) {
  std::vector<uint32_t> r(K.n);
  perm_std_stable(K, desc, r.data());
  return r;
}

static void check(const std::string& what, const Keys& K, const uint8_t* desc,
                  const std::vector<uint32_t>& got, const std::vector<uint32_t>& ref,
                  bool claims_stable) {
  if (!is_permutation(got, K.n)) return fail(what, "not a permutation of 0..n-1");
  if (!is_ordered(K, desc, got)) return fail(what, "sequence is not ordered");
  if (claims_stable && got != ref) return fail(what, "stable order differs from reference");
}

static void verify_case(uint32_t n, uint32_t axes, Layout L, Entropy e, Order o, bool desc_mix) {
  Keys K = gen_keys(n, axes, L, e, o, n * 31 + axes);
  std::vector<uint8_t> desc(axes, 0);
  if (desc_mix) {
    for (uint32_t a = 0; a < axes; a++) desc[a] = (uint8_t)(a % 2);
  }
  std::string tag = std::to_string(n) + "/" + std::to_string(axes) + "ax/" + name(L) + "/" +
                    name(e) + "/" + name(o) + (desc_mix ? "/mixdesc" : "/asc");
  std::vector<uint32_t> ref = reference(K, desc.data());

  std::vector<uint32_t> p(n);
  perm_std_sort(K, desc.data(), p.data());
  check("std::sort_perm " + tag, K, desc.data(), p, ref, false);

  perm_radix_gather(K, desc.data(), p.data(), nullptr, true);
  check("radix_gather " + tag, K, desc.data(), p, ref, true);

  perm_radix_carry(K, desc.data(), p.data(), nullptr, true);
  check("radix_carry " + tag, K, desc.data(), p, ref, true);

  perm_radix_prehist(K, desc.data(), p.data(), nullptr, true);
  check("radix_prehist " + tag, K, desc.data(), p, ref, true);

  perm_insertion(K, desc.data(), p.data());
  check("insertion_perm " + tag, K, desc.data(), p, ref, true);

  // top-k: the window must equal the first k of the full order, element for element.
  for (uint32_t k : {1u, 7u, 64u}) {
    if (k > n) continue;
    std::vector<uint32_t> got(k);
    perm_radix_topk(K, desc.data(), got.data(), k, nullptr, true);
    std::vector<uint32_t> want(ref.begin(), ref.begin() + k);
    RowLess less{&K, desc.data()};
    for (uint32_t i = 0; i < k; i++) {
      if (less(got[i], want[i]) || less(want[i], got[i])) {
        fail("radix_msd_topk " + tag, "k=" + std::to_string(k) + " window differs at " +
                                          std::to_string(i));
        break;
      }
    }
  }

  // the scalar vary mask and the vector one must agree bit for bit
  for (uint32_t a = 0; a < axes; a++) {
    Lane ln = K.lane(a);
    uint64_t s = vary_scalar(ln.base, n, ln.stride).vary();
    uint64_t v = vary_simd(ln.base, n, ln.stride).vary();
    if (s != v) fail("vary_simd " + tag, "disagrees with vary_scalar on lane " + std::to_string(a));
  }
}

static void verify_networks() {
  for (size_t n = 2; n <= 512; n <<= 1) {
    Rng rng(n * 7 + 1);
    std::vector<uint64_t> base(n);
    for (size_t i = 0; i < n; i++) base[i] = rng.next() & 0xFFFF;  // force ties
    std::vector<uint64_t> want = base, a = base, b = base;
    std::sort(want.begin(), want.end());
    bitonic_sort_scalar(a.data(), n);
    bitonic_sort_simd(b.data(), n);
    if (a != want) fail("bitonic_scalar n=" + std::to_string(n), "not sorted");
    if (b != want) fail("bitonic_simd n=" + std::to_string(n), "not sorted");
  }
}

static void verify_encode() {
  size_t n = 1000;
  Rng rng(5);
  std::vector<int64_t> src(n);
  for (size_t i = 0; i < n; i++) src[i] = (int64_t)rng.next();
  std::vector<uint64_t> a(n), b(n);
  encode_i64_scalar(src.data(), a.data(), n);
  encode_i64_simd(src.data(), b.data(), n);
  if (a != b) fail("encode_simd", "disagrees with encode_scalar");

  // the encode must be order-preserving: that is the only reason a radix may replace
  // a comparator on a signed axis at all.
  std::vector<int64_t> s = src;
  std::sort(s.begin(), s.end());
  std::vector<uint64_t> c(n);
  encode_i64_scalar(s.data(), c.data(), n);
  for (size_t i = 1; i < n; i++) {
    if (c[i - 1] > c[i]) fail("encode_order", "code order does not match value order");
  }
}

int main() {
  const uint32_t sizes[] = {1, 2, 31, 32, 33, 257, 5000};
  const Layout layouts[] = {Layout::RowMajor, Layout::LaneMajor};
  const Entropy entropies[] = {Entropy::Bits64, Entropy::Bits24, Entropy::Bits8, Entropy::Equal};
  const Order orders[] = {Order::Random, Order::Sorted, Order::Reverse, Order::Nearly};

  for (uint32_t n : sizes) {
    for (uint32_t axes : {1u, 2u, 4u}) {
      for (Layout L : layouts) {
        for (Entropy e : entropies) {
          for (Order o : orders) {
            verify_case(n, axes, L, e, o, false);
            verify_case(n, axes, L, e, o, true);
          }
        }
      }
    }
  }
  verify_networks();
  verify_encode();

  if (g_failures == 0) {
    std::printf("ok — every contender agrees with the reference order\n");
    return 0;
  }
  std::printf("%d failures\n", g_failures);
  return 1;
}
