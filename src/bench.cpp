// Copyright 2026 TETRA CA PBC (https://tetra-ca.com)
// SPDX-License-Identifier: Apache-2.0

// bench.cpp — the sweeps.
//
// Each sweep varies exactly one thing. The cost model is not linear in any of them,
// and the point of separating them is to see which term explodes:
//
//   scale     n                        comparator carries a log n the radix has not
//   entropy   bits of key in use       radix work is bytes of key, not size of set
//   lanes     ORDER BY a, b, c, d      comparator branches per lane, radix adds passes
//   layout    row-major vs lane-major  the stride the byte pass reads at
//   payload   bytes per row            moved log n times, or moved once
//   presort   sortedness               pdqsort's best case, the radix's indifference
//   smalln    below the floor          network vs comparator vs counting
//   topk      LIMIT k                  O(n log n) vs O(n + k·depth)
//   simd      the two vectorisable steps
#include <cstdio>
#include <string>
#include <vector>

#include "common.hpp"
#include "gen.hpp"
#include "sorts.hpp"

using Perm = std::vector<uint32_t>;

// Results have to be observed or the optimiser deletes the work that produced them,
// and a benchmark that reports zero is reporting the compiler, not the machine.
static volatile uint64_t g_sink = 0;

static std::vector<uint8_t> ascending(uint32_t nAxes) { return std::vector<uint8_t>(nAxes, 0); }

static std::string cfg(const std::string& a, const std::string& b = "") {
  return b.empty() ? a : a + "/" + b;
}

static std::string with_isa(const char* stem) { return std::string(stem) + simd_isa(); }

// A permutation contender does not mutate its input, so a batch is just the same call
// repeated; only the mutating contenders below need independent copies.
template <class Fn>
static void run_perm(const char* bench, const std::string& cont, const std::string& config,
                     const Keys& K, Fn fn, double passes = 0) {
  Perm out(K.n);
  Result r = timeit(bench, cont.c_str(), config, K.n, batch_for(K.n), [] {},
                    [&](uint32_t) { fn(out.data()); });
  r.axes = K.nAxes;
  r.passes = passes;
  emit(r);
}

// A mutating contender gets `batch` independent copies, restored untimed between batches.
template <class Fn>
static void run_inplace(const char* bench, const std::string& cont, const std::string& config,
                        const std::vector<uint64_t>& keys, uint64_t n, Fn fn) {
  uint32_t batch = batch_for(n);
  std::vector<uint64_t> work((size_t)n * batch);
  auto prep = [&] {
    for (uint32_t b = 0; b < batch; b++) std::memcpy(&work[(size_t)b * n], keys.data(), n * 8);
  };
  Result r = timeit(bench, cont.c_str(), config, n, batch, prep,
                    [&](uint32_t b) { fn(&work[(size_t)b * n], n); });
  emit(r);
}

static double probe_passes(const Keys& K, const uint8_t* desc) {
  RadixStats st;
  Perm out(K.n);
  perm_radix_carry(K, desc, out.data(), &st, true);
  return st.passes;
}

// ------------------------------------------------------------------- scale
static void sweep_scale() {
  table_header("scale — n, single 64-bit lane, random keys, permutation out");
  const uint64_t sizes[] = {16, 64, 128, 256, 1024, 10000, 100000, 1000000, 10000000};
  for (uint64_t n : sizes) {
    Keys K = gen_keys((uint32_t)n, 1, Layout::LaneMajor, Entropy::Bits64, Order::Random, n);
    std::vector<uint8_t> d = ascending(1);
    std::string c = cfg("e64", "random");
    double p = probe_passes(K, d.data());
    RadixScratch pool;

    run_inplace("scale", "std::sort_direct", c, K.k, n, sort_keys_direct);
    run_perm("scale", "std::sort_perm", c, K, [&](uint32_t* o) { perm_std_sort(K, d.data(), o); });
    run_perm("scale", "std::stable_perm", c, K,
             [&](uint32_t* o) { perm_std_stable(K, d.data(), o); });
    if (n <= 1024)
      run_perm("scale", "insertion_perm", c, K,
               [&](uint32_t* o) { perm_insertion(K, d.data(), o); });
    run_perm("scale", "radix_gather", c, K,
             [&](uint32_t* o) { perm_radix_gather(K, d.data(), o, nullptr, true, pool); }, p);
    run_perm("scale", "radix_carry", c, K,
             [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true, pool); }, p);
    run_perm("scale", "radix_prehist", c, K,
             [&](uint32_t* o) { perm_radix_prehist(K, d.data(), o, nullptr, true, pool); }, p);
    run_perm("scale", "radix_carry_alloc", c, K,
             [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true); }, p);
  }
}

// ----------------------------------------------------------------- entropy
static void sweep_entropy(uint32_t n) {
  table_header("entropy — bits of key in use decide the pass count, n does not");
  const Entropy es[] = {Entropy::Bits64, Entropy::Bits32, Entropy::Bits24, Entropy::Bits8,
                        Entropy::Equal};
  for (Entropy e : es) {
    Keys K = gen_keys(n, 1, Layout::LaneMajor, e, Order::Random, 7);
    std::vector<uint8_t> d = ascending(1);
    std::string c = cfg(name(e), "random");
    double p = probe_passes(K, d.data());
    RadixScratch pool;
    run_perm("entropy", "std::sort_perm", c, K,
             [&](uint32_t* o) { perm_std_sort(K, d.data(), o); });
    run_perm("entropy", "radix_gather", c, K,
             [&](uint32_t* o) { perm_radix_gather(K, d.data(), o, nullptr, true, pool); }, p);
    run_perm("entropy", "radix_carry", c, K,
             [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true, pool); }, p);
    run_perm("entropy", "radix_prehist", c, K,
             [&](uint32_t* o) { perm_radix_prehist(K, d.data(), o, nullptr, true, pool); }, p);
  }
}

// ------------------------------------------------------- lanes and layout
static void sweep_lanes(uint32_t n) {
  table_header("lanes — ORDER BY over 1..8 keys, in both matrix layouts");
  const uint32_t axes[] = {1, 2, 4, 8};
  const Layout layouts[] = {Layout::RowMajor, Layout::LaneMajor};
  for (uint32_t a : axes) {
    for (Layout L : layouts) {
      Keys K = gen_keys(n, a, L, Entropy::Bits64, Order::Random, 11 + a);
      std::vector<uint8_t> d = ascending(a);
      std::string c = cfg(std::to_string(a) + "axis", name(L));
      double p = probe_passes(K, d.data());
      RadixScratch pool;
      run_perm("lanes", "std::sort_perm", c, K,
               [&](uint32_t* o) { perm_std_sort(K, d.data(), o); });
      run_perm("lanes", "radix_gather", c, K,
               [&](uint32_t* o) { perm_radix_gather(K, d.data(), o, nullptr, true, pool); }, p);
      run_perm("lanes", "radix_carry", c, K,
               [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true, pool); }, p);
      run_perm("lanes", "radix_prehist", c, K,
               [&](uint32_t* o) { perm_radix_prehist(K, d.data(), o, nullptr, true, pool); }, p);
    }
  }
}

// ----------------------------------------------------------------- payload
template <int W>
static void payload_case(uint32_t n) {
  Keys K = gen_keys(n, 1, Layout::LaneMajor, Entropy::Bits64, Order::Random, 23);
  std::vector<uint8_t> d = ascending(1);
  std::string c = std::to_string(W) + "B";
  uint32_t batch = batch_for(n);
  RadixScratch pool;

  std::vector<uint8_t> src = gen_payload(n, W, 5), dst((size_t)n * W);
  Perm perm(n);
  perm_radix_carry(K, d.data(), perm.data(), nullptr, true, pool);

  Result g = timeit("payload", "gather_only", c, n, batch, [] {}, [&](uint32_t) {
    gather_payload(dst.data(), src.data(), perm.data(), n, W);
    g_sink += dst[(size_t)n * W - 1];
  });
  g.payload = W;
  emit(g);

  Result t = timeit("payload", "radix_then_gather", c, n, batch, [] {}, [&](uint32_t) {
    perm_radix_carry(K, d.data(), perm.data(), nullptr, true, pool);
    gather_payload(dst.data(), src.data(), perm.data(), n, W);
    g_sink += dst[(size_t)n * W - 1];
  });
  t.payload = W;
  emit(t);

  std::vector<Row<W>> rows(n), work;
  for (uint32_t i = 0; i < n; i++) rows[i].key = K.at(i, 0);
  Result s = timeit("payload", "std::sort_rows", c, n, 1, [&] { work = rows; },
                    [&](uint32_t) { sort_rows_inplace<W>(work.data(), n); });
  s.payload = W;
  emit(s);
}

static void sweep_payload(uint32_t n) {
  table_header("payload — moved log n times by a comparator, or moved once by a gather");
  payload_case<8>(n);
  payload_case<32>(n);
  payload_case<128>(n);
}

// ---------------------------------------------------------------- presort
static void sweep_presort(uint32_t n) {
  table_header("presortedness — pdqsort detects runs; a radix does not care");
  const Order os[] = {Order::Random, Order::Sorted, Order::Reverse, Order::Nearly};
  for (Order o : os) {
    Keys K = gen_keys(n, 1, Layout::LaneMajor, Entropy::Bits64, o, 31);
    std::vector<uint8_t> d = ascending(1);
    std::string c = name(o);
    double p = probe_passes(K, d.data());
    RadixScratch pool;
    run_perm("presort", "std::sort_perm", c, K,
             [&](uint32_t* out) { perm_std_sort(K, d.data(), out); });
    run_perm("presort", "std::stable_perm", c, K,
             [&](uint32_t* out) { perm_std_stable(K, d.data(), out); });
    run_perm("presort", "radix_carry", c, K,
             [&](uint32_t* out) { perm_radix_carry(K, d.data(), out, nullptr, true, pool); }, p);
  }
}

// ----------------------------------------------------------------- small n
static void sweep_smalln() {
  table_header("small n — below the radix floor, where the network lives");
  for (uint64_t n = 8; n <= 512; n <<= 1) {
    Keys K = gen_keys((uint32_t)n, 1, Layout::LaneMajor, Entropy::Bits64, Order::Random, n + 3);
    std::vector<uint8_t> d = ascending(1);
    std::string c = cfg("e64", "random");
    RadixScratch pool;

    run_inplace("smalln", "std::sort_direct", c, K.k, n, sort_keys_direct);
    run_inplace("smalln", "bitonic_scalar", c, K.k, n, bitonic_sort_scalar);
    run_inplace("smalln", with_isa("bitonic_"), c, K.k, n, bitonic_sort_simd);
    run_perm("smalln", "insertion_perm", c, K,
             [&](uint32_t* o) { perm_insertion(K, d.data(), o); });
    run_perm("smalln", "std::sort_perm", c, K, [&](uint32_t* o) { perm_std_sort(K, d.data(), o); });
    run_perm("smalln", "radix_carry", c, K,
             [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true, pool); });
    run_perm("smalln", "radix_carry_alloc", c, K,
             [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true); });
  }
}

// -------------------------------------------------------------------- topk
static void sweep_topk(uint32_t n) {
  table_header("top-k — LIMIT is not a sort");
  Keys K = gen_keys(n, 1, Layout::LaneMajor, Entropy::Bits64, Order::Random, 41);
  std::vector<uint8_t> d = ascending(1);
  RadixScratch pool;
  double p = probe_passes(K, d.data());
  run_perm("topk", "radix_carry_full", "k=all", K,
           [&](uint32_t* o) { perm_radix_carry(K, d.data(), o, nullptr, true, pool); }, p);
  run_perm("topk", "std::sort_perm_full", "k=all", K,
           [&](uint32_t* o) { perm_std_sort(K, d.data(), o); });
  for (uint32_t k : {1u, 10u, 100u, 1000u, 10000u}) {
    std::string c = "k=" + std::to_string(k);
    run_perm("topk", "std::partial_sort", c, K,
             [&](uint32_t* o) { perm_partial_sort(K, d.data(), o, k); });
    run_perm("topk", "radix_msd_topk", c, K,
             [&](uint32_t* o) { perm_radix_topk(K, d.data(), o, k, nullptr, true, pool); });
  }
}

// -------------------------------------------------------------------- simd
static void sweep_simd(uint32_t n) {
  table_header("simd — the two steps the vector unit actually owns");
  Rng rng(97);
  std::vector<uint64_t> lane(n), wide((size_t)n * 4);
  for (uint32_t i = 0; i < n; i++) lane[i] = rng.next();
  for (size_t i = 0; i < wide.size(); i++) wide[i] = rng.next();
  std::vector<int64_t> src(n);
  std::vector<uint64_t> dst(n);
  for (uint32_t i = 0; i < n; i++) src[i] = (int64_t)rng.next();
  volatile uint64_t& sink = g_sink;

  // Three baselines, not two: the compiler auto-vectorises the plain loop, so a
  // "SIMD vs scalar" claim that omits the auto-vectorised middle is not a claim at all.
  enum Kind { NoVec, Auto, Hand };
  auto reduce = [&](Kind kind, const std::string& c, const uint64_t* p, size_t stride) {
    const char* cont = kind == NoVec ? "vary_novec" : (kind == Auto ? "vary_autovec" : "hand");
    std::string hand = with_isa("vary_");
    Result r = timeit("simd", kind == Hand ? hand.c_str() : cont, c, n, 1, [] {}, [&](uint32_t) {
      switch (kind) {
        case NoVec: sink = vary_novec(p, n, stride).vary(); break;
        case Auto: sink = vary_scalar(p, n, stride).vary(); break;
        default: sink = vary_simd(p, n, stride).vary(); break;
      }
    });
    emit(r);
  };
  reduce(NoVec, "stride1", lane.data(), 1);
  reduce(Auto, "stride1", lane.data(), 1);
  reduce(Hand, "stride1", lane.data(), 1);
  reduce(NoVec, "stride4", wide.data(), 4);
  reduce(Auto, "stride4", wide.data(), 4);
  reduce(Hand, "stride4", wide.data(), 4);

  // dst is consumed, or the whole encode folds away and the row reads zero.
  Result f = timeit("simd", "encode_novec", "i64", n, 1, [] {}, [&](uint32_t) {
    encode_i64_scalar(src.data(), dst.data(), n);
    sink += dst[n - 1];
  });
  emit(f);
  Result g = timeit("simd", with_isa("encode_").c_str(), "i64", n, 1, [] {}, [&](uint32_t) {
    encode_i64_simd(src.data(), dst.data(), n);
    sink += dst[n - 1];
  });
  emit(g);
}

// -------------------------------------------------------------------- main
int main(int argc, char** argv) {
  std::string which = argc > 1 ? argv[1] : "all";
  const uint32_t mid = 1000000;

  std::printf("sortlab — geometric (radix/SIMD) vs scalar (comparator) sort\n");
  std::printf("isa=%s vector_lanes=%d pointer=%zub\n", simd_isa(), kVecLanes, sizeof(void*) * 8);

  bool all = (which == "all");
  if (all || which == "scale") sweep_scale();
  if (all || which == "entropy") sweep_entropy(mid);
  if (all || which == "lanes") sweep_lanes(mid);
  if (all || which == "payload") sweep_payload(mid);
  if (all || which == "presort") sweep_presort(mid);
  if (all || which == "smalln") sweep_smalln();
  if (all || which == "topk") sweep_topk(10000000);
  if (all || which == "simd") sweep_simd(mid);

  write_csv(("results/" + which + ".csv").c_str());
  return 0;
}
