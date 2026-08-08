# sortlab

**Geometric sorting vs per-unit comparison, measured.** Self-contained C++17, no dependencies,
one `make` away from reproducing every number below on your own machine.

---

## The idea, before the code

The usual mental model of sorting is *measurement*: take two items, ask which is bigger, act on
the answer, repeat. Every general-purpose sort in every standard library is built on that one
primitive — `std::sort`, `sort.Slice`, `Arrays.sort`, `list.sort()`. They differ in how cleverly
they choose the pairs. They do not differ in asking.

Nobody sorts a pile of rocks that way. Nobody picks up two rocks, holds them side by side, decides,
puts one down, picks up a third. You get a **sieve** — a mesh pan — and shake. Everything smaller
than the mesh falls through; everything bigger stays. Stack the sieves coarse to fine and the pile
comes out graded, and *no rock was ever compared to another rock*. The mesh size did the comparing,
once, for all of them, before the first rock was picked up.

That is a radix sort. The mesh is a byte of the key: 256 holes, laid down before any data is read.
Each row looks at its own digit and falls into its own hole. Nothing is compared. The ordering of
the holes *is* the answer.

Two more properties come free from the physical picture and matter more than they look:

- **The rocks never travel through the stack.** You carry a *tag* with each rock's number through
  the sieves and fetch the rocks once, at the end. In code that is a permutation of indices, and on
  128-byte rows it is worth more than the sort itself (see [payload](#5-payload)).
- **The stack can be built in either direction.** Coarse-to-fine descends only into the pan still
  in contention — that is MSD radix, and it is what makes `LIMIT 10` cost far less than a sort.
  Fine-to-coarse does one full pass per sieve and relies on each pass preserving the previous
  pass's order — that is LSD radix, stable by construction, and it is the main contender here.

The interesting claim is not "radix sort is fast." It is that **the cost stops being a function of
how many rows there are relative to each other, and becomes a function of how many bytes of
coordinate are occupied.** That is why the numbers below move when the *entropy* changes and barely
move when *n* changes. It is a different cost model, not a constant-factor win.

---

## The machine these numbers came from

Every number in this README is a property of this machine as much as of the algorithm. Reproduce
before you cite.

| | |
|---|---|
| Chip | Apple M4 (`Mac16,1`, MacBook Pro), 10 cores — 4 performance + 6 efficiency |
| P-core cache | L1d 128 KB, L1i 192 KB, **L2 16 MB shared across the 4 P-cores** |
| E-core cache | L1d 64 KB, L1i 128 KB, L2 4 MB shared across the 6 E-cores |
| Cache line | 128 B (twice the x86 norm — relevant to every scatter number here) |
| Page size | 16 KB |
| Memory | 16 GB unified LPDDR5X |
| Vector unit | NEON, 128-bit — **2 lanes of uint64**, no gather, no scatter, no mask registers |
| OS | macOS 15.6.1, Darwin 24.6.0 arm64 |
| Compiler | Apple clang 17.0.0, `-O3 -mcpu=native -std=c++17` |
| Timer | `steady_clock`, ~41 ns granularity (hence the batched timing, below) |

Three of those facts do real work in the results:

1. **L2 is 16 MB.** A 1M-row uint64 key lane is 8 MB and lives in L2; the 10M-row case is 80 MB and
   does not. The single largest jump in the whole suite happens at exactly that boundary, and only
   for the contender that reads through an index.
2. **The vector is 2 lanes wide for uint64.** Every SIMD claim here is a claim about a 2× machine,
   not a 8× one. On AVX2 it is 4 lanes, on AVX-512 8 — the shape of the result holds, the magnitude
   scales.
3. **NEON has no gather.** A permutation apply is a scalar loop no matter what. That is not a gap in
   this implementation; it is the ISA. It is also why "SIMD sort" claims deserve suspicion.

---

## Build and run

```sh
make            # builds bin/bench and bin/verify
make verify     # correctness oracle — must pass before any number is meaningful
make bench      # every sweep, into results/all.txt + results/all.csv
make scale      # or one sweep at a time: scale entropy lanes payload presort smalln topk simd
./run.sh        # verify + all sweeps + records the machine into results/machine.txt
```

Portability: the vector layer has three back ends selected at compile time — **NEON** (aarch64),
**AVX2** (amd64), and a **scalar fallback** that compiles anywhere. On amd64 the Makefile passes
`-march=native -mavx2`; AVX2 compares `epi64` as signed, so the unsigned compare is synthesised
with one bias XOR (see `src/simd.hpp`). No intrinsic is used that lacks all three paths.

---

## What is being compared

Every contender has the **same contract**: given `n` rows of keys, emit a permutation of row
indices. Payload movement is excluded from the sort and timed separately, so the two distinct wins
can be told apart.

### Comparator family — order is a *relation between two rows*

| contender | what it is |
|---|---|
| `std::sort_direct` | `std::sort` over the raw keys. No indirection, no permutation. The honest floor of the comparator world — the analogue of `sort.Ints`/`slices.Sort`, not `sort.Slice`. |
| `std::sort_perm` | `std::sort` over an index vector with a row comparator. This is what `ORDER BY` actually compiles to, and the analogue of `sort.Slice`. |
| `std::stable_perm` | same, `std::stable_sort`. The only comparator that matches the radix's stability guarantee. |
| `insertion_perm` | the classic small-n floor. |
| `std::partial_sort` | heap selection for `LIMIT k`. |

### Radix family — order is *counted*, never compared

All are stable LSD by default, DESC carried as a per-lane XOR mask (`^k == k ^ ~0`, so the
inversion commutes with digit extraction and is applied at read — no inverted copy of the key
matrix is ever materialised), and all skip byte passes proven uniform by the **vary mask**:

```
vary = (⋁ all keys) ⊕ (⋀ all keys)      // the bits that are not constant
```

A byte of a lane is uniform exactly when its vary byte is zero, and a uniform byte orders nothing —
so it is dropped **before** it runs, from one sequential sweep, rather than discovered by paying for
a full gather and then throwing the result away.

| contender | what it is |
|---|---|
| `radix_gather` | the key matrix stays put; every pass reads it through the current order. Fewest bytes moved, **broken stride** — the address of the next read is not known until the previous index lands. |
| `radix_carry` | the lane is materialised into order-sequence **once per axis**, then every byte pass of that axis reads it at stride 1. More bytes moved, **unbroken stride**. |
| `radix_prehist` | `carry`, plus: a histogram counts a multiset, and a permutation does not change a multiset — so every byte histogram of an axis is knowable from the single sweep that gathers it. The counting half of each later pass disappears. |
| `radix_carry_alloc` | `radix_carry` allocating its own buffers per call instead of reusing a pool. Reported separately because below a few hundred rows the allocation *is* the measurement. |
| `radix_msd_topk` | coarse-to-fine descent for `LIMIT k`: the first digit level buckets all n once, every level below touches only rows still contending. |

### Vector family

| contender | what it is |
|---|---|
| `bitonic_scalar` / `bitonic_neon` | the same sorting network, once with scalar compare-exchange and once in vector registers. A network has **no data-dependent branch at all** — the compare-exchanges are fixed by n. |
| `vary_novec` / `vary_autovec` / `vary_neon` | the OR/AND reduce, with auto-vectorisation disabled, with it enabled, and hand-written with 4 independent accumulators. Three baselines, not two — see [§8](#8-what-simd-actually-does). |
| `encode_novec` / `encode_neon` | signed→order-preserving-unsigned fold, the width-independent lane arithmetic the vector unit owns outright. |

---

## Method, and what it controls for

- **Minimum of many repetitions**, not the mean: the least-disturbed run is the one that measures
  the machine rather than the scheduler.
- **Batched timing.** The clock ticks at ~41 ns, which is longer than an 8-element sort takes. Small
  cases run a batch of independent operations inside one timed region and divide; mutating
  contenders get `batch` independent copies, restored untimed between batches.
- **Results are consumed.** `volatile` sinks on every reduce and gather. Two early rows in this
  suite read `0.000 ns` before those sinks existed — the optimiser had deleted the work. A benchmark
  reporting zero is reporting the compiler.
- **Allocation is charged.** Global `operator new`/`delete` are overridden, so `B/op` and `allocs/op`
  include the temporaries `std::stable_sort` allocates internally.
- **Foreground, not background.** macOS demotes background-QoS processes to E-cores; multi-lane
  cases moved by up to 2× between a backgrounded and foregrounded run. Every number below was taken
  in the foreground. This laptop was not otherwise quiesced — treat ±10% as noise, and everything
  below is a factor of 2 or more.

Not controlled: thermal state, other userland activity, and hardware counters. macOS does not expose
PMU counters without a signed tool, so the branch-miss argument here rests on the wall-clock shape
alone. `run.sh` emits `perf stat` output automatically when run on Linux.

**Correctness first.** `make verify` runs every contender against a `std::stable_sort` reference
across 1,344 configurations — n ∈ {1, 2, 31, 32, 33, 257, 5000} × {1,2,4} axes × both layouts ×
{64,24,8}-bit and all-equal keys × {random, sorted, reverse, 99%-sorted} × ascending and mixed DESC —
checking three properties separately, because they fail separately:

1. the output is a permutation of `0..n-1`,
2. the emitted sequence is non-decreasing under the row comparator,
3. rows equal on every lane come out in input order (asserted only where stability is claimed).

Plus: the top-k window equals the first k of the full order element-for-element; the vector and
scalar vary masks agree bit for bit; the networks sort tie-heavy input identically to `std::sort`;
and the signed encode is verified **order-preserving**, which is the only reason a radix may replace
a comparator on a signed axis at all.

---

# Results

All figures **ns per element**, lower is better. Full tables in `results/`.

## 1. Scale

Single 64-bit lane, random keys, permutation out.

| n | `std::sort_direct` | `std::sort_perm` | `radix_gather` | `radix_carry` | `radix_prehist` |
|---:|---:|---:|---:|---:|---:|
| 16 | **2.10** | 4.11 | 43.90 | 44.18 | 56.24 |
| 64 | **2.44** | 4.82 | 15.41 | 15.17 | 15.84 |
| 128 | **2.59** | 5.46 | 11.38 | 10.96 | 11.61 |
| 256 | **3.14** | 7.19 | 8.77 | 8.65 | 8.61 |
| 1 024 | 4.60 | 9.42 | 7.22 | **6.86** | 6.88 |
| 10 K | 17.33 | 16.39 | 8.44 | 7.79 | **7.72** |
| 100 K | 31.96 | 50.88 | 8.64 | 7.51 | **7.38** |
| 1 M | 40.87 | 67.35 | 11.33 | **9.52** | 10.12 |
| 10 M | 54.70 | 100.79 | 43.90 | **9.53** | 9.56 |

The comparator's per-element cost **grows by 24× from n=16 to n=10M**. The radix's *falls* by 6.4×
and then flattens: 7.5 ns at 100 K, 9.5 ns at 10 M — a 100× increase in n for a 27% increase in unit
cost. That flatness is the whole point. `log n` is not a small term; it is the term.

Against the like-for-like contender (`std::sort_perm`, which is what an `ORDER BY` actually is):
**7.1× at 1M, 10.6× at 10M.** Against the monomorphic floor (`std::sort_direct`, which does strictly
less work — no permutation at all): **4.3× at 1M, 5.7× at 10M.**

**The crossover is n ≈ 256.** Below it, the comparator wins outright and no amount of counting will
change that; a radix has a fixed cost — the vary sweep, 256-entry histograms, prefix sums — that 16
rows cannot amortise. At n=16 the radix is **21× slower**. Report the floor or the comparison is
propaganda.

**And the stride axiom, in one row.** At 10M, `radix_gather` collapses from 8.6 → 43.9 ns/element,
a 5.1× cliff, while `radix_carry` — which moves *strictly more bytes* — does not move at all. The
80 MB key lane no longer fits the 16 MB L2, and `gather`'s read address is `keys[order[i]]`: not
computable until the previous index arrives, so the prefetcher cannot run ahead and every touch is a
full miss. `carry` pays extra writes to keep every read at stride 1 and the front never stalls.
*Fewer bytes touched is not the goal; an unbroken stride is.* This is the single most important
number in the suite, and it is invisible at 1M.

## 2. Entropy

n = 1M, single lane. `passes` is the number of byte-scatters actually executed.

| key | passes | `std::sort_perm` | `radix_carry` | speed-up |
|---|---:|---:|---:|---:|
| 64-bit random | 8 | 66.94 | 9.25 | 7.2× |
| 32-bit | 4 | 67.10 | 4.86 | **13.8×** |
| 24-bit (node-id realistic) | 3 | 67.65 | 3.79 | **17.8×** |
| 8-bit (few distinct) | 1 | 31.38 | 1.60 | 19.6× |
| all equal | 0 | 1.16 | 0.12 | 9.8× |

**The comparator's cost is flat across four orders of magnitude of key entropy** — 66.9 → 67.6 ns —
because it does `n log n` comparisons regardless of what is in the keys. The radix's cost is *linear
in occupied bytes*: 8 → 4 → 3 → 1 → 0 passes, 9.25 → 4.86 → 3.79 → 1.60 → 0.12 ns.

This is the non-linearity that matters in practice, because **real keys are not 64 bits of entropy.**
Node ids, dictionary codes, dates, counts, enum tags — all live in 1–3 bytes. The 24-bit row is the
realistic one, and it is **17.8×**.

The all-equal row is the vary mask paying for itself: zero passes, because one sequential sweep
proved no byte orders anything. The comparator still walks the whole set to discover the same fact.

## 3. Lanes and layout

`ORDER BY a, b, …` at n = 1M. Leading lanes are deliberately low-cardinality (a coarse primary, a
fine tiebreak — what real queries look like).

| axes | layout | passes | `std::sort_perm` | `radix_gather` | `radix_carry` |
|---:|---|---:|---:|---:|---:|
| 1 | row-major | 8 | 77.86 | 10.77 | **9.60** |
| 1 | lane-major | 8 | 66.75 | 10.82 | **9.30** |
| 2 | row-major | 9 | 91.47 | 15.20 | **12.23** |
| 2 | lane-major | 9 | 87.32 | 12.25 | **11.36** |
| 4 | row-major | 11 | 108.97 | 55.63 | **24.33** |
| 4 | lane-major | 11 | 130.82 | 22.17 | **17.54** |
| 8 | row-major | 15 | 119.42 | 88.24 | **50.11** |
| 8 | lane-major | 15 | 125.90 | 30.53 | **27.04** |

The radix advantage narrows with lane count — 8.1× at one axis, 4.7× at eight — and that is the
honest shape: each extra lane is extra passes for the radix, while the comparator often *short-
circuits* on lane 0 and never reads the rest.

**Layout is the story here.** At 8 axes, `radix_gather` costs 88.2 ns row-major and 30.5 ns
lane-major — **2.9× for changing nothing but where the bytes sit.** Row-major (`keys[r*nAxes+a]`)
makes a byte pass stride `nAxes*8 = 64 B`; lane-major (`keys[a*n+r]`) makes it stride 8 B, so a
128-byte cache line delivers 16 useful keys instead of 2. `radix_carry` is far less sensitive (50.1
vs 27.0) precisely because it only touches the matrix once per axis. The comparator is nearly
indifferent — it is branch-bound, not bandwidth-bound, and cannot exploit either layout.

## 4. Presortedness

n = 1M, single lane. The comparator's best case, included because omitting it would be dishonest.

| input | `std::sort_perm` | `std::stable_perm` | `radix_carry` |
|---|---:|---:|---:|
| random | 66.23 | 76.42 | **9.15** |
| already sorted | **1.23** | 7.99 | 9.08 |
| reverse sorted | **1.84** | 34.49 | 9.48 |
| 99% sorted | 9.38 | 15.48 | **9.36** |

**pdqsort wins outright on sorted input — 7.4×.** It detects the run and goes near-linear. The radix
does not care and cannot care: it counts the same digits either way, 9.1 ns whatever you hand it.

The crossover is at roughly **99% sorted**, where the two are a dead heat. Below that the radix wins;
above it the comparator does. Note also that `std::stable_sort` — the only comparator with the same
guarantee the radix provides for free — is 6.5× worse than `std::sort` on already-sorted input and
never competitive anywhere else.

## 5. Payload

n = 1M. `std::sort_rows` sorts records in place, so every swap carries the whole row. The radix path
emits a permutation and gathers once.

| row width | `gather_only` | `radix_then_gather` | `std::sort_rows` | speed-up |
|---:|---:|---:|---:|---:|
| 8 B | 0.53 | **10.09** | 40.12 | 4.0× |
| 32 B | 2.91 | **12.62** | 43.84 | 3.5× |
| 128 B | 19.74 | **29.15** | 59.77 | 2.1× |

Widening the row 16× costs the permutation path **19.1 ns** (all of it in the single gather) and the
comparator **19.7 ns** — nearly identical *absolute* growth, which is the surprise. `std::sort`'s
`log n ≈ 20` swap depth does not multiply the payload cost 20× because pdqsort's swaps are mostly
sequential and the hardware streams them. The permutation's one gather is *random* by construction —
it is the definition of a permutation — so it pays 128 B of cache line for 128 B of useful data at
best, and a full line for a partial row at worst.

The correct reading: **"payload never moves" is worth a solid constant factor, not an order of
magnitude,** and it shrinks as rows widen. The ordering win is the durable one. Claims that a
permutation sort wins *because* it avoids payload movement are overstating the smaller of the two
effects.

## 6. Small n — below the floor

| n | `std::sort_direct` | `std::sort_perm` | `insertion_perm` | `bitonic_scalar` | `bitonic_neon` | `radix_carry` |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | **0.78** | 2.00 | 1.24 | 3.09 | 1.41 | 84.85 |
| 16 | **1.63** | 3.84 | 2.68 | 5.78 | 2.01 | 44.44 |
| 32 | **1.74** | 4.27 | 4.81 | 8.84 | 2.63 | 25.18 |
| 64 | **2.10** | 4.38 | 9.55 | 12.23 | 3.54 | 15.57 |
| 128 | **2.51** | 5.40 | 20.78 | 16.89 | 4.36 | 11.23 |
| 256 | **3.30** | 6.85 | 36.62 | 21.67 | 5.37 | 8.34 |
| 512 | **3.94** | 8.10 | 74.36 | 26.54 | 6.79 | **7.53** |

The sorting network vectorises cleanly — **NEON beats the identical scalar network by 2.2× to 4.0×**,
rising with n as more of the network's stages land at stride ≥ 2 (stages at stride 1 need an
intra-register shuffle whose direction alternates inside the vector, and run scalar; that tail is in
the measurement, not hidden from it). On a 4-lane AVX2 or 8-lane AVX-512 machine the gap widens.

But `std::sort` on raw keys beats everything below 512 and it is not close. `pdqsort` bottoms out in
hand-tuned insertion networks of its own, with no permutation indirection to pay for. **The network's
real use is as the base case *inside* a larger sort**, not as a standalone contender — and the
`insertion_perm` column shows why something must fill that role: it degrades to 74 ns/element by
n=512, 60× worse than where it started.

`radix_carry` at n=8 is **108× slower** than `std::sort`. Two hundred rows of fixed cost cannot be
amortised over eight elements. Any honest dispatch takes the comparator below ~256.

## 7. Top-k — `LIMIT` is not a sort

n = 10M, single lane. Full-sort baselines: `radix_carry` 9.20 ns/element, `std::sort_perm` 94.69.

| k | `std::partial_sort` | `radix_msd_topk` |
|---:|---:|---:|
| 1 | **1.00** | 1.16 |
| 10 | **1.01** | 1.15 |
| 100 | **1.01** | 1.16 |
| 1 000 | **1.06** | 1.14 |
| 10 000 | 1.80 | **1.22** |

**Both beat a full sort by 8× (against a full radix) to 95× (against a full comparator sort)**, and the headline is that number, not the gap between them.
`LIMIT 10` over 10M rows costs 1.0–1.2 ns/element against 94.7 for sorting the whole set — asking
for ten rows should never cost a total order, and neither contender makes you pay for one.

Between them the comparator wins at small k and loses at large k, which is exactly what the cost
models predict. Heap selection is `O(n log k)` with a single sequential pass — at k=10 that is one
memory sweep and effectively nothing else, which is unbeatable. MSD radix pays a full histogram +
scatter of all 10M rows at its first level regardless of k, then descends only into contending
buckets; that fixed first level is the 1.15 ns floor it cannot go below, and it is why the MSD line
is **flat in k while the comparator's rises**. They cross between k = 1 000 and k = 10 000, and by k = 10 000 the
radix is 1.5× ahead and pulling away.

## 8. What SIMD actually does

n = 1M.

| step | `novec` | `autovec` | `neon` (hand, 4 accumulators) |
|---|---:|---:|---:|
| vary reduce, stride 1 | 0.256 | 0.067 | **0.064** |
| vary reduce, stride 4 (row-major) | **0.378** | 0.437 | 0.405 |
| signed encode | 0.244 | — | **0.118** |

Three baselines, not two — and the middle one is the honest finding: **clang already vectorises the
plain loop, and the hand-written intrinsics beat it by 5%.** The 4× gap is between vectorised and
non-vectorised, and the compiler collects almost all of it unaided. Hand intrinsics buy control and
the guarantee, not the win. (The first version of `vary_neon` here was *3× slower* than the scalar
loop: one accumulator chain, latency-bound at 2 lanes. Four independent chains fixed it. That failure
mode is the normal outcome of writing intrinsics without measuring.)

The stride-4 row is the more useful one: **at stride 4 every approach converges to ~0.4 ns and
vectorisation buys nothing at all** — auto-vectorisation is 16% *worse* than the scalar loop. There
is no contiguous vector to load. Layout decides whether the vector unit can participate; the
intrinsics cannot conjure a load that the memory layout does not permit.

The honest summary of SIMD in a sort:

| step | vectorisable? |
|---|---|
| encode value → order-preserving coordinate | **yes**, 2× here, scales with lane width |
| vary/OR-AND reduce over a lane | **yes**, 4× — and the compiler does it for you |
| sorting network at small n | **yes**, 2.2–4.0× |
| histogram (256 counters, data-dependent increment) | **no** — scatter-increment has no vector form |
| scatter to bucket positions | **no** — no NEON scatter; AVX-512 has one and it is slow |
| permutation gather | **no** on NEON; AVX2/512 gather exists and rarely wins |

**The sort's inner loops are exactly the operations SIMD does not have.** Everything above 3× in this
document comes from the algebra — counting instead of comparing, skipping proven-uniform bytes,
keeping the stride unbroken — and not from the vector unit. Treat "SIMD sort" claims accordingly.

---

## The cost model, assembled

| | comparator | geometric (radix) |
|---|---|---|
| work | `O(n log n)` comparisons | `O(n · P)`, P = occupied byte lanes |
| what P depends on | — | **key entropy**, not n |
| per unit | closure/interface call + data-dependent branch | histogram increment + scatter store |
| branch behaviour | ~50% mispredict on random keys | **no data-dependent branch anywhere** |
| reads | binary-search-like walk | sequential per pass — *if* carried, not gathered |
| writes | `O(n log n)` payload swaps | n scatter-stores into 256 buckets per pass |
| payload | moved `log n` times | moved **once**, at the gather |
| extra memory | none | `n·8` carried key + `n·4` permutation + scratch |
| stability | not without `stable_sort`'s cost | **free**, by construction |
| presorted input | near-linear (its best case) | unchanged |
| already-equal keys | full `n log n` | **zero passes** |

The one-line version: **a comparator's cost is a function of the number of rows; a radix's cost is a
function of the number of occupied bytes in the key.** Everything measured above follows from that
sentence.

## Where the comparator wins, plainly

1. **n below ~256.** By up to 110×. Fixed setup cannot amortise.
2. **Already-sorted or near-sorted input.** By up to 7.4×. `pdqsort` detects runs; a radix cannot.
3. **Small-k `LIMIT`.** Heap selection at k ≤ 1000 is a single sequential pass.
4. **Many lanes with a decisive first lane.** A comparator short-circuits on lane 0; a radix pays
   for every lane it was told to order by.

A production dispatcher should route on exactly those four conditions. Every other case in this
document goes the other way, by 2× to 20×.

## Layout of the source

```
src/common.hpp   timing (batched, min-of-reps), splitmix64 rng, allocation accounting, tables/CSV
src/common.cpp   global operator new/delete overrides, output formatting
src/gen.hpp      the input space: entropy, presortedness, lane count, layout, payload widths
src/simd.hpp     NEON / AVX2 / scalar vector layer; vary reduce, bitonic network, encode
src/sorts.hpp    the contenders — comparator family, radix family, MSD top-k, gather
src/bench.cpp    the eight sweeps
src/verify.cpp   the correctness oracle
```

Roughly 1,500 lines total, no dependencies, one translation unit per binary.

## License and attribution

Copyright 2026 [TETRA CA PBC](https://tetra-ca.com). Licensed under the Apache License,
Version 2.0 — see [LICENSE](LICENSE) for the full terms.

Use it, fork it, lift the techniques into your own code. Two conditions worth stating
plainly, since they are the ones people miss:

- **Keep the `NOTICE`.** If you redistribute this, in source or binary form, Apache 2.0
  section 4(d) requires you to carry a readable copy of [NOTICE](NOTICE) with it. The
  per-file SPDX headers only cover source redistribution; NOTICE is what follows the code
  into a compiled product.
- **The name is not part of the grant.** "TETRA CA" and the TETRA CA name and logo are
  trademarks of TETRA CA PBC, and section 6 of the License explicitly does not license
  them. A fork is welcome to the code, but not to implying it is a TETRA CA product.

If you cite these numbers in a post or paper, [CITATION.cff](CITATION.cff) has the
structured metadata — GitHub turns it into a copyable BibTeX entry via the
*Cite this repository* button.
