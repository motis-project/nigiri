#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/it_range.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/breadcrumb.h"
#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/journey_pod.h"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

// ============================================================================
// GPU McRAPTOR (multi-criteria RAPTOR, CPU reference: raptor/mcraptor.h)
//
// The CPU algorithm keeps one accumulating pareto bag per stop (labels
// tagged with their transfer round, so the round is an implicit pareto
// dimension) and an append-only breadcrumb arena for reconstruction. The
// GPU port keeps exactly this design but maps it onto fixed-capacity,
// lock-free structures:
//
//  * label = ONE 64-bit word  [arr_key:16 | extras:16 | round:4 |
//    by_route:1 | bc:27], so a pareto bag is a small array of u64 slots
//    updated with plain atomicCAS - no locks, no torn reads.
//    - arr_key is direction-biased (like device_times) so that a smaller
//      key always means a better arrival; dominance then reads the same
//      for both search directions: key_a <= key_b && extras_a <= extras_b
//      (strict (arr, extras) pareto, see arr_cost_criteria; with
//      extras == 0 everywhere it degenerates to the arrival-only
//      configuration).
//    - kEmptySlot = ~0 (round 0xF never occurs).
//  * bag insert implements the CPU bag_insert rules 1:1 (REJECTED only by
//    lower-or-equal-round labels - a by-route candidate only by by-route
//    labels; EVICTS only higher-or-equal-round labels - a by-transfer
//    candidate only by-transfer labels). Insert races can transiently
//    leave mutually dominated labels in a bag (both writers pass the
//    reject scan before either lands). That is deliberately tolerated:
//    dominated labels only spawn dominated work, and the final journey
//    extraction pareto-filters at journey level, so results stay exact.
//    What can never happen is the LOSS of a non-dominated label
//    (rejection requires an already-present dominator; eviction requires
//    the candidate to dominate the victim).
//  * the round-eviction rule doubles as the phase-safety property: round-k
//    inserts never evict round-(k-1) labels, so the route scan can read
//    the previous round's labels while concurrently inserting this
//    round's, and the footpath phase (which inserts by-transfer labels)
//    never disturbs the by-route labels it iterates.
//  * breadcrumbs: append-only arena of {payload48, parent, arr} entries
//    (atomicAdd bump). Entries are only read by the reconstruction kernel
//    (separate launch = full sync), never during the search - no fences.
//  * destination pruning: a small global pareto frontier over
//    (round, criteria) with the same insert machinery; capacity overflow
//    simply skips the insert (pruning gets weaker, results stay correct).
//    GPU has no lower bounds, so the CPU's lb-projection degenerates to
//    the plain criteria (lb = 0 is a valid lower bound).
//  * route scan: one thread per marked route walking the stop sequence
//    like the CPU loop, with the route bag (pareto over total trip order
//    x carried extras) in registers/local memory.
//
// Capacity overflows (stop bag, route bag, arena) set a device canary
// that the host checks after every query - a lossy search never goes
// unnoticed (validation would also catch it, but this fails fast).
// ============================================================================

#define kInvalid (kInvalidDelta<SearchDir>)
#define kFwd (SearchDir == direction::kForward)
#define kUnreachable (std::numeric_limits<std::uint16_t>::max())
#define kIntermodalTarget (get_special_station(special_station::kEnd))

// ---- packed label ----------------------------------------------------------
using mc_label_t = std::uint64_t;

inline constexpr auto kMcEmptySlot = ~mc_label_t{0};
inline constexpr auto kMcNoBc = std::uint32_t{(1U << 27U) - 1U};
// stop bag capacity: RUNTIME (env NIGIRI_GPU_MC_BAG_CAP, default 64,
// max 255 = hwm range). The strict (arr, extras) pareto keeps every
// walk-class trade-off, so frontier sizes vary strongly with the dataset;
// a fixed compile-time cap either wastes memory or trips the canary.
inline constexpr auto kMcBagCapDefault = 64U;
inline constexpr auto kMcDestCap = 64U;  // dest frontier entries
// cross-start reuse frontier (rRAPTOR range reuse, GPU form): CPU reuse
// only ever uses labels of previously processed (later-departing) starts
// to REJECT current candidates - they are never boarded or extracted. So
// instead of tagging the per-start labels with their departure (no room
// in the packed label), a separate persistent per-stop frontier stores
// (arr_key, extras, dep, round, by_route) of every accepted label and
// bag_insert consults it with the departure-discounted rule
// (arr <= && extras -/+ dep <= && round <= && flag rules) - semantically
// identical to the CPU's cross-departure reuse rejections, which is
// result-neutral under the strict dominance rule. Cleared per QUERY
// (reset_arrivals), not per start. Capacity overflow drops the entry
// (pruning only - never results).
inline constexpr auto kMcReuseCap = 8U;

CISTA_CUDA_COMPAT inline std::uint64_t mc_reuse_pack(
    std::uint32_t const arr_key,
    std::uint32_t const extras,
    delta_t const dep,
    std::uint32_t const round,
    bool const by_route) {
  return (static_cast<std::uint64_t>(arr_key & 0xFFFFU) << 48U) |
         (static_cast<std::uint64_t>(extras & 0xFFFFU) << 32U) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(dep))
          << 16U) |
         (static_cast<std::uint64_t>(round & 0xFU) << 4U) |
         (by_route ? 1U : 0U);
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_reuse_key(std::uint64_t const x) {
  return static_cast<std::uint32_t>(x >> 48U);
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_reuse_extras(
    std::uint64_t const x) {
  return static_cast<std::uint32_t>(x >> 32U) & 0xFFFFU;
}
CISTA_CUDA_COMPAT inline delta_t mc_reuse_dep(std::uint64_t const x) {
  return static_cast<delta_t>(static_cast<std::uint16_t>(x >> 16U));
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_reuse_round(std::uint64_t const x) {
  return static_cast<std::uint32_t>(x >> 4U) & 0xFU;
}
CISTA_CUDA_COMPAT inline bool mc_reuse_by_route(std::uint64_t const x) {
  return (x & 1U) != 0U;
}
inline constexpr auto kMcRouteBagCap = 32U;  // boardings per route scan

// two-pass route scan: pass 1 (redundant per lane, register-only) walks
// the stops replaying the route-bag merges from the precomputed et
// blocks and records each boarding's ACTIVE SEGMENT - the scan-position
// range [start_, end_) in which it produces arrivals, with the
// board/parent valid for that range ("board closest to exit" updates
// close the old segment and open a new one; evictions close). Pass 2
// fans the warp's 32 lanes over the stop positions and executes the
// memory-heavy alight work (event-time load, window/dest pruning, locked
// bag insert) for every segment active at the lane's position - the
// sequential dependency lives entirely in the cheap pass 1.
struct mc_seg {
  std::uint32_t et_;
  std::uint32_t parent_;
  std::uint16_t extras_;
  std::uint16_t board_;
  std::uint16_t start_;
  std::uint16_t end_;
};
static_assert(sizeof(mc_seg) == 16U);
inline constexpr auto kMcMaxSegs = 128U;  // per route; overflow -> seq path
inline constexpr auto kMcEtGatherCap = 32U;  // sorted lookup candidates
inline constexpr auto kMcScanThreads = 128U;  // 4 warps per block
// per route-stop earliest-transport result block (et phase): one entry per
// boardable round-(k-1) label of the stop, filled densely with a
// terminator. The collect phase only resets entry [0] per stop (the
// terminator position of an empty block); the lookup phase overwrites the
// prefix and re-terminates, so entries beyond the terminator may be stale
// garbage from earlier rounds/starts but are never read.
// [et:32 | extras:16 | slot:8 | valid:1]; block size is RUNTIME
// (env NIGIRI_GPU_MC_ET_PER_STOP, default 32, canary on overflow).
inline constexpr auto kMcEtPerStopDefault = 32U;
inline constexpr auto kMcEtInvalid = ~std::uint64_t{0};

CISTA_CUDA_COMPAT inline std::uint64_t mc_et_pack(std::uint32_t const et,
                                                  std::uint32_t const extras,
                                                  std::uint32_t const slot) {
  return (static_cast<std::uint64_t>(et) << 32U) |
         (static_cast<std::uint64_t>(extras & 0xFFFFU) << 16U) |
         (static_cast<std::uint64_t>(slot & 0xFFU) << 8U) | 1U;
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_et_key(std::uint64_t const x) {
  return static_cast<std::uint32_t>(x >> 32U);
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_et_extras(std::uint64_t const x) {
  return static_cast<std::uint32_t>(x >> 16U) & 0xFFFFU;
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_et_slot(std::uint64_t const x) {
  return static_cast<std::uint32_t>(x >> 8U) & 0xFFU;
}

CISTA_CUDA_COMPAT inline mc_label_t mc_pack(std::uint32_t const arr_key,
                                            std::uint32_t const extras,
                                            std::uint32_t const round,
                                            bool const by_route,
                                            std::uint32_t const bc) {
  return (static_cast<mc_label_t>(arr_key & 0xFFFFU) << 48U) |
         (static_cast<mc_label_t>(extras & 0xFFFFU) << 32U) |
         (static_cast<mc_label_t>(round & 0xFU) << 28U) |
         (static_cast<mc_label_t>(by_route ? 1U : 0U) << 27U) |
         static_cast<mc_label_t>(bc & kMcNoBc);
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_arr_key(mc_label_t const x) {
  return static_cast<std::uint32_t>(x >> 48U);
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_extras(mc_label_t const x) {
  return static_cast<std::uint32_t>(x >> 32U) & 0xFFFFU;
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_round(mc_label_t const x) {
  return static_cast<std::uint32_t>(x >> 28U) & 0xFU;
}
CISTA_CUDA_COMPAT inline bool mc_by_route(mc_label_t const x) {
  return ((x >> 27U) & 0x1U) != 0U;
}
CISTA_CUDA_COMPAT inline std::uint32_t mc_bc(mc_label_t const x) {
  return static_cast<std::uint32_t>(x) & kMcNoBc;
}

// breadcrumb arena entry: everything the reconstruction chase needs.
// arr_ is the arrival at the label's location (post transfer buffer /
// footpath), used for traffic-day recovery exactly like the CPU.
struct mc_bc_entry {
  breadcrumb_t payload_;
  std::uint32_t parent_;
  delta_t arr_;
  std::uint16_t pad_{0U};
};
static_assert(sizeof(mc_bc_entry) == 16U);

// device canary bits (checked by the host after each query)
inline constexpr auto kMcOverflowBag = 1U;
inline constexpr auto kMcOverflowArena = 2U;
inline constexpr auto kMcOverflowRouteBag = 4U;
inline constexpr auto kMcOverflowRec = 8U;
inline constexpr auto kMcOverflowEtBlock = 16U;

template <direction SearchDir, bool WithCost>
struct mcraptor_impl {
  // OTP-default generalized cost parameters (see arr_cost_criteria):
  // total walk reluctance = 1 (elapsed charge) + surcharge
  static constexpr auto const kBoardCost = WithCost ? 10U : 0U;

  __device__ __forceinline__ bool is_better(auto a, auto b) const {
    return kFwd ? a < b : a > b;
  }
  __device__ __forceinline__ bool is_better_or_eq(auto a, auto b) const {
    return kFwd ? a <= b : a >= b;
  }
  __device__ __forceinline__ auto dir(auto a) const {
    return (kFwd ? 1 : -1) * a;
  }

  // strict pareto over (arr, extras) - earliness is never traded against
  // extras (see arr_cost_criteria::dominates: the earlier label must be
  // assumed to wait for the same connection, and the waiting charge
  // cancels the arrival advantage). WithCost=false: extras are all 0 and
  // this degenerates to the plain arrival comparison.
  __device__ __forceinline__ bool dominates(std::uint32_t const a_key,
                                            std::uint32_t const a_extras,
                                            std::uint32_t const b_key,
                                            std::uint32_t const b_extras)
      const {
    return a_key <= b_key && a_extras <= b_extras;
  }

  // dominance for COMPLETED journeys (the destination frontier): the
  // elapsed part is realized at the destination, so (key, key + extras)
  // IS the journey-level cost dominance and stays valid against
  // lb-projections (see arr_cost_criteria::completed_dominates). Only
  // the stop bags need the strict rule.
  __device__ __forceinline__ bool completed_dominates(
      std::uint32_t const a_key,
      std::uint32_t const a_extras,
      std::uint32_t const b_key,
      std::uint32_t const b_extras) const {
    return a_key <= b_key && a_key + a_extras <= b_key + b_extras;
  }

  // biased arrival key (identical bias to device_times<SearchDir>)
  CISTA_CUDA_COMPAT static std::uint32_t to_key(delta_t const t) {
    return kFwd ? static_cast<std::uint16_t>(static_cast<int>(t) + 32768)
                : static_cast<std::uint16_t>(32767 - static_cast<int>(t));
  }
  CISTA_CUDA_COMPAT static delta_t from_key(std::uint32_t const k) {
    return kFwd ? static_cast<delta_t>(static_cast<int>(k) - 32768)
                : static_cast<delta_t>(32767 - static_cast<int>(k));
  }

  // ---- bag ops -------------------------------------------------------------

  __device__ __forceinline__ mc_label_t* bag(std::uint32_t const l) const {
    return bags_ + static_cast<std::size_t>(l) * bag_cap_;
  }

  // CPU bag_insert rules 1:1, serialized by a per-stop spinlock: without
  // the lock, concurrent inserts that mutually pass the reject scan both
  // land, and such transiently dominated junk sticks around occupying
  // slots (hub stops see hundreds of concurrent writers per round). Under
  // the lock the bag always holds exactly the true pareto frontier, so
  // the fixed capacity carries the CPU-measured frontier sizes. Evictions
  // punch holes (no compaction!) so concurrent lock-free READERS never
  // see a surviving label move slots; round-(k-1) labels are never
  // evicted by round-k inserts, which keeps the boarding/footpath
  // iterations stable (see header comment). 64-bit aligned stores do not
  // tear, so readers see each slot either old or new, never mixed.
  // with_bc = allocate the breadcrumb arena entry {payload, parent, arr}
  // INSIDE the critical section, only once the insert is accepted -
  // rejected candidates are the common case and must not consume arena
  // (the CPU pushes its breadcrumb only after a successful merge, too).
  // Arena entries persist after label eviction: child labels reference
  // their evicted parents' entries, exactly like the CPU's append-only
  // breadcrumbs_ vector.
  // departure-discounted extras term of the reuse rule: for a shared
  // continuation the final costs differ by (extras - dep) fwd /
  // (extras + dep) bwd, independent of the arrival gap
  __device__ __forceinline__ int reuse_term(std::uint32_t const extras,
                                            delta_t const dep) const {
    return static_cast<int>(extras) -
           (kFwd ? static_cast<int>(dep) : -static_cast<int>(dep));
  }

  // cross-start rejection: some previously accepted label (any start)
  // with round <= and compatible flags reuse-dominates the candidate.
  // Lock-free racy reads: missed entries only weaken pruning.
  __device__ bool reuse_rejected(std::uint64_t const* const rslots,
                                 std::uint32_t const arr_key,
                                 std::uint32_t const extras,
                                 std::uint32_t const round,
                                 bool const by_route) const {
    auto const cand_term = reuse_term(extras, dep_);
    for (auto i = 0U; i != kMcReuseCap; ++i) {
      auto const v = rslots[i];
      if (v == kMcEmptySlot) {
        continue;
      }
      if (mc_reuse_round(v) <= round &&
          (!by_route || mc_reuse_by_route(v)) &&
          mc_reuse_key(v) <= arr_key &&
          reuse_term(mc_reuse_extras(v), mc_reuse_dep(v)) <= cand_term) {
        return true;
      }
    }
    return false;
  }

  // upsert an accepted label into the reuse frontier (under the caller's
  // bag lock); frontier entries prune each other with the same rules
  __device__ void reuse_upsert(std::uint64_t* const rslots,
                               std::uint32_t const arr_key,
                               std::uint32_t const extras,
                               std::uint32_t const round,
                               bool const by_route) {
    auto const cand_term = reuse_term(extras, dep_);
    auto free_slot = ~0U;
    for (auto i = 0U; i != kMcReuseCap; ++i) {
      auto const v = rslots[i];
      if (v == kMcEmptySlot) {
        free_slot = free_slot == ~0U ? i : free_slot;
        continue;
      }
      if (mc_reuse_round(v) >= round && (by_route || !mc_reuse_by_route(v)) &&
          arr_key <= mc_reuse_key(v) &&
          cand_term <= reuse_term(mc_reuse_extras(v), mc_reuse_dep(v))) {
        rslots[i] = kMcEmptySlot;  // evicted by the new entry
        free_slot = free_slot == ~0U ? i : free_slot;
      }
    }
    if (free_slot != ~0U) {
      rslots[free_slot] = mc_reuse_pack(arr_key, extras, dep_, round,
                                        by_route);
      return;
    }
    // Full of mutually non-dominated entries: evict the LATEST arrival
    // among entries with round >= the newcomer's. Starts are processed
    // latest-first, so future candidates arrive ever earlier - the
    // arrival clause (arr_e <= arr_c) is the binding one and
    // late-arriving entries are the least likely to reject again. The
    // round restriction keeps the frontier's round coverage intact: a
    // low-round entry is the broadest rejector (round_e <= round_c), and
    // evicting only rounds >= the newcomer's never shrinks coverage.
    // Any policy is correct (rejection-only); this one keeps the
    // frontier useful.
    auto worst = ~0U;
    for (auto i = 0U; i != kMcReuseCap; ++i) {
      if (mc_reuse_round(rslots[i]) >= round &&
          (worst == ~0U ||
           mc_reuse_key(rslots[i]) > mc_reuse_key(rslots[worst]))) {
        worst = i;
      }
    }
    if (worst != ~0U && arr_key < mc_reuse_key(rslots[worst])) {
      rslots[worst] = mc_reuse_pack(arr_key, extras, dep_, round, by_route);
    }
  }

  __device__ bool bag_insert(std::uint32_t const l,
                             std::uint32_t const arr_key,
                             std::uint32_t const extras,
                             std::uint32_t const round,
                             bool const by_route,
                             bool const with_bc,
                             breadcrumb_t const payload,
                             std::uint32_t const parent,
                             delta_t const arr) {
    auto* const rslots = reuse_bags_ + static_cast<std::size_t>(l) *
                                           kMcReuseCap;
    if (reuse_rejected(rslots, arr_key, extras, round, by_route)) {
      return false;
    }
    auto* const slots = bag(l);
    // lock-free reject pre-scan: most candidates are dominated and never
    // need the lock. Racy reads are safe: 64-bit aligned loads do not
    // tear, a missed dominator is caught by the locked re-scan below,
    // and rejecting on a concurrently evicted label is still valid (its
    // evictor dominates the candidate transitively).
    {
      auto const pre_hwm = static_cast<std::uint32_t>(bag_hwm_[l]);
      for (auto i = 0U; i != pre_hwm; ++i) {
        auto const v = slots[i];
        if (v != kMcEmptySlot && mc_round(v) <= round &&
            (!by_route || mc_by_route(v)) &&
            dominates(mc_arr_key(v), mc_extras(v), arr_key, extras)) {
          return false;
        }
      }
    }
    auto* const lock = bag_locks_ + l;
    while (atomicCAS(lock, 0U, 1U) != 0U) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
    __threadfence();  // acquire: see prior holders' plain stores

    auto inserted = false;
    auto free_slot = ~0U;
    auto const hwm = static_cast<std::uint32_t>(bag_hwm_[l]);
    for (auto i = 0U; i != hwm; ++i) {
      auto const v = slots[i];
      if (v == kMcEmptySlot) {
        free_slot = free_slot == ~0U ? i : free_slot;
        continue;
      }
      // REJECT: dominated by a lower-or-equal-round label? (a by-route
      // candidate only by another by-route label - footpaths are not
      // transitive)
      if (mc_round(v) <= round && (!by_route || mc_by_route(v)) &&
          dominates(mc_arr_key(v), mc_extras(v), arr_key, extras)) {
        goto unlock;
      }
      // EVICT to a hole: a dominated higher-or-equal-round label (a
      // by-transfer candidate only evicts by-transfer labels)
      if (mc_round(v) >= round && (by_route || !mc_by_route(v)) &&
          dominates(arr_key, extras, mc_arr_key(v), mc_extras(v))) {
        slots[i] = kMcEmptySlot;
        free_slot = free_slot == ~0U ? i : free_slot;
      }
    }
    if (free_slot == ~0U && hwm < bag_cap_) {
      free_slot = hwm;
    }
    if (free_slot == ~0U) {
      atomicOr(overflow_, kMcOverflowBag);  // full of non-dominated labels
      atomicCAS(overflow_loc_, ~0U, l);  // diagnostics: first culprit
    } else {
      auto bc = kMcNoBc;
      if (with_bc) {
        bc = bc_append(payload, parent, arr);
        if (bc == kMcNoBc) {
          goto unlock;  // arena overflow (canary set)
        }
      }
      slots[free_slot] = mc_pack(arr_key, extras, round, by_route, bc);
      if (free_slot + 1U > hwm) {
        bag_hwm_[l] = static_cast<std::uint8_t>(free_slot + 1U);
      }
      reuse_upsert(rslots, arr_key, extras, round, by_route);
      inserted = true;
    }
  unlock:
    __threadfence();  // release: publish stores before unlocking
    atomicExch(lock, 0U);
    return inserted;
  }

  // append a breadcrumb; returns kMcNoBc on arena overflow (caller then
  // drops the insert - the canary makes that loud)
  __device__ std::uint32_t bc_append(breadcrumb_t const payload,
                                     std::uint32_t const parent,
                                     delta_t const arr) {
    auto const idx = atomicAdd(bc_count_, 1U);
    if (idx >= bc_cap_ || idx >= kMcNoBc) {
      atomicOr(overflow_, kMcOverflowArena);
      return kMcNoBc;
    }
    bc_arena_[idx] = {payload, parent, arr, 0U};
    return idx;
  }

  // ---- destination frontier (pruning only) ----------------------------------
  // pareto over (round, criteria); persists across start times within a
  // query (reset in reset_arrivals). Overflow-safe: a skipped insert only
  // weakens pruning.

  __device__ bool dest_dominates(std::uint32_t const round,
                                 std::uint32_t const arr_key,
                                 std::uint32_t const extras) const {
    // validation switch mirroring the CPU's NIGIRI_NO_DEST_PRUNING:
    // disabling must not change results, only slow the search down
    if (dest_prune_disabled_) {
      return false;
    }
    // necessary conditions first (2 loads); the full frontier scan only
    // runs when domination is still possible
    if (dest_best_key_[round] > arr_key ||
        dest_best_total_[round] > arr_key + extras) {
      return false;
    }
    for (auto i = 0U; i != kMcDestCap; ++i) {
      auto const v = dest_bag_[i];
      if (v != kMcEmptySlot && mc_round(v) <= round &&
          completed_dominates(mc_arr_key(v), mc_extras(v), arr_key, extras)) {
        return true;
      }
    }
    return false;
  }

  __device__ void dest_bag_add(std::uint32_t const round,
                               std::uint32_t const arr_key,
                               std::uint32_t const extras) {
    while (atomicCAS(dest_lock_, 0U, 1U) != 0U) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
    __threadfence();
    auto free_slot = ~0U;
    for (auto i = 0U; i != kMcDestCap; ++i) {
      auto const v = dest_bag_[i];
      if (v == kMcEmptySlot) {
        free_slot = free_slot == ~0U ? i : free_slot;
        continue;
      }
      if (mc_round(v) <= round &&
          completed_dominates(mc_arr_key(v), mc_extras(v), arr_key, extras)) {
        goto unlock;
      }
      if (mc_round(v) >= round &&
          completed_dominates(arr_key, extras, mc_arr_key(v), mc_extras(v))) {
        dest_bag_[i] = kMcEmptySlot;
        free_slot = free_slot == ~0U ? i : free_slot;
      }
    }
    if (free_slot != ~0U) {
      dest_bag_[free_slot] = mc_pack(arr_key, extras, round, false, kMcNoBc);
      for (auto r = round; r != kMaxTransfers + 2U; ++r) {
        dest_best_key_[r] = min(dest_best_key_[r], arr_key);
        dest_best_total_[r] = min(dest_best_total_[r], arr_key + extras);
      }
    }
    // else: full of non-dominated entries -> skip (weaker pruning only)
  unlock:
    __threadfence();
    atomicExch(dest_lock_, 0U);
  }

  // ---- phases ---------------------------------------------------------------

  __device__ void init_arrivals(delta_t const d_start) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    if (gid == 0U) {
      *done_ = 0U;
    }
    for (auto i = gid; i < starts_.size(); i += stride) {
      auto const l = to_idx(starts_[i].first);
      auto const arr = starts_[i].second;
      // at_start: ingress = walking between query start and seeded stop
      auto const ingress = static_cast<std::uint32_t>(dir(arr - d_start));
      auto const extras = WithCost ? ingress * walk_surcharge_ : 0U;
      if (bag_insert(l, to_key(arr), extras, 0U, false, /*with_bc=*/false,
                     0U, kMcNoBc, 0)) {
        touched_.mark(l);
        station_mark_.mark(l);
      }
    }
  }

  __device__ void mark_routes() {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    for (auto i = gid; i < tt_.n_locations_; i += stride) {
      if (station_mark_[i]) {
        if (!tt_.location_routes_[location_idx_t{i}].empty() && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
        for (auto r : tt_.location_routes_[location_idx_t{i}]) {
          route_mark_.mark(to_idx(r));
        }
      }
    }
  }

  __device__ void begin_transit_phase() {
    prev_station_mark_.swap_reset(station_mark_);
    if (get_global_thread_id() == 0U) {
      *route_list_count_ = 0U;
      *et_task_count_ = 0U;
    }
  }

  // NOTE: no mark swap here (unlike the raptor): the same-station transfer
  // is folded into the route-scan inserts, so the scan's arrival marks in
  // station_mark_ must survive into the next round's route collection; the
  // footpath phase iterates station_mark_ directly and extends it with its
  // targets (their labels are by-transfer and flag-skipped by the
  // iteration, exactly like the CPU update_footpaths).

  // compact the marked routes into route_list_ (gouda et phase 1); also
  // resets any_marked_ for the scan. NOTE: route-length bucketing (LPT
  // scheduling of long routes first) was tried and measured a ~5% net
  // LOSS - warp oversubscription already absorbs the length imbalance.
  __device__ void build_route_list() {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    if (gid == 0U) {
      *any_marked_ = 0U;
    }
    for (auto w = gid; w < route_mark_.blocks_.size(); w += stride) {
      auto const word = route_mark_.blocks_[w];
      if (word == 0U) {
        continue;
      }
      auto pos =
          atomicAdd(route_list_count_, static_cast<unsigned>(__popc(word)));
      for_each_set_bit(
          word, [&](unsigned const b) { route_list_[pos++] = w * 32U + b; });
    }
  }

  // dest-aware same-station transfer buffer (0 at a non-intermodal
  // destination, like the CPU transfer_buffer)
  __device__ __forceinline__ int transfer_buffer(std::uint32_t const l) const {
    return (!is_intermodal_dest() && is_dest_[l])
               ? 0
               : dir(adjusted_transfer_time(
                     transfer_time_settings_,
                     tt_.transfer_time_[location_idx_t{l}].count()));
  }

  __device__ __forceinline__ bool is_intermodal_dest() const {
    return !dist_to_end_.empty();
  }

  // et PHASE 1 (cf. gouda et_collect_tasks): warp-cooperative stream
  // compaction of the marked routes' boardable stops into a flat task list
  template <bool IsWheelchair>
  __device__ void et_collect_tasks() {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    auto const lane = gid % kWarpSize;
    auto const warp_id = gid / kWarpSize;
    auto const n_warps = stride / kWarpSize;
    auto const n_marked = *route_list_count_;
    for (auto idx = warp_id; idx < n_marked; idx += n_warps) {
      auto const ri = route_list_[idx];
      auto const r = route_idx_t{ri};
      auto const base_flat = tt_.route_stop_offset_[ri];
      auto const stop_seq = tt_.route_location_seq_[r];
      auto const n = static_cast<unsigned>(stop_seq.size());
      if (lane == 0U) {
        route_entry_count_[ri] = 0U;
      }
      for (auto chunk = 0U; chunk < n; chunk += kWarpSize) {
        auto const i = chunk + lane;
        auto is_task = false;
        if (i < n) {
          auto const stop_idx =
              static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);
          // terminate the block; the lookup phase overwrites for task stops
          et_blocks_[static_cast<std::size_t>(base_flat + stop_idx) *
                     et_per_stop_] = kMcEtInvalid;
          if (i + 1U != n) {
            auto const stp = stop{stop_seq[stop_idx]};
            auto const l = to_idx(stp.location_idx());
            is_task = prev_station_mark_[l] &&
                      stp.can_start<SearchDir>(IsWheelchair) &&
                      lb_[l] != kUnreachable && bag_hwm_[l] != 0U;
          }
        }
        auto const ballot = __ballot_sync(kAllLanes, is_task);
        if (ballot != 0U) {
          auto const leader =
              static_cast<int>(__ffs(static_cast<int>(ballot))) - 1;
          auto base_pos = 0U;
          if (lane == static_cast<unsigned>(leader)) {
            base_pos = atomicAdd(et_task_count_,
                                 static_cast<unsigned>(__popc(ballot)));
          }
          base_pos = __shfl_sync(kAllLanes, base_pos, leader);
          if (is_task) {
            auto const off =
                static_cast<unsigned>(__popc(ballot & ((1U << lane) - 1U)));
            auto const stop_idx =
                static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);
            et_task_list_[base_pos + off] = base_flat + stop_idx;
          }
        }
      }
    }
  }

  // et PHASE 2: one thread per task; the earliest-transport lookups for
  // all boardable labels of the task's stop go into its result block.
  // Dropping the CPU's scan-order skip heuristic here only adds redundant
  // lookups - the route-bag merge in the scan phase rejects them anyway.
  __device__ void et_run_lookups(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    auto const n_tasks = *et_task_count_;
    for (auto t = gid; t < n_tasks; t += stride) {
      auto const flat = et_task_list_[t];
      auto const r = route_idx_t{tt_.route_of_stop_[flat]};
      auto const stop_idx =
          static_cast<stop_idx_t>(flat - tt_.route_stop_offset_[to_idx(r)]);
      auto const stop_seq = tt_.route_location_seq_[r];
      auto const l_idx = to_idx(stop{stop_seq[stop_idx]}.location_idx());
      auto const l_lb = lb_[l_idx];
      auto const* const stop_bag = bag(l_idx);
      auto const stop_hwm = static_cast<std::uint32_t>(bag_hwm_[l_idx]);

      // gather the boardable labels SORTED BY ARRIVAL: the earliest
      // transport depends only on the arrival time and is monotone in it,
      // so equal arrivals share one lookup and ascending processing lets
      // the previous result be reused whenever its departure is still
      // catchable (a stop's labels usually cluster before the same next
      // departure). Seek work drops from L full seeks to ~1 + (L-1)
      // comparisons in the common case; results are bit-identical.
      struct cand {
        std::uint32_t key_;
        std::uint16_t extras_;
        std::uint8_t sl_;
      };
      cand cands[kMcEtGatherCap];
      auto n_cands = 0U;
      for (auto sl = 0U; sl != stop_hwm; ++sl) {
        auto const lab = stop_bag[sl];
        if (lab == kMcEmptySlot || mc_round(lab) != k - 1U) {
          continue;
        }
        auto const pe_key = mc_arr_key(lab);
        auto const pe_extras = WithCost ? mc_extras(lab) : 0U;
        if (dest_dominates(k, pe_key + l_lb, pe_extras)) {
          continue;
        }
        if (n_cands == kMcEtGatherCap) {
          atomicOr(overflow_, kMcOverflowEtBlock);
          break;
        }
        auto j = n_cands;
        while (j != 0U && cands[j - 1U].key_ > pe_key) {
          cands[j] = cands[j - 1U];
          --j;
        }
        cands[j] = {pe_key, static_cast<std::uint16_t>(pe_extras),
                    static_cast<std::uint8_t>(sl)};
        ++n_cands;
      }

      auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;
      auto out = 0U;
      auto prev_key = ~0U;
      auto prev_valid = false;
      auto prev_et = transport{};
      auto prev_dep = delta_t{0};
      for (auto ci = 0U; ci != n_cands; ++ci) {
        auto const key = cands[ci].key_;
        auto need_seek = true;
        if (key == prev_key) {
          need_seek = false;  // identical arrival: identical result
        } else if (prev_valid &&
                   is_better_or_eq(from_key(key), prev_dep)) {
          need_seek = false;  // previous trip still catchable: same result
        }
        if (need_seek) {
          auto const [day, mam] = split(from_key(key));
          auto const et = get_earliest_transport(r, stop_idx, day, mam, l_lb);
          prev_valid = et.is_valid();
          if (prev_valid) {
            prev_et = et;
            prev_dep = time_at_stop(r, et, stop_idx, dep_ev);
          }
        }
        prev_key = key;
        if (!prev_valid) {
          continue;  // nothing catchable from this (or an equal) arrival
        }
        if (out == et_per_stop_) {
          atomicOr(overflow_, kMcOverflowEtBlock);
          break;
        }
        auto const packed =
            mc_et_pack(pack_et(r, prev_et), cands[ci].extras_, cands[ci].sl_);
        et_blocks_[static_cast<std::size_t>(flat) * et_per_stop_ + out] =
            packed;
        if (atomicAdd(route_entry_count_ + to_idx(r), 1U) == 0U) {
          route_single_entry_[to_idx(r)] = packed;
          route_single_flat_[to_idx(r)] = flat;
        }
        ++out;
      }
      // stale-terminate the block (entries are round-stamped, but the scan
      // stops at the first non-current entry, so mark the boundary)
      if (out != et_per_stop_) {
        et_blocks_[static_cast<std::size_t>(flat) * et_per_stop_ + out] =
            kMcEtInvalid;
      }
    }
  }

  // route bag entry (registers/local): pareto frontier over the total trip
  // order x carried extras while scanning one route
  struct route_label {
    std::uint32_t et_;  // packed (rel day | transport offset), scan order
    std::uint32_t parent_;  // breadcrumb arena index of the boarded label
    delta_t board_dep_;  // departure at the boarding stop
    std::uint16_t board_;  // boarding stop index
    std::uint16_t extras_;  // carried extras (cost config; 0 otherwise)
  };

  // warp per marked route: pass 1 replays the boarding merges into
  // shared-memory segments, pass 2 parallelizes the alight work across
  // the lanes. Falls back to the sequential single-lane scan when a
  // route needs more than kMcMaxSegs segments.
  template <bool WithClaszFilter, bool IsWheelchair>
  __device__ void scan_routes(unsigned const k, mc_seg* const seg_smem) {
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const warp_id = get_global_thread_id() / kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto* const segs =
        seg_smem + (threadIdx.x / kWarpSize) * kMcMaxSegs;
    auto const n_marked = *route_list_count_;
    auto local_marked = false;

    for (auto idx = warp_id; idx < n_marked; idx += n_warps) {
      auto const r = route_idx_t{route_list_[idx]};
      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          continue;
        }
      }
      auto const n_entries = route_entry_count_[to_idx(r)];
      if (n_entries == 0U) {
        continue;  // all lookups pruned: nothing boards this route
      }
      auto n_segs = 0U;
      if (n_entries == 1U) {
        // single boarding: the lookup phase recorded the entry - emit its
        // one segment directly, no sequential stop walk needed
        auto const packed = route_single_entry_[to_idx(r)];
        auto const flat = route_single_flat_[to_idx(r)];
        auto const stop_idx = static_cast<stop_idx_t>(
            flat - tt_.route_stop_offset_[to_idx(r)]);
        auto const stop_seq = tt_.route_location_seq_[r];
        auto const n = static_cast<unsigned>(stop_seq.size());
        auto const pos = kFwd ? static_cast<unsigned>(stop_idx)
                              : n - 1U - static_cast<unsigned>(stop_idx);
        if (lane == 0U) {
          auto const l = to_idx(stop{stop_seq[stop_idx]}.location_idx());
          segs[0] = {mc_et_key(packed),
                     mc_bc(bag(l)[mc_et_slot(packed)]),
                     static_cast<std::uint16_t>(
                         WithCost ? mc_et_extras(packed) : 0U),
                     static_cast<std::uint16_t>(stop_idx),
                     static_cast<std::uint16_t>(pos + 1U),
                     static_cast<std::uint16_t>(n)};
        }
        n_segs = 1U;
      } else {
        n_segs = scan_pass1(k, r, segs, lane);
      }
      __syncwarp();
      if (n_segs > kMcMaxSegs) {  // overflow: sequential fallback
        auto m = false;
        if (lane == 0U) {
          m = scan_route<IsWheelchair>(k, r);
        }
        local_marked |= __shfl_sync(kAllLanes, m, 0) != 0;
      } else {
        local_marked |= scan_pass2<IsWheelchair>(k, r, segs, n_segs, lane);
      }
      __syncwarp();
    }
    if (__any_sync(kAllLanes, local_marked) && lane == 0U &&
        !*any_marked_) {
      atomicOr(any_marked_, 1U);
    }
  }

  // pass 1: replay the route-bag merges (identical rules to the
  // sequential scan) and emit segments. All lanes run this redundantly
  // on identical data (warp-uniform, block loads broadcast); lane 0
  // writes the shared segments. Returns the segment count, or
  // kMcMaxSegs + 1 on overflow.
  __device__ unsigned scan_pass1(unsigned const k,
                                 route_idx_t const r,
                                 mc_seg* const segs,
                                 unsigned const lane) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto const base_flat = tt_.route_stop_offset_[to_idx(r)];

    struct open_entry {  // a route-bag entry and its open segment
      std::uint32_t et_;
      std::uint16_t extras_;
      std::uint16_t seg_;
    };
    open_entry rb[kMcRouteBagCap];
    auto bag_size = 0U;
    auto n_segs = 0U;

    for (auto i = 0U; i + 1U < n; ++i) {  // last stop never boards
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);
      auto const flat = static_cast<std::size_t>(base_flat + stop_idx);
      auto const first = et_blocks_[flat * et_per_stop_];
      if (first == kMcEtInvalid) {
        continue;  // not a boarding stop this round
      }
      auto const l_idx = to_idx(stop{stop_seq[stop_idx]}.location_idx());
      auto const* const stop_bag = bag(l_idx);
      for (auto e = 0U; e != et_per_stop_; ++e) {
        auto const blk =
            e == 0U ? first : et_blocks_[flat * et_per_stop_ + e];
        if (blk == kMcEtInvalid) {
          break;  // dense fill up to the terminator
        }
        auto const et_key = mc_et_key(blk);
        auto const pe_extras = WithCost ? mc_et_extras(blk) : 0U;
        auto const pe_parent = mc_bc(stop_bag[mc_et_slot(blk)]);

        // merge into the route bag: pareto over (trip order, extras)
        auto merged = false;
        for (auto b = 0U; b != bag_size; ++b) {
          auto& rl = rb[b];
          if (rl.et_ == et_key && rl.extras_ == pe_extras) {
            // same trip, same carried extras: board closest to the exit
            // -> close the old segment, open a new one from here
            if (n_segs == kMcMaxSegs) {
              return kMcMaxSegs + 1U;
            }
            if (lane == 0U) {
              segs[rl.seg_].end_ = static_cast<std::uint16_t>(i + 1U);
              segs[n_segs] = {et_key,
                              pe_parent,
                              static_cast<std::uint16_t>(pe_extras),
                              static_cast<std::uint16_t>(stop_idx),
                              static_cast<std::uint16_t>(i + 1U),
                              static_cast<std::uint16_t>(n)};
            }
            rl.seg_ = static_cast<std::uint16_t>(n_segs);
            ++n_segs;
            merged = true;
            break;
          }
          if (et_key >= rl.et_ && rl.extras_ <= pe_extras) {
            merged = true;  // dominated: not-earlier trip, extras no better
            break;
          }
        }
        if (merged) {
          continue;
        }
        auto w = 0U;
        for (auto b = 0U; b != bag_size; ++b) {
          if (!(et_key <= rb[b].et_) || !(pe_extras <= rb[b].extras_)) {
            rb[w] = rb[b];
            ++w;
          } else if (lane == 0U) {  // evicted: close its segment
            segs[rb[b].seg_].end_ = static_cast<std::uint16_t>(i + 1U);
          }
        }
        bag_size = w;
        if (bag_size == kMcRouteBagCap || n_segs == kMcMaxSegs) {
          if (bag_size == kMcRouteBagCap) {
            atomicOr(overflow_, kMcOverflowRouteBag);
          }
          return kMcMaxSegs + 1U;
        }
        if (lane == 0U) {
          segs[n_segs] = {et_key,
                          pe_parent,
                          static_cast<std::uint16_t>(pe_extras),
                          static_cast<std::uint16_t>(stop_idx),
                          static_cast<std::uint16_t>(i + 1U),
                          static_cast<std::uint16_t>(n)};
        }
        rb[bag_size] = {et_key, static_cast<std::uint16_t>(pe_extras),
                         static_cast<std::uint16_t>(n_segs)};
        ++bag_size;
        ++n_segs;
      }
    }
    return n_segs;
  }

  // pass 2: lanes stride the stop positions; each processes every
  // segment active at its position (the alight work of the sequential
  // scan, unchanged rules).
  template <bool IsWheelchair>
  __device__ bool scan_pass2(unsigned const k,
                             route_idx_t const r,
                             mc_seg const* const segs,
                             unsigned const n_segs,
                             unsigned const lane) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
    auto const worst_key = to_key(worst_at_dest_);
    auto any_marked = false;

    for (auto p = 1U + lane; p < n; p += kWarpSize) {
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? p : n - 1U - p);
      auto const stp = stop{stop_seq[stop_idx]};
      if (!stp.can_finish<SearchDir>(IsWheelchair)) {
        continue;
      }
      auto const l_idx = to_idx(stp.location_idx());
      auto const l_lb = lb_[l_idx];
      if (l_lb == kUnreachable) {
        continue;
      }
      auto const buf = transfer_buffer(l_idx);
      auto const is_dest = is_dest_[l_idx];

      for (auto si = 0U; si != n_segs; ++si) {
        auto const& sg = segs[si];
        if (p < sg.start_ || p >= sg.end_) {
          continue;
        }
        auto const t = unpack_et(r, sg.et_);
        auto const by_transport = time_at_stop(r, t, stop_idx, arr_ev);
        auto const ride_key = to_key(by_transport);
        // window bound with the lb projection (CPU update_route)
        if (ride_key >= worst_key || ride_key + l_lb >= worst_key) {
          continue;
        }
        auto const ride_extras =
            WithCost ? static_cast<std::uint32_t>(sg.extras_) + kBoardCost
                     : 0U;
        // destination pareto pruning: optimistic projection (key + lb)
        if (dest_dominates(k, ride_key + l_lb, ride_extras)) {
          continue;
        }
        auto const post_arr = clamp(by_transport + buf);
        if (!bag_insert(
                l_idx, to_key(post_arr), ride_extras, k, true,
                /*with_bc=*/true,
                make_transport_payload(to_idx(t.t_idx_), sg.board_, stop_idx),
                sg.parent_, post_arr)) {
          continue;
        }
        touched_.mark(l_idx);
        station_mark_.mark(l_idx);
        any_marked = true;
        if (is_dest) {
          // buffer is 0 at a station destination -> ride criteria are
          // the journey's arrival criteria
          dest_bag_add(k, ride_key, ride_extras);
        }
      }
    }
    return __any_sync(kAllLanes, any_marked);
  }

  // sequential single-lane scan: fallback for segment overflow
  template <bool WithClaszFilter, bool IsWheelchair>
  __device__ void scan_routes_seq(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    auto const n_marked = *route_list_count_;
    auto local_marked = false;

    for (auto idx = gid; idx < n_marked; idx += stride) {
      auto const r = route_idx_t{route_list_[idx]};
      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          continue;
        }
      }
      local_marked |= scan_route<IsWheelchair>(k, r);
    }
    if (local_marked && !*any_marked_) {
      atomicOr(any_marked_, 1U);
    }
  }

  template <bool IsWheelchair>
  __device__ bool scan_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto any_marked = false;

    route_label route_bag[kMcRouteBagCap];
    auto bag_size = 0U;

    auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
    auto const worst_key = to_key(worst_at_dest_);

    for (auto i = 0U; i != n; ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = to_idx(stp.location_idx());
      auto const is_last = i == n - 1U;

      // ---- alight: write arrivals of carried boardings -----------------
      if (i != 0U && bag_size != 0U && stp.can_finish<SearchDir>(IsWheelchair)) {
        auto const buf = transfer_buffer(l_idx);
        auto const l_lb = lb_[l_idx];
        for (auto b = 0U; b != bag_size; ++b) {
          auto const& rl = route_bag[b];
          auto const t = unpack_et(r, rl.et_);
          auto const by_transport = time_at_stop(r, t, stop_idx, arr_ev);
          auto const ride_key = to_key(by_transport);
          // window bound with the lb projection (CPU update_route)
          if (ride_key >= worst_key || l_lb == kUnreachable ||
              ride_key + l_lb >= worst_key) {
            continue;
          }
          auto const ride_extras =
              WithCost ? static_cast<std::uint32_t>(rl.extras_) + kBoardCost
                       : 0U;
          // destination pareto pruning: optimistic projection (key + lb)
          if (dest_dominates(k, ride_key + l_lb, ride_extras)) {
            continue;
          }
          auto const post_arr = clamp(by_transport + buf);
          if (!bag_insert(
                  l_idx, to_key(post_arr), ride_extras, k, true,
                  /*with_bc=*/true,
                  make_transport_payload(to_idx(t.t_idx_), rl.board_,
                                         stop_idx),
                  rl.parent_, post_arr)) {
            continue;
          }
          touched_.mark(l_idx);
          station_mark_.mark(l_idx);
          any_marked = true;
          if (is_dest_[l_idx]) {
            // buffer is 0 at a station destination -> ride criteria are
            // the journey's arrival criteria
            dest_bag_add(k, ride_key, ride_extras);
          }
        }
      }

      // ---- board: merge this stop's round-(k-1) labels ------------------
      if (is_last || !stp.can_start<SearchDir>(IsWheelchair) ||
          !prev_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {  // CPU update_route: scan abort
        break;
      }

      auto const* const stop_bag = bag(l_idx);
      auto const flat = tt_.route_stop_offset_[to_idx(r)] + stop_idx;

      for (auto e = 0U; e != et_per_stop_; ++e) {
        auto const blk =
            et_blocks_[static_cast<std::size_t>(flat) * et_per_stop_ + e];
        if (blk == kMcEtInvalid) {
          break;  // dense fill up to the terminator
        }
        auto const et_key = mc_et_key(blk);
        auto const pe_extras = WithCost ? mc_et_extras(blk) : 0U;
        auto const pe_parent = mc_bc(stop_bag[mc_et_slot(blk)]);

        // merge into the route bag: pareto over (trip order, extras)
        auto merged = false;
        for (auto b = 0U; b != bag_size; ++b) {
          auto& rl = route_bag[b];
          if (rl.et_ == et_key && rl.extras_ == pe_extras) {
            // same trip, same carried extras: board closest to the exit
            rl.board_ = stop_idx;
            rl.parent_ = pe_parent;
            merged = true;
            break;
          }
          if (et_key >= rl.et_ && rl.extras_ <= pe_extras) {
            merged = true;  // dominated: not-earlier trip, extras no better
            break;
          }
        }
        if (merged) {
          continue;
        }
        auto w = 0U;
        for (auto b = 0U; b != bag_size; ++b) {
          if (!(et_key <= route_bag[b].et_) ||
              !(pe_extras <= route_bag[b].extras_)) {
            route_bag[w] = route_bag[b];
            ++w;
          }
        }
        bag_size = w;
        if (bag_size == kMcRouteBagCap) {
          atomicOr(overflow_, kMcOverflowRouteBag);
          continue;
        }
        route_bag[bag_size] = {et_key, pe_parent, 0, stop_idx,
                               static_cast<std::uint16_t>(pe_extras)};
        ++bag_size;
      }
    }
    return any_marked;
  }

  // relax one footpath from a recovered transit arrival (window + lb +
  // dest pruning like the CPU update_footpaths)
  __device__ __forceinline__ bool relax_fp(unsigned const k,
                                           std::uint32_t const source,
                                           footpath const fp,
                                           delta_t const te_arr,
                                           std::uint32_t const te_extras,
                                           std::uint32_t const te_bc,
                                           std::uint32_t const worst_key) {
    auto const target = to_idx(fp.target());
    if (target == source) {
      return false;
    }
    auto const fp_duration = adjusted_transfer_time(transfer_time_settings_,
                                                    fp.duration().count());
    auto const fp_arr = clamp(te_arr + dir(fp_duration));
    auto const fp_key = to_key(fp_arr);
    auto const target_lb = lb_[target];
    if (fp_key >= worst_key || target_lb == kUnreachable ||
        fp_key + target_lb >= worst_key) {
      return false;
    }
    auto const fp_extras =
        WithCost ? te_extras + static_cast<std::uint32_t>(fp_duration *
                                                          walk_surcharge_)
                 : 0U;
    if (dest_dominates(k, fp_key + target_lb, fp_extras)) {
      return false;
    }
    auto const src = bc_arena_[te_bc];
    if (!bag_insert(target, fp_key, fp_extras, k, false, /*with_bc=*/true,
                    src.payload_, src.parent_, fp_arr)) {
      return false;
    }
    touched_.mark(target);
    station_mark_.mark(target);
    if (is_dest_[target]) {
      dest_bag_add(k, fp_key, fp_extras);
    }
    return true;
  }

  // fused same-round intermodal egress + footpath relaxation over the
  // stops the route scan marked (the same-station transfer buffer is
  // already folded into the route arrivals). Warp-cooperative like the
  // gouda raptor: short footpath lists inline per lane, hub stops
  // deferred to the whole warp (lanes stride the list; the deferred
  // stop's labels are re-read by every lane - no shuffles needed).
  __device__ void update_transfers_and_footpaths(unsigned const k) {
    constexpr auto const kWarpFpThreshold = 8U;
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const warp_id = get_global_thread_id() / kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto const intermodal = is_intermodal_dest();
    auto const worst_key = to_key(worst_at_dest_);
    auto const n_blocks =
        static_cast<unsigned>(station_mark_.blocks_.size());
    auto local_marked = false;

    for (auto w = warp_id; w < n_blocks; w += n_warps) {
      auto const bits = station_mark_.blocks_[w];
      if (bits == 0U) {  // uniform: all lanes read the same word
        continue;
      }
      auto const base = w * kWarpSize;
      auto const my_i = base + lane;
      auto const my_marked = ((bits >> lane) & 1U) != 0U;

      auto defer = false;
      if (my_marked && bag_hwm_[my_i] != 0U) {
        auto const l = location_idx_t{my_i};
        auto const fps = kFwd ? tt_.footpaths_out_[prf_idx_][l]
                              : tt_.footpaths_in_[prf_idx_][l];
        auto const n_fps = static_cast<unsigned>(fps.size());
        auto const egress_ok =
            intermodal && dist_to_end_[my_i] != kUnreachable;
        if (n_fps != 0U || egress_ok) {
          auto const buf = transfer_buffer(my_i);
          auto const* const stop_bag = bag(my_i);
          auto const stop_hwm = static_cast<std::uint32_t>(bag_hwm_[my_i]);
          if (n_fps > kWarpFpThreshold) {
            defer = true;  // hub: the whole warp strides the list below
          }
          for (auto sl = 0U; sl != stop_hwm; ++sl) {
            auto const lab = stop_bag[sl];
            if (lab == kMcEmptySlot || mc_round(lab) != k ||
                !mc_by_route(lab)) {
              continue;
            }
            auto const te_arr = clamp(from_key(mc_arr_key(lab)) - buf);
            auto const te_extras = WithCost ? mc_extras(lab) : 0U;
            auto const te_bc = mc_bc(lab);

            if (egress_ok) {
              auto const end_arr = clamp(te_arr + dir(dist_to_end_[my_i]));
              auto const end_key = to_key(end_arr);
              auto const end_extras =
                  WithCost
                      ? te_extras + static_cast<std::uint32_t>(
                                        dist_to_end_[my_i] * walk_surcharge_)
                      : 0U;
              // window bound: pong's reverse searches must not write
              // journeys departing beyond the ping's start time
              if (end_key < worst_key) {
                auto const src = bc_arena_[te_bc];
                if (bag_insert(to_idx(kIntermodalTarget), end_key,
                               end_extras, k, false, /*with_bc=*/true,
                               src.payload_, src.parent_, end_arr)) {
                  touched_.mark(to_idx(kIntermodalTarget));
                  dest_bag_add(k, end_key, end_extras);
                }
              }
            }

            if (!defer) {
              for (auto f = 0U; f != n_fps; ++f) {
                local_marked |= relax_fp(k, my_i, fps[f], te_arr, te_extras,
                                         te_bc, worst_key);
              }
            }
          }
        }
      }

      // hubs: all lanes stride one deferred stop's footpath list per label
      auto const deferred = __ballot_sync(kAllLanes, defer);
      for_each_set_bit(deferred, [&](unsigned const b) {
        auto const i = base + b;
        auto const l = location_idx_t{i};
        auto const fps = kFwd ? tt_.footpaths_out_[prf_idx_][l]
                              : tt_.footpaths_in_[prf_idx_][l];
        auto const n_fps = static_cast<unsigned>(fps.size());
        auto const buf = transfer_buffer(i);
        auto const* const stop_bag = bag(i);
        auto const stop_hwm = static_cast<std::uint32_t>(bag_hwm_[i]);
        for (auto sl = 0U; sl != stop_hwm; ++sl) {
          auto const lab = stop_bag[sl];
          if (lab == kMcEmptySlot || mc_round(lab) != k ||
              !mc_by_route(lab)) {
            continue;
          }
          auto const te_arr = clamp(from_key(mc_arr_key(lab)) - buf);
          auto const te_extras = WithCost ? mc_extras(lab) : 0U;
          auto const te_bc = mc_bc(lab);
          for (auto f = lane; f < n_fps; f += kWarpSize) {
            local_marked |= relax_fp(k, i, fps[f], te_arr, te_extras, te_bc,
                                     worst_key);
          }
        }
      });
    }
    if (local_marked && !*any_marked_) {
      atomicOr(any_marked_, 1U);
    }
  }

  // debug: dump the traced stops' bag contents after round k
  // (NIGIRI_MC_TRACE + NIGIRI_MC_TRACE_START, single-thread printf)
  __device__ void dump_traced(unsigned const k) const {
    if (get_global_thread_id() != 0U) {
      return;
    }
    for (auto t = 0U; t != trace_locs_.size(); ++t) {
      auto const l = to_idx(trace_locs_[t]);
      auto const* const slots = bag(l);
      for (auto i = 0U; i != bag_cap_; ++i) {
        auto const v = slots[i];
        if (v == kMcEmptySlot) {
          continue;
        }
        printf("GPUTRACE k=%u stop=%u slot=%u round=%u route=%d arr=%d "
               "extras=%u bc=%u\n",
               k, static_cast<unsigned>(l), i, mc_round(v),
               mc_by_route(v) ? 1 : 0,
               static_cast<int>(from_key(mc_arr_key(v))), mc_extras(v),
               mc_bc(v));
      }
    }
  }

  // clear the touched bags + arena between start times (selective sweep)
  __device__ void clear_bags() {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    if (gid == 0U) {
      *bc_count_ = 0U;
    }
    for (auto w = gid; w < touched_.blocks_.size(); w += stride) {
      auto const word = touched_.blocks_[w];
      if (word == 0U) {
        continue;
      }
      for_each_set_bit(word, [&](unsigned const b) {
        auto const l = w * 32U + b;
        auto* const slots = bag(l);
        auto const n = static_cast<std::uint32_t>(bag_hwm_[l]);
        for (auto s = 0U; s != n; ++s) {
          slots[s] = kMcEmptySlot;
        }
        bag_hwm_[l] = 0U;
      });
      touched_.blocks_[w] = 0U;
    }
  }

  // ---- reconstruction (breadcrumb chase, cf. CPU materialize) --------------

  __device__ void reconstruct_label(location_idx_t const dest_l,
                                    mc_label_t const lab,
                                    gpu_journey* out) const {
    out->state_ = reconstruction_result::kNotReconstructed;
    auto const K = mc_round(lab);
    if (K == 0U) {
      return;  // round-0 seed at the destination: no transit legs
    }
    out->dest_l_ = dest_l;
    out->dest_time_ = from_key(mc_arr_key(lab));
    out->transfers_ = static_cast<std::uint8_t>(K - 1U);
    out->criteria_cost_ =
        WithCost ? static_cast<std::uint16_t>(mc_extras(lab)) : 0U;

    auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
    auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;

    // chase: each arena entry stores the arrival at its own location
    // (cur_l), so the loop reads everything from the entry - CPU
    // materialize 1:1
    auto cur_l = dest_l;
    auto li = mc_bc(lab);
    auto n = 0U;
    while (li != kMcNoBc) {
      auto const bc = bc_arena_[li];
      auto const cur_arr = bc.arr_;
      auto const t_idx = transport_idx_t{bc_transport(bc.payload_)};
      auto const board = static_cast<stop_idx_t>(bc_board(bc.payload_));
      auto const alight = static_cast<stop_idx_t>(bc_alight(bc.payload_));
      auto const r = tt_.transport_route_[t_idx];

      // recover the traffic day: a single footpath/transfer crosses
      // midnight at most once -> two candidate days
      auto const event_mam_full =
          tt_.event_mam(r, t_idx, alight, arr_ev).count();
      auto const arr_day = static_cast<int>(split(cur_arr).first.v_);
      auto day = day_idx_t::invalid();
      auto train_arr = kInvalid;
      for (auto off = 0; off != 2; ++off) {
        auto const cand =
            arr_day - event_mam_full / 1440 - (kFwd ? off : -off);
        if (cand < 0) {
          continue;
        }
        if (!is_transport_active(t_idx, static_cast<std::size_t>(cand))) {
          continue;
        }
        auto const tr =
            transport{t_idx, day_idx_t{static_cast<day_idx_t::value_t>(cand)}};
        auto const ev = time_at_stop(r, tr, alight, arr_ev);
        if (is_better_or_eq(ev, cur_arr)) {
          day = day_idx_t{static_cast<day_idx_t::value_t>(cand)};
          train_arr = ev;
          break;
        }
      }
      if (day == day_idx_t::invalid()) {
        out->state_ = reconstruction_result::kReconstructionFailed;
        return;
      }

      auto const tr = transport{t_idx, day};
      auto const dep_at_board = time_at_stop(r, tr, board, dep_ev);
      auto const stop_seq = tt_.route_location_seq_[r];
      auto const board_loc = stop{stop_seq[board]}.location_idx();
      auto const alight_loc = stop{stop_seq[alight]}.location_idx();

      auto const is_egress = is_intermodal_dest() && cur_l == kIntermodalTarget;
      if (is_egress) {
        // no footpath leg: the last mile is the host's mumo leg; the
        // journey's terminal is the ride's alighting stop
        out->dest_l_ = alight_loc;
      } else if (n != 0U || alight_loc != cur_l || train_arr != cur_arr) {
        if (n >= kMaxRecLegs) {
          out->state_ = reconstruction_result::kReconstructionFailed;
          return;
        }
        auto& lg = out->legs_[n++];
        lg.is_footpath_ = true;
        lg.from_l_ = alight_loc;
        lg.to_l_ = cur_l;
        lg.dep_ = train_arr;
        lg.arr_ = cur_arr;
        lg.fp_duration_ = static_cast<std::uint16_t>(
            kFwd ? (cur_arr - train_arr) : (train_arr - cur_arr));
      }

      if (n >= kMaxRecLegs) {
        out->state_ = reconstruction_result::kReconstructionFailed;
        return;
      }
      auto& lg = out->legs_[n++];
      lg.is_footpath_ = false;
      lg.from_l_ = board_loc;
      lg.to_l_ = alight_loc;
      lg.dep_ = dep_at_board;
      lg.arr_ = train_arr;
      lg.transport_ = t_idx;
      lg.rt_transport_ = rt_transport_idx_t::invalid();
      lg.day_ = day;
      lg.enter_stop_ = board;
      lg.exit_stop_ = alight;

      cur_l = board_loc;
      li = bc.parent_;
    }

    out->start_l_ = cur_l;
    out->n_legs_ = static_cast<std::uint8_t>(n);
    out->state_ = (n != 0U) ? reconstruction_result::kOk
                            : reconstruction_result::kReconstructionFailed;
  }

  // ---- shared helpers (1:1 gouda raptor_impl) --------------------------------

  __device__ transport get_earliest_transport(route_idx_t const r,
                                              stop_idx_t const stop_idx,
                                              day_idx_t const day_at_stop,
                                              minutes_after_midnight_t const
                                                  mam_at_stop,
                                              std::uint16_t const l_lb) {
    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    constexpr auto const kNDaysToIterate = static_cast<day_idx_t::value_t>(
        kMaxTravelTime / std::chrono::days{1} + 1U);
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      if (!is_route_active(r, day)) {
        continue;
      }

      auto const ev_time_range =
          it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                   get_end_it(event_times)};
      if (ev_time_range.empty()) {
        continue;
      }

      for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = ev.mam();

        // window pruning with the lb projection (CPU form)
        if (is_better_or_eq(worst_at_dest_,
                            to_delta(day, ev_mam) + dir(l_lb))) {
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(static_cast<int>(day.v_) - ev_day_offset);
        if (!is_transport_active(t, start_day)) {
          continue;
        }
        return {t, static_cast<day_idx_t>(static_cast<int>(day.v_) -
                                          ev_day_offset)};
      }
    }
    return {};
  }

  __device__ __forceinline__ bool is_transport_active(
      transport_idx_t const t, std::size_t const day) const {
    return tt_.bitfields_[tt_.transport_traffic_days_[t]].test(day);
  }

  __device__ __forceinline__ bool is_route_active(route_idx_t const r,
                                                  day_idx_t const day) const {
    return static_cast<int>(day.v_) >= 0 &&
           tt_.bitfields_[tt_.route_traffic_days_[r]].test(
               static_cast<std::size_t>(static_cast<int>(day.v_)));
  }

  // packed earliest transport (gouda): [rel_day : 6 | t_offset : 26];
  // plain unsigned comparison = total trip order in scan direction
  static constexpr auto kEtRelDayShift = 26U;
  static constexpr auto kEtReverseBase = (~std::uint32_t{0}) - 1U;

  __device__ __forceinline__ int et_day_lo() const {
    return static_cast<int>(base_.v_) - 28;
  }

  __device__ __forceinline__ std::uint32_t pack_et(route_idx_t const r,
                                                   transport const t) const {
    auto const rel_day =
        static_cast<std::uint32_t>(static_cast<int>(t.day_.v_) - et_day_lo());
    auto const t_offset =
        to_idx(t.t_idx_) - to_idx(tt_.route_transport_ranges_[r].from_);
    auto const x = (rel_day << kEtRelDayShift) | t_offset;
    return kFwd ? x : kEtReverseBase - x;
  }

  __device__ __forceinline__ transport unpack_et(route_idx_t const r,
                                                 std::uint32_t const p) const {
    auto const x = kFwd ? p : kEtReverseBase - p;
    auto const rel_day = x >> kEtRelDayShift;
    auto const t_offset = x & ((1U << kEtRelDayShift) - 1U);
    return transport{
        transport_idx_t{to_idx(tt_.route_transport_ranges_[r].from_) +
                        t_offset},
        day_idx_t{static_cast<day_idx_t::value_t>(static_cast<int>(rel_day) +
                                                  et_day_lo())}};
  }

  __device__ delta_t time_at_stop(route_idx_t const r,
                                  transport const t,
                                  stop_idx_t const stop_idx,
                                  event_type const ev_type) const {
    return to_delta(t.day_,
                    tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  __device__ delta_t to_delta(day_idx_t const day,
                              std::int16_t const mam) const {
    return clamp((static_cast<int>(day.v_) - static_cast<int>(base_.v_)) *
                     1440 +
                 mam);
  }

  __device__ std::pair<day_idx_t, minutes_after_midnight_t> split(
      delta_t const x) const {
    return split_day_mam(base_, x);
  }

  template <typename T>
  __device__ __forceinline__ auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  __device__ __forceinline__ auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  // ---- members ---------------------------------------------------------------

  std::uint32_t* any_marked_;
  std::uint32_t* done_;
  std::uint32_t* overflow_;
  std::uint32_t* overflow_loc_;
  device_timetable tt_;
  transfer_time_settings transfer_time_settings_;
  clasz_mask_t allowed_claszes_;
  profile_idx_t prf_idx_;
  day_idx_t base_;
  delta_t worst_at_dest_;
  std::uint32_t walk_surcharge_;

  cuda::std::span<std::pair<location_idx_t, delta_t> const> starts_;
  device_bitvec<std::uint64_t const> is_dest_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  // per-query lower bounds to the destination (minutes; kUnreachable =
  // cannot reach). Pruning only - never changes results. In the biased key
  // domain the optimistic projection is uniformly key + lb for both
  // search directions.
  cuda::std::span<std::uint16_t const> lb_;
  cuda::std::span<location_idx_t const> trace_locs_;  // debug dump only

  mc_label_t* bags_;  // n_locations x bag_cap_
  std::uint64_t* reuse_bags_;  // n_locations x kMcReuseCap, per-query
  std::uint32_t bag_cap_;
  std::uint32_t et_per_stop_;
  delta_t dep_;  // this execute's departure (for the reuse rule)
  std::uint32_t* bag_locks_;  // n_locations spinlocks
  // per-bag high-water mark (used slots upper bound): shortcuts the
  // 32-slot scans; maintained under the bag lock, reset with the bags
  std::uint8_t* bag_hwm_;
  mc_label_t* dest_bag_;  // kMcDestCap
  std::uint32_t* dest_lock_;
  // scalar prefilters over the dest frontier, indexed by round:
  // min arr key / min (key + extras) among entries with round <= k.
  // Racy-relaxed reads: stale values only weaken pruning, never results.
  std::uint32_t* dest_best_key_;
  std::uint32_t* dest_best_total_;
  bool dest_prune_disabled_;
  mc_bc_entry* bc_arena_;
  std::uint32_t* bc_count_;
  std::uint32_t bc_cap_;

  device_bitvec<std::uint32_t> touched_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;

  cuda::std::span<std::uint32_t> route_list_;
  std::uint32_t* route_list_count_;

  // et phase buffers (gouda raptor structure): flat (route,stop) task list
  // + per route-stop result blocks consumed by the scan
  cuda::std::span<std::uint32_t> et_task_list_;
  std::uint32_t* et_task_count_;
  cuda::std::span<std::uint64_t> et_blocks_;  // n_route_stops x et_per_stop_
  // single-boarding fast path: per-route entry tally + the (single) entry
  // recorded by the lookup phase. Most marked routes have exactly one
  // boardable (stop, label) in a round; the scan then skips the
  // sequential pass-1 walk entirely (count 0: skip the route, count 1:
  // emit the one segment directly).
  std::uint32_t* route_entry_count_;
  std::uint64_t* route_single_entry_;
  std::uint32_t* route_single_flat_;
};

#undef kInvalid
#undef kFwd
#undef kUnreachable
#undef kIntermodalTarget

}  // namespace nigiri::routing::gpu
