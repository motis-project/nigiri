#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/mc_criteria.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/device_td.cuh"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/journey_pod.h"
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

// ============================================================================
// GPU McRAPTOR, structured as the CPU reference (raptor/mcraptor.cc) on
// fixed-capacity, lock-free structures:
//  * A label is one u64 [arr_key:16 | extras:16 | round:4 | by_route:1 | bc:27]
//    (arr_key direction-biased: smaller is better), inserted under a per-stop
//    spinlock. Bag rules as the CPU; round-k inserts never evict round-(k-1)
//    labels and footpath inserts never evict by-route labels, so the labels a
//    phase reads are stable while it writes.
//  * Breadcrumbs: append-only arena (mc_breadcrumb, as the CPU).
//  * Destination pruning: a small pareto frontier over (round, criteria).
//  * Route scan: one warp per route, a warp prefix scan over the stops'
//    boardings (sequential on overflow); rt transports one thread each.
// Capacity overflows set a canary the host checks after every query.
// ============================================================================

#define kFwd (SearchDir == direction::kForward)
#define kUnreachable (std::numeric_limits<std::uint16_t>::max())
#define kIntermodalTarget (get_special_station(special_station::kEnd))

// ---- packed label ----------------------------------------------------------
using mc_label_t = std::uint64_t;

inline constexpr auto kMcEmptySlot = ~mc_label_t{0};
inline constexpr auto kMcNoBc = std::uint32_t{(1U << 27U) - 1U};
// Capacities: all preallocated per state; exceeding one trips its canary and
// fails the query (nothing is silently dropped).
// Inline slots per stop; bags outgrowing them chain blocks from a global pool
// (bag_ovf_ = head, bag_next_ = links).
inline constexpr auto kMcBagCap = 8U;
inline constexpr auto kMcBagBlock = 16U;  // slots per chained block
// pool blocks (132 B each): n_locations / 2, clamped
inline constexpr auto kMcBagPoolMin = 1U << 20U;  // ~138 MB
inline constexpr auto kMcBagPoolMax = 1U << 22U;  // ~553 MB
// u8 hwm: inline 8 + 15 blocks; more trips the bag canary
inline constexpr auto kMcMaxBagSlots = 248U;
// breadcrumbs of one start time (16 B each): per location, clamped (DE all
// criteria needs up to ~21 per location)
inline constexpr auto kMcArenaPerLocation = 32U;
inline constexpr auto kMcArenaMin = 8'000'000U;  // 128 MB
inline constexpr auto kMcArenaMax = 32'000'000U;  // 512 MB
static_assert(kMcBagCap + kMcBagBlock <= 255U);  // u8 hwm
static_assert(kMcArenaMax < kMcNoBc);  // 27-bit breadcrumb index
// reconstructed journeys copied back with the count (more cost a 2nd sync)
inline constexpr auto kMcRecPrefix = std::uint32_t{64U};
inline constexpr auto kMcDestCap = 64U;  // dest frontier entries
// rRAPTOR range reuse (CPU RangeReuse): the bags are cleared per start time,
// so a per-stop frontier of accepted labels (bc-less) rejects candidates of
// later starts by the bag rules. Cleared per query; overflow only weakens
// pruning.
inline constexpr auto kMcReuseCap = 6U;
inline constexpr auto kMcPrefixK = 4U;  // route-bag entries per lane
inline constexpr auto kMcRouteBagCap = 64U;  // sequential / rt route bag
inline constexpr auto kSbOverflow = ~0U;  // stop bag over kMcPrefixK entries
// invalid: no pool block (bag_ovf_ / bag_next_ links, filled with 0xFF bytes),
// slot, td egress group or lookup
inline constexpr auto kMcInvalid = ~0U;

// a round-0 label: seeded stop time, minutes walked to the stop
struct mc_seed {
  location_idx_t l_;
  delta_t arr_;
  std::uint16_t walk_;
};

// device canary bits (checked by the host after each query)
inline constexpr auto kMcOverflowBag = 1U;
inline constexpr auto kMcOverflowArena = 2U;
inline constexpr auto kMcOverflowRouteBag = 4U;

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

template <direction SearchDir, mc_crit Crit>
struct mcraptor_impl {
  // Which criteria of the 16 bit extras slot (mc_extras) are live.
  static constexpr bool kHasNonTransit = has(Crit, mc_crit::non_transit);
  static constexpr bool kHasModeFilter = has(Crit, mc_crit::mode_filter);
  static constexpr bool kHasModeSwitches = has(Crit, mc_crit::mode_switches);
  static constexpr bool kHasExtras = Crit != mc_crit::arr;
  // arrival-only route bags hold a single trip
  static constexpr auto kPrefixK = kHasExtras ? kMcPrefixK : 1U;

  static constexpr auto kArrEv = kFwd ? event_type::kArr : event_type::kDep;
  static constexpr auto kDepEv = kFwd ? event_type::kDep : event_type::kArr;

  __device__ __forceinline__ bool is_better(auto a, auto b) const {
    return kFwd ? a < b : a > b;
  }
  __device__ __forceinline__ bool is_better_or_eq(auto a, auto b) const {
    return kFwd ? a <= b : a >= b;
  }
  __device__ __forceinline__ auto dir(auto a) const {
    return (kFwd ? 1 : -1) * a;
  }

  // ---- criteria (extras slot), as the CPU dimensions --------------------
  // a round-0 seed: `ingress` minutes walked to the seeded stop
  __device__ __forceinline__ static std::uint32_t crit_at_start(
      std::uint32_t const ingress) {
    return kHasNonTransit ? mc_extras::pack(0U, non_transit_add(0U, ingress),
                                            kNoModeGroup, 0U)
                          : 0U;
  }
  // after a footpath / mumo offset of `dur` minutes
  __device__ __forceinline__ static std::uint32_t crit_after_walk(
      std::uint32_t const crit, std::uint32_t const dur) {
    if constexpr (kHasNonTransit) {
      return mc_extras::pack(mc_extras::mode_filter(crit),
                             non_transit_add(mc_extras::non_transit(crit), dur),
                             mc_extras::group(crit), mc_extras::switches(crit));
    } else {
      return crit;
    }
  }
  // after riding a trip of class cl
  __device__ __forceinline__ std::uint32_t crit_after_ride(
      std::uint32_t const carried, clasz const cl) const {
    auto mf = mc_extras::mode_filter(carried);
    auto group = mc_extras::group(carried);
    auto sw = mc_extras::switches(carried);
    if constexpr (kHasModeFilter) {
      mf |= is_allowed(avoided_, cl) ? 1U : 0U;
    }
    if constexpr (kHasModeSwitches) {
      auto const g = mode_group(cl);
      sw = switches_after_ride(group, sw, g);
      group = g;
    }
    return mc_extras::pack(mf, mc_extras::non_transit(carried), group, sw);
  }

  // field-wise (mc_extras::dominates), never as integers
  __device__ __forceinline__ static bool crit_le(std::uint32_t const a,
                                                 std::uint32_t const b) {
    return mc_extras::dominates(a, b);
  }

  // pareto over (arr, crit)
  __device__ __forceinline__ static bool dominates(
      std::uint32_t const a_key,
      std::uint32_t const a_extras,
      std::uint32_t const b_key,
      std::uint32_t const b_extras) {
    return a_key <= b_key && crit_le(a_extras, b_extras);
  }

  // label v rejects a candidate (by-route ones only by by-route labels)
  __device__ __forceinline__ static bool rejects(std::uint64_t const v,
                                                 std::uint32_t const arr_key,
                                                 std::uint32_t const extras,
                                                 std::uint32_t const round,
                                                 bool const by_route) {
    return v != kMcEmptySlot && mc_round(v) <= round &&
           (!by_route || mc_by_route(v)) &&
           dominates(mc_arr_key(v), mc_extras(v), arr_key, extras);
  }
  // a candidate evicts label v (by-transfer ones only by-transfer labels)
  __device__ __forceinline__ static bool evicts(std::uint64_t const v,
                                                std::uint32_t const arr_key,
                                                std::uint32_t const extras,
                                                std::uint32_t const round,
                                                bool const by_route) {
    return v != kMcEmptySlot && mc_round(v) >= round &&
           (by_route || !mc_by_route(v)) &&
           dominates(arr_key, extras, mc_arr_key(v), mc_extras(v));
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

  // ---- bags ----------------------------------------------------------------

  // Cursor over a bag's slots: inline, then the block chain (forward reads hop
  // one block, backward jumps restart at the head). Lock-free reads are safe:
  // a read returns the value at kernel start or a later one, and a free block
  // always links to kMcInvalid (clear_bags), so the cursor never leaves the
  // chain. Slots not reachable yet read as empty.
  struct bag_cursor {
    // nullptr: not reachable (yet)
    __device__ mc_label_t* slot(std::uint32_t const i) {
      if (i < kMcBagCap) {
        return inline_ + i;
      }
      auto const ord = (i - kMcBagCap) / kMcBagBlock;
      if (blk_ == kMcInvalid || ord < ord_) {
        ord_ = 0U;
        blk_ = *head_;
      }
      while (ord_ != ord && blk_ != kMcInvalid) {
        blk_ = next_[blk_];
        ++ord_;
      }
      return blk_ == kMcInvalid
                 ? nullptr
                 : pool_ + static_cast<std::size_t>(blk_) * kMcBagBlock +
                       (i - kMcBagCap) % kMcBagBlock;
    }
    __device__ mc_label_t operator[](std::uint32_t const i) {
      auto const* const s = slot(i);
      return s == nullptr ? kMcEmptySlot : *s;
    }
    __device__ std::uint32_t size() const { return n_; }

    mc_label_t* inline_;
    mc_label_t* pool_;
    std::uint32_t const* next_;
    std::uint32_t const* head_;
    std::uint32_t n_;
    std::uint32_t ord_{0U}, blk_{kMcInvalid};
  };
  __device__ __forceinline__ bag_cursor bag_slots(std::uint32_t const l) const {
    return {.inline_ = bags_ + static_cast<std::size_t>(l) * kMcBagCap,
            .pool_ = bag_pool_,
            .next_ = bag_next_,
            .head_ = bag_ovf_ + l,
            .n_ = bag_hwm_[l]};
  }

  // rejected by an earlier start's label? racy reads only weaken pruning
  __device__ bool reuse_rejected(std::uint64_t const* const rslots,
                                 std::uint32_t const arr_key,
                                 std::uint32_t const extras,
                                 std::uint32_t const round,
                                 bool const by_route) const {
    for (auto i = 0U; i != kMcReuseCap; ++i) {
      if (rejects(rslots[i], arr_key, extras, round, by_route)) {
        return true;
      }
    }
    return false;
  }

  // adds an accepted label to the reuse frontier (under the bag lock)
  __device__ void reuse_upsert(std::uint64_t* const rslots,
                               std::uint32_t const arr_key,
                               std::uint32_t const extras,
                               std::uint32_t const round,
                               bool const by_route) {
    auto const entry = mc_pack(arr_key, extras, round, by_route, kMcNoBc);
    auto free_slot = kMcInvalid;
    for (auto i = 0U; i != kMcReuseCap; ++i) {
      if (evicts(rslots[i], arr_key, extras, round, by_route)) {
        rslots[i] = kMcEmptySlot;
      }
      if (rslots[i] == kMcEmptySlot && free_slot == kMcInvalid) {
        free_slot = i;
      }
    }
    if (free_slot != kMcInvalid) {
      rslots[free_slot] = entry;
      return;
    }
    // full: evict the latest arrival of round >= the newcomer's (any choice is
    // correct, this one keeps the broadest rejectors)
    auto worst = kMcInvalid;
    for (auto i = 0U; i != kMcReuseCap; ++i) {
      if (mc_round(rslots[i]) >= round &&
          (worst == kMcInvalid ||
           mc_arr_key(rslots[i]) > mc_arr_key(rslots[worst]))) {
        worst = i;
      }
    }
    if (worst != kMcInvalid && arr_key < mc_arr_key(rslots[worst])) {
      rslots[worst] = entry;
    }
  }

  // CPU bag_insert; true iff inserted
  __device__ bool bag_insert(std::uint32_t const l,
                             std::uint32_t const arr_key,
                             std::uint32_t const extras,
                             std::uint32_t const round,
                             bool const by_route,
                             mc_breadcrumb const* const bc_entry) {
    auto* const rslots =
        reuse_bags_ + static_cast<std::size_t>(l) * kMcReuseCap;
    if (reuse_rejected(rslots, arr_key, extras, round, by_route)) {
      return false;
    }
    // Lock-free reject pre-scan (most candidates are dominated): a stale label
    // was evicted by one that dominates the candidate too.
    {
      auto pre = bag_slots(l);
      for (auto i = 0U; i != pre.size(); ++i) {
        if (rejects(pre[i], arr_key, extras, round, by_route)) {
          return false;
        }
      }
    }
    return locked(bag_locks_ + l, [&] {
      auto bag = bag_slots(l);
      auto const hwm = bag.size();
      auto free_slot = kMcInvalid;
      for (auto i = 0U; i != hwm; ++i) {
        auto& v = *bag.slot(i);
        if (rejects(v, arr_key, extras, round, by_route)) {
          return false;
        }
        if (evicts(v, arr_key, extras, round, by_route)) {
          v = kMcEmptySlot;
        }
        if (v == kMcEmptySlot && free_slot == kMcInvalid) {
          free_slot = i;
        }
      }
      // grow, appending a pool block when the tail is full
      if (free_slot == kMcInvalid && hwm < kMcMaxBagSlots) {
        if (hwm < kMcBagCap || (hwm - kMcBagCap) % kMcBagBlock != 0U) {
          free_slot = hwm;
        } else if (auto const b = atomicAdd(bag_pool_count_, 1U);
                   b < bag_pool_cap_) {
          // the scan above left the cursor on the tail block
          if (hwm == kMcBagCap) {
            bag_ovf_[l] = b;
          } else {
            bag_next_[bag.blk_] = b;
          }
          free_slot = hwm;
        }
      }
      if (free_slot == kMcInvalid) {
        // inline + blocks full of non-dominated labels, or pool exhausted
        atomicOr(overflow_, kMcOverflowBag);
        return false;
      }
      auto bc = kMcNoBc;
      if (bc_entry != nullptr && (bc = bc_append(*bc_entry)) == kMcNoBc) {
        return false;  // arena overflow (canary set)
      }
      *bag.slot(free_slot) = mc_pack(arr_key, extras, round, by_route, bc);
      if (free_slot + 1U > hwm) {
        bag_hwm_[l] = static_cast<std::uint8_t>(free_slot + 1U);
      }
      reuse_upsert(rslots, arr_key, extras, round, by_route);
      return true;
    });
  }

  // f() under a spin lock; the fences make the previous holder's plain stores
  // visible (acquire) and publish f's (release). The holder runs f inside the
  // loop: without independent thread scheduling (pre-Volta), lanes spinning
  // after the loop would starve a holder of their own warp.
  template <typename Fn>
  __device__ static bool locked(std::uint32_t* const lock, Fn&& f) {
    while (true) {
      if (atomicCAS(lock, 0U, 1U) == 0U) {
        __threadfence();
        auto const r = f();
        __threadfence();
        atomicExch(lock, 0U);
        return r;
      }
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);  // back off from a contended lock
#endif
    }
  }

  // kMcNoBc on arena overflow (insert dropped, canary set)
  __device__ std::uint32_t bc_append(mc_breadcrumb const& entry) {
    auto const idx = atomicAdd(bc_count_, 1U);
    if (idx >= bc_cap_) {  // bc_cap_ <= kMcArenaMax < kMcNoBc
      atomicOr(overflow_, kMcOverflowArena);
      return kMcNoBc;
    }
    bc_[idx] = entry;
    return idx;
  }

  // ---- destination frontier (pruning only, persists across start times) ---

  // destination frontier: no next switch, so without the mode group margin
  __device__ __forceinline__ static bool completed_dominates(
      std::uint32_t const a_key,
      std::uint32_t const a_extras,
      std::uint32_t const b_key,
      std::uint32_t const b_extras) {
    return a_key <= b_key && mc_extras::completed_dominates(a_extras, b_extras);
  }

  __device__ bool dest_dominates(std::uint32_t const round,
                                 std::uint32_t const arr_key,
                                 std::uint32_t const extras) const {
    // cheap necessary conditions before the frontier scan
    if (dest_best_key_[round] > arr_key ||
        dest_best_total_[round] > arr_key + mc_extras::ordered(extras)) {
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
    locked(dest_lock_, [&] {
      auto free_slot = kMcInvalid;
      for (auto i = 0U; i != kMcDestCap; ++i) {
        auto const v = dest_bag_[i];
        if (v == kMcEmptySlot) {
          free_slot = free_slot == kMcInvalid ? i : free_slot;
          continue;
        }
        if (mc_round(v) <= round &&
            completed_dominates(mc_arr_key(v), mc_extras(v), arr_key, extras)) {
          return false;
        }
        if (mc_round(v) >= round &&
            completed_dominates(arr_key, extras, mc_arr_key(v), mc_extras(v))) {
          dest_bag_[i] = kMcEmptySlot;
          free_slot = free_slot == kMcInvalid ? i : free_slot;
        }
      }
      if (free_slot == kMcInvalid) {
        return false;  // full of non-dominated entries: weaker pruning only
      }
      dest_bag_[free_slot] = mc_pack(arr_key, extras, round, false, kMcNoBc);
      for (auto r = round; r != kMaxTransfers + 2U; ++r) {
        dest_best_key_[r] = min(dest_best_key_[r], arr_key);
        dest_best_total_[r] =
            min(dest_best_total_[r], arr_key + mc_extras::ordered(extras));
      }
      return true;
    });
  }

  // ---- rounds (CPU execute) ------------------------------------------------

  // round-0 labels
  __device__ void init_arrivals() {
    auto const gid = get_global_thread_id();
    if (gid == 0U) {
      *done_ = 0U;
      *any_marked_ = 0U;
    }
    for (auto i = gid; i < starts_.size(); i += get_global_stride()) {
      auto const l = to_idx(starts_[i].l_);
      if (bag_insert(l, to_key(starts_[i].arr_),
                     crit_at_start(starts_[i].walk_), 0U, false, nullptr)) {
        touched_.mark(l);
        station_mark_.mark(l);
      }
    }
  }

  // The routes / rt transports of the stops improved last round (as
  // raptor_impl): a warp per mark word, long lists by the whole warp.
  template <typename Lists>
  __device__ void mark_from_locations(Lists const& lists,
                                      device_bitvec<std::uint32_t>& marks) {
    constexpr auto const kInline = 8U;
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto const n_blocks = station_mark_.blocks_.size();
    for (auto w = get_global_thread_id() / kWarpSize; w < n_blocks;
         w += n_warps) {
      auto const bits = station_mark_.blocks_[w];
      if (bits == 0U) {  // uniform: nothing writes the marks here
        continue;
      }
      auto const my_i = w * kWarpSize + lane;
      auto n = 0U;
      if (((bits >> lane) & 1U) != 0U) {
        auto const list = lists[location_idx_t{my_i}];
        n = static_cast<unsigned>(list.size());
        if (n != 0U && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
        if (n <= kInline) {
          for (auto j = 0U; j != n; ++j) {
            marks.mark(to_idx(list[j]));
          }
          n = 0U;  // done on this lane
        }
      }
      for_each_set_bit(
          __ballot_sync(kAllLanes, n != 0U), [&](unsigned const b) {
            auto const src_i =
                __shfl_sync(kAllLanes, my_i, static_cast<int>(b));
            auto const cnt = __shfl_sync(kAllLanes, n, static_cast<int>(b));
            auto const list = lists[location_idx_t{src_i}];
            for (auto j = lane; j < cnt; j += kWarpSize) {
              marks.mark(to_idx(list[j]));
            }
          });
    }
  }

  __device__ void mark_routes() {
    mark_from_locations(tt_.location_routes_, route_mark_);
  }

  __device__ void mark_rt_transports() {
    mark_from_locations(rtt_.location_rt_transports_, rt_transport_mark_);
  }

  // CPU swap + zero_out: the routes board from prev_station_mark_
  __device__ void begin_transit_phase() {
    prev_station_mark_.swap_reset(station_mark_);
    if (get_global_thread_id() == 0U) {
      *et_task_count_ = 0U;
      *route_list_count_ = 0U;
    }
  }

  // the footpaths read the scan's marks and write station_mark_
  __device__ void begin_footpath_phase() {
    prev_station_mark_.swap_reset(station_mark_);
  }

  // ---- route scan (CPU update_route) ---------------------------------------

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

  // CPU alight: rides into l_idx at by_transport with the carried extras of
  // class cl; true iff the arrival (plus transfer buffer) was inserted.
  __device__ bool alight(unsigned const k,
                         std::uint32_t const l_idx,
                         delta_t const by_transport,
                         std::uint32_t const carried,
                         clasz const cl,
                         mc_breadcrumb bc) {
    auto const l_lb = lb_[l_idx];
    auto const ride_key = to_key(by_transport);
    auto const worst_key = to_key(worst_at_dest_);
    if (ride_key >= worst_key || l_lb == kUnreachable ||
        ride_key + l_lb >= worst_key) {
      return false;
    }
    auto const ride_extras = crit_after_ride(carried, cl);
    if (dest_dominates(k, ride_key + l_lb, ride_extras)) {
      return false;
    }
    bc.arr_ = clamp(by_transport + transfer_buffer(l_idx));
    if (!bag_insert(l_idx, to_key(bc.arr_), ride_extras, k, true, &bc)) {
      return false;
    }
    touched_.mark(l_idx);
    station_mark_.mark(l_idx);
    // the buffer is 0 at a station destination: ride_key is the arrival
    if (is_dest_[l_idx]) {
      dest_bag_add(k, ride_key, ride_extras);
    }
    return true;
  }

  // Route bag: pareto over (trip order, carried extras), merged as the CPU
  // board_from (same trip and extras: board closest to the exit).
  template <unsigned K>
  struct route_bag {
    __device__ __forceinline__ static std::uint32_t extras(
        std::uint32_t const bx) {
      return bx & 0xFFFFU;
    }
    __device__ __forceinline__ static std::uint32_t pack_bx(
        stop_idx_t const board, std::uint32_t const ex) {
      return (static_cast<std::uint32_t>(board) << 16U) | ex;
    }
    __device__ __forceinline__ std::uint32_t board(unsigned const i) const {
      return bx_[i] >> 16U;
    }

    // false if full
    __device__ bool add(std::uint32_t const et,
                        std::uint32_t const par,
                        std::uint32_t const bx) {
      auto const ex = extras(bx);
      for (auto i = 0U; i != K; ++i) {
        if (i < n_ && et_[i] == et && extras(bx_[i]) == ex) {
          par_[i] = par;
          bx_[i] = bx;
          return true;
        }
        if (i < n_ && et_[i] <= et && crit_le(extras(bx_[i]), ex)) {
          return true;  // dominated: not-earlier trip, extras no better
        }
      }
      auto w = 0U;
      for (auto i = 0U; i != K; ++i) {
        if (i < n_ && !(et <= et_[i] && crit_le(ex, extras(bx_[i])))) {
          et_[w] = et_[i];
          par_[w] = par_[i];
          bx_[w] = bx_[i];
          ++w;
        }
      }
      n_ = w;
      if (n_ == K) {
        return false;
      }
      et_[n_] = et;
      par_[n_] = par;
      bx_[n_] = bx;
      ++n_;
      return true;
    }

    // the scan operator: later boardings b absorb a (ties go to b); false on
    // overflow
    __device__ static bool combine(route_bag const& a, route_bag& b) {
      auto r = route_bag{};
      for (auto bi = 0U; bi != K; ++bi) {
        auto keep = bi < b.n_;
        for (auto ai = 0U; ai != K; ++ai) {
          if (ai < a.n_ && a.et_[ai] <= b.et_[bi] &&
              crit_le(extras(a.bx_[ai]), extras(b.bx_[bi])) &&
              !(a.et_[ai] == b.et_[bi] &&
                extras(a.bx_[ai]) == extras(b.bx_[bi]))) {
            keep = false;
          }
        }
        if (keep) {
          r.copy_from(b, bi);
        }
      }
      for (auto ai = 0U; ai != K; ++ai) {
        auto keep = ai < a.n_;
        for (auto bi = 0U; bi != K; ++bi) {
          if (bi < b.n_ && b.et_[bi] <= a.et_[ai] &&
              crit_le(extras(b.bx_[bi]), extras(a.bx_[ai]))) {
            keep = false;
          }
        }
        if (keep) {
          if (r.n_ == K) {
            return false;
          }
          r.copy_from(a, ai);
        }
      }
      b = r;
      return true;
    }

    __device__ __forceinline__ void copy_from(route_bag const& o,
                                              unsigned const i) {
      et_[n_] = o.et_[i];
      par_[n_] = o.par_[i];
      bx_[n_] = o.bx_[i];
      ++n_;
    }

    // every field through the warp shuffle f
    __device__ __forceinline__ route_bag shuffled(auto&& f) const {
      auto r = route_bag{};
      for (auto i = 0U; i != K; ++i) {
        r.et_[i] = f(et_[i]);
        r.par_[i] = f(par_[i]);
        r.bx_[i] = f(bx_[i]);
      }
      r.n_ = f(n_);
      return r;
    }
    __device__ route_bag shfl_up(unsigned const d) const {
      return shuffled(
          [&](auto const x) { return __shfl_up_sync(kAllLanes, x, d); });
    }
    __device__ route_bag bcast(int const src) const {
      return shuffled(
          [&](auto const x) { return __shfl_sync(kAllLanes, x, src); });
    }

    // stop bag f (et_run_lookups' output) <-> route bag
    __device__ __forceinline__ void load(mcraptor_impl const& m,
                                         std::uint32_t const f) {
      n_ = m.sb_n_[f];
      for (auto i = 0U; i != K; ++i) {
        if (i < n_) {
          et_[i] = m.sb_et_[f * kPrefixK + i];
          par_[i] = m.sb_par_[f * kPrefixK + i];
          bx_[i] = m.sb_bx_[f * kPrefixK + i];
        }
      }
    }
    __device__ __forceinline__ void store(mcraptor_impl const& m,
                                          std::uint32_t const f) const {
      m.sb_n_[f] = n_;
      for (auto i = 0U; i != n_; ++i) {
        m.sb_et_[f * kPrefixK + i] = et_[i];
        m.sb_par_[f * kPrefixK + i] = par_[i];
        m.sb_bx_[f * kPrefixK + i] = bx_[i];
      }
    }

    std::uint32_t et_[K];  // packed trip (pack_et: scan order)
    std::uint32_t par_[K];  // breadcrumb arena index of the boarded label
    std::uint32_t bx_[K];  // [board stop_idx:16 | carried extras:16]
    unsigned n_;
  };

  // CPU update_route's boarding conditions at stop_idx
  template <bool IsWheelchair>
  __device__ bool boardable(route_idx_t const r,
                            unsigned const n,
                            unsigned const stop_idx) const {
    if (kFwd ? stop_idx + 1U == n : stop_idx == 0U) {
      return false;  // the last stop never boards
    }
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const stp = stop{stop_seq[stop_idx]};
    auto const l = to_idx(stp.location_idx());
    return prev_station_mark_[l] && stp.can_start<SearchDir>(IsWheelchair) &&
           lb_[l] != kUnreachable && bag_hwm_[l] != 0U;
  }

  // CPU board_from at a boardable stop: the earliest trip of every
  // round-(k-1) label into the bag; false on overflow
  template <unsigned K>
  __device__ bool board(unsigned const k,
                        route_idx_t const r,
                        stop_idx_t const stop_idx,
                        route_bag<K>& bag) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const l = to_idx(stop{stop_seq[stop_idx]}.location_idx());
    auto const l_lb = lb_[l];
    // the earliest trip is monotone in the arrival (FIFO): a lookup holds for
    // the same arrival and while its trip is still catchable
    auto et = transport{};
    auto et_key = kMcInvalid;  // arrival of the last lookup
    auto et_dep_key = 0U;  // its trip's departure (0: none)
    auto labels = bag_slots(l);
    for (auto sl = 0U; sl != labels.size(); ++sl) {
      auto const lab = labels[sl];
      auto const key = mc_arr_key(lab);
      auto const ex = kHasExtras ? mc_extras(lab) : 0U;
      // not of the last round, or even the optimistic completion is dominated
      if (lab == kMcEmptySlot || mc_round(lab) != k - 1U ||
          dest_dominates(k, key + l_lb, ex)) {
        continue;
      }
      if (key != et_key && !(et_key <= key && key <= et_dep_key)) {
        auto const [day, mam] = split(from_key(key));
        et = get_earliest_transport(r, stop_idx, day, mam, l_lb);
        et_key = key;
        et_dep_key =
            et.is_valid() ? to_key(time_at_stop(r, et, stop_idx, kDepEv)) : 0U;
      }
      if (et.is_valid() && !bag.add(pack_et(r, et), mc_bc(lab),
                                    route_bag<K>::pack_bx(stop_idx, ex))) {
        return false;
      }
    }
    return true;
  }

  // CPU alight loop at scan position p
  template <bool IsWheelchair, unsigned K>
  __device__ bool alight_all(unsigned const k,
                             route_idx_t const r,
                             unsigned const n,
                             unsigned const p,
                             route_bag<K> const& bag) {
    if (p == 0U || bag.n_ == 0U) {
      return false;
    }
    auto const stop_idx = static_cast<stop_idx_t>(kFwd ? p : n - 1U - p);
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const stp = stop{stop_seq[stop_idx]};
    if (!stp.can_finish<SearchDir>(IsWheelchair)) {
      return false;
    }
    auto const l = to_idx(stp.location_idx());
    auto const cl = ride_clasz_at(r, stop_idx);
    auto any = false;
    for (auto i = 0U; i != K; ++i) {
      if (i < bag.n_) {
        auto const t = unpack_et(r, bag.et_[i]);
        any |= alight(k, l, time_at_stop(r, t, stop_idx, kArrEv),
                      route_bag<K>::extras(bag.bx_[i]), cl,
                      {.transport_ = to_idx(t.t_idx_),
                       .parent_ = bag.par_[i],
                       .day_ = t.day_,
                       .board_ = static_cast<stop_idx_t>(bag.board(i)),
                       .alight_ = stop_idx});
      }
    }
    return any;
  }

  // et phase 1 (as raptor_impl): compact the marked routes into route_list_,
  // resetting any_marked_ for the scans (nothing reads it in this kernel)
  __device__ void et_build_route_list() {
    auto const gid = get_global_thread_id();
    if (gid == 0U) {
      *any_marked_ = 0U;
    }
    for (auto w = gid; w < route_mark_.blocks_.size();
         w += get_global_stride()) {
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

  // et phase 2 (as raptor_impl): the boardable stops of the marked routes
  // into et_task_list_ (warp-aggregated compaction), the other stop bags
  // emptied
  template <bool IsWheelchair>
  __device__ void et_collect_tasks() {
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto const n_marked = *route_list_count_;
    for (auto idx = get_global_thread_id() / kWarpSize; idx < n_marked;
         idx += n_warps) {
      auto const r = route_idx_t{route_list_[idx]};
      auto const base_flat = tt_.route_stop_offset_[to_idx(r)];
      auto const n = static_cast<unsigned>(tt_.route_location_seq_[r].size());
      for (auto chunk = 0U; chunk < n; chunk += kWarpSize) {
        auto const s = chunk + lane;
        auto is_task = false;
        if (s < n) {
          sb_n_[base_flat + s] = 0U;
          is_task = boardable<IsWheelchair>(r, n, s);
        }
        auto const ballot = __ballot_sync(kAllLanes, is_task);
        if (ballot == 0U) {
          continue;
        }
        auto const leader =
            static_cast<int>(__ffs(static_cast<int>(ballot))) - 1;
        auto base_pos = 0U;
        if (lane == static_cast<unsigned>(leader)) {
          base_pos =
              atomicAdd(et_task_count_, static_cast<unsigned>(__popc(ballot)));
        }
        base_pos = __shfl_sync(kAllLanes, base_pos, leader);
        if (is_task) {
          et_task_list_[base_pos + static_cast<unsigned>(
                                       __popc(ballot & ((1U << lane) - 1U)))] =
              base_flat + s;
        }
      }
    }
  }

  // et phase 3: one thread per boardable stop fills its stop bag (up to
  // kPrefixK entries, kSbOverflow if they did not fit)
  __device__ void et_run_lookups(unsigned const k) {
    auto const n_tasks = *et_task_count_;
    for (auto i = get_global_thread_id(); i < n_tasks;
         i += get_global_stride()) {
      auto const f = et_task_list_[i];
      auto const r = route_idx_t{tt_.route_of_stop_[f]};
      auto const stop_idx =
          static_cast<stop_idx_t>(f - tt_.route_stop_offset_[to_idx(r)]);
      auto bag = route_bag<kPrefixK>{};
      if (board(k, r, stop_idx, bag)) {
        bag.store(*this, f);
      } else {
        sb_n_[f] = kSbOverflow;
      }
    }
  }

  // The route bag before position p is an associative pareto prefix over the
  // stops' boardings: a Kogge-Stone shuffle scan gives every lane its state.
  // False on a K overflow (the caller rescans: inserts are idempotent,
  // so `marked` already counts those of the earlier chunks).
  template <unsigned K, bool IsWheelchair>
  __device__ bool scan_prefix(unsigned const k,
                              route_idx_t const r,
                              unsigned const lane,
                              bool& marked) {
    using bag_t = route_bag<K>;
    auto const n = static_cast<unsigned>(tt_.route_location_seq_[r].size());
    auto const base_flat = tt_.route_stop_offset_[to_idx(r)];
    auto carry = bag_t{};
    for (auto base = 0U; base < n; base += kWarpSize) {
      auto const p = base + lane;
      auto incl = bag_t{};
      auto overflow = false;
      if (p + 1U < n) {  // the last stop never boards
        auto const f = base_flat + (kFwd ? p : n - 1U - p);
        overflow = sb_n_[f] == kSbOverflow || sb_n_[f] > K;
        if (!overflow) {
          incl.load(*this, f);
        }
      }
      // no boardings in this chunk: the state everywhere is the carry
      if (__ballot_sync(kAllLanes, incl.n_ != 0U || overflow) == 0U) {
        if (p < n) {
          marked |= alight_all<IsWheelchair>(k, r, n, p, carry);
        }
        continue;
      }
      for (auto d = 1U; d < kWarpSize; d <<= 1U) {
        auto const other = incl.shfl_up(d);
        if (lane >= d) {
          overflow |= !bag_t::combine(other, incl);
        }
      }
      auto excl = incl.shfl_up(1U);
      if (lane == 0U) {
        excl.n_ = 0U;
      }
      overflow |= !bag_t::combine(carry, excl);  // state before p
      auto next_carry = incl.bcast(static_cast<int>(kWarpSize - 1U));
      overflow |= !bag_t::combine(carry, next_carry);
      if (__ballot_sync(kAllLanes, overflow) != 0U) {
        return false;
      }
      if (p < n) {
        marked |= alight_all<IsWheelchair>(k, r, n, p, excl);
      }
      carry = next_carry;
    }
    return true;
  }

  // the CPU update_route loop, for routes overflowing the prefix scan: the
  // stop bags, looked up again only at stops that overflowed their own
  template <bool IsWheelchair>
  __device__ bool scan_sequential(unsigned const k, route_idx_t const r) {
    auto const n = static_cast<unsigned>(tt_.route_location_seq_[r].size());
    auto const base_flat = tt_.route_stop_offset_[to_idx(r)];
    auto bag = route_bag<kMcRouteBagCap>{};
    auto any = false;
    for (auto p = 0U; p != n; ++p) {
      any |= alight_all<IsWheelchair>(k, r, n, p, bag);
      if (p + 1U == n) {
        break;  // the last stop never boards
      }
      auto const f = base_flat + (kFwd ? p : n - 1U - p);
      auto ok = true;
      if (sb_n_[f] == kSbOverflow) {
        ok = board(k, r, static_cast<stop_idx_t>(kFwd ? p : n - 1U - p), bag);
      } else {
        for (auto i = 0U; i != sb_n_[f]; ++i) {
          auto const j = f * kPrefixK + i;
          ok = ok && bag.add(sb_et_[j], sb_par_[j], sb_bx_[j]);
        }
      }
      if (!ok) {
        atomicOr(overflow_, kMcOverflowRouteBag);
      }
    }
    return any;
  }

  // one warp per marked route (CPU loop_routes, as raptor_impl)
  template <bool WithClaszFilter, bool IsWheelchair>
  __device__ void loop_routes(unsigned const k) {
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto marked = false;
    for (auto i = get_global_thread_id() / kWarpSize; i < tt_.n_routes_;
         i += n_warps) {
      auto const r = route_idx_t{i};
      if (!route_mark_[i] ||
          (WithClaszFilter &&
           !is_allowed(allowed_claszes_, tt_.route_clasz_[r]))) {
        continue;
      }
      // K = 1 holds most routes, a wider scan or the sequential one the rest
      if (!scan_prefix<1U, IsWheelchair>(k, r, lane, marked) &&
          (kPrefixK == 1U ||
           !scan_prefix<kPrefixK, IsWheelchair>(k, r, lane, marked)) &&
          lane == 0U) {
        marked |= scan_sequential<IsWheelchair>(k, r);
      }
      __syncwarp();
    }
    if (__any_sync(kAllLanes, marked) && lane == 0U && !*any_marked_) {
      atomicOr(any_marked_, 1U);
    }
  }

  // ---- rt scan (CPU update_rt_transport) -----------------------------------

  // one thread per marked rt transport (CPU loop_rt_transports)
  template <bool WithClaszFilter, bool IsWheelchair>
  __device__ void update_rt_transports(unsigned const k) {
    auto marked = false;
    for (auto i = get_global_thread_id(); i < rtt_.n_rt_transports_;
         i += get_global_stride()) {
      auto const rt_t = rt_transport_idx_t{i};
      if (rt_transport_mark_[i] &&
          (!WithClaszFilter ||
           is_allowed(allowed_claszes_, rtt_.rt_transport_clasz_[rt_t]))) {
        marked |= update_rt_transport<IsWheelchair>(k, rt_t);
      }
    }
    if (marked && !*any_marked_) {
      atomicOr(any_marked_, 1U);
    }
  }

  // update_route for an rt run: a single trip with absolute event times, so
  // the route bag reduces to the carried extras
  template <bool IsWheelchair>
  __device__ bool update_rt_transport(unsigned const k,
                                      rt_transport_idx_t const rt_t) {
    auto const stop_seq = rtt_.rt_transport_location_seq_[rt_t];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto bag = route_bag<kMcRouteBagCap>{};
    auto any = false;
    for (auto i = 0U; i != n; ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l = to_idx(stp.location_idx());

      if (i != 0U && bag.n_ != 0U && stp.can_finish<SearchDir>(IsWheelchair)) {
        auto const by_transport = rt_time_at_stop(rt_t, stop_idx, kArrEv);
        auto const cl = rt_ride_clasz_at(rt_t, stop_idx);
        for (auto b = 0U; b != bag.n_; ++b) {
          any |= alight(k, l, by_transport,
                        route_bag<kMcRouteBagCap>::extras(bag.bx_[b]), cl,
                        {.transport_ = to_idx(rt_t),
                         .parent_ = bag.par_[b],
                         .day_ = day_idx_t::invalid(),
                         .board_ = static_cast<stop_idx_t>(bag.board(b)),
                         .alight_ = stop_idx});
        }
      }

      if (i == n - 1U || !stp.can_start<SearchDir>(IsWheelchair) ||
          !prev_station_mark_[l]) {
        continue;
      }
      auto const l_lb = lb_[l];
      if (l_lb == kUnreachable) {
        break;
      }
      auto const dep_key = to_key(rt_time_at_stop(rt_t, stop_idx, kDepEv));
      auto labels = bag_slots(l);
      for (auto sl = 0U; sl != labels.size(); ++sl) {
        auto const lab = labels[sl];
        if (lab == kMcEmptySlot || mc_round(lab) != k - 1U ||
            mc_arr_key(lab) > dep_key) {
          continue;  // not of the last round, or misses this run
        }
        auto const ex = kHasExtras ? mc_extras(lab) : 0U;
        if (dest_dominates(k, mc_arr_key(lab) + l_lb, ex)) {
          continue;
        }
        if (!bag.add(0U, mc_bc(lab),
                     route_bag<kMcRouteBagCap>::pack_bx(stop_idx, ex))) {
          atomicOr(overflow_, kMcOverflowRouteBag);
          break;
        }
      }
    }
    return any;
  }

  // class of a ride alighted at stop_idx, as the CPU (only read by the mode
  // criteria)
  __device__ __forceinline__ clasz
  ride_clasz_at(route_idx_t const r, stop_idx_t const stop_idx) const {
    if constexpr (kHasModeFilter || kHasModeSwitches) {
      return ride_clasz(tt_.route_section_clasz_[r], tt_.route_clasz_[r],
                        static_cast<unsigned>(kFwd ? stop_idx - 1 : stop_idx));
    } else {
      return clasz::kOther;
    }
  }
  __device__ __forceinline__ clasz rt_ride_clasz_at(
      rt_transport_idx_t const rt_t, stop_idx_t const stop_idx) const {
    if constexpr (kHasModeFilter || kHasModeSwitches) {
      return ride_clasz(rtt_.rt_transport_section_clasz_[rt_t], clasz::kOther,
                        static_cast<unsigned>(kFwd ? stop_idx - 1 : stop_idx));
    } else {
      return clasz::kOther;
    }
  }

  // ---- footpaths (CPU update_footpaths) ------------------------------------

  // a walk keeps the ride's breadcrumb, only the arrival changes
  __device__ __forceinline__ mc_breadcrumb walk_bc(std::uint32_t const te_bc,
                                                   delta_t const arr) const {
    auto bc = bc_[te_bc];
    bc.arr_ = arr;
    return bc;
  }

  // CPU relax_egress: to the intermodal target
  __device__ void relax_egress(unsigned const k,
                               std::uint32_t const dur,
                               std::uint32_t const walk,
                               delta_t const te_arr,
                               std::uint32_t const te_extras,
                               std::uint32_t const te_bc) {
    auto const end_arr = clamp(te_arr + dir(static_cast<int>(dur)));
    auto const end_key = to_key(end_arr);
    if (end_key >= to_key(worst_at_dest_)) {
      return;  // window bound
    }
    auto const end_extras = crit_after_walk(te_extras, walk);
    auto const bc = walk_bc(te_bc, end_arr);
    if (bag_insert(to_idx(kIntermodalTarget), end_key, end_extras, k, false,
                   &bc)) {
      touched_.mark(to_idx(kIntermodalTarget));
      dest_bag_add(k, end_key, end_extras);
    }
  }

  // CPU relax_fp: td durations (incl. waiting) verbatim, static ones adjusted
  __device__ void relax_fp(unsigned const k,
                           std::uint32_t const source,
                           std::uint32_t const target,
                           int const dur,
                           std::uint32_t const walk,
                           delta_t const te_arr,
                           std::uint32_t const te_extras,
                           std::uint32_t const te_bc) {
    if (target == source) {
      return;
    }
    auto const fp_arr = clamp(te_arr + dir(dur));
    auto const fp_key = to_key(fp_arr);
    auto const target_lb = lb_[target];
    auto const worst_key = to_key(worst_at_dest_);
    if (fp_key >= worst_key || target_lb == kUnreachable ||
        fp_key + target_lb >= worst_key) {
      return;
    }
    auto const fp_extras = crit_after_walk(te_extras, walk);
    if (dest_dominates(k, fp_key + target_lb, fp_extras)) {
      return;
    }
    auto const bc = walk_bc(te_bc, fp_arr);
    if (!bag_insert(target, fp_key, fp_extras, k, false, &bc)) {
      return;
    }
    touched_.mark(target);
    station_mark_.mark(target);
    if (is_dest_[target]) {
      dest_bag_add(k, fp_key, fp_extras);
    }
  }

  // index of l in the sorted td egress list (kMcInvalid = none)
  __device__ __forceinline__ std::uint32_t td_dest_group_of(
      location_idx_t const l) const {
    auto lo = std::uint32_t{0U};
    auto hi = static_cast<std::uint32_t>(td_dest_locs_.size());
    while (lo < hi) {
      auto const mid = lo + (hi - lo) / 2U;
      if (to_idx(td_dest_locs_[mid]) < to_idx(l)) {
        lo = mid + 1U;
      } else {
        hi = mid;
      }
    }
    return (lo < td_dest_locs_.size() && td_dest_locs_[lo] == l) ? lo
                                                                 : kMcInvalid;
  }

  __device__ __forceinline__ bool has_td_fps(location_idx_t const l) const {
    if (rtt_.td_ == nullptr) {
      return false;
    }
    auto const& bv =
        kFwd ? rtt_.td_->has_out_[prf_idx_] : rtt_.td_->has_in_[prf_idx_];
    return !bv.blocks_.empty() && bv[to_idx(l)];
  }

  __device__ __forceinline__ auto static_fps(location_idx_t const l) const {
    return kFwd ? tt_.footpaths_out_[prf_idx_][l]
                : tt_.footpaths_in_[prf_idx_][l];
  }

  // Egress and footpaths from stop i's round-k by-route arrivals. The static
  // footpaths are strided (first, step) so a warp can share a hub; the rest
  // is done by first == 0.
  __device__ void relax_stop(unsigned const k,
                             std::uint32_t const i,
                             unsigned const first,
                             unsigned const step) {
    auto const l = location_idx_t{i};
    auto const use_td_fps = has_td_fps(l);  // replace the static ones
    auto const fps = static_fps(l);
    auto const n_fps = use_td_fps ? 0U : static_cast<unsigned>(fps.size());
    auto const egress_ok =
        is_intermodal_dest() && dist_to_end_[i] != kUnreachable;
    auto const td_grp = is_intermodal_dest() ? td_dest_group_of(l) : kMcInvalid;
    if (n_fps == 0U && !egress_ok && td_grp == kMcInvalid && !use_td_fps) {
      return;
    }
    auto const buf = transfer_buffer(i);
    auto labels = bag_slots(i);
    // a hub's lanes share the trip count (by-route labels lie below every
    // lane's hwm) and reconverge per label, so the label reads coalesce
    auto const warp = step != 1U;
    auto const n =
        warp ? __shfl_sync(kAllLanes, labels.size(), 0) : labels.size();
    for (auto sl = 0U; sl != n; ++sl) {
      if (warp) {
        __syncwarp();
      }
      auto const lab = labels[sl];
      if (lab == kMcEmptySlot || mc_round(lab) != k || !mc_by_route(lab)) {
        continue;
      }
      // without the transfer buffer
      auto const te_arr = clamp(from_key(mc_arr_key(lab)) - buf);
      auto const te_extras = kHasExtras ? mc_extras(lab) : 0U;
      auto const te_bc = mc_bc(lab);
      if (first == 0U) {
        if (egress_ok) {
          relax_egress(k, dist_to_end_[i], dist_to_end_[i], te_arr, te_extras,
                       te_bc);
        }
        // td durations depend on the arrival, so they are evaluated per label
        if (td_grp != kMcInvalid) {
          auto const offsets = td_dest_[td_dest_group_idx_t{td_grp}];
          auto const rr = d_get_td_duration<SearchDir>(
              offsets, 0U, static_cast<std::uint32_t>(offsets.size()),
              to_unix(te_arr));
          if (rr.valid_) {
            relax_egress(
                k, static_cast<std::uint32_t>(rr.duration_.count()),
                static_cast<std::uint32_t>(offsets[rr.idx_].duration_.count()),
                te_arr, te_extras, te_bc);
          }
        }
        if (use_td_fps) {
          d_for_each_td_footpath<SearchDir>(
              kFwd ? rtt_.td_->out_[prf_idx_][l] : rtt_.td_->in_[prf_idx_][l],
              to_unix(te_arr),
              [&](location_idx_t const target, duration_t const d,
                  duration_t const walk) {
                relax_fp(k, i, to_idx(target), static_cast<int>(d.count()),
                         static_cast<std::uint32_t>(walk.count()), te_arr,
                         te_extras, te_bc);
              });
        }
      }
      for (auto f = first; f < n_fps; f += step) {
        auto const dur = adjusted_transfer_time(transfer_time_settings_,
                                                fps[f].duration().count());
        relax_fp(k, i, to_idx(fps[f].target()), dur,
                 static_cast<std::uint32_t>(dur), te_arr, te_extras, te_bc);
      }
    }
  }

  // one warp per stop mark word: a lane per stop, hubs by the whole warp
  __device__ void update_footpaths(unsigned const k) {
    constexpr auto const kWarpFpThreshold = 4U;
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto const n_words =
        static_cast<unsigned>(prev_station_mark_.blocks_.size());
    for (auto w = get_global_thread_id() / kWarpSize; w < n_words;
         w += n_warps) {
      auto const bits = prev_station_mark_.blocks_[w];
      if (bits == 0U) {  // uniform: nothing writes prev_station_mark_ here
        continue;
      }
      // the scan's stops stay marked for the next round (CPU: no swap before
      // the footpaths; raptor_impl re-marks them by the same-stop transfer)
      if (lane == 0U) {
        atomicOr(&station_mark_.blocks_[w], bits);
      }
      auto const i = w * kWarpSize + lane;
      auto const mine = ((bits >> lane) & 1U) != 0U && bag_hwm_[i] != 0U;
      auto const hub = mine && !has_td_fps(location_idx_t{i}) &&
                       static_fps(location_idx_t{i}).size() > kWarpFpThreshold;
      if (mine && !hub) {
        relax_stop(k, i, 0U, 1U);
      }
      for_each_set_bit(__ballot_sync(kAllLanes, hub), [&](unsigned const b) {
        relax_stop(k, w * kWarpSize + b, lane, kWarpSize);
      });
    }
  }

  // clear the touched bags + arena between start times (selective sweep)
  __device__ void clear_bags() {
    auto const gid = get_global_thread_id();
    if (gid == 0U) {
      *bc_count_ = 0U;
    }
    for (auto w = gid; w < touched_.blocks_.size(); w += get_global_stride()) {
      auto const word = touched_.blocks_[w];
      if (word == 0U) {
        continue;
      }
      for_each_set_bit(word, [&](unsigned const b) {
        auto const l = w * 32U + b;
        auto* const slots = bags_ + static_cast<std::size_t>(l) * kMcBagCap;
        for (auto s = 0U; s != umin(bag_hwm_[l], kMcBagCap); ++s) {
          slots[s] = kMcEmptySlot;
        }
        // free the chain: blocks back to empty slots, linking to kMcInvalid
        for (auto blk = bag_ovf_[l]; blk != kMcInvalid;) {
          auto* const p =
              bag_pool_ + static_cast<std::size_t>(blk) * kMcBagBlock;
          for (auto s = 0U; s != kMcBagBlock; ++s) {
            p[s] = kMcEmptySlot;
          }
          auto const next = bag_next_[blk];
          bag_next_[blk] = kMcInvalid;
          blk = next;
        }
        bag_ovf_[l] = kMcInvalid;
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
    {
      // as the CPU dimensions' apply_to()
      auto const crit = kHasExtras ? mc_extras(lab) : 0U;
      out->criteria_cost_ = static_cast<std::uint16_t>(
          mc_extras::non_transit(crit) * kNonTransitBin);
      out->criteria_mode_filter_ =
          static_cast<std::uint8_t>(mc_extras::mode_filter(crit));
      out->criteria_mode_switches_ =
          static_cast<std::uint8_t>(mc_extras::switches(crit));
    }

    auto cur_l = dest_l;
    auto li = mc_bc(lab);
    auto n = 0U;
    while (li != kMcNoBc) {
      auto const& bc = bc_[li];
      auto const cur_arr = bc.arr_;
      auto const is_rt = bc.day_ == day_idx_t::invalid();
      auto const t_idx =
          is_rt ? transport_idx_t::invalid() : transport_idx_t{bc.transport_};
      auto const rt_t = is_rt ? rt_transport_idx_t{bc.transport_}
                              : rt_transport_idx_t::invalid();
      auto const board = bc.board_;
      auto const alight = bc.alight_;
      auto const day = is_rt ? day_idx_t{0U} : bc.day_;

      auto train_arr = delta_t{};
      auto dep_at_board = delta_t{};
      auto board_loc = location_idx_t::invalid();
      auto alight_loc = location_idx_t::invalid();
      if (is_rt) {
        // rt event times are absolute: no traffic-day recovery
        auto const stop_seq = rtt_.rt_transport_location_seq_[rt_t];
        board_loc = stop{stop_seq[board]}.location_idx();
        alight_loc = stop{stop_seq[alight]}.location_idx();
        train_arr = rt_time_at_stop(rt_t, alight, kArrEv);
        dep_at_board = rt_time_at_stop(rt_t, board, kDepEv);
      } else {
        auto const r = tt_.transport_route_[t_idx];
        train_arr = time_at_stop(r, transport{t_idx, day}, alight, kArrEv);
        dep_at_board = time_at_stop(r, transport{t_idx, day}, board, kDepEv);
        auto const stop_seq = tt_.route_location_seq_[r];
        board_loc = stop{stop_seq[board]}.location_idx();
        alight_loc = stop{stop_seq[alight]}.location_idx();
      }

      if (is_intermodal_dest() && cur_l == kIntermodalTarget) {
        // the last mile is the host's offset leg
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
      lg.rt_transport_ = rt_t;
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

  // ---- helpers as in raptor_impl ----------------------------------------

  __device__ transport
  get_earliest_transport(route_idx_t const r,
                         stop_idx_t const stop_idx,
                         day_idx_t const day_at_stop,
                         minutes_after_midnight_t const mam_at_stop,
                         std::uint16_t const l_lb) {
    auto const event_times = tt_.event_times_at_stop(r, stop_idx, kDepEv);
    auto const n = static_cast<unsigned>(event_times.size());

    // transport offset of the i-th event in search direction
    auto const t_offset_at = [&](unsigned const i) {
      return kFwd ? i : n - 1U - i;
    };

    // day 0: the first event at or after the time at the stop (FIFO routes,
    // so the column is sorted by mam)
    auto const seek_first_day = [&]() {
      auto lo = 0U;
      auto hi = n;
      while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2U;
        if (is_better(event_times[t_offset_at(mid)].mam(),
                      mam_at_stop.count())) {
          lo = mid + 1U;
        } else {
          hi = mid;
        }
      }
      return lo;
    };

    constexpr auto const kNDaysToIterate = static_cast<day_idx_t::value_t>(
        kMaxTravelTime / std::chrono::days{1} + 1U);
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      if (!is_route_active(r, day)) {
        continue;
      }

      for (auto pos = i == 0U ? seek_first_day() : 0U; pos < n; ++pos) {
        auto const t_offset = t_offset_at(pos);
        auto const ev = event_times[t_offset];
        auto const ev_mam = ev.mam();

        // window pruning with the lb projection (CPU form)
        if (is_better_or_eq(worst_at_dest_,
                            to_delta(day, ev_mam) + dir(l_lb))) {
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = event_times.transport(t_offset);
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const start_day = static_cast<int>(day.v_) - ev.days();
        if (!is_transport_active(t, static_cast<std::size_t>(start_day))) {
          continue;
        }
        return {t, static_cast<day_idx_t>(start_day)};
      }
    }
    return {};
  }

  __device__ __forceinline__ bool is_transport_active(
      transport_idx_t const t, std::size_t const day) const {
    // kRtBitfieldFlag entries resolve into rtt_.bitfields_ (as in raptor_impl)
    auto const i = to_idx(tt_.transport_traffic_days_[t]);
    return ((i & kRtBitfieldFlag) != 0U
                ? rtt_.bitfields_[bitfield_idx_t{i & ~kRtBitfieldFlag}]
                : tt_.bitfields_[bitfield_idx_t{i}])
        .test(day);
  }

  __device__ __forceinline__ bool is_route_active(route_idx_t const r,
                                                  day_idx_t const day) const {
    return static_cast<int>(day.v_) >= 0 &&
           tt_.bitfields_[tt_.route_traffic_days_[r]].test(
               static_cast<std::size_t>(static_cast<int>(day.v_)));
  }

  // packed trip [rel_day:6 | t_offset:26], ordered as unsigned in scan order
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
        to_idx(t.t_idx_) - to_idx(tt_.route_times_[r].transports_.from_);
    auto const x = (rel_day << kEtRelDayShift) | t_offset;
    return kFwd ? x : kEtReverseBase - x;
  }

  __device__ __forceinline__ transport unpack_et(route_idx_t const r,
                                                 std::uint32_t const p) const {
    auto const x = kFwd ? p : kEtReverseBase - p;
    auto const rel_day = x >> kEtRelDayShift;
    auto const t_offset = x & ((1U << kEtRelDayShift) - 1U);
    return transport{
        transport_idx_t{to_idx(tt_.route_times_[r].transports_.from_) +
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

  // rt event times are stored absolute on the rt base day: no traffic day
  __device__ delta_t rt_time_at_stop(rt_transport_idx_t const rt_t,
                                     stop_idx_t const stop_idx,
                                     event_type const ev_type) const {
    return to_delta(rtt_.base_day_idx_,
                    rtt_.event_time(rt_t, stop_idx, ev_type));
  }

  __device__ date::sys_days base() const {
    return tt_.internal_interval_days().from_ +
           static_cast<int>(base_.v_) * date::days{1};
  }

  __device__ unixtime_t to_unix(delta_t const t) const {
    return delta_to_unix(base(), t);
  }

  __device__ delta_t to_delta(day_idx_t const day,
                              std::int16_t const mam) const {
    return clamp(
        (static_cast<int>(day.v_) - static_cast<int>(base_.v_)) * 1440 + mam);
  }

  __device__ std::pair<day_idx_t, minutes_after_midnight_t> split(
      delta_t const x) const {
    return split_day_mam(base_, x);
  }

  // ---- members ----------------------------------------------------------

  std::uint32_t* any_marked_;
  std::uint32_t* done_;
  std::uint32_t* overflow_;
  device_timetable tt_;
  // empty (n_rt_transports_ == 0) unless the query carries an rt_timetable
  device_rt_timetable rtt_;
  transfer_time_settings transfer_time_settings_;
  clasz_mask_t allowed_claszes_;
  clasz_mask_t avoided_;  // mode_filter: the CPU's mode_filter_dim mask
  profile_idx_t prf_idx_;
  day_idx_t base_;
  delta_t worst_at_dest_;

  cuda::std::span<mc_seed const> starts_;

  device_bitvec<std::uint64_t const> is_dest_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  // td egress offsets (q.td_dest_), sparse groups; empty = none
  cuda::std::span<location_idx_t const> td_dest_locs_;
  d_vecvec_view<td_dest_offsets_t> td_dest_;
  // lower bounds to the destination (pruning only; projection = key + lb)
  cuda::std::span<std::uint16_t const> lb_;

  mc_label_t* bags_;  // n_locations x kMcBagCap inline slots
  mc_label_t* bag_pool_;  // bag_pool_cap_ x kMcBagBlock chained blocks
  std::uint32_t* bag_ovf_;  // n_locations chain head (~0 = none)
  std::uint32_t* bag_next_;  // bag_pool_cap_ chain links (~0 = end / free)
  std::uint32_t* bag_pool_count_;  // bump allocator
  std::uint32_t bag_pool_cap_;
  std::uint8_t* bag_hwm_;  // per bag used-slot bound (under the bag lock)
  std::uint32_t* bag_locks_;  // n_locations spinlocks
  std::uint64_t* reuse_bags_;  // n_locations x kMcReuseCap, per query
  mc_label_t* dest_bag_;  // kMcDestCap
  std::uint32_t* dest_lock_;
  // per round: min key / min key + extras of the dest frontier (racy reads
  // only weaken pruning)
  std::uint32_t* dest_best_key_;
  std::uint32_t* dest_best_total_;
  mc_breadcrumb* bc_;  // arena
  std::uint32_t* bc_count_;
  std::uint32_t bc_cap_;

  device_bitvec<std::uint32_t> touched_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;
  device_bitvec<std::uint32_t> rt_transport_mark_;

  // marked routes this round, boardable route stops (et phase)
  cuda::std::span<std::uint32_t> route_list_;
  std::uint32_t* route_list_count_;
  cuda::std::span<std::uint32_t> et_task_list_;
  std::uint32_t* et_task_count_;

  // per route stop (route_of_stop_ order): boarding count and kMcPrefixK
  // route-bag entries (et_run_lookups -> loop_routes)
  std::uint32_t* sb_n_;
  std::uint32_t* sb_et_;
  std::uint32_t* sb_par_;
  std::uint32_t* sb_bx_;
};

#undef kFwd
#undef kUnreachable
#undef kIntermodalTarget

}  // namespace nigiri::routing::gpu
