#pragma once

#include "nigiri/routing/component_graph.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

// Device views of routing::component_graph (see component_graph.h).
struct device_component_graph {
  std::uint32_t n_components_{0U};
  d_vecmap_view<location_idx_t, component_idx_t> location_component_;
  d_vecvec_view<vecvec<comp_route_idx_t, component_idx_t>> seqs_;
  d_vecvec_view<vecvec<comp_route_idx_t, std::uint16_t>> durations_;
  d_vecvec_view<vecvec<component_idx_t, comp_route_idx_t>> comp_routes_;
};

// GPU port of routing::compute_component_lb (component_graph.cc is the
// reference implementation): a time-independent RAPTOR on the component
// graph. One round per boarding; tt_cur_[c] = travel time lower bound,
// ic_[c] = round of the first finite value = boarding count lower bound.
// The result is packed into comp_lb_[c] = [tt:16 | ic:16]
// (0xFFFFFFFF = unreachable) consumed by raptor_impl::lb_allows_*.
template <direction SearchDir>
struct component_lb_impl {
  static constexpr auto const kInf = ~std::uint32_t{0U};

  __device__ __forceinline__ void seed(component_idx_t const c,
                                       std::uint32_t const val) {
    atomicMin(&tt_cur_[to_idx(c)], val);
    ic_[to_idx(c)] = 0U;
    comp_marked_.mark(to_idx(c));
  }

  __device__ void init() {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    if (gid == 0U) {
      *lb_done_ = 0U;
    }
    for (auto i = gid; i < n_locations_; i += stride) {
      auto const l = location_idx_t{i};
      if (is_dest_[i]) {
        seed(g_.location_component_[l], 0U);
      }
      if (!dist_to_end_.empty() && dist_to_end_[i] != kUnreachable) {
        seed(g_.location_component_[l], dist_to_end_[i]);
      }
    }
    for (auto i = gid; i < td_dest_locs_.size(); i += stride) {
      seed(g_.location_component_[td_dest_locs_[i]], 0U);
    }
  }

  // marks the routes of all components improved in the previous round;
  // comp_marked_ is cleared by the host (memset) after this kernel
  __device__ void mark_routes() {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    if (gid == 0U) {
      *any_improved_ = 0U;
    }
    for (auto w = gid; w < comp_marked_.blocks_.size(); w += stride) {
      auto const word = comp_marked_.blocks_[w];
      if (word == 0U) {
        continue;
      }
      for_each_set_bit(word, [&](unsigned const b) {
        auto const c =
            component_idx_t{static_cast<std::uint32_t>(w * 32U + b)};
        for (auto const cr : g_.comp_routes_[c]) {
          route_marked_.mark(to_idx(cr));
        }
      });
    }
  }

  __device__ __forceinline__ void relax(component_idx_t const c,
                                        std::uint32_t const val,
                                        unsigned const k) {
    auto const old = atomicMin(&tt_cur_[to_idx(c)], val);
    if (val < old) {
      if (old == kInf) {
        ic_[to_idx(c)] = k;  // first finite value -> boarding count lb
      }
      comp_marked_.mark(to_idx(c));
      if (!*any_improved_) {
        atomicOr(any_improved_, 1U);
      }
    }
  }

  // one component route per thread; reads only the previous round's values
  // (tt_prev_) so ic_ counts boardings exactly
  __device__ void scan(unsigned const k) {
    constexpr auto const kFwdLb = SearchDir == direction::kForward;
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    auto const n_routes = static_cast<std::uint32_t>(g_.seqs_.size());
    for (auto i = gid; i < n_routes; i += stride) {
      if (!route_marked_.test(i)) {
        continue;
      }
      auto const cr = comp_route_idx_t{i};
      auto const seq = g_.seqs_[cr];
      auto const durs = g_.durations_[cr];
      auto const len = static_cast<unsigned>(seq.size());
      auto carry = kInf;
      if constexpr (kFwdLb) {
        // bounds measure the distance TO the seeds: board at j-1, ride
        // forward, alight at any j' >= j, continue with tt_prev_[c_j']
        for (auto j = len - 1U; j != 0U; --j) {
          auto const m = min(tt_prev_[to_idx(seq[j])], carry);
          if (m != kInf) {
            carry = m + durs[j - 1U];
            relax(seq[j - 1U], carry, k);
          }
        }
      } else {
        // distance FROM the seeds: mirrored
        for (auto j = 0U; j != len - 1U; ++j) {
          auto const m = min(tt_prev_[to_idx(seq[j])], carry);
          if (m != kInf) {
            carry = m + durs[j];
            relax(seq[j + 1U], carry, k);
          }
        }
      }
    }
  }

  __device__ void check_done() {
    if (get_global_thread_id() == 0U && *any_improved_ == 0U) {
      *lb_done_ = 1U;
    }
  }

  // pack [tt:16|ic:16]; values beyond the travel time limit or never
  // reached -> 0xFFFFFFFF ("prune unconditionally")
  __device__ void finalize(std::uint32_t const max_travel_time) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    for (auto c = gid; c < g_.n_components_; c += stride) {
      auto const t = tt_cur_[c];
      comp_lb_[c] = (t == kInf || t > max_travel_time)
                        ? kInf
                        : ((t << 16U) | (ic_[c] & 0xFFFFU));
    }
  }

  __device__ __forceinline__ static std::uint32_t min(std::uint32_t const a,
                                                      std::uint32_t const b) {
    return a < b ? a : b;
  }

  device_component_graph g_;
  std::uint32_t n_locations_;

  // query (destination side of the search consuming the bounds)
  device_bitvec<std::uint64_t const> is_dest_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  cuda::std::span<location_idx_t const> td_dest_locs_;

  // state
  cuda::std::span<std::uint32_t> tt_cur_;
  cuda::std::span<std::uint32_t> tt_prev_;
  cuda::std::span<std::uint32_t> ic_;
  device_bitvec<std::uint32_t> comp_marked_;
  device_bitvec<std::uint32_t> route_marked_;
  std::uint32_t* any_improved_;
  std::uint32_t* lb_done_;

  // output
  cuda::std::span<std::uint32_t> comp_lb_;
};

}  // namespace nigiri::routing::gpu
