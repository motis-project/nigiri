#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/it_range.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/device_td.cuh"
#include "nigiri/routing/gpu/device_times.h"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/journey_pod.h"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

#ifndef NIGIRI_CUDA_DEBUG
#define debug(...)
#else
#define debug(...) printf(__VA_ARGS__)
#endif

#define kInvalid (kInvalidDelta<SearchDir>)
#define kFwd (SearchDir == direction::kForward)
#define kBwd (SearchDir == direction::kBackward)
#define kUnreachable (std::numeric_limits<std::uint16_t>::max())
#define kIntermodalTarget (get_special_station(special_station::kEnd))

using td_dest_group_idx_t = cista::strong<std::uint32_t, struct td_dest_group_>;
using td_dest_offsets_t = vecvec<td_dest_group_idx_t, td_offset>;

template <direction SearchDir>
struct raptor_impl {
  static constexpr via_offset_t Vias = 0U;

  __device__ __forceinline__ bool is_better(auto a, auto b) {
    return kFwd ? a < b : a > b;
  }
  __device__ __forceinline__ bool is_better_or_eq(auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  }
  __device__ __forceinline__ auto get_best(auto a, auto b) {
    return is_better(a, b) ? a : b;
  }
  __device__ __forceinline__ auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  __device__ __forceinline__ auto min(auto x, auto y) { return x <= y ? x : y; }
  __device__ __forceinline__ auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  __device__ void init_arrivals(unixtime_t const worst_time_at_dest) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    if (global_t_id == 0U) {
      *done_ = 0U;
    }

    for (auto i = global_t_id; i < starts_.size(); i += global_stride) {
      auto const l = starts_[i].first;
      auto const t = unix_to_delta(base(), starts_[i].second);
      auto const v = via_offset_t{0};
      best_.update_min(l, v, t);
      round_times_.update_min(0U, l, v, t, make_start_bc());
      station_mark_.mark(to_idx(l));
    }

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto i = global_t_id; i < kMaxTransfers + 2U; i += global_stride) {
      time_at_dest_.update_min(i, d_worst_at_dest);
    }
  }

  __device__ void reuse_previous_arrivals(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
      debug("round %u: location %d / %u\n", k, i, tt_.n_locations_);
      auto const l = location_idx_t{i};
      for (auto v = 0U; v != Vias + 1; ++v) {
        best_.update_min(l, v, round_times_.get(k, l, v));
      }
      if (is_dest_[i]) {
        update_time_at_dest(k, best_.get(l, Vias));
      }
    }
    if (global_t_id == 0U) {
      *any_marked_ = 0U;
    }
  }

  __device__ void mark_routes(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
      if (station_mark_[i]) {
        if (!tt_.location_routes_[location_idx_t{i}].empty() && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
        for (auto r : tt_.location_routes_[location_idx_t{i}]) {
          debug("round %u: marking route %u\n", k, to_idx(r));
          route_mark_.mark(to_idx(r));
        }
      }
    }
  }

  __device__ void mark_rt_transports(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
      if (station_mark_[i]) {
        auto const rt_transports =
            rtt_.location_rt_transports_[location_idx_t{i}];
        if (!rt_transports.empty() && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
        for (auto const rt_t : rt_transports) {
          debug("round %u: marking rt transport %u\n", k, to_idx(rt_t));
          rt_transport_mark_.mark(to_idx(rt_t));
        }
      }
    }
  }

  __device__ void begin_transit_phase() {
    prev_station_mark_.swap_reset(station_mark_);
    if (get_global_thread_id() == 0U) {
      *et_task_count_ = 0U;
      *route_list_count_ = 0U;
    }
  }

  __device__ void begin_footpath_phase() {
    prev_station_mark_.swap_reset(station_mark_);
  }

  __device__ void reconstruct_journey(location_idx_t const dest,
                                      unsigned const K,
                                      gpu_journey* out) {
    out->state_ = reconstruction_result::kNotReconstructed;
    auto cur_v = static_cast<via_offset_t>(Vias);
    auto const dest_time =
        round_times_.get(static_cast<std::uint8_t>(K), dest, cur_v);
    if (dest_time == kInvalid) {
      return;
    }
    out->dest_l_ = dest;
    out->dest_time_ = dest_time;
    out->transfers_ = static_cast<std::uint8_t>(K - 1U);

    auto const ev_arr_type = kFwd ? event_type::kArr : event_type::kDep;
    auto const ev_dep_type = kFwd ? event_type::kDep : event_type::kArr;

    auto cur_l = dest;
    auto cur_k = K;
    auto n = 0U;
    while (cur_k >= 1U) {
      auto const bc =
          round_times_.get_bc(static_cast<std::uint8_t>(cur_k), cur_l, cur_v);

      if (bc_is_start(bc)) {
        break;
      }

      auto const bc_t = bc_transport(bc);
      auto const board = static_cast<stop_idx_t>(bc_board(bc));
      auto const alight = static_cast<stop_idx_t>(bc_alight(bc));
      auto const arr_at_cur =
          round_times_.get(static_cast<std::uint8_t>(cur_k), cur_l, cur_v);

      auto const is_rt = is_rt_bc_transport(bc_t, rtt_.n_rt_transports_);
      auto day = day_idx_t{0U};
      auto train_arr = kInvalid;
      auto dep_at_board = kInvalid;
      auto board_loc = location_idx_t::invalid();
      auto alight_loc = location_idx_t::invalid();

      if (is_rt) {
        // rt transport: event times are stored absolute (relative to the rt
        // base day) -> exact, no traffic-day recovery needed
        auto const rt_t = rt_transport_idx_t{decode_rt_bc_transport(bc_t)};
        auto const stop_seq = rtt_.rt_transport_location_seq_[rt_t];
        board_loc = stop{stop_seq[board]}.location_idx();
        alight_loc = stop{stop_seq[alight]}.location_idx();
        train_arr = rt_time_at_stop(rt_t, alight, ev_arr_type);
        dep_at_board = rt_time_at_stop(rt_t, board, ev_dep_type);
      } else {
        auto const t_idx = transport_idx_t{bc_t};
        auto const r = tt_.transport_route_[t_idx];

        // Longest transfer/footpath/offset <24h
        // -> at most one midnight crossing
        // -> iterate 2 days
        auto const event_mam_full =
            tt_.event_mam(r, t_idx, alight, ev_arr_type).count();
        auto const [arr_day, _] = split(arr_at_cur);
        auto found_day = false;
        for (auto off = 0; off != 2; ++off) {
          auto const cand =
              as_int(arr_day) - event_mam_full / 1440 - (kFwd ? off : -off);
          if (cand < 0) {
            continue;
          }
          if (!is_transport_active(t_idx, static_cast<std::size_t>(cand))) {
            continue;
          }
          auto const tr = transport{
              t_idx, day_idx_t{static_cast<day_idx_t::value_t>(cand)}};
          auto const ev = time_at_stop(r, tr, alight, ev_arr_type);
          if (is_better_or_eq(ev, arr_at_cur)) {
            day = day_idx_t{static_cast<day_idx_t::value_t>(cand)};
            train_arr = ev;
            found_day = true;
            break;
          }
        }
        if (!found_day) {
          out->state_ = reconstruction_result::kReconstructionFailed;
          return;
        }

        auto const tr = transport{t_idx, day};
        dep_at_board = time_at_stop(r, tr, board, ev_dep_type);
        auto const stop_seq = tt_.route_location_seq_[r];
        board_loc = stop{stop_seq[board]}.location_idx();
        alight_loc = stop{stop_seq[alight]}.location_idx();
      }

      auto const is_egress = is_intermodal_dest() && cur_l == kIntermodalTarget;
      if (is_egress) {
        // no footpath leg: the last mile alight -> kEnd is the host's mumo
        // leg; the GPU journey's terminal is the ride's alighting stop
        out->dest_l_ = alight_loc;
      } else if (n != 0U || alight_loc != cur_l ||
                 train_arr != arr_at_cur /* skip 0min last leg */) {
        // Footpaths are always emitted (even zero minute reflexive transfers).
        // Journey structure: [WALK]? TRANSIT [TRANSIT WALK]*
        if (n >= kMaxRecLegs) {
          out->state_ = reconstruction_result::kReconstructionFailed;
          return;
        }
        auto& lg = out->legs_[n++];
        lg.is_footpath_ = true;
        lg.from_l_ = alight_loc;
        lg.to_l_ = cur_l;
        lg.dep_ = train_arr;
        lg.arr_ = arr_at_cur;
        lg.fp_duration_ = static_cast<std::uint16_t>(
            kFwd ? (arr_at_cur - train_arr) : (train_arr - arr_at_cur));
      }

      // transport leg board_loc -> alight_loc
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
      lg.transport_ =
          is_rt ? transport_idx_t::invalid() : transport_idx_t{bc_t};
      lg.rt_transport_ = is_rt
                             ? rt_transport_idx_t{decode_rt_bc_transport(bc_t)}
                             : rt_transport_idx_t::invalid();
      lg.day_ = day;
      lg.enter_stop_ = board;
      lg.exit_stop_ = alight;

      cur_l = board_loc;
      cur_k -= 1U;
    }

    out->start_l_ = cur_l;
    out->n_legs_ = static_cast<std::uint8_t>(n);
    // n == 0 with a valid dest label = inconsistent chain, flag loudly
    out->state_ = (n != 0U) ? reconstruction_result::kOk
                            : reconstruction_result::kReconstructionFailed;
  }

  __device__ date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  // Combines the query's required modes (bike/car runtime flags +
  // IsWheelchair compile-time), delegating the 2-bit decode to each filter.
  // Only called from the WithFilters kernel variants.
  template <bool IsWheelchair, typename Key>
  __device__ __forceinline__ bool transport_allowed(
      device_transport_filters<Key> const& f,
      std::uint32_t const i,
      unsigned& section_mask) const {
    if (require_bike_transport_ &&
        !f.bike_.allows(i, kBikeSections, section_mask)) {
      return false;
    }
    if (require_car_transport_ &&
        !f.car_.allows(i, kCarSections, section_mask)) {
      return false;
    }
    if constexpr (IsWheelchair) {
      if (!f.wheelchair_.allows(i, kWheelchairSections, section_mask)) {
        return false;
      }
    }
    return true;
  }

  template <bool WithClaszFilter, bool IsWheelchair, bool WithFilters>
  __device__ void loop_routes(unsigned const k) {
    // One warp per route:
    // - lanes cooperate on the route's stops via update_route_warp
    // - all lanes share warp_id for the warp shuffles and __any_sync
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const warp_id = get_global_thread_id() / kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;

    for (auto i = warp_id; i < tt_.n_routes_; i += n_warps) {
      if (!route_mark_.test(i)) {
        continue;
      }

      auto const r = route_idx_t{i};
      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          continue;
        }
      }

      if constexpr (WithFilters) {
        auto section_mask = 0U;
        if (!transport_allowed<IsWheelchair>(*tt_.filters_, i, section_mask)) {
          continue;
        }
        auto const marked =
            section_mask != 0U
                ? update_route_warp<IsWheelchair, true>(k, r, lane,
                                                        section_mask)
                : update_route_warp<IsWheelchair, false>(k, r, lane, 0U);
        if (marked && lane == 0U && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
      } else {
        if (update_route_warp<false, false>(k, r, lane, 0U) && lane == 0U &&
            !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
      }
    }
  }

  template <bool WithClaszFilter, bool IsWheelchair, bool WithFilters>
  __device__ void update_rt_transports(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    for (auto i = gid; i < rtt_.n_rt_transports_; i += stride) {
      if (!rt_transport_mark_.test(i)) {
        continue;
      }

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_,
                        rtt_.rt_transport_clasz_[rt_transport_idx_t{i}])) {
          continue;
        }
      }

      if constexpr (WithFilters) {
        auto section_mask = 0U;
        if (!transport_allowed<IsWheelchair>(*rtt_.filters_, i, section_mask)) {
          continue;
        }
        auto const marked = section_mask != 0U
                                ? update_rt_transport<IsWheelchair, true>(
                                      k, rt_transport_idx_t{i}, section_mask)
                                : update_rt_transport<IsWheelchair, false>(
                                      k, rt_transport_idx_t{i}, 0U);
        if (marked && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
      } else {
        if (update_rt_transport<false, false>(k, rt_transport_idx_t{i}, 0U) &&
            !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
      }
    }
  }

  template <bool IsWheelchair, bool WithSections>
  __device__ bool update_rt_transport(
      unsigned const k,
      rt_transport_idx_t const rt_t,
      [[maybe_unused]] unsigned const section_mask) {
    auto const stop_seq = rtt_.rt_transport_location_seq_[rt_t];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto any_marked = false;

    auto et = false;
    auto boarding_stop_idx = stop_idx_t{0};
    for (auto i = 0U; i != n; ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l = stp.location_idx();
      auto const l_idx = to_idx(l);

      // the carried transport cannot cross an inaccessible section
      if constexpr (WithSections) {
        if (i != 0U && et) {
          auto const sec =
              static_cast<unsigned>(kFwd ? stop_idx - 1 : stop_idx);
          if (rtt_.filters_->section_killed(section_mask, rt_t, sec)) {
            et = false;
          }
        }
      }

      if (i != 0U && et && stop_idx <= kBcStopMask &&
          stp.can_finish<SearchDir>(IsWheelchair)) {
        auto const by_transport = rt_time_at_stop(
            rt_t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        if (is_better(by_transport, time_at_dest_.get(k))) {
          tmp_.update_min(
              l, 0U, by_transport,
              make_transport_payload(encode_rt_bc_transport(to_idx(rt_t)),
                                     boarding_stop_idx, stop_idx));
          station_mark_.mark(l_idx);
          any_marked = true;
        }
      }

      if (i == n - 1U || stop_idx > kBcStopMask ||
          !stp.can_start<SearchDir>(IsWheelchair) ||
          !prev_station_mark_[l_idx]) {
        continue;
      }

      auto const dep = rt_time_at_stop(
          rt_t, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
      if (is_better_or_eq(round_times_.get(k - 1, l, 0U), dep)) {
        et = true;
        boarding_stop_idx = stop_idx;
      }
    }
    return any_marked;
  }

  __device__ __forceinline__ void relax_footpath(unsigned const k,
                                                 footpath const fp,
                                                 delta_t const tmp_time,
                                                 breadcrumb_t const bc,
                                                 delta_t const t_at_dest) {
    relax_fp_target(
        k, fp.target(),
        adjusted_transfer_time(transfer_time_settings_, fp.duration().count()),
        tmp_time, bc, t_at_dest);
  }

  __device__ __forceinline__ void relax_fp_target(unsigned const k,
                                                  location_idx_t const target_l,
                                                  int const duration,
                                                  delta_t const tmp_time,
                                                  breadcrumb_t const bc,
                                                  delta_t const t_at_dest) {
    auto const target = to_idx(target_l);
    auto const fp_target_time = clamp(tmp_time + dir(duration));
    if (!is_better(fp_target_time, t_at_dest)) {
      return;
    }

    if (is_better(fp_target_time, best_.get(target_l, Vias))) {
      round_times_.update_min(k, target_l, Vias, fp_target_time, bc);
      best_.update_min(target_l, Vias, fp_target_time);
      station_mark_.mark(target);
      if (is_dest_[target]) {
        update_time_at_dest(k, fp_target_time);
      }
    }
  }

  __device__ __forceinline__ bool has_td_fps(location_idx_t const l) const {
    auto const& bv =
        kFwd ? rtt_.td_->has_out_[prf_idx_] : rtt_.td_->has_in_[prf_idx_];
    return !bv.blocks_.empty() && bv[to_idx(l)];
  }

  __device__ void update_td_dest_offsets(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    for (auto g = gid; g < td_dest_locs_.size(); g += stride) {
      auto const l = td_dest_locs_[g];

      if (!prev_station_mark_[to_idx(l)]) {
        continue;
      }

      auto const tmp_time = tmp_.get(l, Vias);
      if (tmp_time == kInvalid) {
        continue;
      }

      auto const offsets = td_dest_[td_dest_group_idx_t{g}];
      auto const r = d_get_td_duration<SearchDir>(
          offsets, 0U, static_cast<std::uint32_t>(offsets.size()),
          to_unix(tmp_time));
      if (!r.valid_) {
        continue;
      }

      auto const end_time =
          clamp(tmp_time + dir(static_cast<int>(r.duration_.count())));
      if (is_better(end_time, best_.get(kIntermodalTarget, Vias))) {
        auto const bc = tmp_.get_bc(0U, l, Vias);
        round_times_.update_min(k, kIntermodalTarget, Vias, end_time, bc);
        best_.update_min(kIntermodalTarget, Vias, end_time);
        update_time_at_dest(k, end_time);
      }
    }
  }

  template <bool WithTdDest, bool WithTdFootpaths>
  __device__ void update_transfers_and_footpaths(unsigned const k) {
    constexpr auto const kWarpFpThreshold = 8U;
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const warp_id = get_global_thread_id() / kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto const intermodal = is_intermodal_dest();
    auto const n_blocks =
        static_cast<unsigned>(prev_station_mark_.blocks_.size());

    if constexpr (WithTdDest) {
      if (intermodal && !td_dest_locs_.empty()) {
        update_td_dest_offsets(k);
      }
    }

    for (auto w = warp_id; w < n_blocks; w += n_warps) {
      auto const bits = prev_station_mark_.blocks_[w];
      if (bits == 0U) {  // uniform: all lanes read the same word
        continue;
      }

      auto const base = w * kWarpSize;  // lane i <-> bit i of the mark word
      auto const my_i = base + lane;
      auto const my_marked = ((bits >> lane) & 1U) != 0U;

      // per-lane state; sourced via shuffle by the cooperative hub path
      auto tmp_time = kInvalid;
      auto bc = breadcrumb_t{0U};
      auto n_fps = 0U;
      auto defer = false;

      auto const t_at_dest = time_at_dest_.get(k);

      if (my_marked) {
        auto const l = location_idx_t{my_i};
        tmp_time = tmp_.get(l, Vias);
        if (tmp_time != kInvalid) {
          bc = tmp_.get_bc(0U, l, Vias);
          auto const is_dest = is_dest_[my_i];
          auto const loc_transfer_time = dir(adjusted_transfer_time(
              transfer_time_settings_, tt_.transfer_time_[l].count()));

          // same-station transfer (former update_transfers)
          {
            auto const fp_target_time = static_cast<delta_t>(
                tmp_time + ((!intermodal && is_dest) ? 0 : loc_transfer_time));
            if (is_better(fp_target_time, t_at_dest) &&
                is_better(fp_target_time, best_.get(l, Vias))) {
              round_times_.update_min(k, l, Vias, fp_target_time, bc);
              best_.update_min(l, Vias, fp_target_time);
              station_mark_.mark(my_i);
              if (is_dest) {
                update_time_at_dest(k, fp_target_time);
              }
            }
          }

          // intermodal egress (former update_intermodal_footpaths)
          if (intermodal && dist_to_end_[my_i] != kUnreachable) {
            auto const end_time = clamp(tmp_time + dir(dist_to_end_[my_i]));
            if (is_better(end_time, t_at_dest)) {
              round_times_.update_min(
                  k, kIntermodalTarget, Vias, end_time,
                  bc /* write breadcrumb of last arriving transport */);
              best_.update_min(kIntermodalTarget, Vias, end_time);
              update_time_at_dest(k, end_time);
            }
          }

          // footpaths: short lists inline, hubs deferred to the whole warp
          auto use_td_fps = false;
          if constexpr (WithTdFootpaths) {
            use_td_fps = has_td_fps(l);
          }
          if (use_td_fps) {
            if constexpr (WithTdFootpaths) {
              auto const td_fps = kFwd ? rtt_.td_->out_[prf_idx_][l]
                                       : rtt_.td_->in_[prf_idx_][l];
              d_for_each_td_footpath<SearchDir>(
                  td_fps, to_unix(tmp_time),
                  [&](location_idx_t const target, duration_t const d) {
                    relax_fp_target(k, target, static_cast<int>(d.count()),
                                    tmp_time, bc, t_at_dest);
                  });
            }
          } else {
            auto const fps = kFwd ? tt_.footpaths_out_[prf_idx_][l]
                                  : tt_.footpaths_in_[prf_idx_][l];
            n_fps = static_cast<unsigned>(fps.size());
            if (n_fps <= kWarpFpThreshold) {
              for (auto j = 0U; j != n_fps; ++j) {
                relax_footpath(k, fps[j], tmp_time, bc, t_at_dest);
              }
            } else {
              defer = true;
            }
          }
        }
      }

      // hubs: all 32 lanes stride one deferred location's footpath list
      auto const deferred = __ballot_sync(kAllLanes, defer);
      for_each_set_bit(deferred, [&](unsigned const b) {
        auto const l = location_idx_t{base + b};
        // full-mask shuffles also reconverge the warp for the strided loop
        auto const l_tmp = static_cast<delta_t>(__shfl_sync(
            kAllLanes, static_cast<int>(tmp_time), static_cast<int>(b)));
        auto const l_bc = __shfl_sync(kAllLanes, bc, static_cast<int>(b));
        auto const l_n = __shfl_sync(kAllLanes, n_fps, static_cast<int>(b));
        auto const fps = kFwd ? tt_.footpaths_out_[prf_idx_][l]
                              : tt_.footpaths_in_[prf_idx_][l];
        for (auto j = lane; j < l_n; j += kWarpSize) {
          relax_footpath(k, fps[j], l_tmp, l_bc, t_at_dest);
        }
      });
    }
  }

  // IsWheelchair == the kernel's dispatch flag -> stop accessibility checks
  // constant-fold. WithSections + section_mask = the per-route section
  // filters of the query's required modes (bike/car/wheelchair).
  template <bool IsWheelchair, bool WithSections>
  __device__ bool update_route_warp(
      unsigned const k,
      route_idx_t const r,
      unsigned const lane,
      [[maybe_unused]] unsigned const section_mask) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto const base_flat = tt_.route_stop_offset_[to_idx(r)];
    auto local_marked = false;

    // The 64bit key:
    //  - 32bit MSB: earliest transport: (day, transport) = total order in route
    //  - 32bit LSB: scan-order index (not stop index)
    //    (forward: boarding index, backward: alighting index)
    //
    // Plain unsigned integer comparison yields lexicographical order
    // (transport, stop) selects earliest transport at earliest boarding index
    constexpr auto kEtKeyInvalid = ~std::uint64_t{0};
    auto carried_et = kEtKeyInvalid;

    for (auto chunk = 0U; chunk < n; chunk += kWarpSize) {
      // Note: continue for i >= n would be UB for __shfl_up_sync etc

      auto const i = chunk + lane;
      auto stop_idx = stop_idx_t{};
      auto my_key = kEtKeyInvalid;
      [[maybe_unused]] auto kill = false;

      if (i < n) {
        stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);

        auto const et = et_result_[base_flat + stop_idx];
        if (et != kEtInvalid) {
          my_key = (static_cast<std::uint64_t>(et) << 32U) | i;
        }

        if constexpr (WithSections) {
          if (i != 0U) {
            auto const sec =
                static_cast<unsigned>(kFwd ? stop_idx - 1 : stop_idx);
            kill = tt_.filters_->section_killed(section_mask, r, sec);
          }
        }
      }

      // Inclusive prefix-min over previous keys in the chunk (including this).
      auto incl = my_key;
      [[maybe_unused]] auto prefix_contains_kill = 0U;
      if constexpr (WithSections) {
        prefix_contains_kill = kill ? 1U : 0U;
        for (auto off = 1U; off < kWarpSize; off <<= 1) {
          auto const prev_incl = __shfl_up_sync(kAllLanes, incl, off);
          auto const prev_prefix_contains_kill =
              __shfl_up_sync(kAllLanes, prefix_contains_kill, off);
          if (lane >= off) {
            if (prefix_contains_kill == 0U) {
              // no kill between lane-off and me
              incl = min(incl, prev_incl);
            }
            prefix_contains_kill |= prev_prefix_contains_kill;
          }
        }
      } else {
        // no section filters -> plain prefix-min
        for (auto off = 1U; off < kWarpSize; off <<= 1) {
          auto const prev_incl = __shfl_up_sync(kAllLanes, incl, off);
          if (lane >= off) {
            incl = min(incl, prev_incl);
          }
        }
      }

      // Get earliest transport from previous stop.
      auto et = __shfl_up_sync(kAllLanes, incl, 1);
      if constexpr (WithSections) {
        auto prev_prefix_contains_kill =
            __shfl_up_sync(kAllLanes, prefix_contains_kill, 1);
        if (lane == 0U) {
          et = kEtKeyInvalid;
          prev_prefix_contains_kill = 0U;
        }
        et =  // if no kill between chunk start and incl here -> min(et, carry)
            kill ? kEtKeyInvalid
                 : (prev_prefix_contains_kill != 0U ? et : min(et, carried_et));
      } else {
        if (lane == 0U) {
          et = kEtKeyInvalid;
        }
        et = min(et, carried_et);
      }

      // Update stop time.
      auto const et_board_i = static_cast<unsigned>(et & 0xFFFF'FFFFU);
      // stop_idx <= kBcStopMask: no alight at positions the 11-bit
      // breadcrumb cannot represent (see et_run_lookups)
      if (i < n && et != kEtKeyInvalid && et_board_i < i &&
          stop_idx <= kBcStopMask) {
        auto const stp = stop{stop_seq[stop_idx]};
        if (stp.can_finish<SearchDir>(IsWheelchair)) {
          auto const l = stp.location_idx();
          auto const l_idx = to_idx(l);
          auto const t = unpack_et(r, static_cast<std::uint32_t>(et >> 32U));
          auto const by_transport = time_at_stop(
              r, t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
          if (is_better(by_transport, time_at_dest_.get(k))) {
            auto const board_stop = static_cast<stop_idx_t>(
                kFwd ? et_board_i : n - 1U - et_board_i);
            tmp_.update_min(
                l, 0U, by_transport,
                make_transport_payload(t.t_idx_.v_, board_stop, stop_idx));
            station_mark_.mark(l_idx);
            local_marked = true;
          }
        }
      }

      // Carry across chunks (sections: reset if the chunk contains a kill).
      if constexpr (WithSections) {
        auto const incl_last = __shfl_sync(kAllLanes, incl, kWarpSize - 1U);
        auto const chunk_contains_kill =
            __shfl_sync(kAllLanes, prefix_contains_kill, kWarpSize - 1U);
        carried_et =
            chunk_contains_kill != 0U ? incl_last : min(incl_last, carried_et);
      } else {
        carried_et =
            __shfl_sync(kAllLanes, min(incl, carried_et), kWarpSize - 1U);
      }
    }

    return __any_sync(kAllLanes, local_marked);
  }

  __device__ transport
  get_earliest_transport(unsigned const k,
                         route_idx_t const r,
                         stop_idx_t const stop_idx,
                         day_idx_t const day_at_stop,
                         minutes_after_midnight_t const mam_at_stop) {
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

        if (is_better_or_eq(time_at_dest_.get(k), to_delta(day, ev_mam))) {
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active(t, start_day)) {
          continue;
        }
        return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
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
    return as_int(day) >= 0 && tt_.bitfields_[tt_.route_traffic_days_[r]].test(
                                   static_cast<std::size_t>(as_int(day)));
  }

  // Packed earliest transport, 32 bits: [rel_day : 6 | t_offset : 26].
  //  - rel_day: traffic day relative to (base_ - 28)
  //  - t_offset: transport index relative to the route's first transport
  //
  // Bwd: encode t_offset as (kEtReverseBase - x [including 0]) != kEtInvalid
  // -> taking min() is still correct (yields later transport for bwd search)
  static constexpr auto kEtInvalid = ~std::uint32_t{0};
  static constexpr auto kEtRelDayShift = 26U;
  static constexpr auto kEtReverseBase = kEtInvalid - 1U;

  __device__ __forceinline__ int et_day_lo() const {
    return as_int(base_) - 28;
  }

  __device__ __forceinline__ std::uint32_t pack_et(route_idx_t const r,
                                                   transport const t) const {
    if (!t.is_valid()) {
      return kEtInvalid;
    }

    auto const rel_day =
        static_cast<std::uint32_t>(as_int(t.day_) - et_day_lo());
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

  // et PHASE 1: compact the marked routes into route_list_. Also resets
  // any_marked_ for loop_routes: doing it here (nothing reads it in this
  // kernel) instead of in begin_transit_phase avoids racing that kernel's
  // grid-wide convergence read.
  __device__ void et_build_route_list() {
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

  // et PHASE 2: Write list of marked route stops (flat offsets).
  // -> warp-aggregated stream compaction:
  // 1) Filter: check if route is "boardable" (fwd) / "alightable" (bwd)
  // 2) Count: __ballot_sync(is_task) collapses 32 lanes to a 32bit mask
  // 3) Reserve: make space for popcount(ballot sync mask) entries
  // 4) Rank + Scatter: write flat route stop to ballot & ((1 << lane) - 1)
  // IsWheelchair: kernel-level (see loop_routes) -> can_start constant-folds
  template <bool IsWheelchair>
  __device__ void et_collect_tasks(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    {
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
        for (auto chunk = 0U; chunk < n; chunk += kWarpSize) {
          auto const s = chunk + lane;
          auto is_task = false;
          if (s < n) {
            et_result_[base_flat + s] = kEtInvalid;
            auto const is_dir_last = kFwd ? (s + 1U == n) : (s == 0U);
            if (!is_dir_last) {
              auto const stp = stop{stop_seq[s]};
              auto const l = stp.location_idx();
              is_task = prev_station_mark_[to_idx(l)] &&
                        stp.can_start<SearchDir>(IsWheelchair) &&
                        round_times_.get(k - 1, l, 0U) != kInvalid;
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
              et_task_list_[base_pos + off] = base_flat + s;
            }
          }
        }
      }
    }
  }

  // et PHASE 3: Do one earliest-transport lookup per task.
  __device__ void et_run_lookups(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    auto const n_tasks = *et_task_count_;
    for (auto i = gid; i < n_tasks; i += stride) {
      auto const flat = et_task_list_[i];
      auto const r = route_idx_t{tt_.route_of_stop_[flat]};
      auto const stop_idx =
          static_cast<stop_idx_t>(flat - tt_.route_stop_offset_[to_idx(r)]);
      auto const stop_seq = tt_.route_location_seq_[r];
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l = stp.location_idx();
      auto const [day, mam] = split(round_times_.get(k - 1, l, 0U));
      // the breadcrumb stores stop positions in 11 bits: no boarding at
      // positions beyond that (a handful of >2048-stop routes exist in
      // worldwide data; those tails are unreachable for reconstruction)
      et_result_[flat] =
          stop_idx > kBcStopMask
              ? kEtInvalid
              : pack_et(r, get_earliest_transport(k, r, stop_idx, day, mam));
    }
  }

  __device__ delta_t time_at_stop(route_idx_t const r,
                                  transport const t,
                                  stop_idx_t const stop_idx,
                                  event_type const ev_type) {
    return to_delta(t.day_,
                    tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  __device__ delta_t rt_time_at_stop(rt_transport_idx_t const rt_t,
                                     stop_idx_t const stop_idx,
                                     event_type const ev_type) {
    return clamp((as_int(rtt_.base_day_idx_) - as_int(base_)) * 1440 +
                 rtt_.event_time(rt_t, stop_idx, ev_type));
  }

  __device__ delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
    return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  __device__ unixtime_t to_unix(delta_t const t) {
    return delta_to_unix(base(), t);
  }

  __device__ std::pair<day_idx_t, minutes_after_midnight_t> split(
      delta_t const x) {
    return split_day_mam(base_, x);
  }

  __device__ __forceinline__ bool is_intermodal_dest() const {
    return !dist_to_end_.empty();
  }

  __device__ void update_time_at_dest(unsigned const k, delta_t const t) {
    for (auto i = k; i != max_transfers_ + 1U; ++i) {
      time_at_dest_.update_min(i, t);
    }
  }

  __device__ __forceinline__ int as_int(day_idx_t const d) const {
    return static_cast<int>(d.v_);
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

  std::uint32_t* any_marked_;
  std::uint32_t* done_;
  device_timetable tt_;
  device_rt_timetable rtt_;
  transfer_time_settings transfer_time_settings_;
  std::uint8_t max_transfers_;
  clasz_mask_t allowed_claszes_;
  profile_idx_t prf_idx_;
  bool require_bike_transport_;
  bool require_car_transport_;
  day_idx_t base_;
  cuda::std::span<std::pair<location_idx_t, unixtime_t> const> starts_;
  device_bitvec<std::uint64_t const> is_dest_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  cuda::std::span<location_idx_t const> td_dest_locs_;
  d_vecvec_view<td_dest_offsets_t> td_dest_;
  device_times<SearchDir, Vias + 1> round_times_;
  device_times<SearchDir, Vias + 1> best_;
  device_times<SearchDir, Vias + 1> tmp_;
  device_times<SearchDir, 1U> time_at_dest_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;
  device_bitvec<std::uint32_t> rt_transport_mark_;

  // earliest transports per flat (route,stop)
  cuda::std::span<std::uint32_t> et_result_;
  cuda::std::span<std::uint32_t> et_task_list_;
  std::uint32_t* et_task_count_;  // number of tasks this round

  // marked routes this round
  cuda::std::span<std::uint32_t> route_list_;
  std::uint32_t* route_list_count_;
};

}  // namespace nigiri::routing::gpu