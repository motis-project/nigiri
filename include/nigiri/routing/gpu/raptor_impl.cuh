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

inline constexpr auto kWarpSize = 32U;

inline constexpr auto kAllLanes = ~std::uint32_t{0};

using td_dest_group_idx_t = cista::strong<std::uint32_t, struct td_dest_group_>;
using td_dest_offsets_t = vecvec<td_dest_group_idx_t, td_offset>;

template <direction SearchDir, bool WithBounds, std::uint8_t NWorlds = 1U>
struct raptor_impl {
  static constexpr via_offset_t Vias = 0U;
  static constexpr auto const kNSlots = NWorlds;

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

  // For ping: comparison has to be <= instead of < so cells with equal arrival
  // times are still populated. Otherwise, the ping bounds are too strict for
  // pong to still find all optimal journeys.
  //
  // Loose pruning with <= instead of < is always enabled on the GPU.
  // Measured to have ~ the same performance as strict pruning.
  __device__ __forceinline__ bool is_better_loose(auto a, auto b) {
    return is_better_or_eq(a, b);
  }

  __device__ bool within_bounds(unsigned const k,
                                location_idx_t const l,
                                delta_t const t) {
    if constexpr (!WithBounds) {
      return true;
    } else {
      // FWD arrival + 5min transfer = earliest departure
      //   10:00      10:05
      // >>>>>|-------->*>>>>>
      //
      // BWD departure - 5min transfer = latest arrival
      //   10:00      10:05
      // <<<<<*<--------|<<<<<
      //
      // -> 10:00 < 10:05 would get rejected.
      // -> ping journey would not be found in pong
      auto const* const row = bounds_ + (bounds_last_k_ - k) * tt_.n_locations_;
      auto const transfer = dir(adjusted_transfer_time(
          transfer_time_settings_,
          static_cast<int>(tt_.transfer_time_[l].count())));
      return is_better_or_eq(static_cast<int>(t),
                             static_cast<int>(row[to_idx(l)]) + transfer);
    }
  }

  __device__ void init_arrivals(unixtime_t const worst_time_at_dest) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    if (global_t_id == 0U) {
      *done_ = 0U;
    }

    for (auto i = global_t_id; i < starts_.size(); i += global_stride) {
      auto const l = starts_[i].first;
      auto const t = unix_to_delta(base(), starts_[i].second);
      for (auto v = via_offset_t{0}; v != kNSlots; ++v) {
        best_.update_min(l, v, t);
        round_times_.update_min(0U, l, v, t, make_start_bc());
      }
      touch_round(0U, l);
      station_mark_.mark(to_idx(l));
    }

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto i = global_t_id; i < (kMaxTransfers + 2U) * NWorlds;
         i += global_stride) {
      time_at_dest_.update_min(i, d_worst_at_dest);
    }
  }

  __device__ void reuse_previous_arrivals(unsigned const k) {
    auto const global_t_id = get_global_thread_id();

    if (has_reusable_round_times_ != 0U) {
      auto const lane = global_t_id % kWarpSize;
      auto const warp_id = global_t_id / kWarpSize;
      auto const n_warps = get_global_stride() / kWarpSize;
      auto const* row = &round_touched_[k * round_touched_stride_];

      for (auto w = warp_id; w < round_touched_stride_; w += n_warps) {
        auto const bits = row[w];
        if (bits == 0U) {
          continue;
        }

        auto const my_i = w * kWarpSize + lane;
        if (((bits >> lane) & 1U) != 0U && my_i < tt_.n_locations_) {
          auto const l = location_idx_t{my_i};
          for (auto v = 0U; v != kNSlots; ++v) {
            best_.update_min(l, v, round_times_.get(k, l, v));
          }
        }
      }

      for (auto d = global_t_id; d < n_dest_locs_; d += get_global_stride()) {
        if constexpr (NWorlds == 1U) {
          update_time_at_dest<0U>(k, best_.get(dest_locs_[d], 0U));
        } else {
          update_time_at_dest<0U>(k, best_.get(dest_locs_[d], 0U));
          update_time_at_dest<1U>(k, best_.get(dest_locs_[d], 1U));
        }
      }
    }

    if (global_t_id == 0U) {
      *any_marked_ = 0U;
    }
  }

  __device__ void touch_round(unsigned const k, location_idx_t const l) {
    auto const i = to_idx(l);
    atomicOr(&round_touched_[k * round_touched_stride_ + (i / 32U)],
             std::uint32_t{1U} << (i % 32U));
  }

  template <typename Lists>
  __device__ void mark_from_locations(Lists const& lists,
                                      device_bitvec<std::uint32_t>& marks) {
    constexpr auto const kInline = 8U;
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const warp_id = get_global_thread_id() / kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    auto const n_blocks = station_mark_.blocks_.size();

    for (auto w = warp_id; w < n_blocks; w += n_warps) {
      auto const bits = station_mark_.blocks_[w];
      if (bits == 0U) {  // uniform: all lanes read the same word
        continue;
      }

      auto const my_i = w * kWarpSize + lane;
      auto const my_marked =
          ((bits >> lane) & 1U) != 0U && my_i < tt_.n_locations_;

      auto n = 0U;
      if (my_marked) {
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

      // long lists: all 32 lanes stride one deferred location's list
      auto const deferred = __ballot_sync(kAllLanes, n != 0U);
      for_each_set_bit(deferred, [&](unsigned const b) {
        auto const src_i = __shfl_sync(kAllLanes, my_i, static_cast<int>(b));
        auto const cnt = __shfl_sync(kAllLanes, n, static_cast<int>(b));
        auto const list = lists[location_idx_t{src_i}];
        for (auto j = lane; j < cnt; j += kWarpSize) {
          marks.mark(to_idx(list[j]));
        }
      });
    }
  }

  __device__ void mark_routes(unsigned const) {
    mark_from_locations(tt_.location_routes_, route_mark_);
  }

  __device__ void mark_rt_transports(unsigned const) {
    mark_from_locations(rtt_.location_rt_transports_, rt_transport_mark_);
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

  // runtime world check for reconstruct (thread-indexed world)
  __device__ __forceinline__ bool is_transport_active_rt(
      transport_idx_t const t, std::size_t const day, unsigned const w) const {
    if constexpr (NWorlds == 2U) {
      if (w == 0U) {
        return tt_.bitfields_[sched_transport_traffic_days_[t]].test(day);
      }
    }
    return is_transport_active(t, day);
  }

  __device__ void reconstruct_journey(location_idx_t const dest,
                                      unsigned const K,
                                      unsigned const w,
                                      gpu_journey* out) {
    out->state_ = reconstruction_result::kNotReconstructed;
    auto cur_v = static_cast<via_offset_t>(w);
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

        constexpr auto const kRecMaxDayShift =
            static_cast<int>(routing::kMaxTravelTime.count() / 1440 + 1);
        auto const event_mam_full =
            tt_.event_mam(r, t_idx, alight, ev_arr_type).count();
        auto const [arr_day, _] = split(arr_at_cur);
        auto found_day = false;
        for (auto off = 0; off != kRecMaxDayShift; ++off) {
          auto const cand =
              as_int(arr_day) - event_mam_full / 1440 - (kFwd ? off : -off);
          if (cand < 0) {
            continue;
          }
          if (!is_transport_active_rt(t_idx, static_cast<std::size_t>(cand),
                                      w)) {
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

  // Combines the query's required modes (bike/car/reservation runtime flags +
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
    if (no_compulsory_reservation_ &&
        !f.reservation_not_required_.allows(i, kNoCompulsoryReservationSections,
                                            section_mask)) {
      return false;
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
    // one warp per marked rt transport: lanes cooperate over the stop
    // sequence (coalesced stop-time loads; boarding chain = warp prefix-max)
    auto const lane = get_global_thread_id() % kWarpSize;
    auto const warp_id = get_global_thread_id() / kWarpSize;
    auto const n_warps = get_global_stride() / kWarpSize;
    for (auto i = warp_id; i < rtt_.n_rt_transports_; i += n_warps) {
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
        auto const marked =
            section_mask != 0U
                ? update_rt_transport_warp<IsWheelchair, true>(
                      k, rt_transport_idx_t{i}, lane, section_mask)
                : update_rt_transport_warp<IsWheelchair, false>(
                      k, rt_transport_idx_t{i}, lane, 0U);
        if (marked && lane == 0U && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
      } else {
        if (update_rt_transport_warp<false, false>(k, rt_transport_idx_t{i},
                                                   lane, 0U) &&
            lane == 0U && !*any_marked_) {
          atomicOr(any_marked_, 1U);
        }
      }
    }
  }

  template <bool IsWheelchair, bool WithSections>
  __device__ bool update_rt_transport_warp(
      unsigned const k,
      rt_transport_idx_t const rt_t,
      unsigned const lane,
      [[maybe_unused]] unsigned const section_mask) {
    auto const stop_seq = rtt_.rt_transport_location_seq_[rt_t];
    auto const n = static_cast<unsigned>(stop_seq.size());
    auto local_marked = false;

    // rt world = last slot; a single concrete transport, so the boarding
    // chain is just "latest boardable scan index so far" (any valid boarding
    // is a correct witness; latest = shortest ride, matches the sequential
    // version's behavior)
    constexpr auto kNoBoard = -1;
    auto carried_board = kNoBoard;

    for (auto chunk = 0U; chunk < n; chunk += kWarpSize) {
      auto const i = chunk + lane;
      auto stop_idx = stop_idx_t{};
      auto my_board = kNoBoard;
      [[maybe_unused]] auto kill = false;

      if (i < n) {
        stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);
        auto const stp = stop{stop_seq[stop_idx]};
        auto const is_dir_last = i + 1U == n;
        if (!is_dir_last && stp.can_start<SearchDir>(IsWheelchair) &&
            prev_station_mark_[to_idx(stp.location_idx())]) {
          auto const dep = rt_time_at_stop(
              rt_t, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
          if (is_better_or_eq(
                  round_times_.get(k - 1, stp.location_idx(), kNSlots - 1U),
                  dep)) {
            my_board = static_cast<int>(i);
          }
        }
        if constexpr (WithSections) {
          if (i != 0U) {
            auto const sec =
                static_cast<unsigned>(kFwd ? stop_idx - 1 : stop_idx);
            kill = rtt_.filters_->section_killed(section_mask, rt_t, sec);
          }
        }
      }

      // Inclusive prefix-max over boardable indices (kill resets the chain).
      auto incl = my_board;
      [[maybe_unused]] auto prefix_contains_kill = 0U;
      if constexpr (WithSections) {
        prefix_contains_kill = kill ? 1U : 0U;
        for (auto off = 1U; off < kWarpSize; off <<= 1) {
          auto const prev_incl = __shfl_up_sync(kAllLanes, incl, off);
          auto const prev_prefix_contains_kill =
              __shfl_up_sync(kAllLanes, prefix_contains_kill, off);
          if (lane >= off) {
            if (prefix_contains_kill == 0U) {
              incl = incl > prev_incl ? incl : prev_incl;
            }
            prefix_contains_kill |= prev_prefix_contains_kill;
          }
        }
      } else {
        for (auto off = 1U; off < kWarpSize; off <<= 1) {
          auto const prev_incl = __shfl_up_sync(kAllLanes, incl, off);
          if (lane >= off) {
            incl = incl > prev_incl ? incl : prev_incl;
          }
        }
      }

      // Boarding available when entering this stop = chain up to the
      // previous scan index.
      auto board = __shfl_up_sync(kAllLanes, incl, 1);
      if constexpr (WithSections) {
        auto prev_prefix_contains_kill =
            __shfl_up_sync(kAllLanes, prefix_contains_kill, 1);
        if (lane == 0U) {
          board = kNoBoard;
          prev_prefix_contains_kill = 0U;
        }
        board = kill ? kNoBoard
                     : (prev_prefix_contains_kill != 0U
                            ? board
                            : (board > carried_board ? board : carried_board));
      } else {
        if (lane == 0U) {
          board = kNoBoard;
        }
        board = board > carried_board ? board : carried_board;
      }

      // Arrival update.
      if (i != 0U && i < n && board != kNoBoard &&
          board < static_cast<int>(i)) {
        auto const stp = stop{stop_seq[stop_idx]};
        if (stp.can_finish<SearchDir>(IsWheelchair)) {
          auto const l = stp.location_idx();
          auto const by_transport = rt_time_at_stop(
              rt_t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
          if (is_better_loose(by_transport, t_at_dest(k)) &&
              within_bounds(k, l, by_transport)) {
            auto const board_stop = static_cast<stop_idx_t>(
                kFwd ? static_cast<unsigned>(board)
                     : n - 1U - static_cast<unsigned>(board));
            tmp_.update_min(
                l, kNSlots - 1U, by_transport,
                make_transport_payload(encode_rt_bc_transport(to_idx(rt_t)),
                                       board_stop, stop_idx));
            station_mark_.mark(to_idx(l));
            local_marked = true;
          }
        }
      }

      // Carry across chunks.
      if constexpr (WithSections) {
        auto const incl_last = __shfl_sync(kAllLanes, incl, kWarpSize - 1U);
        auto const chunk_contains_kill =
            __shfl_sync(kAllLanes, prefix_contains_kill, kWarpSize - 1U);
        carried_board =
            chunk_contains_kill != 0U
                ? incl_last
                : (incl_last > carried_board ? incl_last : carried_board);
      } else {
        auto const m = incl > carried_board ? incl : carried_board;
        carried_board = __shfl_sync(kAllLanes, m, kWarpSize - 1U);
      }
    }

    return __any_sync(kAllLanes, local_marked);
  }

  // both worlds share value+breadcrumb: one evaluation, writes to both
  // slots (admission with the looser bound / either-world improvement:
  // bounds only prune, so the wider write stays correct)
  __device__ __forceinline__ void relax_fp_target_mirror(
      unsigned const k,
      location_idx_t const target_l,
      int const duration,
      delta_t const tmp_time,
      breadcrumb_t const bc,
      delta_t const t_at_dest_loose) {
    auto const target = to_idx(target_l);
    auto const fp_target_time = clamp(tmp_time + dir(duration));

    auto const improves = is_better(fp_target_time, best_.get(target_l, 0U)) ||
                          is_better(fp_target_time, best_.get(target_l, 1U));
    if constexpr (!WithBounds) {
      if (improves) {
        round_times_.update_min(k, target_l, 0U, fp_target_time, bc);
        round_times_.update_min(k, target_l, 1U, fp_target_time, bc);
        touch_round(k, target_l);
      }
    }

    if (!is_better_loose(fp_target_time, t_at_dest_loose)) {
      return;
    }

    if (improves && within_bounds(k, target_l, fp_target_time)) {
      round_times_.update_min(k, target_l, 0U, fp_target_time, bc);
      round_times_.update_min(k, target_l, 1U, fp_target_time, bc);
      touch_round(k, target_l);
      best_.update_min(target_l, 0U, fp_target_time);
      best_.update_min(target_l, 1U, fp_target_time);
      station_mark_.mark(target);
      if (is_dest_[target]) {
        update_time_at_dest<0U>(k, fp_target_time);
        update_time_at_dest<1U>(k, fp_target_time);
      }
    }
  }

  __device__ __forceinline__ void relax_footpath_mirror(
      unsigned const k,
      footpath const fp,
      delta_t const tmp_time,
      breadcrumb_t const bc,
      delta_t const t_at_dest_loose) {
    relax_fp_target_mirror(
        k, fp.target(),
        adjusted_transfer_time(transfer_time_settings_, fp.duration().count()),
        tmp_time, bc, t_at_dest_loose);
  }

  template <std::uint8_t W>
  __device__ __forceinline__ void relax_footpath(unsigned const k,
                                                 footpath const fp,
                                                 delta_t const tmp_time,
                                                 breadcrumb_t const bc,
                                                 delta_t const t_at_dest) {
    relax_fp_target<W>(
        k, fp.target(),
        adjusted_transfer_time(transfer_time_settings_, fp.duration().count()),
        tmp_time, bc, t_at_dest);
  }

  template <std::uint8_t W>
  __device__ __forceinline__ void relax_fp_target(unsigned const k,
                                                  location_idx_t const target_l,
                                                  int const duration,
                                                  delta_t const tmp_time,
                                                  breadcrumb_t const bc,
                                                  delta_t const t_at_dest) {
    auto const target = to_idx(target_l);
    auto const fp_target_time = clamp(tmp_time + dir(duration));

    if constexpr (!WithBounds) {
      // Required for pong search. Target pruning to save writes.
      if (is_better(fp_target_time, best_.get(target_l, W))) {
        round_times_.update_min(k, target_l, W, fp_target_time, bc);
        touch_round(k, target_l);
      }
    }

    if (!is_better_loose(fp_target_time, t_at_dest)) {
      return;
    }

    if (is_better(fp_target_time, best_.get(target_l, W)) &&
        within_bounds(k, target_l, fp_target_time)) {
      round_times_.update_min(k, target_l, W, fp_target_time, bc);
      touch_round(k, target_l);
      best_.update_min(target_l, W, fp_target_time);
      station_mark_.mark(target);
      if (is_dest_[target]) {
        update_time_at_dest<W>(k, fp_target_time);
      }
    }
  }

  __device__ __forceinline__ bool has_td_fps(location_idx_t const l) const {
    auto const& bv =
        kFwd ? rtt_.td_->has_out_[prf_idx_] : rtt_.td_->has_in_[prf_idx_];
    return !bv.blocks_.empty() && bv[to_idx(l)];
  }

  template <std::uint8_t W>
  __device__ void update_td_dest_offsets(unsigned const k) {
    auto const gid = get_global_thread_id();
    auto const stride = get_global_stride();
    for (auto g = gid; g < td_dest_locs_.size(); g += stride) {
      auto const l = td_dest_locs_[g];

      if (!prev_station_mark_[to_idx(l)]) {
        continue;
      }

      auto const tmp_time = tmp_.get(l, W);
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
      if (is_better_loose(end_time, t_at_dest<W>(k)) &&
          is_better(end_time, best_.get(kIntermodalTarget, W))) {
        auto const bc = tmp_.get_bc(0U, l, W);
        round_times_.update_min(k, kIntermodalTarget, W, end_time, bc);
        touch_round(k, location_idx_t{kIntermodalTarget});
        best_.update_min(kIntermodalTarget, W, end_time);
        update_time_at_dest<W>(k, end_time);
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
        update_td_dest_offsets<0U>(k);
        if constexpr (NWorlds == 2U) {
          update_td_dest_offsets<1U>(k);
        }
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
      auto tmp0 = kInvalid;
      auto bc0 = breadcrumb_t{0U};
      [[maybe_unused]] auto tmp1 = kInvalid;
      [[maybe_unused]] auto bc1 = breadcrumb_t{0U};
      auto mirror = false;
      auto n_fps = 0U;
      auto defer = false;

      auto const t0 = t_at_dest<0U>(k);
      [[maybe_unused]] auto const t1 = NWorlds == 2U ? t_at_dest<1U>(k) : t0;
      auto const t_loose = NWorlds == 2U ? (is_better(t0, t1) ? t1 : t0) : t0;

      // relax one footpath/transfer edge for whatever world state the
      // source stop carries (mirrored when both worlds agree)
      auto const relax_edge = [&](location_idx_t const target, int const dur,
                                  delta_t const s0, breadcrumb_t const b0,
                                  delta_t const s1, breadcrumb_t const b1,
                                  bool const mir) {
        if constexpr (NWorlds == 1U) {
          relax_fp_target<0U>(k, target, dur, s0, b0, t0);
        } else if (mir) {
          relax_fp_target_mirror(k, target, dur, s0, b0, t_loose);
        } else {
          if (s0 != kInvalid) {
            relax_fp_target<0U>(k, target, dur, s0, b0, t0);
          }
          if (s1 != kInvalid) {
            relax_fp_target<1U>(k, target, dur, s1, b1, t1);
          }
        }
      };

      if (my_marked) {
        auto const l = location_idx_t{my_i};
        auto const raw0 = tmp_.raw(0U, l, 0U);
        tmp0 = device_times<SearchDir, kNSlots>::from_key(
            static_cast<std::uint16_t>(raw0 >> kBcBits));
        bc0 = raw0 & kBcMask;
        if constexpr (NWorlds == 2U) {
          auto const raw1 = tmp_.raw(0U, l, 1U);
          tmp1 = device_times<SearchDir, kNSlots>::from_key(
              static_cast<std::uint16_t>(raw1 >> kBcBits));
          bc1 = raw1 & kBcMask;
          mirror = raw0 == raw1;
        }
        auto const any_valid =
            tmp0 != kInvalid || (NWorlds == 2U && tmp1 != kInvalid);
        if (any_valid) {
          auto const is_dest = is_dest_[my_i];

          // same-station transfer (former update_transfers)
          relax_edge(l,
                     (!intermodal && is_dest)
                         ? 0
                         : adjusted_transfer_time(
                               transfer_time_settings_,
                               static_cast<int>(tt_.transfer_time_[l].count())),
                     tmp0, bc0, tmp1, bc1, mirror);

          // intermodal egress (former update_intermodal_footpaths)
          if (intermodal && dist_to_end_[my_i] != kUnreachable) {
            auto const egress = [&]<std::uint8_t W>(delta_t const st,
                                                    breadcrumb_t const sb) {
              if (st == kInvalid) {
                return;
              }
              auto const end_time = clamp(st + dir(dist_to_end_[my_i]));
              if (is_better_loose(end_time, t_at_dest<W>(k))) {
                round_times_.update_min(
                    k, kIntermodalTarget, W, end_time,
                    sb /* write breadcrumb of last arriving transport */);
                touch_round(k, location_idx_t{kIntermodalTarget});
                best_.update_min(kIntermodalTarget, W, end_time);
                update_time_at_dest<W>(k, end_time);
              }
            };
            egress.template operator()<0U>(tmp0, bc0);
            if constexpr (NWorlds == 2U) {
              egress.template operator()<1U>(tmp1, bc1);
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
              // td footpaths serve the rt world only on the CPU at rt
              // stops; on the device both worlds use them (superset,
              // pruning-neutral) -> per-world relax, never mirrored with
              // the static list semantics
              d_for_each_td_footpath<SearchDir>(
                  td_fps, to_unix(tmp0 != kInvalid ? tmp0 : tmp1),
                  [&](location_idx_t const target, duration_t const d) {
                    relax_edge(target, static_cast<int>(d.count()), tmp0, bc0,
                               tmp1, bc1, mirror);
                  });
            }
          } else {
            auto const fps = kFwd ? tt_.footpaths_out_[prf_idx_][l]
                                  : tt_.footpaths_in_[prf_idx_][l];
            n_fps = static_cast<unsigned>(fps.size());
            if (n_fps <= kWarpFpThreshold) {
              for (auto j = 0U; j != n_fps; ++j) {
                auto const fp = fps[j];
                relax_edge(fp.target(),
                           adjusted_transfer_time(transfer_time_settings_,
                                                  fp.duration().count()),
                           tmp0, bc0, tmp1, bc1, mirror);
              }
              n_fps = 0U;
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
        auto const l_tmp0 = static_cast<delta_t>(__shfl_sync(
            kAllLanes, static_cast<int>(tmp0), static_cast<int>(b)));
        auto const l_bc0 = __shfl_sync(kAllLanes, bc0, static_cast<int>(b));
        [[maybe_unused]] auto const l_tmp1 = static_cast<delta_t>(__shfl_sync(
            kAllLanes, static_cast<int>(tmp1), static_cast<int>(b)));
        [[maybe_unused]] auto const l_bc1 =
            __shfl_sync(kAllLanes, bc1, static_cast<int>(b));
        auto const l_mirror =
            __shfl_sync(kAllLanes, mirror ? 1U : 0U, static_cast<int>(b)) != 0U;
        auto const l_n = __shfl_sync(kAllLanes, n_fps, static_cast<int>(b));
        auto const fps = kFwd ? tt_.footpaths_out_[prf_idx_][l]
                              : tt_.footpaths_in_[prf_idx_][l];
        for (auto j = lane; j < l_n; j += kWarpSize) {
          auto const fp = fps[j];
          relax_edge(fp.target(),
                     adjusted_transfer_time(transfer_time_settings_,
                                            fp.duration().count()),
                     l_tmp0, l_bc0, l_tmp1, l_bc1, l_mirror);
        }
      });
    }
  }

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
    //
    // scheduled+rt (NWorlds == 2): both worlds' chains ride one warp pass;
    // where the carried transports coincide, one arrival evaluation feeds a
    // mirrored write into both slots
    constexpr auto kEtKeyInvalid = ~std::uint64_t{0};
    auto carried_et = kEtKeyInvalid;
    [[maybe_unused]] auto carried_et1 = kEtKeyInvalid;

    auto const arrival = [&](std::uint64_t const et, unsigned const i,
                             stop_idx_t const stop_idx, auto const& stp,
                             delta_t const dest_bound, auto&& write) {
      auto const et_board_i = static_cast<unsigned>(et & 0xFFFF'FFFFU);
      if (et == kEtKeyInvalid || et_board_i >= i) {
        return;
      }
      auto const l = stp.location_idx();
      auto const t = unpack_et(r, static_cast<std::uint32_t>(et >> 32U));
      auto const by_transport = time_at_stop(
          r, t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      if (is_better_loose(by_transport, dest_bound) &&
          within_bounds(k, l, by_transport)) {
        auto const board_stop =
            static_cast<stop_idx_t>(kFwd ? et_board_i : n - 1U - et_board_i);
        write(l, by_transport,
              make_transport_payload(t.t_idx_.v_, board_stop, stop_idx));
        station_mark_.mark(to_idx(l));
        local_marked = true;
      }
    };

    for (auto chunk = 0U; chunk < n; chunk += kWarpSize) {
      // Note: continue for i >= n would be UB for __shfl_up_sync etc

      auto const i = chunk + lane;
      auto stop_idx = stop_idx_t{};
      auto my_key = kEtKeyInvalid;
      [[maybe_unused]] auto my_key1 = kEtKeyInvalid;
      [[maybe_unused]] auto kill = false;

      if (i < n) {
        stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - 1U - i);

        auto const et = et_result_[(base_flat + stop_idx) * NWorlds];
        if (et != kEtInvalid) {
          my_key = (static_cast<std::uint64_t>(et) << 32U) | i;
        }
        if constexpr (NWorlds == 2U) {
          auto const et1 = et_result_[(base_flat + stop_idx) * NWorlds + 1U];
          if (et1 != kEtInvalid) {
            my_key1 = (static_cast<std::uint64_t>(et1) << 32U) | i;
          }
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
      [[maybe_unused]] auto incl1 = my_key1;
      [[maybe_unused]] auto prefix_contains_kill = 0U;
      if constexpr (WithSections) {
        prefix_contains_kill = kill ? 1U : 0U;
        for (auto off = 1U; off < kWarpSize; off <<= 1) {
          auto const prev_incl = __shfl_up_sync(kAllLanes, incl, off);
          [[maybe_unused]] auto const prev_incl1 =
              NWorlds == 2U ? __shfl_up_sync(kAllLanes, incl1, off)
                            : kEtKeyInvalid;
          auto const prev_prefix_contains_kill =
              __shfl_up_sync(kAllLanes, prefix_contains_kill, off);
          if (lane >= off) {
            if (prefix_contains_kill == 0U) {
              // no kill between lane-off and me
              incl = min(incl, prev_incl);
              if constexpr (NWorlds == 2U) {
                incl1 = min(incl1, prev_incl1);
              }
            }
            prefix_contains_kill |= prev_prefix_contains_kill;
          }
        }
      } else {
        // no section filters -> plain prefix-min
        for (auto off = 1U; off < kWarpSize; off <<= 1) {
          auto const prev_incl = __shfl_up_sync(kAllLanes, incl, off);
          if constexpr (NWorlds == 2U) {
            auto const prev_incl1 = __shfl_up_sync(kAllLanes, incl1, off);
            if (lane >= off) {
              incl1 = min(incl1, prev_incl1);
            }
          }
          if (lane >= off) {
            incl = min(incl, prev_incl);
          }
        }
      }

      // Get earliest transport from previous stop.
      auto et = __shfl_up_sync(kAllLanes, incl, 1);
      [[maybe_unused]] auto et1 =
          NWorlds == 2U ? __shfl_up_sync(kAllLanes, incl1, 1) : kEtKeyInvalid;
      if constexpr (WithSections) {
        auto prev_prefix_contains_kill =
            __shfl_up_sync(kAllLanes, prefix_contains_kill, 1);
        if (lane == 0U) {
          et = kEtKeyInvalid;
          et1 = kEtKeyInvalid;
          prev_prefix_contains_kill = 0U;
        }
        et =  // if no kill between chunk start and incl here -> min(et, carry)
            kill ? kEtKeyInvalid
                 : (prev_prefix_contains_kill != 0U ? et : min(et, carried_et));
        if constexpr (NWorlds == 2U) {
          et1 = kill
                    ? kEtKeyInvalid
                    : (prev_prefix_contains_kill != 0U ? et1
                                                       : min(et1, carried_et1));
        }
      } else {
        if (lane == 0U) {
          et = kEtKeyInvalid;
          et1 = kEtKeyInvalid;
        }
        et = min(et, carried_et);
        if constexpr (NWorlds == 2U) {
          et1 = min(et1, carried_et1);
        }
      }

      // Update stop time.
      if (i < n) {
        auto const stp = stop{stop_seq[stop_idx]};
        if (stp.can_finish<SearchDir>(IsWheelchair)) {
          if constexpr (NWorlds == 1U) {
            arrival(
                et, i, stop_idx, stp, t_at_dest<0U>(k),
                [&](location_idx_t const l, delta_t const v,
                    breadcrumb_t const bc) { tmp_.update_min(l, 0U, v, bc); });
          } else if (et == et1) {
            // both worlds ride the same transport: one evaluation, mirrored
            // write (looser dest bound: pruning only)
            arrival(et, i, stop_idx, stp, t_at_dest_worse(k),
                    [&](location_idx_t const l, delta_t const v,
                        breadcrumb_t const bc) {
                      tmp_.update_min(l, 0U, v, bc);
                      tmp_.update_min(l, 1U, v, bc);
                    });
          } else {
            arrival(
                et, i, stop_idx, stp, t_at_dest<0U>(k),
                [&](location_idx_t const l, delta_t const v,
                    breadcrumb_t const bc) { tmp_.update_min(l, 0U, v, bc); });
            arrival(
                et1, i, stop_idx, stp, t_at_dest<1U>(k),
                [&](location_idx_t const l, delta_t const v,
                    breadcrumb_t const bc) { tmp_.update_min(l, 1U, v, bc); });
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
        if constexpr (NWorlds == 2U) {
          auto const incl1_last = __shfl_sync(kAllLanes, incl1, kWarpSize - 1U);
          carried_et1 = chunk_contains_kill != 0U
                            ? incl1_last
                            : min(incl1_last, carried_et1);
        }
      } else {
        carried_et =
            __shfl_sync(kAllLanes, min(incl, carried_et), kWarpSize - 1U);
        if constexpr (NWorlds == 2U) {
          carried_et1 =
              __shfl_sync(kAllLanes, min(incl1, carried_et1), kWarpSize - 1U);
        }
      }
    }

    return __any_sync(kAllLanes, local_marked);
  }

  template <std::uint8_t W = NWorlds - 1U>
  __device__ transport
  get_earliest_transport(unsigned const k,
                         route_idx_t const r,
                         stop_idx_t const stop_idx,
                         day_idx_t const day_at_stop,
                         minutes_after_midnight_t const mam_at_stop,
                         delta_t const dest_bound) {
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

        // an event equal to time-at-dest must still be boardable
        // (equal-arrival coverage for the pong)
        auto const ev_t = to_delta(day, ev_mam);
        if (is_better(dest_bound, ev_t)) {
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active_w<W>(t, start_day)) {
          continue;
        }
        return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
      }
    }
    return {};
  }

  struct dual_et {
    transport t0_{};
    transport t1_{};
  };

  // one event iteration serving both worlds on rt-touched routes: equal
  // boarding labels, per-event activity split by world; transports without
  // re-pointed traffic days share one bitfield test (the common case)
  __device__ dual_et
  get_earliest_transport_dual(unsigned const k,
                              route_idx_t const r,
                              stop_idx_t const stop_idx,
                              day_idx_t const day_at_stop,
                              minutes_after_midnight_t const mam_at_stop) {
    auto const b0 = t_at_dest<0U>(k);
    auto const b1 = t_at_dest<NWorlds - 1U>(k);
    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    auto out = dual_et{};
    auto done0 = false;
    auto done1 = false;

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

        auto const ev_t = to_delta(day, ev_mam);
        done0 = done0 || is_better(b0, ev_t);
        done1 = done1 || is_better(b1, ev_t);
        if (done0 && done1) {
          return out;
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        auto const tr =
            transport{t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};

        auto const idx = to_idx(tt_.transport_traffic_days_[t]);
        if ((idx & kRtBitfieldFlag) == 0U) {
          // traffic days untouched by rt: one test serves both worlds
          if (tt_.bitfields_[bitfield_idx_t{idx}].test(start_day)) {
            if (!done0) {
              out.t0_ = tr;
              done0 = true;
            }
            if (!done1) {
              out.t1_ = tr;
              done1 = true;
            }
          }
        } else {
          if (!done0 && tt_.bitfields_[sched_transport_traffic_days_[t]].test(
                            start_day)) {
            out.t0_ = tr;
            done0 = true;
          }
          if (!done1 &&
              rtt_.bitfields_[bitfield_idx_t{idx & ~kRtBitfieldFlag}].test(
                  start_day)) {
            out.t1_ = tr;
            done1 = true;
          }
        }
        if (done0 && done1) {
          return out;
        }
      }
    }
    return out;
  }

  __device__ __forceinline__ bool is_transport_active(
      transport_idx_t const t, std::size_t const day) const {
    auto const i = to_idx(tt_.transport_traffic_days_[t]);
    return ((i & kRtBitfieldFlag) != 0U
                ? rtt_.bitfields_[bitfield_idx_t{i & ~kRtBitfieldFlag}]
                : tt_.bitfields_[bitfield_idx_t{i}])
        .test(day);
  }

  // scheduled+rt: world 0 rides the original schedule (cancelled trips
  // included), world 1 = the rt-replaced traffic days (single-world default)
  template <std::uint8_t W>
  __device__ __forceinline__ bool is_transport_active_w(
      transport_idx_t const t, std::size_t const day) const {
    if constexpr (NWorlds == 1U || W == 1U) {
      return is_transport_active(t, day);
    } else {
      return tt_.bitfields_[sched_transport_traffic_days_[t]].test(day);
    }
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
            for (auto w = 0U; w != NWorlds; ++w) {
              et_result_[(base_flat + s) * NWorlds + w] = kEtInvalid;
            }
            auto const is_dir_last = kFwd ? (s + 1U == n) : (s == 0U);
            if (!is_dir_last) {
              auto const stp = stop{stop_seq[s]};
              auto const l = stp.location_idx();
              auto any_valid = round_times_.get(k - 1, l, 0U) != kInvalid;
              if constexpr (NWorlds == 2U) {
                any_valid =
                    any_valid || round_times_.get(k - 1, l, 1U) != kInvalid;
              }
              is_task = prev_station_mark_[to_idx(l)] &&
                        stp.can_start<SearchDir>(IsWheelchair) && any_valid;
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
      auto const lookup = [&]<std::uint8_t W>() {
        auto const label = round_times_.get(k - 1, l, W);
        et_result_[flat * NWorlds + W] =
            label == kInvalid ? kEtInvalid : pack_et(r, [&] {
              auto const [day, mam] = split(label);
              return get_earliest_transport<W>(k, r, stop_idx, day, mam,
                                               t_at_dest<W>(k));
            }());
      };
      if constexpr (NWorlds == 2U) {
        auto const l0 = round_times_.get(k - 1, l, 0U);
        auto const l1 = round_times_.get(k - 1, l, 1U);
        if (l0 == l1 && route_untouched(r)) {
          // identical labels + untouched route: one walk serves both worlds
          // (looser dest bound: pruning only, stays correct for both)
          auto const res = l0 == kInvalid ? kEtInvalid : pack_et(r, [&] {
            auto const [day, mam] = split(l0);
            return get_earliest_transport<0U>(k, r, stop_idx, day, mam,
                                              t_at_dest_worse(k));
          }());
          et_result_[flat * NWorlds] = res;
          et_result_[flat * NWorlds + 1U] = res;
        } else if (l0 == l1 && l0 != kInvalid) {
          // rt-touched route, equal labels: one event iteration, per-world
          // activity (untouched transports need only one bitfield test)
          auto const [day, mam] = split(l0);
          auto const ets =
              get_earliest_transport_dual(k, r, stop_idx, day, mam);
          et_result_[flat * NWorlds] = pack_et(r, ets.t0_);
          et_result_[flat * NWorlds + 1U] = pack_et(r, ets.t1_);
        } else if (l0 == l1) {
          et_result_[flat * NWorlds] = kEtInvalid;
          et_result_[flat * NWorlds + 1U] = kEtInvalid;
        } else {
          lookup.template operator()<0U>();
          lookup.template operator()<1U>();
        }
      } else {
        lookup.template operator()<0U>();
      }
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

  __device__ __forceinline__ delta_t t_at_dest_worse(unsigned const k) {
    if constexpr (NWorlds == 1U) {
      return t_at_dest<0U>(k);
    } else {
      auto const a = t_at_dest<0U>(k);
      auto const b = t_at_dest<1U>(k);
      return is_better(a, b) ? b : a;  // looser bound admits the union
    }
  }

  __device__ __forceinline__ bool route_untouched(route_idx_t const r) const {
    if constexpr (NWorlds == 1U) {
      return true;
    } else {
      return !rtt_.route_has_rt_[to_idx(r)];
    }
  }

  template <std::uint8_t W = NWorlds - 1U>
  __device__ __forceinline__ delta_t t_at_dest(unsigned const k) {
    return time_at_dest_.get(static_cast<std::uint8_t>(k * NWorlds + W));
  }

  template <std::uint8_t W = NWorlds - 1U>
  __device__ void update_time_at_dest(unsigned const k, delta_t const t) {
    for (auto i = k; i != max_transfers_ + 1U; ++i) {
      time_at_dest_.update_min(i * NWorlds + W, t);
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
  bool no_compulsory_reservation_;
  day_idx_t base_;
  cuda::std::span<std::pair<location_idx_t, unixtime_t> const> starts_;
  device_bitvec<std::uint64_t const> is_dest_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  cuda::std::span<location_idx_t const> td_dest_locs_;
  d_vecvec_view<td_dest_offsets_t> td_dest_;
  device_times<SearchDir, kNSlots> round_times_;
  device_times<SearchDir, kNSlots> best_;
  device_times<SearchDir, kNSlots> tmp_;
  device_times<SearchDir, 1U> time_at_dest_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;
  device_bitvec<std::uint32_t> rt_transport_mark_;

  // tracking for efficient reset in reuse_previous_arrivals
  cuda::std::span<std::uint32_t> round_touched_;
  std::uint32_t round_touched_stride_{0U};
  // true once a round loop has written round_times_; cleared by
  // reset_arrivals(). Gates the reuse_previous_arrivals kernel.
  std::uint32_t has_reusable_round_times_{0U};
  cuda::std::span<location_idx_t const> dest_locs_;
  std::uint32_t n_dest_locs_{0U};

  // earliest transports per flat (route,stop)
  cuda::std::span<std::uint32_t> et_result_;
  cuda::std::span<std::uint32_t> et_task_list_;
  std::uint32_t* et_task_count_;  // number of tasks this round

  // marked routes this round
  cuda::std::span<std::uint32_t> route_list_;
  std::uint32_t* route_list_count_;

  // scheduled+rt: original static traffic days for world 0
  d_vecmap_view<transport_idx_t, bitfield_idx_t> sched_transport_traffic_days_;

  // ping bounds
  delta_t const* bounds_{nullptr};
  std::uint32_t bounds_last_k_{0U};
};

}  // namespace nigiri::routing::gpu