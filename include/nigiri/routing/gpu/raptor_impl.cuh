#pragma once

#include "cooperative_groups.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/it_range.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
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

#define start_timing() (void)0
#define debug_timing(...)

// #define start_timing() auto const round_start = clock64()
// #define debug_timing(str, ...)                        \
//  if (global_t_id == 0) {                             \
//    printf(str ": %" PRIi64 " cycles\n", __VA_ARGS__, \
//           clock64() - round_start);                  \
//  }

#define kInvalid (kInvalidDelta<SearchDir>)
#define kFwd (SearchDir == direction::kForward)
#define kBwd (SearchDir == direction::kBackward)
#define kUnreachable (std::numeric_limits<std::uint16_t>::max())
#define kIntermodalTarget (get_special_station(special_station::kEnd))

__device__ char const* b2s(bool const b) { return b ? "true" : "false"; }

template <direction SearchDir, bool Rt, via_offset_t Vias>
struct raptor_impl {
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

  __device__ void execute(unixtime_t const start_time,
                          std::uint8_t const max_transfers,
                          unixtime_t const worst_time_at_dest) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    for (auto i = global_t_id; i < dist_to_end_.size(); i += global_stride) {
      if (dist_to_end_[i] != kUnreachable) {
        end_reachable_.mark(i);
      }
    }

    for (auto i = global_t_id; i < starts_.size(); i += global_stride) {
      auto const l = starts_[i].first;
      auto const t = unix_to_delta(base(), starts_[i].second);
      auto const v = (Vias != 0 && is_via_[0][to_idx(l)]) ? 1U : 0U;
      best_.update_min(l, v, t);
      round_times_.update_min(0U, l, v, t, make_start_bc());
      station_mark_.mark(to_idx(l));
    }

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto i = global_t_id; i < kMaxTransfers + 2U; i += global_stride) {
      time_at_dest_.update_min(i, d_worst_at_dest);
    }

    sync();

    // +2 (not +1): rounds run 1..max_transfers+1 so that journeys with exactly
    // max_transfers transfers (reached in round max_transfers+1) are found. CPU
    // raptor uses the same +2; the GPU previously dropped the last round, which
    // was invisible at high max_transfers but emptied pong (max_transfers=0).
    auto const end_k = min(max_transfers, kMaxTransfers) + 2U;
    for (auto k = 1U; k != end_k; ++k) {
      start_timing();

      // Reuse best time from previous time at start (for range queries).
      for (auto i = global_t_id; i < n_locations_; i += global_stride) {
        debug("round %u: location %d / %u\n", k, i, n_locations_);
        auto const l = location_idx_t{i};
        for (auto v = 0U; v != Vias + 1; ++v) {
          best_.update_min(l, v, round_times_.get(k, l, v));
        }
        if (is_dest_[i]) {
          update_time_at_dest(k, best_.get(l, Vias));
        }
      }

      // Mark every route at all stations marked in the previous round.
      if (global_t_id == 0) {
        *any_marked_ = 0U;
      }

      sync();
      debug_timing("round %u: after reuse times from prev start", k);

      for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
        if (station_mark_[i]) {
          if (!tt_.location_routes_[location_idx_t{i}].empty() &&
              !*any_marked_) {
            atomicOr(any_marked_, 1U);
          }
          for (auto r : tt_.location_routes_[location_idx_t{i}]) {
            debug("round %u: marking route %u\n", k, to_idx(r));
            route_mark_.mark(to_idx(r));
          }
        }
      }

      sync();
      debug_timing("round %u: after station flags -> route flags", k);

      if (!*any_marked_) {
        debug("round %d: no route marked -> break;\n", k);
        debug_timing("round %u: TOTAL", k);
        break;
      }

      prev_station_mark_.swap_reset(station_mark_);
      if (global_t_id == 0) {
        *any_marked_ = false;
      }
      sync();

      // Load-balanced boarding pre-pass: precompute earliest transports for all
      // (route,stop) candidates in parallel, so loop_routes/update_route can read
      // them instead of doing the divergent get_earliest_transport per route.
      compute_et(k);
      sync();

      (allowed_claszes_ == all_clasz_allowed())
          ? (require_bike_transport_ ? loop_routes<false, true>(k)
                                     : loop_routes<false, false>(k))
          : (require_bike_transport_ ? loop_routes<true, true>(k)
                                     : loop_routes<true, false>(k));

      sync();
      debug_timing("round %u: after visit routes", k);

      if (!*any_marked_) {
        debug("round %d: no location marked after loop_routes -> break;\n", k);
        debug_timing("round %u: TOTAL", k);
        break;
      }
      prev_station_mark_.swap_reset(station_mark_);
      sync();
      debug_timing("round %u: after swap station marks", k);

      update_transfers(k);
      update_intermodal_footpaths(k);
      update_footpaths(k);

      route_mark_.reset();
      sync();
      debug_timing("round %u: TOTAL", k);
    }
  }

  __device__ __forceinline__ void sync() const {
    cooperative_groups::this_grid().sync();
  }

  // Reconstruct the journey arriving at `dest` using exactly `K` rounds by
  // following the breadcrumbs back to the start. Pure pointer-chase: each
  // breadcrumb carries (transport, board_stop, alight_stop); the route comes
  // from transport_route_, the day is recovered from the arrival time minus the
  // event's over-midnight offset, and the transfer/footpath hop is derived from
  // the alight location. Legs are written in reverse-chronological order; the
  // host reverses them. Writes out->valid_ = 1 on success.
  __device__ void reconstruct_journey(location_idx_t const dest,
                                      unsigned const K,
                                      gpu_journey* out) {
    out->valid_ = 0U;
    auto cur_v = static_cast<via_offset_t>(Vias);
    auto const dest_time =
        round_times_.get(static_cast<std::uint8_t>(K), dest, cur_v);
    if (dest_time == kInvalid) {
      return;
    }
    out->dest_l_ = dest.v_;
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
      // egress only applies to intermodal queries; a real station may share the
      // special kEnd index, so guard with is_intermodal_dest()
      if (bc_is_start(bc) ||
          (is_intermodal_dest() && cur_l == kIntermodalTarget)) {
        break;
      }

      auto const t_idx = transport_idx_t{bc_transport(bc)};
      auto const board = static_cast<stop_idx_t>(bc_board(bc));
      auto const alight = static_cast<stop_idx_t>(bc_alight(bc));
      auto const r = tt_.transport_route_[t_idx];
      auto const arr_at_cur =
          round_times_.get(static_cast<std::uint8_t>(cur_k), cur_l, cur_v);

      // recover the traffic (first-departure) day of the transport
      auto const event_mam_full =
          tt_.event_mam(r, t_idx, alight, ev_arr_type).count();
      auto const split_arr = split(arr_at_cur);
      auto found_day = false;
      auto day = day_idx_t{0U};
      auto train_arr = kInvalid;
      for (auto off = 0; off != 2; ++off) {
        auto const cand = as_int(split_arr.first) - event_mam_full / 1440 -
                          (kFwd ? off : -off);
        if (cand < 0) {
          continue;
        }
        if (!is_transport_active(t_idx, static_cast<std::size_t>(cand))) {
          continue;
        }
        auto const tr =
            transport{t_idx, day_idx_t{static_cast<day_idx_t::value_t>(cand)}};
        auto const ev = time_at_stop(r, tr, alight, ev_arr_type);
        if (is_better_or_eq(ev, arr_at_cur)) {
          day = day_idx_t{static_cast<day_idx_t::value_t>(cand)};
          train_arr = ev;
          found_day = true;
          break;
        }
      }
      if (!found_day) {
        return;  // could not recover -> leave invalid
      }

      auto const tr = transport{t_idx, day};
      auto const dep_at_board = time_at_stop(r, tr, board, ev_dep_type);
      auto const stop_seq = tt_.route_location_seq_[r];
      auto const board_loc = stop{stop_seq[board]}.location_idx();
      auto const alight_loc = stop{stop_seq[alight]}.location_idx();

      // footpath/transfer leg from the train's alighting stop to cur_l. nigiri
      // represents the same-station transfer between two trains as a self
      // footpath (B->B) with the transfer time, so we emit it whenever there is
      // a non-zero cost (a real footpath, or a same-station transfer), and skip
      // it only for a zero-cost direct arrival (e.g. at the destination).
      if (alight_loc != cur_l || train_arr != arr_at_cur) {
        if (n >= kMaxRecLegs) {
          return;
        }
        auto& lg = out->legs_[n++];
        lg.is_footpath_ = 1U;
        lg.from_l_ = alight_loc.v_;
        lg.to_l_ = cur_l.v_;
        lg.dep_ = train_arr;
        lg.arr_ = arr_at_cur;
        lg.fp_duration_ = static_cast<std::uint16_t>(
            kFwd ? (arr_at_cur - train_arr) : (train_arr - arr_at_cur));
      }

      // transport leg board_loc -> alight_loc
      if (n >= kMaxRecLegs) {
        return;
      }
      auto& lg = out->legs_[n++];
      lg.is_footpath_ = 0U;
      lg.from_l_ = board_loc.v_;
      lg.to_l_ = alight_loc.v_;
      lg.dep_ = dep_at_board;
      lg.arr_ = train_arr;
      lg.transport_ = t_idx.v_;
      lg.day_ = static_cast<std::uint16_t>(day.v_);
      lg.enter_stop_ = board;
      lg.exit_stop_ = alight;

      cur_l = board_loc;
      cur_k -= 1U;
    }

    out->start_l_ = cur_l.v_;
    out->n_legs_ = static_cast<std::uint8_t>(n);
    out->valid_ = 1U;
  }

  __device__ date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <bool WithClaszFilter, bool WithBikeFilter>
  __device__ void loop_routes(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    for (auto i = global_t_id; i < tt_.n_routes_; i += global_stride) {
      if (!route_mark_.test(i)) {
        continue;
      }

      auto const r = route_idx_t{i};
      debug("round %u: processing route %d\n", k, i);
      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          continue;
        }
      }

      auto section_bike_filter = false;
      if constexpr (WithBikeFilter) {
        auto const bikes_allowed_on_all_sections =
            tt_.route_bikes_allowed_.test(i * 2);
        if (!bikes_allowed_on_all_sections) {
          auto const bikes_allowed_on_some_sections =
              tt_.route_bikes_allowed_.test(i * 2 + 1);
          if (!bikes_allowed_on_some_sections) {
            continue;
          }
          section_bike_filter = true;
        }
      }

      auto const route_any_marked = section_bike_filter
                                        ? update_route<true>(k, r)
                                        : update_route<false>(k, r);
      if (route_any_marked && !*any_marked_) {
        debug("round %u: route=%d -> any_marked=true\n", k, i);
        atomicOr(any_marked_, 1U);
      }
    }
  }

  __device__ void update_transfers(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      if (!prev_station_mark_.test(i)) {
        continue;
      }

      auto const l = location_idx_t{i};

      for (auto v = 0U; v != Vias + 1; ++v) {
        auto const tmp_time = tmp_.get(l, v);
        if (tmp_time == kInvalid) {
          continue;
        }

        auto const is_via = v != Vias && is_via_[v][i];
        auto const target_v = is_via ? v + 1 : v;
        auto const is_dest = target_v == Vias && is_dest_[i];
        auto const stay = is_via ? via_stops_[v].stay_ : 0_minutes;

        auto const transfer_time =
            (!is_intermodal_dest() && is_dest)
                ? 0
                : dir(adjusted_transfer_time(transfer_time_settings_,
                                             tt_.transfer_time_[l].count()) +
                      stay.count());
        auto const fp_target_time =
            static_cast<delta_t>(tmp_time + transfer_time);

        if (is_better(fp_target_time, best_.get(l, target_v)) &&
            is_better(fp_target_time, time_at_dest_.get(k))) {
          round_times_.update_min(k, l, target_v, fp_target_time,
                                  tmp_.get_bc(0U, l, v));
          best_.update_min(l, target_v, fp_target_time);
          station_mark_.mark(i);
          if (is_dest) {
            update_time_at_dest(k, fp_target_time);
          }
        }
      }
    }
  }

  __device__ void update_footpaths(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      if (!prev_station_mark_.test(i)) {
        continue;
      }

      auto const l = location_idx_t{i};
      auto const& fps = kFwd ? tt_.footpaths_out_[l] : tt_.footpaths_in_[l];

      for (auto const& fp : fps) {
        auto const target = to_idx(fp.target());

        for (auto v = 0U; v != Vias + 1; ++v) {
          auto const tmp_time = tmp_.get(l, v);
          if (tmp_time == kInvalid) {
            continue;
          }

          auto const start_is_via = v != Vias && is_via_[v][i];
          auto const start_v = start_is_via ? v + 1 : v;

          auto const target_is_via =
              start_v != Vias && is_via_[start_v][target];
          auto const target_v = target_is_via ? start_v + 1 : start_v;
          auto stay = 0_minutes;
          if (start_is_via) {
            stay += via_stops_[v].stay_;
          }
          if (target_is_via) {
            stay += via_stops_[start_v].stay_;
          }

          auto const fp_target_time = clamp(
              tmp_time + dir(adjusted_transfer_time(transfer_time_settings_,
                                                    fp.duration().count()) +
                             stay.count()));

          if (is_better(fp_target_time, best_.get(fp.target(), target_v)) &&
              is_better(fp_target_time, time_at_dest_.get(k))) {
            round_times_.update_min(k, fp.target(), target_v, fp_target_time,
                                    tmp_.get_bc(0U, l, v));
            best_.update_min(fp.target(), target_v, fp_target_time);
            station_mark_.mark(target);
            if (target_v == Vias && is_dest_[target]) {
              update_time_at_dest(k, fp_target_time);
            }
          }
        }
      }
    }
  }

  __device__ void update_intermodal_footpaths(unsigned const k) {
    if (dist_to_end_.empty()) {
      return;
    }

    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      if (!prev_station_mark_.test(i)) {
        continue;
      }

      auto const l = location_idx_t{i};
      if (prev_station_mark_[i] || station_mark_[i]) {
        if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
          auto const best_time =
              get_best(best_.get(l, Vias), tmp_.get(l, Vias));
          if (best_time == kInvalid) {
            continue;
          }
          auto const stay = Vias != 0 && is_via_[Vias - 1U][i]
                                ? via_stops_[Vias - 1U].stay_
                                : 0_minutes;
          auto const end_time =
              clamp(best_time + stay.count() + dir(dist_to_end_[i]));

          if (is_better(end_time, best_.get(kIntermodalTarget, Vias))) {
            round_times_.update_min(k, kIntermodalTarget, Vias, end_time,
                                    make_egress_bc(i));
            best_.update_min(kIntermodalTarget, Vias, end_time);
            update_time_at_dest(k, end_time);
          }
        }
      }
    }
  }

  template <bool WithSectionBikeFilter>
  __device__ bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto et = std::array<transport, Vias + 1>{};
    auto v_offset = std::array<std::size_t, Vias + 1>{};
    // boarding stop_idx of the transport currently in et[v] (for breadcrumbs)
    auto et_board_stop = cuda::std::array<stop_idx_t, Vias + 1>{};

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l = stp.location_idx();
      auto const l_idx = cista::to_idx(l);
      auto const is_first = i == 0U;
      auto const is_last = i == stop_seq.size() - 1U;

      auto current_best = cuda::std::array<delta_t, Vias + 1>{};
      current_best.fill(kInvalid);

      // v = via state when entering the transport
      // v + v_offset = via state at the current stop after entering the
      // transport (v_offset > 0 if the transport passes via stops)
      for (auto j = 0U; j != Vias + 1; ++j) {
        auto const v = Vias - j;
        if (!et[v].is_valid() && !prev_station_mark_[l_idx]) {
          continue;
        }

        if constexpr (WithSectionBikeFilter) {
          if (!is_first &&
              !const_cast<device_timetable const&>(tt_)
                   .route_bikes_allowed_per_section_[r][kFwd ? stop_idx - 1
                                                             : stop_idx]) {
            et[v] = {};
            v_offset[v] = 0;
          }
        }

        auto target_v = v + v_offset[v];

        if (et[v].is_valid() && stp.can_finish<SearchDir>(is_wheelchair_)) {
          auto const by_transport = time_at_stop(
              r, et[v], stop_idx, kFwd ? event_type::kArr : event_type::kDep);

          auto const is_via = target_v != Vias && is_via_[target_v][l_idx];
          auto const is_no_stay_via =
              is_via && via_stops_[target_v].stay_ == 0_minutes;

          // special case: stop is via with stay > 0m + destination
          auto const is_via_and_dest =
              is_via && !is_no_stay_via &&
              (is_dest_[l_idx] ||
               (is_intermodal_dest() && end_reachable_[l_idx]));

          if (is_no_stay_via) {
            ++v_offset[v];
            ++target_v;
          }

          current_best[v] =
              get_best(round_times_.get(k - 1, l, target_v),
                       tmp_.get(l, target_v), best_.get(l, target_v));

          auto higher_v_best = kInvalid;
          for (auto higher_v = Vias; higher_v != target_v; --higher_v) {
            higher_v_best =
                get_best(higher_v_best, round_times_.get(k - 1, l, higher_v),
                         tmp_.get(l, higher_v), best_.get(l, higher_v));
          }

          if (is_better(by_transport, current_best[v]) &&
              is_better(by_transport, time_at_dest_.get(k)) &&
              is_better(by_transport, higher_v_best)) {
            tmp_.update_min(l, target_v, by_transport,
                            make_transport_payload(et[v].t_idx_.v_,
                                                   et_board_stop[v], stop_idx));
            station_mark_.mark(l_idx);
            current_best[v] = by_transport;
            any_marked = true;
            debug("round %u: route=%u, marking l=%u\n", k, to_idx(r), l_idx);
          }

          if (is_via_and_dest) {
            auto const dest_v = target_v + 1;
            assert(dest_v == Vias);
            auto const best_dest =
                get_best(round_times_.get(k - 1, l, dest_v),
                         tmp_.get(l, dest_v), best_.get(l, dest_v));

            if (is_better(by_transport, best_dest) &&
                is_better(by_transport, time_at_dest_.get(k))) {
              tmp_.update_min(l, dest_v, by_transport,
                              make_transport_payload(et[v].t_idx_.v_,
                                                     et_board_stop[v], stop_idx));
              station_mark_.mark(l_idx);
              any_marked = true;
              debug("round %u: route=%u, marking l=%u\n", k, to_idx(r), l_idx);
            }
          }
        }
      }

      if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
          !prev_station_mark_[l_idx]) {
        debug(
            "round %u: route=%u, stop=%u [l=%u] -> "
            "is_last=%s, can_start=%s, prev_station_mark=%s   => no new et\n",
            k, r.v_, stop_idx, l_idx, b2s(is_last),
            b2s(stp.can_start<SearchDir>(is_wheelchair_)),
            b2s(prev_station_mark_[l_idx]));
        continue;
      }

      for (auto v = 0U; v != Vias + 1; ++v) {
        if (!et[v].is_valid() && !prev_station_mark_[l_idx]) {
          debug(
              "round %u: route=%u, stop=%u [l=%u] -> et_valid=%s, "
              "prev_station_mark=%s   => no new et\n",
              k, r.v_, stop_idx, l_idx, b2s(et[v].is_valid()),
              b2s(prev_station_mark_[l_idx]));
          continue;
        }

        auto const target_v = v + v_offset[v];
        auto const et_time_at_stop =
            et[v].is_valid()
                ? time_at_stop(r, et[v], stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)
                : kInvalid;
        auto const prev_round_time = round_times_.get(k - 1, l, target_v);
        if (prev_round_time != kInvalid &&
            is_better_or_eq(prev_round_time, et_time_at_stop)) {
          transport new_et{};
          if constexpr (Vias == 0) {
            // boarding et precomputed by compute_et (load-balanced pre-pass)
            new_et = unpack_et(
                et_result_[tt_.route_stop_offset_[to_idx(r)] + stop_idx]);
          } else {
            auto const [day, mam] = split(prev_round_time);
            new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                            stp.location_idx());
          }
          current_best[v] = get_best(current_best[v], best_.get(l, target_v),
                                     tmp_.get(l, target_v));
          if (new_et.is_valid() &&
              (current_best[v] == kInvalid ||
               is_better_or_eq(
                   time_at_stop(r, new_et, stop_idx,
                                kFwd ? event_type::kDep : event_type::kArr),
                   et_time_at_stop))) {
            et[v] = new_et;
            v_offset[v] = 0;
            et_board_stop[v] = stop_idx;
            debug("round %u: route=%u, stop=%u [l=%u] -> transport=%u\n", k,
                  r.v_, stop_idx, l_idx, new_et.t_idx_.v_);
          }
        } else {
          debug(
              "round %u: route=%u, stop=%u [l=%u] -> prev_round_time=%d, "
              "time_at_stop=%d => no new et\n",
              k, r.v_, stop_idx, l_idx, prev_round_time, et_time_at_stop);
        }
      }
    }
    return any_marked;
  }

  __device__ transport
  get_earliest_transport(unsigned const k,
                         route_idx_t const r,
                         stop_idx_t const stop_idx,
                         day_idx_t const day_at_stop,
                         minutes_after_midnight_t const mam_at_stop,
                         location_idx_t const l) {
    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    // Must match the CPU raptor: look up to kMaxTravelTime days ahead for the
    // next departure. The previous value (2) silently missed connections when
    // the next active service was >2 days out (sparse/rural service, or reduced
    // service near the timetable-start dates) -> GPU returned empty where CPU
    // found journeys.
    constexpr auto const kNDaysToIterate = static_cast<day_idx_t::value_t>(
        kMaxTravelTime / std::chrono::days{1} + 1U);
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      // Skip days on which this route runs no service before iterating its
      // events (matches the CPU raptor). Without this the wider day window
      // (kNDaysToIterate) would scan every event on every inactive day.
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
    return as_int(day) >= 0 &&
           tt_.bitfields_[tt_.route_traffic_days_[r]].test(
               static_cast<std::size_t>(as_int(day)));
  }

  static constexpr std::uint64_t kEtInvalid = ~std::uint64_t{0};

  __device__ __forceinline__ std::uint64_t pack_et(transport const t) const {
    return t.is_valid() ? ((static_cast<std::uint64_t>(to_idx(t.day_)) << 32) |
                           static_cast<std::uint64_t>(to_idx(t.t_idx_)))
                        : kEtInvalid;
  }

  __device__ __forceinline__ transport unpack_et(std::uint64_t const p) const {
    return p == kEtInvalid
               ? transport{}
               : transport{transport_idx_t{static_cast<std::uint32_t>(
                               p & 0xFFFFFFFFU)},
                           day_idx_t{static_cast<std::uint16_t>(p >> 32)}};
  }

  // Load-balanced boarding pre-pass (the "compute_et" optimization): instead of
  // one thread per route scanning all its stops (huge per-route work variance ->
  // warps stall at the round barrier), parallelize the earliest-transport lookup
  // over a compacted list of (route,stop) boarding candidates -- one task per
  // thread. update_route then just reads et_result_ instead of calling
  // get_earliest_transport inline.
  __device__ void compute_et(unsigned const k) {
    if constexpr (Vias == 0) {
      auto const gid = get_global_thread_id();
      auto const stride = get_global_stride();
      auto const total = tt_.route_stop_offset_[tt_.n_routes_];

      // phase 1a: compact candidate tasks (default non-candidates to invalid).
      if (gid == 0U) {
        *et_task_count_ = 0U;
      }
      sync();
      for (auto flat = gid; flat < total; flat += stride) {
        auto const r = route_idx_t{tt_.route_of_stop_[flat]};
        if (!route_mark_.test(to_idx(r))) {
          continue;
        }
        et_result_[flat] = kEtInvalid;
        auto const stop_seq = tt_.route_location_seq_[r];
        auto const stop_idx =
            static_cast<stop_idx_t>(flat - tt_.route_stop_offset_[to_idx(r)]);
        auto const is_dir_last =
            kFwd ? (stop_idx + 1U == stop_seq.size()) : (stop_idx == 0U);
        if (is_dir_last) {
          continue;
        }
        auto const stp = stop{stop_seq[stop_idx]};
        auto const l = stp.location_idx();
        if (!prev_station_mark_[cista::to_idx(l)] ||
            !stp.can_start<SearchDir>(is_wheelchair_) ||
            round_times_.get(k - 1, l, 0U) == kInvalid) {
          continue;
        }
        et_task_list_[atomicAdd(et_task_count_, 1U)] = flat;
      }
      sync();

      // phase 1b: one earliest-transport lookup per task (no route divergence).
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
        et_result_[flat] =
            pack_et(get_earliest_transport(k, r, stop_idx, day, mam, l));
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
  device_timetable tt_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  transfer_time_settings transfer_time_settings_;
  std::uint8_t max_transfers_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  day_idx_t base_;
  unixtime_t worst_time_at_dest_;
  bool is_intermodal_dest_;
  bool is_wheelchair_;
  cuda::std::span<std::pair<location_idx_t, unixtime_t> const> starts_;
  cuda::std::array<device_bitvec<std::uint64_t const>, kMaxVias> is_via_;
  cuda::std::span<via_stop const> via_stops_;
  device_bitvec<std::uint64_t const> is_dest_;
  device_bitvec<std::uint32_t> end_reachable_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  device_times<SearchDir, Vias + 1> round_times_;
  device_times<SearchDir, Vias + 1> best_;
  device_times<SearchDir, Vias + 1> tmp_;
  device_times<SearchDir, 1U> time_at_dest_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;
  cuda::std::span<std::uint64_t> et_result_;  // packed et per flat (route,stop)
  cuda::std::span<std::uint32_t> et_task_list_;  // compacted boarding candidates
  std::uint32_t* et_task_count_;  // number of tasks this round
};

}  // namespace nigiri::routing::gpu