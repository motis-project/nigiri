#pragma once

#include "cooperative_groups.h"

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/bitvec.cuh"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

template <direction SearchDir, bool Rt, via_offset_t Vias>
struct raptor_impl {
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  constexpr static bool is_better(auto a, auto b) {
    return kFwd ? a < b : a > b;
  }
  constexpr static bool is_better_or_eq(auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  }
  constexpr static auto get_best(auto a, auto b) {
    return is_better(a, b) ? a : b;
  }
  constexpr static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  constexpr auto min(auto x, auto y) { return x <= y ? x : y; }
  constexpr static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  __device__ void execute(unixtime_t const start_time,
                          std::uint8_t const max_transfers,
                          unixtime_t const worst_time_at_dest) {
    namespace cg = cooperative_groups;

    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    for (auto i = global_t_id; i < starts_.size(); i += global_stride) {
      auto const [l, t] = starts_[i];
      auto const v = (Vias != 0 && is_via_[0][to_idx(l)]) ? 1U : 0U;
      best_[to_idx(l)][v] = unix_to_delta(base(), t);
      round_times_[0U][to_idx(l)][v] = unix_to_delta(base(), t);
      station_mark_.mark(to_idx(l));
    }

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto i = global_t_id; i < kMaxTransfers + 1U; i += global_stride) {
      time_at_dest_[i] = get_best(d_worst_at_dest, time_at_dest_[i]);
    }

    auto const end_k = min(max_transfers, kMaxTransfers) + 1U;
    for (auto k = 1U; k != end_k; ++k) {
      // ==================
      // RAPTOR ROUND START
      // ------------------

      // Reuse best time from previous time at start (for range queries).
      for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
        for (auto v = 0U; v != Vias + 1; ++v) {
          best_[i][v] = get_best(round_times_[k][i][v], best_[i][v]);
        }
        if (is_dest_[i]) {
          update_time_at_dest(k, best_[i][Vias]);
        }
      }

      // Mark every route at all stations marked in the previous round.
      auto any_marked = cuda::std::atomic_bool{false};
      for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
        if (station_mark_[i]) {
          if (!any_marked) {
            any_marked = true;
          }
          for (auto r : tt_.location_routes_[location_idx_t{i}]) {
            route_mark_.mark(to_idx(r));
          }
        }
      }

      cg::this_grid().sync();

      if (!any_marked) {
        break;
      }

      cuda::std::swap(prev_station_mark_, station_mark_);
      station_mark_.zero_out();

      any_marked =
          (allowed_claszes_ == all_clasz_allowed())
              ? (require_bike_transport_ ? loop_routes<false, true>(k)
                                         : loop_routes<false, false>(k))
              : (require_bike_transport_ ? loop_routes<true, true>(k)
                                         : loop_routes<true, false>(k));

      if (!any_marked) {
        break;
      }

      route_mark_.zero_out();

      cuda::std::swap(prev_station_mark_, station_mark_);
      station_mark_.zero_out();

      update_transfers(k);
      update_intermodal_footpaths(k);
      update_footpaths(k);
    }

    // TODO
    //    is_dest_.for_each_set_bit([&](auto const i) {
    //      for (auto k = 1U; k != end_k; ++k) {
    //        auto const dest_time = round_times_[k][i][Vias];
    //        if (dest_time != kInvalid) {
    //          auto const [optimal, it, dominated_by] = results.add(
    //              journey{.legs_ = {},
    //                      .start_time_ = start_time,
    //                      .dest_time_ = delta_to_unix(base(), dest_time),
    //                      .dest_ = location_idx_t{i},
    //                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
    //        }
    //      }
    //    });
  }

  __device__ date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <bool WithClaszFilter, bool WithBikeFilter>
  __device__ bool loop_routes(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    auto any_marked = cuda::std::atomic_bool{};
    for (auto i = global_t_id; i < tt_.n_routes_; i += global_stride) {
      auto const r = route_idx_t{i};

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

      any_marked |= section_bike_filter ? update_route<true>(k, r)
                                        : update_route<false>(k, r);
    }
    return any_marked;
  }

  __device__ void update_transfers(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
      for (auto v = 0U; v != Vias + 1; ++v) {
        auto const tmp_time = tmp_[i][v];
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
                : dir(adjusted_transfer_time(
                          transfer_time_settings_,
                          tt_.transfer_time_[location_idx_t{i}].count()) +
                      stay.count());
        auto const fp_target_time =
            static_cast<delta_t>(tmp_time + transfer_time);

        if (is_better(fp_target_time, best_[i][target_v]) &&
            is_better(fp_target_time, time_at_dest_[k])) {
          if (lb_[i] == kUnreachable ||
              !is_better(fp_target_time + dir(lb_[i]), time_at_dest_[k])) {
            return;
          }

          round_times_[k][i][target_v] = fp_target_time;
          best_[i][target_v] = fp_target_time;
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
      auto const l_idx = location_idx_t{i};
      auto const& fps =
          kFwd ? tt_.footpaths_out_[l_idx] : tt_.footpaths_in_[l_idx];

      for (auto const& fp : fps) {
        auto const target = to_idx(fp.target());

        for (auto v = 0U; v != Vias + 1; ++v) {
          auto const tmp_time = tmp_[i][v];
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

          if (is_better(fp_target_time, best_[target][target_v]) &&
              is_better(fp_target_time, time_at_dest_[k])) {
            auto const lower_bound = lb_[target];
            if (lower_bound == kUnreachable ||
                !is_better(fp_target_time + dir(lower_bound),
                           time_at_dest_[k])) {
              continue;
            }

            round_times_[k][target][target_v] = fp_target_time;
            best_[target][target_v] = fp_target_time;
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
      if (prev_station_mark_[i] || station_mark_[i]) {
        auto const l = location_idx_t{i};
        if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
          auto const best_time = get_best(best_[i][Vias], tmp_[i][Vias]);
          if (best_time == kInvalid) {
            continue;
          }
          auto const stay = Vias != 0 && is_via_[Vias - 1U][i]
                                ? via_stops_[Vias - 1U].stay_
                                : 0_minutes;
          auto const end_time =
              clamp(best_time + stay.count() + dir(dist_to_end_[i]));

          if (is_better(end_time, best_[kIntermodalTarget][Vias])) {
            round_times_[k][kIntermodalTarget][Vias] = end_time;
            best_[kIntermodalTarget][Vias] = end_time;
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

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_first = i == 0U;
      auto const is_last = i == stop_seq.size() - 1U;

      auto current_best = std::array<delta_t, Vias + 1>{};
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
              get_best(round_times_[k - 1][l_idx][target_v],
                       tmp_[l_idx][target_v], best_[l_idx][target_v]);

          auto higher_v_best = kInvalid;
          for (auto higher_v = Vias; higher_v != target_v; --higher_v) {
            higher_v_best =
                get_best(higher_v_best, round_times_[k - 1][l_idx][higher_v],
                         tmp_[l_idx][higher_v], best_[l_idx][higher_v]);
          }

          if (is_better(by_transport, current_best[v]) &&
              is_better(by_transport, time_at_dest_[k]) &&
              is_better(by_transport, higher_v_best) &&
              lb_[l_idx] != kUnreachable &&
              is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k])) {
            tmp_[l_idx][target_v] =
                get_best(by_transport, tmp_[l_idx][target_v]);
            station_mark_.mark(l_idx);
            current_best[v] = by_transport;
            any_marked = true;
          }

          if (is_via_and_dest) {
            auto const dest_v = target_v + 1;
            assert(dest_v == Vias);
            auto const best_dest =
                get_best(round_times_[k - 1][l_idx][dest_v],
                         tmp_[l_idx][dest_v], best_[l_idx][dest_v]);

            if (is_better(by_transport, best_dest) &&
                is_better(by_transport, time_at_dest_[k]) &&
                lb_[l_idx] != kUnreachable &&
                is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k])) {

              tmp_[l_idx][dest_v] = get_best(by_transport, tmp_[l_idx][dest_v]);
              station_mark_.mark(l_idx);
              any_marked = true;
            }
          }
        }
      }

      if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
          !prev_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      for (auto v = 0U; v != Vias + 1; ++v) {
        if (!et[v].is_valid() && !prev_station_mark_[l_idx]) {
          continue;
        }

        auto const target_v = v + v_offset[v];
        auto const et_time_at_stop =
            et[v].is_valid()
                ? time_at_stop(r, et[v], stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)
                : kInvalid;
        auto const prev_round_time = round_times_[k - 1][l_idx][target_v];
        if (prev_round_time != kInvalid &&
            is_better_or_eq(prev_round_time, et_time_at_stop)) {
          auto const [day, mam] = split(prev_round_time);
          auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                                     stp.location_idx());
          current_best[v] = get_best(current_best[v], best_[l_idx][target_v],
                                     tmp_[l_idx][target_v]);
          if (new_et.is_valid() &&
              (current_best[v] == kInvalid ||
               is_better_or_eq(
                   time_at_stop(r, new_et, stop_idx,
                                kFwd ? event_type::kDep : event_type::kArr),
                   et_time_at_stop))) {
            et[v] = new_et;
            v_offset[v] = 0;
          }
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

    constexpr auto const kNDaysToIterate = day_idx_t::value_t{2U};
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
      auto const ev_time_range =
          it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                   get_end_it(event_times)};
      if (ev_time_range.empty()) {
        continue;
      }

      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = ev.mam();

        if (is_better_or_eq(time_at_dest_[k],
                            to_delta(day, ev_mam) + dir(lb_[to_idx(l)]))) {
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
    for (auto i = k; i != time_at_dest_.size(); ++i) {
      time_at_dest_[i] = get_best(time_at_dest_[i], t);
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
  cuda::std::array<device_bitvec_view, kMaxVias> is_via_;
  cuda::std::span<via_stop const> via_stops_;
  device_bitvec_view is_dest_;
  device_bitvec_view end_reachable_;
  cuda::std::span<std::uint16_t const> dist_to_end_;
  cuda::std::span<std::uint16_t const> lb_;
  device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>> round_times_;
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> best_;
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> tmp_;
  cuda::std::span<delta_t> time_at_dest_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
};

}  // namespace nigiri::routing::gpu