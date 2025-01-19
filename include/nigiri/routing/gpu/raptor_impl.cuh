#pragma once

#include "cooperative_groups.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/device_times.h"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

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
      round_times_.update_min(0U, l, v, t);
      station_mark_.mark(to_idx(l));
    }

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto i = global_t_id; i < kMaxTransfers + 1U; i += global_stride) {
      time_at_dest_.update_min(i, d_worst_at_dest);
    }

    sync();

    auto const end_k = min(max_transfers, kMaxTransfers) + 1U;
    for (auto k = 1U; k != end_k; ++k) {
      // Reuse best time from previous time at start (for range queries).
      for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
        printf("round %u: location %d\n", k, i);
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

      for (auto i = global_t_id; i < tt_.n_locations_; i += global_stride) {
        if (station_mark_[i]) {
          if (!tt_.location_routes_[location_idx_t{i}].empty() &&
              !*any_marked_) {
            atomicOr(any_marked_, 1U);
          }
          for (auto r : tt_.location_routes_[location_idx_t{i}]) {
            printf("round %u: marking route %u\n", k, to_idx(r));
            route_mark_.mark(to_idx(r));
          }
        }
      }

      sync();

      if (!*any_marked_) {
        printf("round %d: no route marked -> break;\n", k);
        break;
      }

      if (global_t_id == 0) {
        cuda::std::swap(prev_station_mark_, station_mark_);
        *any_marked_ = false;

        for (auto l = location_idx_t{0U}; l != n_locations_; ++l) {
          printf("l=%u: %s\n", l.v_,
                 prev_station_mark_.test(l.v_) ? "marked" : "-");
        }
      }
      sync();
      station_mark_.zero_out();
      sync();

      (allowed_claszes_ == all_clasz_allowed())
          ? (require_bike_transport_ ? loop_routes<false, true>(k)
                                     : loop_routes<false, false>(k))
          : (require_bike_transport_ ? loop_routes<true, true>(k)
                                     : loop_routes<true, false>(k));

      sync();

      if (!*any_marked_) {
        printf("round %d: no location marked after loop_routes -> break;\n", k);
        break;
      }

      if (global_t_id == 0) {
        cuda::std::swap(prev_station_mark_, station_mark_);
      }
      sync();
      station_mark_.zero_out();
      sync();

      update_transfers(k);
      update_intermodal_footpaths(k);
      update_footpaths(k);

      route_mark_.zero_out();
      sync();
    }
  }

  __device__ __forceinline__ void sync() const {
    cooperative_groups::this_grid().sync();
  }

  __device__ date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <bool WithClaszFilter, bool WithBikeFilter>
  __device__ void loop_routes(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();

    for (auto i = global_t_id; i < tt_.n_routes_; i += global_stride) {
      auto const r = route_idx_t{i};

      printf("round %u: processing route %d\n", k, i);
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
        printf("round %u: route=%d -> any_marked=true\n", k, i);
        atomicOr(any_marked_, 1U);
      }
    }
  }

  __device__ void update_transfers(unsigned const k) {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < n_locations_; i += global_stride) {
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
          if (lb_[i] == kUnreachable ||
              !is_better(fp_target_time + dir(lb_[i]), time_at_dest_.get(k))) {
            continue;
          }

          round_times_.update_min(k, l, target_v, fp_target_time);
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
            auto const lower_bound = lb_[target];
            if (lower_bound == kUnreachable ||
                !is_better(fp_target_time + dir(lower_bound),
                           time_at_dest_.get(k))) {
              continue;
            }

            round_times_.update_min(k, fp.target(), target_v, fp_target_time);
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
            round_times_.update_min(k, kIntermodalTarget, Vias, end_time);
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
              is_better(by_transport, higher_v_best) &&
              lb_[l_idx] != kUnreachable &&
              is_better(by_transport + dir(lb_[l_idx]), time_at_dest_.get(k))) {
            tmp_.update_min(l, target_v, by_transport);
            station_mark_.mark(l_idx);
            current_best[v] = by_transport;
            any_marked = true;
            printf("round %u: route=%u, marking l=%u\n", k, to_idx(r), l_idx);
          }

          if (is_via_and_dest) {
            auto const dest_v = target_v + 1;
            assert(dest_v == Vias);
            auto const best_dest =
                get_best(round_times_.get(k - 1, l, dest_v),
                         tmp_.get(l, dest_v), best_.get(l, dest_v));

            if (is_better(by_transport, best_dest) &&
                is_better(by_transport, time_at_dest_.get(k)) &&
                lb_[l_idx] != kUnreachable &&
                is_better(by_transport + dir(lb_[l_idx]),
                          time_at_dest_.get(k))) {
              tmp_.update_min(l, dest_v, by_transport);
              station_mark_.mark(l_idx);
              any_marked = true;
              printf("round %u: route=%u, marking l=%u\n", k, to_idx(r), l_idx);
            }
          }
        }
      }

      if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
          !prev_station_mark_[l_idx]) {
        printf(
            "round %u: route=%u, stop=%u [l=%u] -> "
            "is_last=%s, can_start=%s, prev_station_mark=%s   => no new et\n",
            k, r.v_, stop_idx, l_idx, b2s(is_last),
            b2s(stp.can_start<SearchDir>(is_wheelchair_)),
            b2s(prev_station_mark_[l_idx]));
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      for (auto v = 0U; v != Vias + 1; ++v) {
        if (!et[v].is_valid() && !prev_station_mark_[l_idx]) {
          printf(
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
          auto const [day, mam] = split(prev_round_time);
          auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                                     stp.location_idx());
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
            printf("round %u: route=%u, stop=%u [l=%u] -> transport=%u\n", k,
                   r.v_, stop_idx, l_idx, new_et.t_idx_.v_);
          }
        } else {
          printf(
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

        if (is_better_or_eq(time_at_dest_.get(k),
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
  cuda::std::span<std::uint16_t const> lb_;
  device_times<SearchDir, Vias + 1> round_times_;
  device_times<SearchDir, Vias + 1> best_;
  device_times<SearchDir, Vias + 1> tmp_;
  device_times<SearchDir, 1U> time_at_dest_;
  device_bitvec<std::uint32_t> station_mark_;
  device_bitvec<std::uint32_t> prev_station_mark_;
  device_bitvec<std::uint32_t> route_mark_;
};

}  // namespace nigiri::routing::gpu