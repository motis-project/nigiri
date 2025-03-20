#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct mcraptor_label {
  delta_t arr_t_{};

  location_idx_t trip_l_{};
  location_idx_t fp_l_{};

  bool dominates(mcraptor_label const& l) const {
    return this->arr_t_ < l.arr_t_;
  }
};

struct mcraptor_bag {
  std::vector<mcraptor_label> labels_{};

  bool dominates(mcraptor_label const& other_label) const {
    return std::any_of(labels_.begin(), labels_.end(),
                       [&](mcraptor_label l) {
                         return l.dominates(other_label);
                       });
  }

  void add(mcraptor_label const& new_label) {
    if (this->dominates(new_label)) {
      return;
    }
    auto new_end = std::remove_if(labels_.begin(), labels_.end(),
                                  [&](auto l) {
                                    return new_label.dominates(l);
                                  });
    labels_.erase(new_end, labels_.end());
    labels_.emplace_back(new_label);
  }
};

struct mcraptor_dest_bag {
  std::vector<std::pair<unsigned, mcraptor_label>> labels_;

  bool dominates(mcraptor_label const& other_label,
                 unsigned const& k) const {
    return std::any_of(labels_.begin(),
                       labels_.end(),
                       [&](std::pair<unsigned, mcraptor_label> pair) {
                         return pair.first <= k &&
                                pair.second.dominates(other_label);
                       });
  }

  void add(mcraptor_label const& new_label, unsigned const& k) {
    if (this->dominates(new_label, k)) {
      return;
    }
    auto new_end = std::remove_if(
        labels_.begin(),
        labels_.end(),
        [&](std::pair<unsigned, mcraptor_label> pair) {
          return k <= pair.first &&
                 new_label.dominates(pair.second);
        });
    labels_.erase(new_end, labels_.end());
    labels_.emplace_back(k, new_label);
  }
};

template <direction SearchDir = direction::kForward, bool Rt = false, via_offset_t Vias = 0>
struct mcraptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;

  static constexpr auto const kFwd = SearchDir == direction::kForward;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;

  static bool is_better(auto a, auto b) { return a < b;}
  static bool is_better_or_eq(auto a, auto b) { return a <= b;}
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }

  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }


  mcraptor(timetable const& tt,
               rt_timetable const* rtt,
               raptor_state& state,
               bitvec& is_dest,
               std::array<bitvec, kMaxVias>& is_via,
               std::vector<std::uint16_t>& dist_to_dest,
               hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
               std::vector<std::uint16_t>& lb,
               std::vector<via_stop> const& via_stops,
               day_idx_t const base,
               clasz_mask_t const allowed_claszes,
               bool const require_bike_transport,
               bool const is_wheelchair,
               transfer_time_settings const& tts)
      : tt_{tt},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        state_{state.resize(n_locations_, n_routes_, 0)},
        is_dest_{is_dest},
        lb_{lb},
        base_{base},
        transfer_time_settings_{tts} {

    prev_round_station_mark_.resize(n_locations_);
    tmp_station_mark_.resize(n_locations_);
    station_mark_.resize(n_locations_);
    route_mark_.resize(n_routes_);

    transfer_times_.reserve(n_locations_);
    for (auto l = 0U; l != n_locations_; ++l) {
      auto const transfer_time =
          is_dest_[l]
              ? 0
              : dir(adjusted_transfer_time(
                    transfer_time_settings_,
                    tt_.locations_.transfer_time_[location_idx_t{l}]
                        .count()));
      transfer_times_.emplace_back(transfer_time);
    }

    tmp_ = {n_locations_, mcraptor_label{kInvalid}};
    best_bag_ = std::vector<mcraptor_bag>{n_locations_, mcraptor_bag{}};

    round_bags_ = {n_locations_ * (kMaxTransfers + 1),
                   mcraptor_bag{}};;
    location_bags_ = {{reinterpret_cast<mcraptor_bag*>(
                           round_bags_.data()),
                       n_locations_ * (kMaxTransfers + 1)},
                      n_locations_,
                      kMaxTransfers + 1U};

  }

  [[nodiscard]] algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals() {
    dest_bag_.labels_.clear();
    std::for_each(location_bags_.entries_.begin(), location_bags_.entries_.end(),
                  [&](mcraptor_bag& bag) {
                    bag.labels_.clear();});
  }

  void next_start_time() {
    std::for_each(best_bag_.begin(), best_bag_.end(), [&](mcraptor_bag& bag) {
      bag.labels_.clear();
    });
    std::for_each(tmp_.begin(), tmp_.end(), [&](mcraptor_label& label) {
      label.arr_t_ = kInvalid;
    });
    utl::fill(prev_round_station_mark_.blocks_, 0U);
    utl::fill(tmp_station_mark_.blocks_, 0U);
    utl::fill(station_mark_.blocks_, 0U);
    utl::fill(route_mark_.blocks_, 0U);
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    location_bags_[to_idx(l)][0U].add({.arr_t_ = unix_to_delta(base(), t)});
    prev_round_station_mark_.set(to_idx(l), true);
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results) {

    auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    dest_bag_.add({.arr_t_ = get_best(d_worst_at_dest, kInvalid)}, 0);



    for (auto k = 1U; k != end_k; ++k) {
      is_dest_.for_each_set_bit([&](std::uint64_t const i) {
        dest_bag_.add({.arr_t_ = get_best_time(i)}, k);
      });

      auto any_marked = false;
      prev_round_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
          any_marked = true;
          route_mark_.set(to_idx(r), true);
        }
      });

      if (!any_marked) {
        break;
      }

      any_marked = loop_routes(k);

      if (!any_marked) {
        break;
      }

      update_transfers(k);
      update_footpaths(k, prf_idx);

      utl::fill(route_mark_.blocks_, 0U);
      std::swap(prev_round_station_mark_, station_mark_);
      utl::fill(tmp_station_mark_.blocks_, 0U);
      utl::fill(station_mark_.blocks_, 0U);
    }

    is_dest_.for_each_set_bit([&](auto const i) {

      for (auto k = 1U; k != end_k; ++k) {
        auto const dest_time = get_round_time(i, k);
        if (dest_time != kInvalid) {
          results.add(
              journey{.legs_ = {},
                      .start_time_ = start_time,
                      .dest_time_ = delta_to_unix(base(), dest_time),
                      .dest_ = location_idx_t{i},
                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
        }
      }
    });
  }

  void reconstruct(query const& q, journey& j) {
    auto l = j.dest_;
    for (auto i = 0U; i <= j.transfers_; ++i) {
      auto const k = j.transfers_ + 1 - i;
      auto [fp_leg, transport_leg] = get_legs(k, l, q.prf_idx_);
      l = kFwd ? transport_leg.from_ : transport_leg.to_;
      // don't add a 0-minute footpath at the end (fwd) or beginning (bwd)
      if (i != 0 || fp_leg.from_ != fp_leg.to_ ||
          fp_leg.dep_time_ != fp_leg.arr_time_) {
        j.add(std::move(fp_leg));
      }
      j.add(std::move(transport_leg));
    }
    std::reverse(begin(j.legs_), end(j.legs_));
  }


private:
  bool loop_routes(unsigned const k) {
    auto any_marked = false;
    route_mark_.for_each_set_bit([&](auto const r_idx) {
      auto const r = route_idx_t{r_idx};

      ++stats_.n_routes_visited_;
      any_marked |= update_route(k, r);
    });
    return any_marked;
  }

  bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    auto any_marked = false;

    mcraptor_label et_label{};
    transport et{};

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(i);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      auto current_best = kInvalid;

      if (!et.is_valid() && !prev_round_station_mark_[l_idx]) {
        continue;
      }

      if(et.is_valid() && stp.can_finish<SearchDir>(false)) {
        auto const by_transport = time_at_stop(
            r, et, stop_idx, event_type::kArr);

        current_best = get_best(get_round_time(l_idx, k - 1),
                                tmp_[l_idx].arr_t_,
                                get_best_time(l_idx));

        if(is_better(by_transport, current_best) &&
            !dest_bag_.dominates({.arr_t_ = by_transport}, k) &&
            lb_[l_idx] != kUnreachable &&
            !dest_bag_.dominates({.arr_t_ = static_cast<delta_t>(by_transport + lb_[l_idx])}, k)) {

          ++stats_.n_earliest_arrival_updated_by_route_;
          tmp_[l_idx] = {by_transport, et_label.trip_l_, stp.location_idx()};
          tmp_station_mark_.set(l_idx, true);
          any_marked = true;
        }
      }

      if (is_last || !stp.can_start<SearchDir>(false) || !prev_round_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      if(prev_round_station_mark_[l_idx]) {
        auto const et_time_at_stop =
            et.is_valid()
                ? time_at_stop(r, et, stop_idx, event_type::kDep )
                : kInvalid;
        auto const prev_round_time = get_round_time(l_idx, k - 1);
        if (prev_round_time != kInvalid &&
            is_better_or_eq(prev_round_time, et_time_at_stop)) {
          auto const [day, mam] = split(prev_round_time);
          auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                                     stp.location_idx());
          current_best = get_best(current_best,
                                  get_best_time(l_idx),
                                  tmp_[l_idx].arr_t_);

          if (new_et.is_valid() &&
              (current_best == kInvalid ||
               is_better_or_eq(
                   time_at_stop(r, new_et, stop_idx,event_type::kDep),
                   et_time_at_stop))) {
            et = new_et;
            et_label.trip_l_ = stp.location_idx();
          }
        }
      }
    }
    return any_marked;
  }

  void update_transfers(unsigned const k) {
    tmp_station_mark_.for_each_set_bit([&](auto&& i) {
      mcraptor_label const tmp_label = tmp_[i];
      auto const tmp_time = tmp_label.arr_t_;
      if ((delta_t) tmp_time != kInvalid) {

        auto const fp_target_time =
            static_cast<delta_t>(tmp_time + transfer_times_[i]);

        if (is_better(fp_target_time, get_best_time(i)) &&
            !dest_bag_.dominates({.arr_t_ = fp_target_time}, k)) {
          if (lb_[i] == kUnreachable ||
              dest_bag_.dominates({.arr_t_ = fp_target_time + lb_[i]}, k)) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            return;
          }
          ++stats_.n_earliest_arrival_updated_by_footpath_;

          mcraptor_label new_label = tmp_[i];
          new_label.arr_t_ = fp_target_time;

          location_bags_[i][k].add(new_label);
          station_mark_.set(i, true);
          if (is_dest_[i]) {
            dest_bag_.add({.arr_t_ = fp_target_time}, k);
          }
        }
      }
    });
  }

  void update_footpaths(unsigned const k, profile_idx_t const prf_idx) {
    tmp_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l_idx = location_idx_t{i};
      auto const& fps = tt_.locations_.footpaths_out_[prf_idx][l_idx];

      for (auto const& fp : fps) {
        ++stats_.n_footpaths_visited_;

        auto const target = to_idx(fp.target());
        auto const tmp_time = tmp_[i].arr_t_;
        if (tmp_time == kInvalid) {
          continue;
        }

        auto const fp_target_time = clamp(
            tmp_time + adjusted_transfer_time(transfer_time_settings_,
                                              fp.duration().count()));

        if (is_better(fp_target_time, get_best_time(target)) &&
            !dest_bag_.dominates({.arr_t_ = fp_target_time}, k)) {
          auto const lower_bound = lb_[target];

          if (lower_bound == kUnreachable ||
              dest_bag_.dominates({.arr_t_ = static_cast<delta_t>(fp_target_time + dir(lower_bound))}, k)) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            continue;
          }

          ++stats_.n_earliest_arrival_updated_by_footpath_;
          auto new_label = tmp_[i];
          new_label.arr_t_ = fp_target_time;

          location_bags_[target][k].add(new_label);
          station_mark_.set(target, true);
          if (is_dest_[target]) {
            dest_bag_.add({.arr_t_ = fp_target_time}, k);
          }
        }
      }
    });
  }

  std::pair<journey::leg, journey::leg> get_legs(unsigned const k,
                                                 auto const l,
                                                 auto const prf_idx) {
    auto label = location_bags_[to_idx(l)][k].labels_[0];
    auto const& trip_l_idx = label.trip_l_;
    auto const& fp_l_idx = label.fp_l_;
    auto const& prev_stop_time = get_round_time(to_idx(trip_l_idx), k - 1);
    auto const& arr_t = label.arr_t_;

    delta_t trip_dep_time;
    delta_t trip_arr_fp_dep_time;
    stop_idx_t from_stop_idx;
    stop_idx_t to_stop_idx;
    route_idx_t route_idx;
    transport transport;
    footpath footpath{};

    auto found_end_location = false;
    for (auto const& r : tt_.location_routes_[trip_l_idx]) {
      if (found_end_location) {
        break;
      }
      auto const stop_seq = tt_.route_location_seq_[r];
      struct transport trip{};

      for (auto i = 0U; i != stop_seq.size(); ++i) {
        auto const stop_idx = static_cast<stop_idx_t>(i);
        auto const stp = stop{stop_seq[stop_idx]};
        auto const l_idx = stp.location_idx();

        if (trip.is_valid() && (fp_l_idx.v_ == l_idx.v_)) {
          to_stop_idx = stop_idx;
          transport = trip;
          route_idx = r;
          trip_arr_fp_dep_time = time_at_stop(
              r, trip, stop_idx, event_type::kArr);
          found_end_location = true;
          break;
        }

        if (trip_l_idx.v_ == l_idx.v_) {
          auto const [day, mam] = split(prev_stop_time);
          trip = get_earliest_transport(k - 1, r, stop_idx, day, mam,
                                        stp.location_idx());
          trip_dep_time = time_at_stop(r, trip, stop_idx,event_type::kDep);
          from_stop_idx = stop_idx;
        }
      }
    }

    if (l == fp_l_idx) {
      footpath = {l, tt_.locations_.transfer_time_[location_idx_t{l}]};
    } else {
      auto const& fps = tt_.locations_.footpaths_out_[prf_idx][fp_l_idx];
      for (auto const& fp : fps) {
        if (l == fp.target()) {
          footpath = fp;
          break;
        }
      }
    }

    auto const fp_leg =
        journey::leg{SearchDir, fp_l_idx, l,
                     delta_to_unix(base(), trip_arr_fp_dep_time),
                     delta_to_unix(base(), arr_t),
                     footpath};

    auto const transport_leg =
        journey::leg{SearchDir, trip_l_idx, fp_l_idx,
                     delta_to_unix(base(), trip_dep_time),
                     delta_to_unix(base(), trip_arr_fp_dep_time),
                     journey::run_enter_exit{{.t_ = transport,
                                              .stop_range_ = interval<stop_idx_t>{0, static_cast<stop_idx_t>(
                                                                                         tt_.route_location_seq_[route_idx].size())}},
                                             from_stop_idx, to_stop_idx}};

    return {fp_leg, transport_leg};
  }

  delta_t get_round_time(auto const l, unsigned const k) {
    auto const& rb = location_bags_[l][k];
    return rb.labels_.empty() ? kInvalid :  rb.labels_[0].arr_t_;
  }

  delta_t get_best_time(auto const l) {
    auto const& bb = best_bag_[l];
    return bb.labels_.empty() ? kInvalid :  bb.labels_[0].arr_t_;
  }

  [[nodiscard]] date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <typename T>
  auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  [[nodiscard]] bool is_transport_active(transport_idx_t const t,
                                         std::size_t const day) const {
    return tt_.bitfields_[tt_.transport_traffic_days_[t]].test(day);
  }

  transport get_earliest_transport(unsigned const k,
                                   route_idx_t const r,
                                   stop_idx_t const stop_idx,
                                   day_idx_t const day_at_stop,
                                   minutes_after_midnight_t const mam_at_stop,
                                   location_idx_t const l) {
    ++stats_.n_earliest_trip_calls_;

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

        if(dest_bag_.dominates({.arr_t_ = static_cast<delta_t>(
                                     to_delta(day, ev_mam)
                                     + dir(lb_[to_idx(l)]))},
                                k)) {
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

  delta_t time_at_stop(route_idx_t const r,
                       transport const t,
                       stop_idx_t const stop_idx,
                       event_type const ev_type) {
    return to_delta(t.day_,
                    tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
    return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) {
    return split_day_mam(base_, x);
  }

  static int as_int(day_idx_t const d) { return static_cast<int>(d.v_); }

  timetable const& tt_;
  int n_days_;
  std::uint32_t n_locations_, n_routes_;
  raptor_state& state_;
  std::vector<mcraptor_bag> round_bags_;
  flat_matrix_view<mcraptor_bag> location_bags_;
  std::vector<mcraptor_bag> best_bag_;
  std::vector<mcraptor_label> tmp_;
  bitvec station_mark_;
  bitvec tmp_station_mark_;
  bitvec prev_round_station_mark_;
  bitvec route_mark_;
  bitvec const& is_dest_;
  mcraptor_dest_bag dest_bag_;
  std::vector<std::uint16_t> const& lb_;
  std::vector<std::uint16_t> transfer_times_;
  day_idx_t base_;
  raptor_stats stats_;
  transfer_time_settings transfer_time_settings_;
};

}  // namespace nigiri::routing