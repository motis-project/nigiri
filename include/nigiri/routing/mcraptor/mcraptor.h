/*
 * This file is based on the mcraptor from PumarPap
 * https://github.com/motis-project/nigiri/pull/183
 */

#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::da {

template <bool Rt = false, via_offset_t Vias = 0>
struct mcraptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;

  static constexpr auto const kFwd = false;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kInvalid = kInvalidDelta<direction::kBackward>;

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }

  struct mcraptor_label {
    delta_t t_{};

    location_idx_t trip_l_{};
    location_idx_t fp_l_{};
    transport trip_id{};

    double success_chance{};
    bool over_limit{};

    bool dominates(mcraptor_label const& l) const {
      return (this->t_ >= l.t_ && this->success_chance >= l.success_chance);
    }
  };

  struct mcraptor_bag {
    std::vector<mcraptor_label> labels_{};

    bool dominates(mcraptor_label const& other_label) const {
      return std::any_of(labels_.begin(), labels_.end(), [&](auto const& l) {
        return l.dominates(other_label);
      });
    }

    void add(mcraptor_label const& new_label) {
      if (this->dominates(new_label)) {
        return;
      }
      auto new_end =
          std::remove_if(labels_.begin(), labels_.end(),
                         [&](auto const& l) { return new_label.dominates(l); });
      labels_.erase(new_end, labels_.end());
      labels_.emplace_back(new_label);
    }

    void unchecked_add(mcraptor_label const& new_label) {
      auto new_end =
          std::remove_if(labels_.begin(), labels_.end(),
                         [&](auto const& l) { return new_label.dominates(l); });
      labels_.erase(new_end, labels_.end());
      labels_.emplace_back(new_label);
    }
  };

  struct mcraptor_dest_bag {
    std::vector<std::pair<unsigned, mcraptor_label>> labels_;

    bool dominates(mcraptor_label const& other_label, unsigned const& k) const {
      return std::any_of(labels_.begin(), labels_.end(),
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
          labels_.begin(), labels_.end(),
          [&](std::pair<unsigned, mcraptor_label> pair) {
            return k <= pair.first && new_label.dominates(pair.second);
          });
      labels_.erase(new_end, labels_.end());
      labels_.emplace_back(k, new_label);
    }
  };

  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  mcraptor(timetable const& tt,
           rt_timetable const*,
           raptor_state& state,
           bitvec& is_dest,
           std::array<bitvec, kMaxVias>&,
           std::vector<std::uint16_t>&,
           hash_map<location_idx_t, std::vector<td_offset>> const&,
           std::vector<std::uint16_t>& lb,
           std::vector<via_stop> const&,
           day_idx_t const base,
           clasz_mask_t const,
           bool const,
           bool const,
           bool const,
           transfer_time_settings const& tts)
      : tt_{tt},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        state_{state.resize(n_locations_, n_routes_, 0)},
        is_dest_{is_dest},
        lb_{lb},
        base_{base},
        transfer_time_settings_{tts},
        arr_dist_{state.arr_dist_} {

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
                    tt_.locations_.transfer_time_[location_idx_t{l}].count()));
      transfer_times_.emplace_back(transfer_time);
    }

    tmp_ = {n_locations_, mcraptor_bag{}};
    best_bag_ = std::vector<mcraptor_bag>{n_locations_, mcraptor_bag{}};

    round_bags_ = {n_locations_ * (kMaxTransfers + 1), mcraptor_bag{}};
    location_bags_ = {{reinterpret_cast<mcraptor_bag*>(round_bags_.data()),
                       n_locations_ * (kMaxTransfers + 1)},
                      n_locations_,
                      kMaxTransfers + 1U};
  }

  [[nodiscard]] algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals() {
    dest_bag_.labels_.clear();
    std::for_each(location_bags_.entries_.begin(),
                  location_bags_.entries_.end(),
                  [&](mcraptor_bag& bag) { bag.labels_.clear(); });
  }

  void next_start_time() {
    std::for_each(best_bag_.begin(), best_bag_.end(),
                  [&](mcraptor_bag& bag) { bag.labels_.clear(); });
    std::for_each(tmp_.begin(), tmp_.end(),
                  [&](mcraptor_bag& bag) { bag.labels_.clear(); });
    for (std::size_t i = 0; i < location_bags_.n_rows_; ++i) {
      for (std::size_t j = 0; j < location_bags_.n_columns_; ++j) {
        location_bags_(i, j).labels_.clear();
      }
    }
    dest_bag_.labels_.clear();
    utl::fill(prev_round_station_mark_.blocks_, 0U);
    utl::fill(tmp_station_mark_.blocks_, 0U);
    utl::fill(station_mark_.blocks_, 0U);
    utl::fill(route_mark_.blocks_, 0U);
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    location_bags_[to_idx(l)][0U].add(
        {.t_ = unix_to_delta(base(), t), .success_chance = 1.0f});
    best_bag_[to_idx(l)].add(
        {.t_ = unix_to_delta(base(), t), .success_chance = 1.0f});
    prev_round_station_mark_.set(to_idx(l), true);
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results) {

    auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    dest_bag_.add({.t_ = get_best(d_worst_at_dest, kInvalid)}, 0);

    for (auto k = 1U; k != end_k; ++k) {
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

      prun_flaged();

      update_transfers(k);
      update_footpaths(k, prf_idx);

      station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        std::sort(
            best_bag_[i].labels_.begin(), best_bag_[i].labels_.end(),
            [](mcraptor_label a, mcraptor_label b) { return a.t_ < b.t_; });
        std::sort(
            location_bags_[i][k].labels_.begin(),
            location_bags_[i][k].labels_.end(),
            [](mcraptor_label a, mcraptor_label b) { return a.t_ > b.t_; });
      });

      update_dest_bag(k);

      tmp_station_mark_.for_each_set_bit(
          [&](std::uint64_t const i) { tmp_[i].labels_.clear(); });

      utl::fill(route_mark_.blocks_, 0U);
      std::swap(prev_round_station_mark_, station_mark_);
      utl::fill(tmp_station_mark_.blocks_, 0U);
      utl::fill(station_mark_.blocks_, 0U);
    }

    for (std::pair<unsigned, mcraptor_label> pair : dest_bag_.labels_) {
      auto label = pair.second;
      if (pair.first > 0) {
        results.add(
            journey{.legs_ = {},
                    .start_time_ = start_time,
                    .dest_time_ = delta_to_unix(base(), label.t_),
                    .success_chance_ = label.success_chance,
                    .dest_ = location_idx_t{label.fp_l_},
                    .transfers_ = static_cast<std::uint8_t>(pair.first - 1)});
      }
    }
  }

  void print_leg(journey::leg leg, auto indent) {
    for (int i = 0; i < indent; ++i) std::cout << "\t";
    std::cout << loc{tt_, leg.from_} << " id: " << leg.from_ << " ["
              << leg.dep_time_ << "] TO: " << loc{tt_, leg.to_}
              << " id: " << leg.to_ << " [" << leg.arr_time_ << "] - "
              << leg.success_chance_;
    leg.print(std::cout, tt_);
  }

  bool const should_print = false;
  void reconstruct(query const& q, journey& j) {
    reconstruct_station_based(q, j);
    reconstruct_trip_based(q, j);
  }

  void reconstruct_trip_based(query const& q, journey& j) {
    delta_t possible_start_t = unix_to_delta(base(), j.dest_time_);

    std::vector<mcraptor_label> labels = {};
    std::vector<pair<std::size_t, location_idx_t>> ls{};
    get_labels_after_dest_bag(j.transfers_ + 1U, possible_start_t, labels);
    ls.push_back({labels.size(), j.dest_});

    std::unordered_map<location_idx_t, delta_t> arrivals{};
    auto try_insert = [&](location_idx_t loc, delta_t d) {
      auto it = arrivals.find(loc);
      if (it == arrivals.end() || d < it->second) {
        arrivals[loc] = d;
      }
    };

    auto i = 0U;
    while (true) {
      auto k = j.transfers_ + 1U - i;
      auto it = ls.begin();
      for (unsigned int index = 0; index < labels.size(); ++index) {
        mcraptor_label const& label = labels[index];
        if (index >= it->first) ++it;
        auto l = it->second;
        auto [fp_leg, transport_leg] = get_legs(l, q.prf_idx_, label);
        auto const& next_l = kFwd ? transport_leg.from_ : transport_leg.to_;
        possible_start_t = unix_to_delta(base(), transport_leg.arr_time_);
        if (i != 0 && (fp_leg.from_ != fp_leg.to_ ||
                       fp_leg.dep_time_ != fp_leg.arr_time_)) {
          if (fp_leg.from_ != fp_leg.to_) j.add(std::move(fp_leg));
          possible_start_t = unix_to_delta(base(), fp_leg.arr_time_);
        }
        try_insert(next_l, possible_start_t);
        j.add(std::move(transport_leg));
      }

      if (k <= 0U) break;

      labels.clear();
      ls.clear();
      for (auto const& pair : arrivals) {
        get_labels_after_add(cista::to_idx(pair.first), k - 1U, pair.second,
                             labels, labels.size());
        ls.push_back({labels.size(), pair.first});
      }
      arrivals.clear();
      ++i;
    }
  }

  void reconstruct_leg_station_based(query const& q,
                                     journey& j,
                                     auto i,
                                     auto l,
                                     mcraptor_label const& label,
                                     auto& trips) {
    auto k = j.transfers_ + 1U - i;
    auto const& [fp_leg, transport_leg] = get_legs(l, q.prf_idx_, label);
    auto const& next_l = kFwd ? transport_leg.from_ : transport_leg.to_;
    auto possible_start_t = unix_to_delta(base(), transport_leg.arr_time_);
    if (i != 0 &&
        (fp_leg.from_ != fp_leg.to_ || fp_leg.dep_time_ != fp_leg.arr_time_)) {
      if (fp_leg.from_ != fp_leg.to_) trips.push_back(fp_leg);
      possible_start_t = unix_to_delta(base(), fp_leg.arr_time_);
    }
    trips.push_back(transport_leg);
    if (i < j.transfers_ &&
        !std::any_of(
            q.start_.begin(), q.start_.end(),
            [next_l](offset loc) { return loc.target_ == next_l; })) {
      k = j.transfers_ + 1U - (i + 1U);
      std::vector<mcraptor_label> labels = {};
      get_labels_after_(cista::to_idx(next_l), k, possible_start_t, labels);
      for (auto const& new_label : labels) {
        reconstruct_leg_station_based(q, j, i + 1U, next_l, new_label, trips);
      }
    }
  }

  unsigned int colum_width = 20;
  void reconstruct_station_based(query const& q, journey& j) {
    std::vector<journey::leg> trips;
    delta_t possible_start_t = unix_to_delta(base(), j.dest_time_);

    std::vector<mcraptor_label> labels = {};
    get_labels_after_dest_bag(j.transfers_ + 1U, possible_start_t, labels);
    if (should_print)
      std::cout << "Gesamtwahrscheinlichkeit: " << j.success_chance_
                << " Umstiege: " << static_cast<int>(j.transfers_) << std::endl;

    for (auto label : labels) {
      reconstruct_leg_station_based(q, j, 0U, j.dest_, label, trips);
    }
    std::cout << trips.size() << std::endl;

    std::sort(trips.begin(), trips.end(), [](journey::leg a, journey::leg b) {
      if (a.dep_time_ != b.dep_time_) return a.dep_time_ < b.dep_time_;
      if (a.from_ != b.from_) return a.from_ < b.from_;
      return a.to_ < b.to_;
    });

    for (auto leg : trips) {
      auto copy = leg;
      j.add(std::move(copy));
    }

    if (should_print) {
      std::vector<location_idx_t> locations{};
      for (journey::leg leg : trips) {
        auto from_i =
            std::find_if(locations.begin(), locations.end(),
                         [&](location_idx_t l) { return l == leg.from_; });
        if (from_i == locations.end()) {
          locations.push_back(leg.from_);
        }
        auto to_i =
            std::find_if(locations.begin(), locations.end(),
                         [&](location_idx_t l) { return l == leg.to_; });
        if (to_i == locations.end()) {
          locations.push_back(leg.to_);
        }
      }
      for (location_idx_t lo : locations) {
        std::stringstream stringstream;
        stringstream << loc{tt_, lo};
        auto string = stringstream.str() + " id: " + std::to_string(lo.v_);
        str_to_length(string, colum_width * 2);
        std::cout << string << "|";
      }
      std::cout << std::endl;
      for (journey::leg leg : trips) {
        auto from_i =
            std::find_if(locations.begin(), locations.end(),
                         [&](location_idx_t l) { return l == leg.from_; }) -
            locations.begin();
        auto to_i =
            std::find_if(locations.begin(), locations.end(),
                         [&](location_idx_t l) { return l == leg.to_; }) -
            locations.begin();

        print_leg_station_based(leg, from_i, to_i, locations.size());
      }
      if (kFwd) std::reverse(begin(j.legs_), end(j.legs_));
      std::cout << std::endl;
    }
  }

  template <bool cut_front = false>
  void str_to_length(std::string& str, auto length) {
    while (str.length() < length) {
      str += " ";
    }
    str = !cut_front ? str.substr(0, length)
                     : str.substr(str.length() - length, str.length());
  }

  void print_unixtime(auto t, bool left) {
    if (left) {
      for (unsigned int j = 0; j < colum_width; ++j) {
        std::cout << " ";
      }
    }
    std::stringstream stringstream;
    stringstream << t;
    auto str = stringstream.str();
    str_to_length<true>(str, colum_width);
    std::cout << str;
    if (!left) {
      for (unsigned int j = 0; j < colum_width; ++j) {
        std::cout << " ";
      }
    }
    std::cout << "|";
  }

  void print_column_spaces(auto n) {
    for (auto i = 0U; i < n; ++i) {
      for (unsigned int j = 0; j < colum_width * 2; ++j) {
        std::cout << " ";
      }
      std::cout << "|";
    }
  }

  void print_leg_station_based(journey::leg leg,
                               auto from_i,
                               auto to_i,
                               auto max) {
    auto f = from_i < to_i ? from_i : to_i;
    auto t = (from_i < to_i ? to_i : from_i) - f;
    auto const pos = static_cast<std::size_t>(from_i < to_i ? to_i : from_i);
    print_column_spaces(f);
    print_unixtime(from_i < to_i ? leg.dep_time_ : leg.arr_time_,
                   from_i < to_i);
    print_column_spaces(t - 1);
    print_unixtime(from_i < to_i ? leg.arr_time_ : leg.dep_time_,
                   from_i > to_i);
    print_column_spaces(max - pos - 1U);
    leg.print(std::cout, tt_);
    std::cout << std::endl;
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

  void get_labels_after_(auto l,
                         auto k,
                         delta_t possible_start_t,
                         std::vector<mcraptor_label>& labels) {
    for (auto i = k; i >= 1; --i) {
      std::copy_if(location_bags_[l][i].labels_.begin(),
                   location_bags_[l][i].labels_.end(),
                   std::back_inserter(labels), [&](mcraptor_label val) {
                     return std::none_of(labels.begin(), labels.end(),
                                         [&](mcraptor_label contained) {
                                           return contained.dominates(val);
                                         });
                   });
    }
    std::sort(labels.begin(), labels.end(),
              [](mcraptor_label a, mcraptor_label b) { return a.t_ < b.t_; });
    auto new_end =
        std::lower_bound(labels.begin(), labels.end(), possible_start_t,
                         [](mcraptor_label a, delta_t t) { return a.t_ < t; });
    labels.erase(labels.begin(), new_end);
  }

  void get_labels_after_add(auto l,
                            auto k,
                            delta_t possible_start_t,
                            std::vector<mcraptor_label>& labels,
                            auto start_index) {
    for (auto i = k; i >= 1U; --i) {
      std::copy_if(location_bags_[l][i].labels_.begin(),
                   location_bags_[l][i].labels_.end(),
                   std::back_inserter(labels), [&](mcraptor_label val) {
                     return std::none_of(
                         labels.begin() + static_cast<long>(start_index),
                         labels.end(), [&](mcraptor_label contained) {
                           return contained.dominates(val);
                         });
                   });
    }
    std::sort(labels.begin() + static_cast<long>(start_index), labels.end(),
              [](mcraptor_label a, mcraptor_label b) { return a.t_ < b.t_; });
    auto new_end = std::lower_bound(
        labels.begin() + static_cast<long>(start_index), labels.end(),
        possible_start_t, [](mcraptor_label a, delta_t t) { return a.t_ < t; });
    labels.erase(labels.begin() + static_cast<long>(start_index), new_end);
  }

  void get_labels_after_dest_bag(auto k,
                                 delta_t possible_start_t,
                                 std::vector<mcraptor_label>& labels) {
    std::for_each(dest_bag_.labels_.begin(), dest_bag_.labels_.end(),
                  [&](std::pair<unsigned, mcraptor_label> const& pair) {
                    if (pair.first == k && pair.first > 0U &&
                        possible_start_t <= pair.second.t_ &&
                        pair.second.t_ != kInvalid) {
                      labels.push_back(pair.second);
                    }
                  });
    std::sort(labels.begin(), labels.end(),
              [](mcraptor_label a, mcraptor_label b) { return a.t_ < b.t_; });
    auto new_end =
        std::lower_bound(labels.begin(), labels.end(), possible_start_t,
                         [](mcraptor_label a, delta_t t) { return a.t_ < t; });
    labels.erase(labels.begin(), new_end);
  }

  double delay_distribution_real(delta_t x, clasz c) {
    auto id = static_cast<unsigned int>(c);
    std::vector<std::pair<int, double>> const& delays =
        arr_dist_[id].empty() ? arr_dist_[static_cast<int>(clasz::kOther)]
                              : arr_dist_[id];

    if (x < delays.back().first) {
      return delays[static_cast<unsigned int>(x) + 5U].second;
    } else {
      return delays.back().second;
    }
  }

  delta_t const lookback = delta_t{91};

  double transferProbability(delta_t to, clasz c) {
    auto function = [&](auto x) { return this->delay_distribution_real(x, c); };
    return function(to);
  }

  double cum_success_chance(auto l,
                            auto k,
                            delta_t possible_start_t,
                            clasz const c) {
    auto it = std::lower_bound(
        best_bag_[l].labels_.begin(), best_bag_[l].labels_.end(),
        possible_start_t, [](mcraptor_label a, delta_t t) { return a.t_ < t; });

    auto prob = transferProbability(it->t_ - possible_start_t, c);
    auto result = prob * it->success_chance;
    auto a = (prob / transferProbability(
                         std::numeric_limits<nigiri::delta_t>::max(), c));
    auto b =
        k > 0 ? transferProbability(
                    std::numeric_limits<nigiri::delta_t>::max(),
                    tt_.route_clasz_[tt_.transport_route_[it->trip_id.t_idx_]])
              : 1;
    auto counterprob = (1 - (a * (b)));
    ++it;
    for (; it != best_bag_[l].labels_.end(); ++it) {
      prob = transferProbability(it->t_ - possible_start_t, c);
      result += counterprob * prob * it->success_chance;
      a = (prob /
           transferProbability(std::numeric_limits<nigiri::delta_t>::max(), c));
      b = transferProbability(
          std::numeric_limits<nigiri::delta_t>::max(),
          tt_.route_clasz_[tt_.transport_route_[it->trip_id.t_idx_]]);
      counterprob = counterprob * (1 - (a * (b)));
    }
    return result;
  }

  double cum_enter_probability(auto l, delta_t possible_start_t) {
    auto it = std::lower_bound(
        best_bag_[l].labels_.begin(), best_bag_[l].labels_.end(),
        possible_start_t, [](mcraptor_label a, delta_t t) { return a.t_ < t; });

    auto result = it->success_chance;
    auto counterprob =
        1 - transferProbability(
                std::numeric_limits<nigiri::delta_t>::max(),
                tt_.route_clasz_[tt_.transport_route_[it->trip_id.t_idx_]]);
    ++it;
    for (; it != best_bag_[l].labels_.end(); ++it) {
      result += counterprob * it->success_chance;
      counterprob =
          counterprob *
          (1 - transferProbability(
                   std::numeric_limits<nigiri::delta_t>::max(),
                   tt_.route_clasz_[tt_.transport_route_[it->trip_id.t_idx_]]));
    }
    return result;
  }

  bool update_route(unsigned const k, route_idx_t const r) {
    auto stop_seq = tt_.route_location_seq_[r];
    auto any_marked = false;
    auto ets = std::vector<mcraptor_label>{};
    for (unsigned int i = 0; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      if (!ets.empty() && stp.can_finish<direction::kBackward>(false)) {
        auto it = ets.begin();
        while (it != ets.end()) {
          auto& et_label = (*it);
          auto const by_transport =
              time_at_stop(r, et_label.trip_id, stop_idx,
                           kFwd ? event_type::kArr : event_type::kDep);

          et_label.t_ = by_transport;
          if (!best_bag_[l_idx].dominates(et_label) &&
              lb_[l_idx] != kUnreachable &&
              !dest_bag_.dominates(
                  {.t_ = static_cast<delta_t>(by_transport + lb_[l_idx]),
                   .trip_id = et_label.trip_id,
                   .success_chance = et_label.success_chance},
                  k)) {
            if (!tmp_[l_idx].dominates(et_label)) {
              ++stats_.n_earliest_arrival_updated_by_route_;
              tmp_[l_idx].add({by_transport, et_label.trip_l_,
                               stp.location_idx(), et_label.trip_id,
                               et_label.success_chance, et_label.over_limit});
              tmp_station_mark_.set(l_idx, true);
              any_marked = true;
            }
            ++it;
          } else {
            it = ets.erase(it);
          }
        }
      } else {
        for (auto& et_label : ets) {
          auto const by_transport =
              time_at_stop(r, et_label.trip_id, stop_idx,
                           kFwd ? event_type::kArr : event_type::kDep);
          et_label.t_ = by_transport;
        }
      }

      if (is_last || !stp.can_start<direction::kBackward>(false) ||
          !prev_round_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      if (prev_round_station_mark_[l_idx]) {
        auto const& prev_round_bag = get_round_bag(l_idx, k - 1);
        auto start = prev_round_bag.labels_[0].t_;
        delta_t end;
        auto old_size = ets.end() - ets.begin();
        for (unsigned int j = 0; j < prev_round_bag.labels_.size(); ++j) {
          if (std::any_of(
                  ets.begin(), ets.end(), [&](mcraptor_label label_of_vector) {
                    return label_of_vector.dominates(prev_round_bag.labels_[j]);
                  })) {
            continue;
          }
          end = prev_round_bag.labels_[j].t_ - lookback;

          if (start < end) continue;
          if ((j + 1) < prev_round_bag.labels_.size() &&
              end <= prev_round_bag.labels_[j + 1].t_)
            continue;

          auto const [day_from, mam_from] = split(start);

          if (!get_earliest_transports(k, r, stop_idx, day_from, mam_from,
                                       stp.location_idx(), ets, end))
            break;
          start = ets.back().t_ + static_cast<delta_t>(dir(1));
          if ((j + 1) < prev_round_bag.labels_.size() &&
              start > prev_round_bag.labels_[j + 1].t_) {
            start = prev_round_bag.labels_[j + 1].t_;
          }
        }
        if (ets.size() > 1) {
          std::inplace_merge(
              ets.begin(), ets.begin() + old_size, ets.end(),
              [](auto const& a, auto const& b) { return a.t_ > b.t_; });

          auto it = ets.begin();
          while (it + 1 != ets.end()) {
            if (it->t_ != (it + 1)->t_) {
              if (it->success_chance > (it + 1)->success_chance) {
                it = ets.erase(it + 1) - 1;
              } else {
                ++it;
              }
            } else {
              if (it->success_chance > (it + 1)->success_chance) {
                it = ets.erase(it + 1) - 1;
              } else {
                it = ets.erase(it);
              }
            }
          }
        }
      }
    }
    return any_marked;
  }

  void update_dest_bag(auto k) {
    is_dest_.for_each_set_bit([&](std::uint64_t const i) {
      for (std::pair<unsigned, mcraptor_label> pair : dest_bag_.labels_) {
        if (pair.first == 0 || pair.first >= k ||
            pair.second.fp_l_ != location_idx_t{i})
          continue;
        auto tmp_label = pair.second;
        tmp_label.success_chance = cum_enter_probability(i, tmp_label.t_);
        dest_bag_.add(tmp_label, k);
      }
      for (mcraptor_label tmp_label : location_bags_[i][k].labels_) {
        if (tmp_label.t_ == kInvalid) continue;
        tmp_label.success_chance = cum_enter_probability(i, tmp_label.t_);
        dest_bag_.add(tmp_label, k);
      }
    });
  }

  void update_transfers(unsigned const k) {
    tmp_station_mark_.for_each_set_bit([&](auto&& i) {
      for (mcraptor_label tmp_label : tmp_[i].labels_) {
        mcraptor_label new_label = tmp_label;
        if (new_label.t_ != kInvalid) {
          auto const is_dest = is_dest_[i];

          auto const transfer_time =
              is_dest ? 0
                      : dir(adjusted_transfer_time(
                            transfer_time_settings_,
                            tt_.locations_.transfer_time_[location_idx_t{i}]
                                .count()));

          new_label.t_ = static_cast<delta_t>(new_label.t_ + transfer_time);

          if (!best_bag_[i].dominates(new_label)) {
            if (lb_[i] == kUnreachable) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              continue;
            }
            new_label.t_ = new_label.t_ + static_cast<delta_t>(lb_[i]);
            if (dest_bag_.dominates(new_label, k)) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              continue;
            }
            new_label.t_ = new_label.t_ - static_cast<delta_t>(lb_[i]);
            ++stats_.n_earliest_arrival_updated_by_footpath_;

            location_bags_[i][k].add(new_label);
            best_bag_[i].unchecked_add(new_label);
            station_mark_.set(i, true);
          }
        }
      }
    });
  }

  void update_footpaths(unsigned const k, profile_idx_t const prf_idx) {
    tmp_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      for (mcraptor_label tmp_label : tmp_[i].labels_) {
        auto const l_idx = location_idx_t{i};
        auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx][l_idx]
                               : tt_.locations_.footpaths_in_[prf_idx][l_idx];

        for (auto const& fp : fps) {
          ++stats_.n_footpaths_visited_;

          auto const target = to_idx(fp.target());
          auto const tmp_time = tmp_label.t_;
          if (tmp_time == kInvalid) {
            continue;
          }

          mcraptor_label new_label = tmp_label;
          new_label.t_ = clamp(
              tmp_time + dir(adjusted_transfer_time(transfer_time_settings_,
                                                    fp.duration().count())));

          if (!best_bag_[target].dominates(new_label)) {
            auto const lower_bound = lb_[target];

            if (lower_bound == kUnreachable ||
                dest_bag_.dominates(
                    {.t_ =
                         static_cast<delta_t>(new_label.t_ + dir(lower_bound)),
                     .trip_id = tmp_label.trip_id,
                     .success_chance = tmp_label.success_chance},
                    k)) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              continue;
            }

            ++stats_.n_earliest_arrival_updated_by_footpath_;

            location_bags_[target][k].add(new_label);
            best_bag_[target].unchecked_add(new_label);
            station_mark_.set(target, true);
          }
        }
      }
    });
  }

  void prun_flaged() {
    tmp_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      std::sort(tmp_[i].labels_.begin(), tmp_[i].labels_.end(),
                [](auto a, auto b) { return a.t_ < b.t_; });
      auto first_flagged =
          std::find_if(tmp_[i].labels_.rbegin(), tmp_[i].labels_.rend(),
                       [](auto a) { return a.over_limit; });
      if (first_flagged == tmp_[i].labels_.rend()) {
        --first_flagged;
      }
      auto value = first_flagged->t_ - lookback;

      auto it = tmp_[i].labels_.begin();
      while (it + 1 != tmp_[i].labels_.end()) {
        if (it->t_ < value) {
          it = tmp_[i].labels_.erase(it);
        } else {
          break;
        }
      }
    });
  }

  std::tuple<journey::leg, journey::leg> get_legs(auto const l,
                                                  auto const prf_idx,
                                                  mcraptor_label label) {
    auto const& trip_l_idx = label.trip_l_;
    auto const& fp_l_idx = label.fp_l_;
    auto const& arr_t = label.t_;

    delta_t trip_dep_time = kInvalid;
    delta_t trip_arr_fp_dep_time = kInvalid;
    stop_idx_t from_stop_idx = 0;
    stop_idx_t to_stop_idx = 0;
    route_idx_t route_idx;
    transport transport;
    footpath footpath{};

    auto r = tt_.transport_route_[label.trip_id.t_idx_];

    auto const stop_seq = tt_.route_location_seq_[r];
    struct transport trip{};

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = stp.location_idx();

      if (trip.is_valid() && (fp_l_idx.v_ == l_idx.v_)) {
        to_stop_idx = stop_idx;
        transport = trip;
        route_idx = r;
        trip_arr_fp_dep_time = time_at_stop(
            r, trip, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        break;
      }

      if (trip_l_idx.v_ == l_idx.v_) {
        trip = label.trip_id;
        trip_dep_time = time_at_stop(
            r, trip, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
        from_stop_idx = stop_idx;
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
        journey::leg{direction::kBackward,
                     fp_l_idx,
                     l,
                     delta_to_unix(base(), trip_arr_fp_dep_time),
                     delta_to_unix(base(), arr_t),
                     footpath,
                     label.success_chance};

    auto const transport_leg = journey::leg{
        direction::kBackward,
        trip_l_idx,
        fp_l_idx,
        delta_to_unix(base(), trip_dep_time),
        delta_to_unix(base(), trip_arr_fp_dep_time),
        journey::run_enter_exit{
            {.t_ = transport,
             .stop_range_ =
                 interval<stop_idx_t>{
                     0, static_cast<stop_idx_t>(
                            tt_.route_location_seq_[route_idx].size())}},
            from_stop_idx,
            to_stop_idx},
        label.success_chance};

    return {fp_leg, transport_leg};
  }

  mcraptor_bag const& get_round_bag(auto const l, unsigned const k) {
    return location_bags_[l][k];
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

  bool get_earliest_transports(unsigned const k,
                               route_idx_t const r,
                               stop_idx_t const stop_idx,
                               day_idx_t const day_at_stop_from,
                               minutes_after_midnight_t const mam_at_stop_from,
                               location_idx_t const l,
                               std::vector<mcraptor_label>& ets,
                               delta_t end) {
    ++stats_.n_earliest_trip_calls_;
    auto const [day_at_stop_to, mam_at_stop_to] = split(end);
    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop_from,
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

      auto const day = kFwd ? day_at_stop_from + i : day_at_stop_from - i;
      for (auto it = ev_time_range.begin(); it != ev_time_range.end(); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = ev.mam();

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop_from.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active(t, start_day)) {
          continue;
        }

        transport new_et = {
            t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
        delta_t time = time_at_stop(r, new_et, stop_idx,
                                    kFwd ? event_type::kDep : event_type::kArr);
        mcraptor_label new_et_label = {
            .t_ = time,
            .trip_l_ = l,
            .trip_id = new_et,
            .success_chance = cum_success_chance(cista::to_idx(l), k - 1, time,
                                                 tt_.route_clasz_[r]),
            .over_limit = time < end};
        ets.push_back(new_et_label);
        if (static_cast<day_idx_t>(as_int(day) - ev_day_offset) <=
                day_at_stop_to &&
            ev_mam <= mam_at_stop_to.count())
          return true;
      }
    }
    return false;
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
  std::vector<mcraptor_bag> tmp_;
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
  std::vector<std::vector<std::pair<int, double>>> const& arr_dist_;
};

}  // namespace nigiri::routing::da