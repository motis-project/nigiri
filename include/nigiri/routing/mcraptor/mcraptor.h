#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

template <direction SearchDir = direction::kForward, bool Rt = false, via_offset_t Vias = 0>
struct mcraptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;

  static constexpr auto const kFwd = SearchDir == direction::kForward;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }

  struct mcraptor_label {
    delta_t arr_t_{};

    location_idx_t trip_l_{};
    location_idx_t fp_l_{};
    transport trip_id{};

    double success_chance{};
    bool over_limit{};

    bool dominates(mcraptor_label const& l) const {
      return (this->arr_t_ >= l.arr_t_ && this->success_chance > l.success_chance) ||
             (this->arr_t_ == l.arr_t_ && this->success_chance == l.success_chance) ||
             (this->arr_t_ > l.arr_t_ && this->success_chance >= l.success_chance);
    }
  };

  struct mcraptor_bag {
    std::vector<mcraptor_label> labels_{};

    bool dominates(mcraptor_label const& other_label) const {
      return std::any_of(labels_.begin(), labels_.end(),
                         [&](auto const& l) {
                           return l.dominates(other_label);
                         });
    }

    void add(mcraptor_label const& new_label) {
      if (this->dominates(new_label)) {
        return;
      }
      auto new_end = std::remove_if(labels_.begin(), labels_.end(),
                                    [&](auto const& l) {
                                      return new_label.dominates(l);
                                    });
      labels_.erase(new_end, labels_.end());
      labels_.emplace_back(new_label);
    }

    void unchecked_add(mcraptor_label const& new_label){
      auto new_end = std::remove_if(labels_.begin(), labels_.end(),
                                    [&](auto const& l) {
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
                           //TODO kommt des jemals vor, dass pair.first <= k ???
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
               bool const require_car_transport,
               bool const is_wheelchair,
               transfer_time_settings const& tts,
               std::vector<std::vector<std::pair<int, double>>> const& arr_dist)
      : tt_{tt},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        state_{state.resize(n_locations_, n_routes_, 0)},
        is_dest_{is_dest},
        lb_{lb},
        base_{base},
        transfer_time_settings_{tts},
        arr_dist_{arr_dist}{

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

    tmp_ = {n_locations_, mcraptor_bag{}};
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
    std::for_each(tmp_.begin(), tmp_.end(), [&](mcraptor_bag& bag) {
      bag.labels_.clear();
      //bag.add({kInvalid});
    });
    utl::fill(prev_round_station_mark_.blocks_, 0U);
    utl::fill(tmp_station_mark_.blocks_, 0U);
    utl::fill(station_mark_.blocks_, 0U);
    utl::fill(route_mark_.blocks_, 0U);
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    location_bags_[to_idx(l)][0U].add({.arr_t_ = unix_to_delta(base(), t), .success_chance=1.0f});
    best_bag_[to_idx(l)].add({.arr_t_ = unix_to_delta(base(), t), .success_chance=1.0f});
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


    //todo end_k + 1. Aber dann passt die größe der arrays nicht mehr. wird hier anderst gerechnet als im normalen raptor
    for (auto k = 1U; k != end_k; ++k) {
//      is_dest_.for_each_set_bit([&](std::uint64_t const i) {
//        dest_bag_.add({.arr_t_ = get_best_time(i)}, k);
//      });
      stats_.n_rounds = k;

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
        std::sort(best_bag_[i].labels_.begin(), best_bag_[i].labels_.end(),
                  [](mcraptor_label a, mcraptor_label b) {
                    return a.arr_t_ < b.arr_t_;
                  });
        //TODO Es wird in prun_flaged auch schon vorsortiert.
        std::sort(location_bags_[i][k].labels_.begin(), location_bags_[i][k].labels_.end(),
                  [](mcraptor_label a, mcraptor_label b) {
                    return a.arr_t_ > b.arr_t_;
                  });
      });

      update_dest_bag(k);

      tmp_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        tmp_[i].labels_.clear();
      });

      utl::fill(route_mark_.blocks_, 0U);
      std::swap(prev_round_station_mark_, station_mark_);
      utl::fill(tmp_station_mark_.blocks_, 0U);
      utl::fill(station_mark_.blocks_, 0U);

      for(auto const & [_, label]: dest_bag_.labels_){
        if(stats_.n_rounds_to_percantage == 0 && label.success_chance >= stats_.percantage) stats_.n_rounds_to_percantage = k;
      }

    }


    for(std::pair<unsigned, mcraptor_label> pair: dest_bag_.labels_){
      auto label = pair.second;
      if(pair.first > 0){
        results.add(
          journey{.legs_ = {},
                  .start_time_ = start_time,
                  .dest_time_ = delta_to_unix(base(), label.arr_t_),
                  .success_chance = label.success_chance,
                  .dest_ = location_idx_t{label.fp_l_},
                  .transfers_ = static_cast<std::uint8_t>(pair.first - 1)});
      }
    }
  }

  void print_leg(journey::leg leg, auto indent){
    for(int i = 0; i<indent; ++i) std::cout <<"\t";
    std::cout << location{tt_, leg.from_} << " id: " << leg.from_ << " ["<< leg.dep_time_ << "] TO: " << location{tt_, leg.to_} << " id: " << leg.to_ << " ["<< leg.arr_time_ << "] - " << leg.success_chance;
    leg.print(std::cout, tt_);
  }

  void reconstruct_leg(query const& q, journey& j, auto i, auto l, mcraptor_label label){
    auto k = j.transfers_ + 1 - i;
    auto [fp_leg, transport_leg] = get_legs(k, l, q.prf_idx_, label);
    auto next_l = kFwd ? transport_leg.from_ : transport_leg.to_;
    // don't add a 0-minute footpath at the end (fwd) or beginning (bwd)
    auto possible_start_t = unix_to_delta(base(), transport_leg.arr_time_);
    if (i != 0 && (fp_leg.from_ != fp_leg.to_ ||
        fp_leg.dep_time_ != fp_leg.arr_time_)) {
      j.add(std::move(fp_leg));
      possible_start_t = unix_to_delta(base(), fp_leg.arr_time_);
    }
    j.add(std::move(transport_leg));
    print_leg(transport_leg, i);
    if(i<j.transfers_ && !std::any_of(q.start_.begin(), q.start_.end(),[next_l](offset loc){return loc.target_ == next_l;})) {
      k = j.transfers_ + 1 - (i+1);
      std::vector<mcraptor_label> labels = {};
      get_labels_after_(cista::to_idx(next_l), k, possible_start_t, labels, 1);
      for(auto new_label: labels){
        reconstruct_leg(q, j, i+1, next_l, new_label);
      }
    }
  }

  void reconstruct_trip_based(query const& q, journey& j) {
    auto l = j.dest_;
    delta_t possible_start_t = unix_to_delta(base(), j.dest_time_);

    std::vector<mcraptor_label> labels = {};
    get_labels_after_(cista::to_idx(l), j.transfers_ + 1, possible_start_t, labels, 1);
    std::cout << "Gesamtwahrscheinlichkeit: " << j.success_chance << std::endl;
    std::cout << labels.size() << std::endl;
    for(auto label: labels){
      reconstruct_leg(q, j, 0, l, label);
    }
    if(kFwd) std::reverse(begin(j.legs_), end(j.legs_));
    //j.print(std::cout, tt_);
    std::cout << std::endl;
  }

  const bool should_print = true;
  void reconstruct(query const& q, journey& j) {
//    std::cout << "done: " << std::endl;
//    return;
    reconstruct_station_based(q, j);
  }

  void reconstruct_leg_station_based(query const& q, journey& j, auto i, auto l, mcraptor_label label, auto& trips){
    auto k = j.transfers_ + 1 - i;
    auto [fp_leg, transport_leg] = get_legs(k, l, q.prf_idx_, label);
    auto next_l = kFwd ? transport_leg.from_ : transport_leg.to_;
    auto possible_start_t = unix_to_delta(base(), transport_leg.arr_time_);
    if (i != 0 && (fp_leg.from_ != fp_leg.to_ ||
                   fp_leg.dep_time_ != fp_leg.arr_time_)) {
      //j.add(std::move(fp_leg));
      if(fp_leg.from_ != fp_leg.to_) trips.insert(fp_leg);
      possible_start_t = unix_to_delta(base(), fp_leg.arr_time_);
    }
    //j.add(std::move(transport_leg));
    trips.insert(transport_leg);
    if(i<j.transfers_ && !std::any_of(q.start_.begin(), q.start_.end(),[next_l](offset loc){return loc.target_ == next_l;})) {
      k = j.transfers_ + 1 - (i+1);
      std::vector<mcraptor_label> labels = {};
      get_labels_after_(cista::to_idx(next_l), k, possible_start_t, labels, 1);
      for(auto new_label: labels){
        reconstruct_leg_station_based(q, j, i+1, next_l, new_label, trips);
      }
    }
  }

  int colum_width = 20;
  void reconstruct_station_based(query const& q, journey& j) {
    auto cmp = [](journey::leg a, journey::leg b){
      if (a.dep_time_ != b.dep_time_) return a.dep_time_ < b.dep_time_;
      if (a.from_ != b.from_) return a.from_ < b.from_;
      return a.to_ < b.to_;
    };
    std::set<journey::leg, decltype(cmp)> trips;
    auto l = j.dest_;
    delta_t possible_start_t = unix_to_delta(base(), j.dest_time_);

    std::vector<mcraptor_label> labels = {};
    get_labels_after_dest_bag(cista::to_idx(l), j.transfers_ + 1, possible_start_t, labels, 1);
    if(should_print) std::cout << "Gesamtwahrscheinlichkeit: " << j.success_chance << " Umstiege: " << static_cast<int>(j.transfers_) << std::endl;

    for(auto label: labels){
      reconstruct_leg_station_based(q, j, 0, l, label, trips);
    }

    for(auto leg: trips){
      auto copy = leg;
      j.add(std::move(copy));
    }

    if(should_print){
      std::vector<location_idx_t> locations{};
      for (journey::leg leg : trips) {
        auto from_i =
            std::find_if(locations.begin(), locations.end(),
                         [&](location_idx_t l) { return l == leg.from_; }) -
            locations.begin();
        if (from_i == locations.size()) {
          locations.push_back(leg.from_);
        }
        auto to_i =
            std::find_if(locations.begin(), locations.end(),
                         [&](location_idx_t l) { return l == leg.to_; }) -
            locations.begin();
        if (to_i == locations.size()) {
          locations.push_back(leg.to_);
        }
      }
      for (location_idx_t lo : locations) {
        std::stringstream stringstream;
        stringstream << location{tt_, lo};
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
  void str_to_length(std::string& str, auto length){
    for(int i = 0; str.length() < length; ++i) str += " ";
    str = !cut_front ? str.substr(0, length) : str.substr(str.length() - length, str.length());
  }

  void print_unixtime(auto t, bool left){
    if(left){
      for (int j = 0; j < colum_width; ++j) {
        std::cout << " ";
      }
    }
    std::stringstream stringstream;
    stringstream << t;
    auto str = stringstream.str();
    str_to_length<true>(str, colum_width);
    std::cout << str;
    if(!left){
      for (int j = 0; j < colum_width; ++j) {
        std::cout << " ";
      }
    }
    std::cout << "|";
  }

  void print_column_spaces(int n){
    for (int i = 0; i < n; ++i){
      for (int j = 0; j < colum_width * 2; ++j) {
        std::cout << " ";
      }
      std::cout << "|";
    }
  }

  void print_leg_station_based(journey::leg leg, auto from_i, auto to_i, auto max){
    int f = from_i < to_i ? from_i : to_i;
    int t = (from_i < to_i ? to_i : from_i) - f;
    print_column_spaces(f);
    print_unixtime(from_i < to_i ? leg.dep_time_ : leg.arr_time_, from_i < to_i);
    print_column_spaces(t - 1);
    print_unixtime(from_i < to_i ? leg.arr_time_ : leg.dep_time_, from_i > to_i);
    print_column_spaces(max - (from_i < to_i ? to_i : from_i) - 1);
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

  void get_labels_after_(auto l, auto k, delta_t possible_start_t, std::vector<mcraptor_label>& labels, int ik){
    for (int i = k; ik <= i; --i) {
      std::copy_if(location_bags_[l][i].labels_.begin(), location_bags_[l][i].labels_.end(), std::back_inserter(labels),
                   [&](mcraptor_label val){
                     return std::none_of(labels.begin(), labels.end(), [&](mcraptor_label contained){
                       return contained.dominates(val);
                     });
                   });
    }
    std::sort(labels.begin(), labels.end(),[](mcraptor_label a, mcraptor_label b){
      return a.arr_t_ < b.arr_t_;
    });
    auto new_end = std::lower_bound(labels.begin(), labels.end(), possible_start_t, [](mcraptor_label a, delta_t t){
      return a.arr_t_ < t;
    });
    labels.erase(labels.begin(), new_end);
  }

  void get_labels_after_dest_bag(auto l, auto k, delta_t possible_start_t, std::vector<mcraptor_label>& labels, int ik){
    std::for_each(
        dest_bag_.labels_.begin(),
        dest_bag_.labels_.end(),
        [&](const std::pair<unsigned, mcraptor_label>& pair) {
          if (pair.first <= k && possible_start_t <= pair.second.arr_t_ && pair.second.arr_t_ != kInvalid) {
            labels.push_back(pair.second);
          }
        }
    );
    std::sort(labels.begin(), labels.end(),[](mcraptor_label a, mcraptor_label b){
      return a.arr_t_ < b.arr_t_;
    });
    auto new_end = std::lower_bound(labels.begin(), labels.end(), possible_start_t, [](mcraptor_label a, delta_t t){
      return a.arr_t_ < t;
    });
    labels.erase(labels.begin(), new_end);
  }

  double delay_distribution_real(delta_t x, clasz c){
    int id = static_cast<int>(c);
    const std::vector<std::pair<int, double>> & delays = arr_dist_[id].empty() ? arr_dist_[static_cast<int>(clasz::kOther)] : arr_dist_[id];

    if (x < delays.back().first) return delays[x].second;
    else return delays.back().second;
  }

  const delta_t max_delay = delta_t{90};
  float delay_distribution_paper(delta_t x){
    auto xf = static_cast<float>(x);
    auto cancelation_probability = 0.95f;
    return std::min(cancelation_probability, (31 * xf + 2 * max_delay) / (30 * xf + 3 * max_delay));
  }

  float delay_distribution_linear(delta_t x){
    float gradient = 0.02f;
    float on_time_probability = 0.6;
    return std::min(1.0f, gradient * static_cast<float>(x) + on_time_probability);
  }

  double transferProbability(delta_t to, clasz c){
    auto function = [&](auto x){return this->delay_distribution_real(x, c);};
    return function(to);
  }

  template <bool transfer=true>
  double cum_prob(auto l, auto k, delta_t possible_start_t, clasz const c){
    auto it = std::lower_bound(best_bag_[l].labels_.begin(), best_bag_[l].labels_.end(), possible_start_t, [](mcraptor_label a, delta_t t){
      return a.arr_t_ < t;
    });

    auto result = (transfer ? transferProbability(it->arr_t_ - possible_start_t, c) : 1) * it->success_chance;
    auto counterprob = 1 - (transfer ? transferProbability(it->arr_t_ - possible_start_t, c) : transferProbability(std::numeric_limits<nigiri::delta_t>::max(), c));
    ++it;
    for (; it != best_bag_[l].labels_.end(); ++it) {
      auto prob = transferProbability(it->arr_t_ - possible_start_t, c);
      result += counterprob * (transfer ? prob : 1) * it->success_chance;
      counterprob = counterprob * (1 - (transfer ? prob: transferProbability(std::numeric_limits<nigiri::delta_t>::max(), c)));
    }
    if(result > 1){
      std::cout << "größer: " << result << std::endl;
      std::cout << "k: " << k << std::endl;
      std::cout << "possible_start_t: " << possible_start_t << std::endl;
      SetLastError(0);
    }
    return result;
  }

  double cum_success_chance(auto l, auto k, delta_t possible_start_t, clasz const c){
    return cum_prob(l, k, possible_start_t, c);
  }

  double cum_enter_probability(auto l, auto k, delta_t possible_start_t, clasz const c){
    //TODO ordentlich machen. Es werden hier nur ausfälle berücksichtigt. ist gemacht?
    return cum_prob<false>(l, k, possible_start_t, c);
  }

  bool update_route(unsigned const k, route_idx_t const r) {
    auto stop_seq = tt_.route_location_seq_[r];
    auto any_marked = false;
    auto ets = std::vector<mcraptor_label>{};
    for (int i = 0; i != stop_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      if (!ets.empty() && stp.can_finish<SearchDir>(false)) {
        auto it = ets.begin();
        while (it != ets.end()) {
          auto& et_label = (*it);
          auto const by_transport = time_at_stop(
              r, et_label.trip_id, stop_idx, kFwd ? event_type::kArr : event_type::kDep);

          et_label.arr_t_ = by_transport;
          if (!best_bag_[l_idx].dominates(et_label) &&
              !dest_bag_.dominates(et_label, k) &&
              lb_[l_idx] != kUnreachable &&
              !dest_bag_.dominates({.arr_t_ = static_cast<delta_t>(by_transport + lb_[l_idx]), .trip_id = et_label.trip_id, .success_chance = et_label.success_chance}, k)) {
            if(!tmp_[l_idx].dominates(et_label)) {
              ++stats_.n_earliest_arrival_updated_by_route_;
              tmp_[l_idx].add({by_transport, et_label.trip_l_, stp.location_idx(), et_label.trip_id, et_label.success_chance, et_label.over_limit});
              tmp_station_mark_.set(l_idx, true);
              any_marked = true;
            }
            ++it;
          }else{
            it = ets.erase(it);
          }
        }
      }

      if (is_last || !stp.can_start<SearchDir>(false) || !prev_round_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      //enter transport
      if(prev_round_station_mark_[l_idx]) {
        auto const& prev_round_bag = get_round_bag(l_idx, k - 1);


        auto start = prev_round_bag.labels_[0].arr_t_;
        delta_t end;
        auto old_size = ets.size();
        for (int j = 0; j < prev_round_bag.labels_.size(); ++j){
          if(std::any_of(ets.begin(), ets.end(), [&](mcraptor_label label_of_vector){
                return label_of_vector.dominates(prev_round_bag.labels_[j]);
              })){
            continue;
          }
          end = prev_round_bag.labels_[j].arr_t_ - max_delay;

          if(start < end) continue;
          if((j+1)<prev_round_bag.labels_.size() && end <= prev_round_bag.labels_[j+1].arr_t_) continue;

          auto const [day_from, mam_from] = split(start);

          if(!get_earliest_transports(k, r, stop_idx, day_from, mam_from,
                                                     stp.location_idx(), ets, end)) break;
          start = ets.back().arr_t_ + dir(1);
          if((j+1)<prev_round_bag.labels_.size() && start > prev_round_bag.labels_[j+1].arr_t_){
            start = prev_round_bag.labels_[j+1].arr_t_;
          }
        }
        if(ets.size() > 1) {
          std::inplace_merge(
              ets.begin(), ets.begin() + old_size, ets.end(),
              [](auto const& a, auto const& b) { return a.arr_t_ > b.arr_t_; });

          auto it = ets.begin();
          while (it + 1 != ets.end()) {
            if (it->arr_t_ != (it + 1)->arr_t_) {
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
  
  void update_dest_bag(auto k){
    is_dest_.for_each_set_bit([&](std::uint64_t const i) {
      for (std::pair<unsigned, mcraptor_label> pair :dest_bag_.labels_) {
        if(pair.first == 0 || pair.first >=k || pair.second.fp_l_ != i) continue;
        auto tmp_label = pair.second;
        tmp_label.success_chance = cum_enter_probability(i, k, tmp_label.arr_t_, tt_.route_clasz_[tt_.transport_route_[tmp_label.trip_id.t_idx_]]);
        dest_bag_.add(tmp_label, k);
      }
      for (mcraptor_label tmp_label :tmp_[i].labels_) {
        if(tmp_label.arr_t_ == kInvalid) continue;
        tmp_label.success_chance = cum_enter_probability(i, k, tmp_label.arr_t_, tt_.route_clasz_[tt_.transport_route_[tmp_label.trip_id.t_idx_]]);
        dest_bag_.add(tmp_label, k);
      }
    });
  }

  void update_transfers(unsigned const k) {
    tmp_station_mark_.for_each_set_bit([&](auto&& i) {
      for (mcraptor_label tmp_label :tmp_[i].labels_) {
        mcraptor_label new_label = tmp_label;
        if ((delta_t) new_label.arr_t_ != kInvalid) {
          auto const is_dest = is_dest_[i];

          auto const transfer_time = is_dest ? 0
                  : dir(adjusted_transfer_time(
                            transfer_time_settings_,
                            tt_.locations_.transfer_time_[location_idx_t{i}]
                                .count()));

          new_label.arr_t_ = static_cast<delta_t>(new_label.arr_t_ + transfer_time);

          if (!best_bag_[i].dominates(new_label) &&
              !dest_bag_.dominates(new_label, k)) {
            if (lb_[i] == kUnreachable) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              continue;
            }
            new_label.arr_t_ = new_label.arr_t_ + lb_[i];
            if(dest_bag_.dominates(new_label, k)){
              ++stats_.fp_update_prevented_by_lower_bound_;
              continue;
            }
            new_label.arr_t_ = new_label.arr_t_ - lb_[i];
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
      for (mcraptor_label tmp_label :tmp_[i].labels_) {
        auto const l_idx = location_idx_t{i};
        auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx][l_idx]
                               : tt_.locations_.footpaths_in_[prf_idx][l_idx];


        for (auto const& fp : fps) {
          ++stats_.n_footpaths_visited_;

          auto const target = to_idx(fp.target());
          auto const tmp_time = tmp_label.arr_t_;
          if (tmp_time == kInvalid) {
            continue;
          }

          mcraptor_label new_label = tmp_label;
          new_label.arr_t_ = clamp(
              tmp_time + dir(adjusted_transfer_time(transfer_time_settings_,
                                                fp.duration().count())));

          if (!best_bag_[target].dominates(new_label) &&
              !dest_bag_.dominates(new_label, k)) {
            auto const lower_bound = lb_[target];

            if (lower_bound == kUnreachable ||
                dest_bag_.dominates({.arr_t_ = static_cast<delta_t>(new_label.arr_t_ + dir(lower_bound)), .trip_id = tmp_label.trip_id, .success_chance = tmp_label.success_chance}, k)) {
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

  void prun_flaged(){
    tmp_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      std::sort(tmp_[i].labels_.begin(), tmp_[i].labels_.end(), [](auto a, auto b){
        return a.arr_t_ < b.arr_t_;
      });
      auto first_flagged = std::find_if(tmp_[i].labels_.rbegin(), tmp_[i].labels_.rend(), [](auto a){return a.over_limit;});
      if(first_flagged == tmp_[i].labels_.rend()) --first_flagged;
      auto value = first_flagged->arr_t_ - max_delay;

      auto it = tmp_[i].labels_.begin();
      while (it+1 != tmp_[i].labels_.end()) {
        if(it->arr_t_ < value) it = tmp_[i].labels_.erase(it);
        else break;
      }
    });
  }

  std::tuple<journey::leg, journey::leg> get_legs(unsigned const k,
                                                 auto const l,
                                                 auto const prf_idx,
                                                  mcraptor_label label) {
    auto const& trip_l_idx = label.trip_l_;
    auto const& fp_l_idx = label.fp_l_;
    auto const& arr_t = label.arr_t_;

    delta_t trip_dep_time;
    delta_t trip_arr_fp_dep_time;
    stop_idx_t from_stop_idx;
    stop_idx_t to_stop_idx;
    route_idx_t route_idx;
    transport transport;
    footpath footpath{};

    auto r = tt_.transport_route_[label.trip_id.t_idx_];

    auto const stop_seq = tt_.route_location_seq_[r];
    struct transport trip{};

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = stp.location_idx();

      if (trip.is_valid() && (fp_l_idx.v_ == l_idx.v_)) {
        to_stop_idx = stop_idx;
        transport = trip;
        route_idx = r;
        trip_arr_fp_dep_time = time_at_stop(
            r, trip, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
//        found_end_location = true;
        break;
      }

      if (trip_l_idx.v_ == l_idx.v_) {
//        auto const [day, mam] = split(prev_stop_time);
        trip = label.trip_id;
//        trip = get_earliest_transport(k - 1, r, stop_idx, day, mam,
//                                      stp.location_idx());
        trip_dep_time = time_at_stop(r, trip, stop_idx,kFwd ? event_type::kDep : event_type::kArr);
        from_stop_idx = stop_idx;
      }
//      }
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
                     footpath,
                      label.success_chance};

    auto const transport_leg =
        journey::leg{SearchDir, trip_l_idx, fp_l_idx,
                     delta_to_unix(base(), trip_dep_time),
                     delta_to_unix(base(), trip_arr_fp_dep_time),
                     journey::run_enter_exit{{.t_ = transport,
                                              .stop_range_ = interval<stop_idx_t>{0, static_cast<stop_idx_t>(
                                                                                         tt_.route_location_seq_[route_idx].size())}},
                                             from_stop_idx, to_stop_idx},
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
                                    std::vector<mcraptor_label> & ets,
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

        transport new_et = {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
        delta_t time = time_at_stop(r, new_et, stop_idx,kFwd ? event_type::kDep : event_type::kArr);
        //TODO cummulierter wert nicht jedes mal neu bestimmen sondern alte berechnung wieder verwenden
        mcraptor_label new_et_label = {.arr_t_ = time, .trip_l_ = l,
                                        .trip_id = new_et, .success_chance = cum_success_chance(cista::to_idx(l), k-1, time, tt_.route_clasz_[r]), .over_limit = time < end};
        ets.push_back(new_et_label);
        if(static_cast<day_idx_t>(as_int(day) - ev_day_offset) <= day_at_stop_to && ev_mam <= mam_at_stop_to.count()) return true;
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

}  // namespace nigiri::routing