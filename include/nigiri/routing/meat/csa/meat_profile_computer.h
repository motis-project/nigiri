#pragma once

#include <memory>
#include <queue>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/meat/csa/meat_csa_stats.h"
#include "nigiri/routing/meat/csa/profile.h"
#include "nigiri/routing/meat/delay.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

template <typename ProfileSet>
struct meat_profile_computer {

  struct trip_data {
    meat_t meat_;
    connection_idx_t exit_conn_;
  };

  explicit meat_profile_computer(timetable const& tt,
                                 meat_csa_state<ProfileSet>& state,
                                 day_idx_t const& base,
                                 clasz_mask_t const& allowed_claszes,
                                 profile_idx_t const& prf_idx,
                                 meat_csa_stats& stats)
      : tt_{tt},
        base_{base},
        state_{state},
        allowed_claszes_{allowed_claszes},
        fp_prf_idx_{prf_idx},
        stats_{stats},
        trip_reset_list_(tt.n_transports()),
        trip_reset_list_end_{0},
        profile_set_{state_.profile_set_} {
    assert(std::numeric_limits<meat_t>::has_infinity == true);

    for (auto t_idx = transport_idx_t{0}; t_idx < tt_.n_transports(); ++t_idx) {
      auto const t_size = tt_.travel_duration_days_[t_idx];
      trip_.emplace_back(std::vector<trip_data>(
          t_size, {std::numeric_limits<meat_t>::infinity(),
                   connection_idx_t::invalid()}));
    }
    assert(trip_.size() == tt_.n_transports());
  }

  delta_t tt_to_delta(day_idx_t const day, std::int16_t mam) const {
    return nigiri::tt_to_delta(base_, day, duration_t{mam});
  }
  void reset() {
    reset_trip();
    while (!fp_que_.empty()) {
      fp_que_.pop();
    }
  }
  void reset_trip() {
    for (auto i = 0U; i < trip_reset_list_end_; ++i) {
      for (auto& t : trip_[trip_reset_list_[i]]) {
        t = {std::numeric_limits<meat_t>::infinity(),
             connection_idx_t::invalid()};
      }
    }
    trip_reset_list_end_ = 0;
  }

  template <bool WithClaszFilter>
  void compute_profile_set(
      std::pair<day_idx_t, connection_idx_t> const& conn_begin,
      std::pair<day_idx_t, connection_idx_t> const& conn_end,
      location_idx_t target_stop,
      delta_t source_time,
      delta_t max_delay,
      meat_t fuzzy_dominance_offset,
      meat_t transfer_cost,
      delta_t last_arr) {
    auto const& ea = state_.ea_;

    auto evaluate_profile = [&](location_idx_t stop, delta_t when) {
      meat_t meat = 0.0;
      double assigned_prob = 0.0;

      auto i = std::begin(profile_set_.for_stop(stop));
      while (assigned_prob < 1.0) {
        double new_prob =
            delay_prob(clamp(i->dep_time_ - when),
                       tt_.locations_.transfer_time_[stop].count(), max_delay);
        meat += (new_prob - assigned_prob) * i->meat_;
        assigned_prob = new_prob;
        ++i;
      }
      return meat;
    };

    profile_set_.set_fp_dis_to_target(target_stop, 0.0);
    for (auto const& fp :
         tt_.locations_.footpaths_in_[fp_prf_idx_][target_stop]) {
      profile_set_.set_fp_dis_to_target(fp.target(), fp.duration().count());
    }

    auto conn = conn_end;
    auto& day = conn.first;
    auto& con_idx = conn.second;
    std::uint8_t n_day = 0;
    while (conn >= conn_begin) {
      stats_.meat_n_connections_scanned_++;
      auto const& c = tt_.fwd_connections_[con_idx];
      auto const c_dep_time = tt_to_delta(day, c.dep_time_.mam());
      auto const c_arr_time = tt_to_delta(
          day + static_cast<day_idx_t>(c.arr_time_.days() - c.dep_time_.days()),
          c.arr_time_.mam());

      insert_footpaths_till(c_dep_time, fuzzy_dominance_offset);

      if ((WithClaszFilter
               ? is_allowed(
                     allowed_claszes_,
                     tt_.route_clasz_[tt_.transport_route_[c.transport_idx_]])
               : true) &&
          ea[stop{c.dep_stop_}.location_idx()] <= c_dep_time &&
          c_arr_time <= last_arr && tt_.is_connection_active(c, day)) {
        auto const d_idx = static_cast<day_idx_t::value_t>(
            (c.dep_time_.days() + n_day) %
            tt_.travel_duration_days_[c.transport_idx_]);

        meat_t meat = trip_[c.transport_idx_][d_idx].meat_;

        if (stop{c.arr_stop_}.out_allowed()) {
          auto c_arr_stop_idx = stop{c.arr_stop_}.location_idx();

          meat = std::min(
              meat, profile_set_.fp_dis_to_target_[c_arr_stop_idx] +
                        static_cast<meat_t>(
                            c_arr_time) /*TODO add expected value to it?*/);
          // TODO add expected value to it? add final footpath in
          // graph_extractor would have to be changed

          if (!profile_set_.is_stop_empty(c_arr_stop_idx)) {
            meat = std::min(meat, evaluate_profile(c_arr_stop_idx, c_arr_time) +
                                      transfer_cost);
          }

          if (meat < trip_[c.transport_idx_][d_idx].meat_) {
            if (trip_[c.transport_idx_][d_idx].meat_ ==
                std::numeric_limits<meat_t>::infinity()) {
              trip_reset_list_[trip_reset_list_end_++] = c.transport_idx_;
            }
            trip_[c.transport_idx_][d_idx].meat_ = meat;
            trip_[c.transport_idx_][d_idx].exit_conn_ = con_idx;
          }
        }

        auto const c_dep_stop_idx = stop{c.dep_stop_}.location_idx();
        auto const faster_than_walk =
            meat < profile_set_.fp_dis_to_target_[c_dep_stop_idx] +
                       static_cast<meat_t>(c_dep_time);
        auto const& early_entry = profile_set_.early_stop_entry(c_dep_stop_idx);
        if (stop{c.dep_stop_}.in_allowed() && faster_than_walk &&
            meat < early_entry.meat_ - fuzzy_dominance_offset) {
          auto const new_entry = profile_entry{
              c_dep_time, meat,
              ride{con_idx, trip_[c.transport_idx_][d_idx].exit_conn_}};
          add_or_replace_entry(new_entry, c_dep_stop_idx);
          for (auto const& fp :
               tt_.locations_.footpaths_in_[fp_prf_idx_][c_dep_stop_idx]) {
            auto const fp_dep_time = clamp(c_dep_time - fp.duration().count());
            auto const& ee_fp = profile_set_.early_stop_entry(fp.target());
            auto const faster_than_final_fp =
                meat < profile_set_.fp_dis_to_target_[fp.target()] +
                           static_cast<meat_t>(fp_dep_time);
            if (faster_than_final_fp &&
                meat < ee_fp.meat_ - fuzzy_dominance_offset) {
              fp_que_.push(std::make_unique<profile_entry>(profile_entry{
                  fp_dep_time, meat,
                  walk{fp.target(), footpath(c_dep_stop_idx, fp.duration())}}));
              stats_.meat_n_fp_added_to_que_++;
            }
          }
        }
      }

      if (c.trip_con_idx_ == 0) {
        auto const d_idx = static_cast<day_idx_t::value_t>(
            (c.dep_time_.days() + n_day) %
            tt_.travel_duration_days_[c.transport_idx_]);
        trip_[c.transport_idx_][d_idx] = {
            std::numeric_limits<meat_t>::infinity(),
            connection_idx_t::invalid()};
      }

      if (con_idx == connection_idx_t{0}) {
        --day;
        con_idx = connection_idx_t{tt_.fwd_connections_.size() - 1};
        ++n_day;
      } else {
        --con_idx;
      }
    }
    insert_footpaths_till(source_time, fuzzy_dominance_offset);
    stats_.meat_n_e_in_profile_ = profile_set_.compute_entry_amount();
  }

  void add_or_replace_entry(profile_entry const& new_entry,
                            location_idx_t dep_stop_idx) {
    auto const& early_entry = profile_set_.early_stop_entry(dep_stop_idx);
    if (early_entry.dep_time_ == new_entry.dep_time_) {
      profile_set_.replace_early_entry(dep_stop_idx, new_entry);
    } else {
      profile_set_.add_early_entry(dep_stop_idx, new_entry);
    }
    stats_.meat_n_e_added_or_replaced_to_profile_++;
  }

  void insert_footpaths_till(delta_t time, meat_t f_d_offset) {
    while (!fp_que_.empty() && fp_que_.top()->dep_time_ >= time) {
      auto const& np = fp_que_.top();
      auto const dep_stop_idx = std::get_if<walk>(&np->uses_)->from_;
      auto const& ee = profile_set_.early_stop_entry(dep_stop_idx);
      if (np->meat_ < ee.meat_ - f_d_offset) {
        add_or_replace_entry(*np, dep_stop_idx);
        stats_.meat_n_fp_added_to_profile_++;
      }
      fp_que_.pop();
    }
  }

  timetable const& tt_;
  day_idx_t const& base_;
  meat_csa_state<ProfileSet>& state_;
  clasz_mask_t const& allowed_claszes_;
  profile_idx_t const& fp_prf_idx_;
  meat_csa_stats& stats_;
  std::vector<transport_idx_t> trip_reset_list_;
  transport_idx_t::value_t trip_reset_list_end_;
  ProfileSet& profile_set_;
  vecvec<transport_idx_t, trip_data> trip_;
  std::priority_queue<std::unique_ptr<profile_entry>,
                      std::vector<std::unique_ptr<profile_entry>>,
                      decltype([](std::unique_ptr<profile_entry> const& l,
                                  std::unique_ptr<profile_entry> const& r) {
                        return l->dep_time_ < r->dep_time_;
                      })>
      fp_que_;
};

}  // namespace nigiri::routing::meat::csa