#pragma once

#include <variant>

#include "nigiri/common/delta_t.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

// using val_t_loc = location_idx_t::value_t;
using val_t_con = connection_idx_t::value_t;

struct walk {
  location_idx_t from_;
  footpath fp_;
};

struct profile_entry {
  delta_t dep_time_;
  meat_t meat_;
  std::variant<ride, walk> uses_;
};

struct profile {
  std::vector<profile_entry>::const_iterator begin_, end_;

  std::reverse_iterator<std::vector<profile_entry>::const_iterator> begin()
      const {
    return std::reverse_iterator<std::vector<profile_entry>::const_iterator>{
        end_};
  }

  std::reverse_iterator<std::vector<profile_entry>::const_iterator> end()
      const {
    return std::reverse_iterator<std::vector<profile_entry>::const_iterator>{
        begin_};
  }
};

struct static_profile_set {
  static_profile_set(timetable const& tt);

  profile for_stop(location_idx_t stop_id) const {
    return {entry_.begin() + entry_begin_end_[stop_id].first,
            entry_.begin() + entry_begin_end_[stop_id].second};
  }

  void reset_stop(location_idx_t stop_id) {
    entry_begin_end_[stop_id].second = entry_begin_end_[stop_id].first + 1;
  }

  profile_entry const& early_stop_entry(location_idx_t stop_id) const {
    return entry_[entry_begin_end_[stop_id].second - 1];
  }

  void add_early_entry(location_idx_t stop_id, profile_entry const& e) {
    if (is_stop_empty(stop_id)) {
      stop_reset_list_[stop_reset_list_end_++] = stop_id;
    }
    entry_[entry_begin_end_[stop_id].second++] = e;
  }

  void replace_early_entry(location_idx_t stop_id, profile_entry const& e) {
    entry_[entry_begin_end_[stop_id].second - 1] = e;
  }

  bool is_stop_empty(location_idx_t stop) const {
    return entry_begin_end_[stop].second == entry_begin_end_[stop].first + 1;
  }

  val_t_con compute_entry_amount() const;

  void reset_fp_dis_to_target() {
    for (auto i = 0U; i < fp_to_target_reset_list_end_; ++i) {
      fp_dis_to_target_[fp_to_target_reset_list_[i]] =
          std::numeric_limits<meat_t>::infinity();
    }
    fp_to_target_reset_list_end_ = 0;
  }

  void reset_stop() {
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      reset_stop(stop_reset_list_[i]);
    }
    stop_reset_list_end_ = 0;
  }
  void reset() {
    reset_stop();
    reset_fp_dis_to_target();
  }

  void set_fp_dis_to_target(location_idx_t l_idx, meat_t dis) {
    fp_dis_to_target_[l_idx] = dis;
    fp_to_target_reset_list_[fp_to_target_reset_list_end_++] = l_idx;
  }

  size_t n_entry_idxs() const { return entry_.size(); }

  auto global_index_of(
      std::reverse_iterator<std::vector<profile_entry>::const_iterator> const&
          rit) const {

    assert(&(*rit) == &entry_[rit.base() - entry_.begin() - 1]);
    return rit.base() - entry_.begin() - 1;
  }

  vector_map<location_idx_t, meat_t> fp_dis_to_target_;
  std::vector<location_idx_t> fp_to_target_reset_list_;
  location_idx_t::value_t fp_to_target_reset_list_end_;
  std::vector<location_idx_t> stop_reset_list_;
  location_idx_t::value_t stop_reset_list_end_;
  vector_map<location_idx_t, std::pair<val_t_con, val_t_con>> entry_begin_end_;
  std::vector<profile_entry> entry_;
};

struct dynamic_growth_profile_set {
  dynamic_growth_profile_set(timetable const& tt)
      : tt_{tt},
        recompute_entry_amount_{true},
        l_start_idx_(tt.n_locations(), 0),
        fp_dis_to_target_(tt.n_locations(),
                          std::numeric_limits<meat_t>::infinity()),
        fp_to_target_reset_list_(tt.n_locations()),
        fp_to_target_reset_list_end_{0},
        stop_reset_list_(tt.n_locations()),
        stop_reset_list_end_{0},
        entry_idx_end_(tt.n_locations(),
                       std::pair(std::numeric_limits<size_t>::max(), 0)),
        entry_() {};

  profile for_stop(location_idx_t stop_id) const {
    return {entry_[entry_idx_end_[stop_id].first].begin(),
            entry_[entry_idx_end_[stop_id].first].begin() +
                entry_idx_end_[stop_id].second};
  }

  void reset_stop(location_idx_t stop_id) {
    entry_idx_end_[stop_id].second = 1;
  }

  profile_entry const& early_stop_entry(location_idx_t stop_id) {
    if (is_stop_un_init(stop_id)) {
      init_stop(stop_id);
      return entry_[entry_idx_end_[stop_id].first][0];
    } else {
      return entry_[entry_idx_end_[stop_id].first]
                   [entry_idx_end_[stop_id].second - 1];
    }
  }

  void init_stop(location_idx_t stop_id) {
    if (entry_idx_end_[stop_id].first == std::numeric_limits<size_t>::max()) {
      entry_.push_back(std::vector<profile_entry>());
      entry_idx_end_[stop_id].first = entry_.size() - 1;
    }
    entry_[entry_idx_end_[stop_id].first].push_back(profile_entry{
        std::numeric_limits<delta_t>::max(),
        std::numeric_limits<meat_t>::infinity(),
        ride{connection_idx_t::invalid(), connection_idx_t::invalid()}});
    entry_idx_end_[stop_id].second = 1;
    recompute_entry_amount_ = true;
  }

  void add_early_entry(location_idx_t stop_id, profile_entry const& e) {
    if (is_stop_un_init(stop_id)) {
      init_stop(stop_id);
      stop_reset_list_[stop_reset_list_end_++] = stop_id;
    } else if (is_stop_empty(stop_id)) {
      stop_reset_list_[stop_reset_list_end_++] = stop_id;
    }
    if (entry_[entry_idx_end_[stop_id].first].size() <=
        entry_idx_end_[stop_id].second) {
      entry_[entry_idx_end_[stop_id].first].push_back(e);
    } else {
      entry_[entry_idx_end_[stop_id].first][entry_idx_end_[stop_id].second] = e;
    }
    ++entry_idx_end_[stop_id].second;
    recompute_entry_amount_ = true;
  }

  void replace_early_entry(location_idx_t stop_id, profile_entry const& e) {
    if (!is_stop_un_init(stop_id)) {
      entry_[entry_idx_end_[stop_id].first]
            [entry_idx_end_[stop_id].second - 1] = e;
    }
  }

  bool is_stop_un_init(location_idx_t stop) const {
    return entry_idx_end_[stop].second == 0;
  }

  bool is_stop_empty(location_idx_t stop) const {
    return entry_idx_end_[stop].second <= 1;
  }

  size_t compute_entry_amount() const {
    size_t n_entrys = 0;
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      l_start_idx_[stop_reset_list_[i]] = n_entrys;
      n_entrys += entry_idx_end_[stop_reset_list_[i]].second;
    }
    recompute_entry_amount_ = false;
    return n_entrys - stop_reset_list_end_;
  };

  void reset_fp_dis_to_target() {
    for (auto i = 0U; i < fp_to_target_reset_list_end_; ++i) {
      fp_dis_to_target_[fp_to_target_reset_list_[i]] =
          std::numeric_limits<meat_t>::infinity();
    }
    fp_to_target_reset_list_end_ = 0;
  }

  void reset_stop() {
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      reset_stop(stop_reset_list_[i]);
    }
    stop_reset_list_end_ = 0;
    recompute_entry_amount_ = true;
  }

  void reset() {
    reset_stop();
    reset_fp_dis_to_target();
  }

  void set_fp_dis_to_target(location_idx_t l_idx, meat_t dis) {
    fp_dis_to_target_[l_idx] = dis;
    fp_to_target_reset_list_[fp_to_target_reset_list_end_++] = l_idx;
  }

  size_t n_entry_idxs() const {
    return compute_entry_amount() + stop_reset_list_end_;
  }

  auto global_index_of(
      std::reverse_iterator<std::vector<profile_entry>::const_iterator> const&
          rit) const {
    location_idx_t stop_idx;
    std::visit(
        utl::overloaded{[&](walk const& w) { stop_idx = w.from_; },
                        [&](ride const& r) {
                          auto enter_conn = tt_.fwd_connections_[r.enter_conn_];
                          stop_idx = stop{enter_conn.dep_stop_}.location_idx();
                        }},
        rit->uses_);
    if (recompute_entry_amount_) {
      compute_entry_amount();
    }
    assert(&(*rit) ==
           &entry_[stop_idx.v_][rit.base() -
                                entry_[entry_idx_end_[stop_idx].first].begin() -
                                1]);
    return rit.base() - entry_[entry_idx_end_[stop_idx].first].begin() - 1 +
           l_start_idx_[stop_idx];
  }

  timetable const& tt_;
  mutable bool recompute_entry_amount_;
  mutable vector_map<location_idx_t, size_t> l_start_idx_;
  vector_map<location_idx_t, meat_t> fp_dis_to_target_;
  std::vector<location_idx_t> fp_to_target_reset_list_;
  location_idx_t::value_t fp_to_target_reset_list_end_;
  std::vector<location_idx_t> stop_reset_list_;
  location_idx_t::value_t stop_reset_list_end_;
  vector_map<location_idx_t, std::pair<size_t, size_t>> entry_idx_end_;
  std::vector<std::vector<profile_entry>> entry_;
};

struct dynamic_profile_set {
  dynamic_profile_set(timetable const& tt)
      : tt_{tt},
        recompute_entry_amount_{true},
        l_start_idx_(tt.n_locations(), 0),
        fp_dis_to_target_(tt.n_locations(),
                          std::numeric_limits<meat_t>::infinity()),
        fp_to_target_reset_list_(tt.n_locations()),
        fp_to_target_reset_list_end_{0},
        stop_reset_list_(tt.n_locations()),
        stop_reset_list_end_{0},
        entry_(tt.n_locations(),
               std::vector<profile_entry>(
                   {{std::numeric_limits<delta_t>::max(),
                     std::numeric_limits<meat_t>::infinity(),
                     ride{connection_idx_t::invalid(),
                          connection_idx_t::invalid()}}})) {};

  profile for_stop(location_idx_t stop_id) const {
    return {entry_[to_idx(stop_id)].begin(), entry_[to_idx(stop_id)].end()};
  }

  void reset_stop(location_idx_t stop_id) {
    entry_[to_idx(stop_id)].clear();
    entry_[to_idx(stop_id)].push_back(profile_entry{
        std::numeric_limits<delta_t>::max(),
        std::numeric_limits<meat_t>::infinity(),
        ride{connection_idx_t::invalid(), connection_idx_t::invalid()}});
    entry_[to_idx(stop_id)].shrink_to_fit();
    recompute_entry_amount_ = true;
  }

  profile_entry const& early_stop_entry(location_idx_t stop_id) {
    return entry_[to_idx(stop_id)].back();
  }

  void add_early_entry(location_idx_t stop_id, profile_entry const& e) {
    if (is_stop_empty(stop_id)) {
      stop_reset_list_[stop_reset_list_end_++] = stop_id;
    }
    entry_[to_idx(stop_id)].emplace_back(e);
    recompute_entry_amount_ = true;
  }

  void replace_early_entry(location_idx_t stop_id, profile_entry const& e) {
    entry_[to_idx(stop_id)].back() = e;
  }

  bool is_stop_empty(location_idx_t stop) const {
    return entry_[to_idx(stop)].size() == 1;
  }

  size_t compute_entry_amount() const {
    size_t n_entrys = 0;
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      l_start_idx_[stop_reset_list_[i]] = n_entrys;
      n_entrys += entry_[to_idx(stop_reset_list_[i])].size();
    }
    recompute_entry_amount_ = false;
    return n_entrys - stop_reset_list_end_;
  };

  void reset_fp_dis_to_target() {
    for (auto i = 0U; i < fp_to_target_reset_list_end_; ++i) {
      fp_dis_to_target_[fp_to_target_reset_list_[i]] =
          std::numeric_limits<meat_t>::infinity();
    }
    fp_to_target_reset_list_end_ = 0;
  }

  void reset_stop() {
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      reset_stop(stop_reset_list_[i]);
    }
    stop_reset_list_end_ = 0;
    recompute_entry_amount_ = true;
  }

  void reset() {
    reset_stop();
    reset_fp_dis_to_target();
  }

  void set_fp_dis_to_target(location_idx_t l_idx, meat_t dis) {
    fp_dis_to_target_[l_idx] = dis;
    fp_to_target_reset_list_[fp_to_target_reset_list_end_++] = l_idx;
  }

  size_t n_entry_idxs() const {
    return compute_entry_amount() + stop_reset_list_end_;
  }

  auto global_index_of(
      std::reverse_iterator<std::vector<profile_entry>::const_iterator> const&
          rit) const {
    location_idx_t stop_idx;
    std::visit(
        utl::overloaded{[&](walk const& w) { stop_idx = w.from_; },
                        [&](ride const& r) {
                          auto enter_conn = tt_.fwd_connections_[r.enter_conn_];
                          stop_idx = stop{enter_conn.dep_stop_}.location_idx();
                        }},
        rit->uses_);
    if (recompute_entry_amount_) {
      compute_entry_amount();
    }
    assert(&(*rit) ==
           &entry_[stop_idx.v_]
                  [rit.base() - entry_[to_idx(stop_idx)].begin() - 1]);
    return rit.base() - entry_[to_idx(stop_idx)].begin() - 1 +
           l_start_idx_[stop_idx];
  }

  timetable const& tt_;
  mutable bool recompute_entry_amount_;
  mutable vector_map<location_idx_t, size_t> l_start_idx_;
  vector_map<location_idx_t, meat_t> fp_dis_to_target_;
  std::vector<location_idx_t> fp_to_target_reset_list_;
  location_idx_t::value_t fp_to_target_reset_list_end_;
  std::vector<location_idx_t> stop_reset_list_;
  location_idx_t::value_t stop_reset_list_end_;
  std::vector<std::vector<profile_entry>> entry_;
};

}  // namespace nigiri::routing::meat::csa