#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "nigiri/types.h"

namespace nigiri {

// Constant-space aggregates of the n-to-all search (average, maximum, owner of
// the maximum, coverage, and the minutes per origin), see raptor_state::agg_.
//
// `Start` in the templates below is profile_start_label (raptor_state.h): the
// start label of a stop, with reached(), dur_ (travel time, no waiting) and
// owner_. It is a template parameter only so that this header does not depend
// on raptor_state.h, which includes it.

// n-to-all aggregates of a stop over the minutes t = 0..len of the window
// (t is "a person at an origin at minute t"). Value at t = best-of-all-origins
// travel time, waiting for the departure included. All times are minutes after
// the start of the window.
struct agg_label {
  static constexpr std::int16_t kNone = std::numeric_limits<std::int16_t>::min();

  // front: the newest entry. It is the best journey for the minutes after the
  // departure of the next entry that is still to come, up to front_dep_.
  std::int16_t front_dep_{kNone};
  std::int16_t front_arr_{0};  // "ready to continue", same value as min_label
  std::int16_t first_dep_{0};  // departure of the very first entry of the stop

  // closed minutes
  std::int16_t covered_{0};  // minutes with a value <= the travel time limit
  std::int16_t max_{-1};     // largest value among them
  std::int32_t sum_{0};      // sum of the values

  location_idx_t front_owner_{location_idx_t::invalid()};
  location_idx_t max_owner_{location_idx_t::invalid()};  // start stop at max_
};

struct agg_state {
  static constexpr std::uint8_t kNoOrigin = 255U;

  void clear(std::size_t const n) {
    labels_.assign(n, agg_label{});
    // n_origins_ and origin_of_ are set by the wrapper before the search
    minutes_.assign(n * n_origins_, std::uint16_t{0U});
  }

  // A label of stop i (round >= 1) that improved in some round of the run that
  // starts at run_start. `start` is the start label of the stop (see
  // profile_start_label): reached without a vehicle, constant over the window.
  template <typename Start>
  void offer(std::size_t const i,
             unixtime_t const run_start,
             unixtime_t const arrival,
             location_idx_t const owner,
             Start const& start) {
    auto& a = labels_[i];
    auto const d_n = rel(run_start);
    auto const a_n = rel(arrival);
    if (a.front_dep_ == agg_label::kNone) {  // first entry of the stop
      a.front_dep_ = d_n;
      a.front_arr_ = a_n;
      a.first_dep_ = d_n;
      a.front_owner_ = owner;
      return;
    }
    if (a.front_dep_ == d_n) {  // same run, a later round: overwrite
      if (a_n < a.front_arr_) {
        a.front_arr_ = a_n;
        a.front_owner_ = owner;
      }
      return;
    }
    assert(d_n < a.front_dep_);  // runs go from the latest to the earliest
    if (!(a_n < a.front_arr_)) {  // dominated: earlier departure, no earlier arrival
      return;
    }
    close(i, d_n, start);
    a.front_dep_ = d_n;
    a.front_arr_ = a_n;
    a.front_owner_ = owner;
  }

  // After the last run: close the fronts and add the minutes after the first
  // entry, which only a start label can serve.
  template <typename Start>
  void finalize(std::vector<Start> const& start) {
    for (auto i = std::size_t{0U}; i != labels_.size(); ++i) {
      auto& a = labels_[i];
      auto tail_from = 0;
      if (a.front_dep_ != agg_label::kNone) {
        close(i, -1, start[i]);
        tail_from = std::max(a.first_dep_ + 1, 0);
        a.front_dep_ = agg_label::kNone;
      }
      if (start_usable(start[i])) {
        add_const(i, tail_from, len_, static_cast<int>(start[i].dur_.count()),
                  start[i].owner_);
      }
    }
  }

  // The front of stop i owns the minutes d_n + 1 .. its own departure (d_n is
  // the departure of the entry that replaces it, -1 at the end).
  template <typename Start>
  void close(std::size_t const i, int const d_n, Start const& s) {
    auto& a = labels_[i];
    auto const lo = std::max(d_n + 1, 0);
    auto const hi = std::min<int>(a.front_dep_, len_);
    if (lo > hi) {
      return;
    }
    auto from = lo;
    if (start_usable(s)) {
      // For t <= front_arr_ - c the start label (constant c, no waiting) is at
      // least as good as the vehicle; ties go to the start label.
      auto const c = static_cast<int>(s.dur_.count());
      auto const split = std::min(hi, a.front_arr_ - c);
      add_const(i, lo, split, c, s.owner_);
      from = std::max(lo, split + 1);
    }
    // values a - t above the limit are not covered
    add_vehicle(i, std::max(from, a.front_arr_ - t_max_), hi, a.front_arr_,
                a.front_owner_);
  }

  template <typename Start>
  bool start_usable(Start const& s) const {
    return s.reached() && s.dur_.count() <= t_max_;
  }

  std::int16_t rel(unixtime_t const t) const {
    return static_cast<std::int16_t>((t - t0_).count());
  }

  // minutes lo..hi of stop i, values a - t (largest at lo)
  void add_vehicle(std::size_t const i, int const lo, int const hi,
                   int const a, location_idx_t const owner) {
    if (lo > hi) {
      return;
    }
    auto& g = labels_[i];
    auto const n = hi - lo + 1;
    g.sum_ += n * a - (lo + hi) * n / 2;
    g.covered_ = static_cast<std::int16_t>(g.covered_ + n);
    if (a - lo > g.max_) {  // strict: on a tie the first closed stays
      g.max_ = static_cast<std::int16_t>(a - lo);
      g.max_owner_ = owner;
    }
    count(i, owner, n);
  }

  // minutes lo..hi of stop i, constant value c
  void add_const(std::size_t const i, int const lo, int const hi, int const c,
                 location_idx_t const owner) {
    if (lo > hi) {
      return;
    }
    auto& g = labels_[i];
    auto const n = hi - lo + 1;
    g.sum_ += n * c;
    g.covered_ = static_cast<std::int16_t>(g.covered_ + n);
    if (c > g.max_) {
      g.max_ = static_cast<std::int16_t>(c);
      g.max_owner_ = owner;
    }
    count(i, owner, n);
  }

  // n minutes of stop i belong to the origin of the start stop `owner`
  void count(std::size_t const i, location_idx_t const owner, int const n) {
    if (n_origins_ == 0U) {  // counters not requested
      return;
    }
    auto const o = origin_of_[to_idx(owner)];
    assert(o != kNoOrigin && o < n_origins_);  // an owner is always a start stop
    auto& m = minutes_[i * n_origins_ + o];
    m = static_cast<std::uint16_t>(m + n);
  }

  // set by raptor_n_to_all before the search
  unixtime_t t0_{};   // start of the window
  int len_{0};        // minutes from the start to the end of the window
  int t_max_{0};      // travel time limit in minutes
  std::vector<agg_label> labels_;

  // counters (0 origins = off); set by raptor_n_to_all before the search
  std::uint32_t n_origins_{0U};
  std::vector<std::uint8_t> origin_of_;   // by location: origin index, kNoOrigin if none
  std::vector<std::uint16_t> minutes_;    // [stop * n_origins_ + origin]
};

}  // namespace nigiri
