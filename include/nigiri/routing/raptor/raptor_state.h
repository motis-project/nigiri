#pragma once

#include <array>
#include <span>
#include <vector>
#include <cstdint>
#include <limits>
#include "date/date.h"

#include "cista/containers/bitvec.h"
#include "cista/containers/flat_matrix.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/flat_matrix_view.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/n_to_all_aggregates.h"
#include "nigiri/types.h"
namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

// n-to-all: shortest travel time to a stop over all runs of a range search.
struct min_label {
  bool reached() const { return owner_ != location_idx_t::invalid(); }

  // arrival (as stored in best_) minus the departure of the run
  duration_t dur_{std::numeric_limits<duration_t::rep>::max()};
  // start stop the shortest journey began at (map it to your origin)
  location_idx_t owner_{location_idx_t::invalid()};
  // number of transfers of that journey
  std::uint8_t transfers_{std::numeric_limits<std::uint8_t>::max()};
  // time at the origin of the run that achieves dur_
  unixtime_t departure_{unixtime_t::max()};
};

// n-to-all profile of a stop, two parts:
//
// profile_entry: one entry per run that gives the stop a vehicle journey that
// is not dominated. The person has to wait for the departure of the run.
struct profile_entry {
  unixtime_t departure_;       // time at the origin of the run
  unixtime_t arrival_;         // "ready to continue", same value as min_label
  location_idx_t owner_;       // start stop the journey began at
  std::uint8_t transfers_;
};

// profile_start_label: the stop is a start stop (the origin itself, or a
// footpath/equivalent neighbour of it). The person is already there or walks:
// no vehicle, no waiting, the same duration at every minute.
struct profile_start_label {
  bool reached() const { return owner_ != location_idx_t::invalid(); }

  duration_t dur_{std::numeric_limits<duration_t::rep>::max()};
  location_idx_t owner_{location_idx_t::invalid()};
};


struct raptor_state {
  raptor_state() = default;
  raptor_state(raptor_state const&) = delete;
  raptor_state& operator=(raptor_state const&) = delete;
  raptor_state(raptor_state&&) = default;
  raptor_state& operator=(raptor_state&&) = default;
  ~raptor_state() = default;

  raptor_state& resize(unsigned n_locations,
                       unsigned n_routes,
                       unsigned n_rt_transports);

  template <via_offset_t Vias>
  void print(timetable const& tt, date::sys_days, delta_t invalid);

  template <via_offset_t Vias>
  std::span<std::array<delta_t, Vias + 1>> get_tmp() {
    return {
        reinterpret_cast<std::array<delta_t, Vias + 1>*>(tmp_storage_.data()),
        n_locations_};
  }

  template <via_offset_t Vias>
  std::span<std::array<delta_t, Vias + 1> const> get_tmp() const {
    return {reinterpret_cast<std::array<delta_t, Vias + 1> const*>(
                tmp_storage_.data()),
            n_locations_};
  }

  template <via_offset_t Vias>
  std::span<std::array<delta_t, Vias + 1>> get_best() {
    return {
        reinterpret_cast<std::array<delta_t, Vias + 1>*>(best_storage_.data()),
        n_locations_};
  }

  template <via_offset_t Vias>
  std::span<std::array<delta_t, Vias + 1> const> get_best() const {
    return {reinterpret_cast<std::array<delta_t, Vias + 1> const*>(
                best_storage_.data()),
            n_locations_};
  }

  template <via_offset_t Vias>
  flat_matrix_view<std::array<delta_t, Vias + 1>> get_round_times() {
    return {{reinterpret_cast<std::array<delta_t, Vias + 1>*>(
                 round_times_storage_.data()),
             n_locations_ * (kMaxTransfers + 2)},
            kMaxTransfers + 2U,
            n_locations_};
  }

  template <via_offset_t Vias>
  flat_matrix_view<std::array<delta_t, Vias + 1> const> get_round_times()
      const {
    return {{reinterpret_cast<std::array<delta_t, Vias + 1> const*>(
                 round_times_storage_.data()),
             n_locations_ * (kMaxTransfers + 2)},
            kMaxTransfers + 2U,
            n_locations_};
  }
  template <via_offset_t Vias>
  std::span<std::array<location_idx_t, Vias + 1>> get_owner() {
    return {reinterpret_cast<std::array<location_idx_t, Vias + 1>*>(
                owner_storage_.data()),
            n_locations_};
  }

  template <via_offset_t Vias>
  std::span<std::array<location_idx_t, Vias + 1> const> get_owner() const {
    return {reinterpret_cast<std::array<location_idx_t, Vias + 1> const*>(
                owner_storage_.data()),
            n_locations_};
  }

  template <via_offset_t Vias>
  std::span<std::array<location_idx_t, Vias + 1>> get_tmp_owner() {
    return {reinterpret_cast<std::array<location_idx_t, Vias + 1>*>(
                tmp_owner_storage_.data()),
            n_locations_};
  }

  template <via_offset_t Vias>
  std::span<std::array<location_idx_t, Vias + 1> const> get_tmp_owner() const {
    return {reinterpret_cast<std::array<location_idx_t, Vias + 1> const*>(
                tmp_owner_storage_.data()),
            n_locations_};
  }

  template <via_offset_t Vias>
  flat_matrix_view<std::array<delta_t, Vias + 1> const> get_bounds() const {
    return {{reinterpret_cast<std::array<delta_t, Vias + 1> const*>(
                 bounds_storage_.data()),
             n_locations_ * (kMaxTransfers + 2)},
            kMaxTransfers + 2U,
            n_locations_};
  }
  // you have the const version — also need the mutable one:
  template <via_offset_t Vias>
  flat_matrix_view<std::array<delta_t, Vias + 1>> get_bounds() {
    return {{reinterpret_cast<std::array<delta_t, Vias + 1>*>(
                 bounds_storage_.data()),
             n_locations_ * (kMaxTransfers + 2)},
            kMaxTransfers + 2U,
            n_locations_};
  }
  unsigned n_locations_{};
  bool track_min_{false};
  std::vector<min_label> min_;
  std::vector<delta_t> tmp_storage_;
  std::vector<delta_t> best_storage_;
  std::vector<delta_t> round_times_storage_;
  std::vector<location_idx_t> owner_storage_;      
  std::vector<location_idx_t> tmp_owner_storage_;
  std::vector<delta_t> bounds_storage_;
  bool track_profile_{false};
  std::vector<std::vector<profile_entry>> profile_;
  // start labels: filled if track_profile_ or track_agg_
  std::vector<profile_start_label> profile_start_;
  // aggregates: only recorded together with track_min_
  bool track_agg_{false};
  agg_state agg_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  bitvec rt_transport_mark_;
};

}  // namespace nigiri::routing
