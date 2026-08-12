#pragma once

#include <array>
#include <span>
#include <vector>

#include "date/date.h"

#include "cista/containers/bitvec.h"
#include "cista/containers/flat_matrix.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/flat_matrix_view.h"
#include "nigiri/routing/limits.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

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
  flat_matrix_view<std::array<delta_t, Vias + 1>> get_bounds() {
    return {{reinterpret_cast<std::array<delta_t, Vias + 1>*>(
                 bounds_storage_.data()),
             n_locations_ * (kMaxTransfers + 2)},
            kMaxTransfers + 2U,
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

  unsigned n_locations_{};
  std::vector<delta_t> tmp_storage_;
  std::vector<delta_t> best_storage_;
  std::vector<delta_t> round_times_storage_;
  std::vector<delta_t> bounds_storage_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  bitvec rt_transport_mark_;

  // implicit transfer-group aggregation (see raptor.h update_footpaths):
  // dense per-group-station slots instead of a hash map -- the GPU-shaped
  // layout. group_rank_[station] = dense index (built lazily from the
  // timetable on first use), slots addressed as
  // (rank * 2 + hub) * (kMaxVias + 1) + via. epoch stamps make per-round
  // clearing O(touched) instead of O(all).
  std::vector<std::uint32_t> group_rank_;
  std::uint32_t n_group_stations_{0U};
  std::vector<int> group_slots_;
  std::vector<std::uint32_t> group_slot_stamp_;
  std::vector<std::uint32_t> group_rank_stamp_;
  std::vector<std::uint32_t> group_touched_;
  std::uint32_t group_epoch_{0U};
};

}  // namespace nigiri::routing
