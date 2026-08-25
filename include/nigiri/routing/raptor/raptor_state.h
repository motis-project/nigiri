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

  // copy-on-diverge (combined scheduled+rt search): the flat storage carved
  // into two single-slot planes; plane 0 = scheduled base (identical to the
  // plain Vias=0 layout), plane 1 = rt overlay, valid only where diverged
  std::span<delta_t> get_tmp_plane(std::size_t const p) {
    return {tmp_storage_.data() + p * n_locations_, n_locations_};
  }

  std::span<delta_t> get_best_plane(std::size_t const p) {
    return {best_storage_.data() + p * n_locations_, n_locations_};
  }

  flat_matrix_view<std::array<delta_t, 1>> get_round_times_plane(
      std::size_t const p) {
    auto const n = n_locations_ * (kMaxTransfers + 2U);
    return {{reinterpret_cast<std::array<delta_t, 1>*>(
                 round_times_storage_.data() + p * n),
             n},
            kMaxTransfers + 2U,
            n_locations_};
  }

  flat_matrix_view<std::array<delta_t, 1> const> get_round_times_plane(
      std::size_t const p) const {
    auto const n = n_locations_ * (kMaxTransfers + 2U);
    return {{reinterpret_cast<std::array<delta_t, 1> const*>(
                 round_times_storage_.data() + p * n),
             n},
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
  // locations written since the last reset_arrivals: rows outside this set
  // are guaranteed kInvalid, so resets and the per-round best-merge only
  // have to visit these rows
  bitvec touched_;
  // copy-on-diverge: rt-plane cell (k * n_locations + l) differs from the
  // scheduled base / rt best_+tmp_ at l differ from the base; the lists
  // mirror the set bits so clearing costs O(diverged), not O(n)
  bitvec diverged_;
  std::vector<std::uint64_t> diverged_list_;
  bitvec diverged_best_;
  std::vector<std::uint32_t> diverged_best_list_;
  // union of both divergence kinds per location, never cleared within a
  // query: one probe decides "fully clean stop" on the fast-lane gates
  bitvec diverged_any_;
  std::vector<std::uint32_t> diverged_any_list_;
  // copy-on-diverge: routes with a transport whose traffic days were
  // re-pointed by rt updates; all other routes scan identically in both
  // worlds (clean-route fast lane); cached per rt_timetable instance
  bitvec route_has_rt_;
  void const* route_has_rt_src_{nullptr};
  std::size_t route_has_rt_version_{0U};
  // fresh storage needs one full reset before sparse resets are sound
  bool needs_full_reset_{true};
  // slot count + direction of the last reset: a different slot width
  // reinterprets the flat storage and a different direction flips kInvalid,
  // either invalidates the sparse-reset invariant
  std::uint8_t reset_slots_{0U};
  bool reset_fwd_{true};
};

}  // namespace nigiri::routing
