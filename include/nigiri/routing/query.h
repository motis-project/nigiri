#pragma once

#include <cinttypes>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

// Integer value that enables the caller to know
// which transportation mode was used.
using start_type_t = std::int32_t;

struct transport_mode_t {
  using mode_t = std::uint16_t;
  using payload_t = std::uint32_t;

  CISTA_PRINTABLE(transport_mode_t, "mode", "payload")
  friend auto operator<=>(transport_mode_t, transport_mode_t) = default;

  mode_t mode_{};
  payload_t payload_{};
};

struct offset {
  offset(location_idx_t const l,
         duration_t const d,
         transport_mode_t::payload_t const p)
      : target_{l}, duration_{d}, transport_mode_payload_{p} {}

  offset(location_idx_t const l, duration_t const d, transport_mode_t const m)
      : target_{l},
        duration_{d},
        transport_mode_{m.mode_},
        transport_mode_payload_{m.payload_} {}

  location_idx_t target() const noexcept { return target_; }
  duration_t duration() const noexcept { return duration_; }
  transport_mode_t mode() const noexcept {
    return {transport_mode_, transport_mode_payload_};
  }

  friend bool operator<(offset const& a, offset const& b) {
    return a.duration_ < b.duration_;
  }

  friend bool operator==(offset const&, offset const&) = default;

  location_idx_t target_;
  duration_t duration_;
  transport_mode_t::mode_t transport_mode_{};
  transport_mode_t::payload_t transport_mode_payload_{};
};

static_assert(sizeof(offset) == 12U);

struct td_offset {
  // Kept an aggregate so designated initializers still work; this is the
  // counterpart of `offset`'s `transport_mode_t` constructor.
  static constexpr td_offset make(unixtime_t const valid_from,
                                  duration_t const d,
                                  transport_mode_t const m) {
    return {.valid_from_ = valid_from,
            .duration_ = d,
            .transport_mode_ = m.mode_,
            .transport_mode_payload_ = m.payload_};
  }

  friend bool operator==(td_offset const&, td_offset const&) = default;

  duration_t duration() const noexcept { return duration_; }
  transport_mode_t mode() const noexcept {
    return {transport_mode_, transport_mode_payload_};
  }

  unixtime_t valid_from_;
  duration_t duration_;
  transport_mode_t::mode_t transport_mode_{};
  transport_mode_t::payload_t transport_mode_payload_{};
};

static_assert(sizeof(td_offset) == 12U);

struct via_stop {
  friend bool operator==(via_stop const&, via_stop const&) = default;

  location_idx_t location_{};
  duration_t stay_{};
};

using start_time_t = std::variant<unixtime_t, interval<unixtime_t>>;

using td_offsets_t = hash_map<location_idx_t, std::vector<routing::td_offset>>;

struct query {
  void flip_dir();
  void sanitize(timetable const&);
  bool operator==(query const& o) const;

  start_time_t start_time_{};
  location_match_mode start_match_mode_{
      nigiri::routing::location_match_mode::kExact};
  location_match_mode dest_match_mode_{
      nigiri::routing::location_match_mode::kExact};
  bool use_start_footpaths_{false};
  std::vector<offset> start_{}, destination_{};
  td_offsets_t td_start_{}, td_dest_{};
  duration_t max_start_offset_{kMaxTravelTime};
  std::uint8_t max_transfers_{kMaxTransfers};
  duration_t max_travel_time_{kMaxTravelTime};
  unsigned min_connection_count_{0U};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
  std::optional<interval<unixtime_t>> max_interval_{};
  profile_idx_t prf_idx_{0};
  clasz_mask_t allowed_claszes_{all_clasz_allowed()};
  bool require_bike_transport_{false};
  bool require_car_transport_{false};
  bool no_compulsory_reservation_{false};
  transfer_time_settings transfer_time_settings_{};
  std::vector<via_stop> via_stops_{};
  std::optional<duration_t> fastest_direct_{};
  double fastest_direct_factor_{1.0};
  bool slow_direct_{false};
  double fastest_slow_direct_factor_{2.0};
};

}  // namespace nigiri::routing

template <>
struct fmt::formatter<nigiri::routing::transport_mode_t> : ostream_formatter {};
