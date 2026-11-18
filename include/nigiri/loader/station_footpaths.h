#pragma once

#include <array>
#include <optional>
#include <vector>

#include "nigiri/loader/station_graph.h"
#include "nigiri/footpath.h"
#include "nigiri/footpath_restriction.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

// How a footpath profile walks through station graphs. walk_factor_ scales
// the walking part, forbidden_ removes routes at import (the query settings
// can forbid more).
struct station_profile {
  double walk_factor_;
  std::uint8_t forbidden_;
};

inline constexpr auto const kStationProfiles =
    std::array<std::optional<station_profile>, kNProfiles>{
        station_profile{.walk_factor_ = 1.0, .forbidden_ = 0U},  // default
        station_profile{.walk_factor_ = 1.0, .forbidden_ = 0U},  // foot
        station_profile{// wheelchair
                        .walk_factor_ = 1.5,
                        .forbidden_ = footpath_restriction::kStairs |
                                      footpath_restriction::kEscalator |
                                      footpath_restriction::kStep |
                                      footpath_restriction::kBanWheelchair},
        std::nullopt,  // car
        std::nullopt};  // bike

struct station_footpath {
  // Minutes of the walking part: scaled by the query's transfer time factor.
  duration_t walk_part() const {
    auto const fixed = duration_t{(restriction_.fixed_seconds() + 59U) / 60U};
    return duration_ - std::min(fixed, duration_);
  }

  duration_t fixed_part() const { return duration_ - walk_part(); }

  // At least as fast and as unrestricted for every query setting.
  bool dominates(station_footpath const& o) const {
    return walk_part() <= o.walk_part() && fixed_part() <= o.fixed_part() &&
           is_less_restricted(restriction_, o.restriction_);
  }

  location_idx_t target_;
  duration_t duration_;
  footpath_restriction restriction_;
};

// Per profile and stop: the footpaths through station graphs, sorted by
// target, then duration. Profiles without station_profile stay empty.
using station_footpaths_t =
    std::array<vector_map<location_idx_t, std::vector<station_footpath>>,
               kNProfiles>;

station_footpaths_t get_station_footpaths(timetable const&,
                                          std::vector<station_route> const&);

// Stop pairs whose walk an unqualified transfer rule set (or forbade).
using rule_pairs_t = hash_set<pair<location_idx_t, location_idx_t>>;

// Sorts by target, then duration and removes dominated footpaths.
void remove_dominated(std::vector<station_footpath>&);

}  // namespace nigiri::loader
