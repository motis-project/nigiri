#include "nigiri/loader/link_nearby_stations.h"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/constants.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void link_nearby_stations(timetable& tt) {
  constexpr auto const kLinkNearbyMaxDistance = 300;  // [m];

  auto const locations_rtree =
      geo::make_point_rtree(tt.locations_.coordinates_);

  for (auto l_from_idx = location_idx_t{0U};
       l_from_idx != tt.locations_.src_.size(); ++l_from_idx) {
    auto const from_pos = tt.locations_.coordinates_[l_from_idx];
    if (std::abs(from_pos.lat_) < 2.0 && std::abs(from_pos.lng_) < 2.0) {
      continue;
    }

    auto const from_src = tt.locations_.src_[l_from_idx];
    if (from_src == source_idx_t::invalid()) {
      continue;  // no dummy stations
    }

    for (auto const& to_idx :
         locations_rtree.in_radius(from_pos, kLinkNearbyMaxDistance)) {
      auto const l_to_idx = location_idx_t{static_cast<unsigned>(to_idx)};
      if (l_from_idx == l_to_idx) {
        continue;
      }

      auto const to_src = tt.locations_.src_[l_to_idx];
      auto const to_pos = tt.locations_.coordinates_[l_to_idx];
      if (to_src == source_idx_t::invalid() /* no dummy stations */
          || from_src == to_src /* don't short-circuit */) {
        continue;
      }

      auto const from_transfer_time =
          duration_t{tt.locations_.transfer_time_[l_from_idx]};
      auto const to_transfer_time =
          duration_t{tt.locations_.transfer_time_[l_to_idx]};
      auto const walk_duration = duration_t{static_cast<unsigned>(
          std::round(geo::distance(from_pos, to_pos) / (60 * kWalkSpeed)))};
      auto const duration =
          std::max({from_transfer_time, to_transfer_time, walk_duration});

      tt.locations_.preprocessing_footpaths_out_[l_from_idx].emplace_back(
          l_to_idx, duration);
      tt.locations_.preprocessing_footpaths_in_[l_to_idx].emplace_back(
          l_from_idx, duration);
      tt.locations_.equivalences_[l_from_idx].emplace_back(l_to_idx);
    }
  }
}

}  // namespace nigiri::loader