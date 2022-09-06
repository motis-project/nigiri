#include "nigiri/location.h"

#include "nigiri/timetable.h"

namespace nigiri {

std::ostream& operator<<(std::ostream& out, location const& l) {
  return out << '(' << l.name_ << ", " << l.id_ << ')';
}

location::location(timetable const& tt, location_idx_t idx)
    : id_{tt.locations_.ids_[idx]},
      name_{tt.locations_.names_[idx]},
      pos_{tt.locations_.coordinates_[idx]},
      src_{tt.locations_.src_[idx]},
      type_{tt.locations_.types_[idx]},
      osm_id_{tt.locations_.osm_ids_[idx]},
      parent_{tt.locations_.parents_[idx]},
      timezone_idx_{tt.locations_.location_timezones_[idx]},
      equivalences_{tt.locations_.equivalences_[idx]},
      footpaths_out_{tt.locations_.footpaths_out_[idx]},
      footpaths_in_{tt.locations_.footpaths_in_[idx]} {}

location::location(
    string const& id,
    string const& name,
    geo::latlng pos,
    source_idx_t src,
    location_type type,
    osm_node_id_t osm_id,
    location_idx_t parent,
    timezone_idx_t timezone,
    it_range<vector<location_idx_t>::const_iterator> equivalences,
    it_range<vector<footpath>::const_iterator> footpaths_in,
    it_range<vector<footpath>::const_iterator> footpaths_out)
    : id_{id},
      name_{name},
      pos_{pos},
      src_{src},
      type_{type},
      osm_id_{osm_id},
      parent_{parent},
      timezone_idx_{timezone},
      equivalences_{std::move(equivalences)},
      footpaths_out_{std::move(footpaths_out)},
      footpaths_in_{std::move(footpaths_in)} {}

}  // namespace nigiri