#include "nigiri/location.h"

#include "nigiri/timetable.h"

namespace nigiri {

std::ostream& operator<<(std::ostream& out, location const& l) {
  return out << '(' << l.name_ << ", " << l.id_ << ')';
}

location::location(timetable const& tt, location_idx_t idx)
    : l_{idx},
      id_{tt.locations_.ids_[idx].view()},
      name_{tt.locations_.names_[idx].view()},
      pos_{tt.locations_.coordinates_[idx]},
      src_{tt.locations_.src_[idx]},
      type_{tt.locations_.types_[idx]},
      parent_{tt.locations_.parents_[idx]},
      timezone_idx_{tt.locations_.location_timezones_[idx]},
      transfer_time_{tt.locations_.transfer_time_[idx]},
      equivalences_{tt.locations_.equivalences_[idx]} {}



location::location(
    std::string_view id,
    std::string_view name,
    geo::latlng pos,
    source_idx_t src,
    location_type type,
    location_idx_t parent,
    timezone_idx_t timezone,
    duration_t transfer_time,
    it_range<vector<location_idx_t>::const_iterator> equivalences)
    : l_{location_idx_t::invalid()},
      id_{id},
      name_{name},
      pos_{pos},
      src_{src},
      type_{type},
      parent_{parent},
      timezone_idx_{timezone},
      transfer_time_{transfer_time},
      equivalences_{equivalences} {}

}  // namespace nigiri
