#include "nigiri/lookup/get_transport_stop_tz.h"

#include "nigiri/timetable.h"

namespace nigiri {

timezone_idx_t get_transport_stop_tz(timetable const& tt,
                                     transport_idx_t const t,
                                     location_idx_t const l) {
  auto const location_tz = tt.locations_.location_timezones_.at(l);
  if (location_tz != timezone_idx_t::invalid()) {
    return location_tz;
  }
  return tt.get_provider(t).tz_;
}

}  // namespace nigiri