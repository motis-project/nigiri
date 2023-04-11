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

  auto const provider_idx = tt.transport_section_providers_[t].front();
  utl::verify(provider_idx != provider_idx_t::invalid(),
              "provider of transport {} not set, no timezone at stop {}", t,
              location{tt, l});

  return tt.providers_[provider_idx].tz_;
}

}  // namespace nigiri