#include "nigiri/routing/mc_context.h"

#include <optional>
#include <variant>
#include <vector>

#include "utl/overloaded.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace {

// the location's timezone, its parent's, else the agency's of a trip there or
// at a child
timezone_idx_t tz_of(timetable const& tt, location_idx_t const l) {
  auto const& locs = tt.locations_;
  for (auto const x : {l, locs.parents_[l]}) {
    if (x != location_idx_t::invalid() &&
        locs.location_timezones_[x] != timezone_idx_t::invalid()) {
      return locs.location_timezones_[x];
    }
  }
  auto const of_trips = [&](location_idx_t const x) {
    for (auto const r : tt.location_routes_[x]) {
      for (auto const p :
           tt.transport_section_providers_[tt.route_transport_ranges_[r]
                                               .from_]) {
        if (tt.providers_[p].tz_ != timezone_idx_t::invalid()) {
          return tt.providers_[p].tz_;
        }
      }
    }
    return timezone_idx_t::invalid();
  };
  if (auto const tz = of_trips(l); tz != timezone_idx_t::invalid()) {
    return tz;
  }
  for (auto const c : locs.children_[l]) {
    if (auto const tz = of_trips(c); tz != timezone_idx_t::invalid()) {
      return tz;
    }
  }
  return timezone_idx_t::invalid();
}

}  // namespace

std::int16_t night_offset(timetable const& tt, query const& q) {
  auto const t = std::visit(
      utl::overloaded{[](interval<unixtime_t> const i) { return i.from_; },
                      [](unixtime_t const x) { return x; }},
      q.start_time_);
  auto const offset_of = [&](std::vector<offset> const& v) {
    auto const tz = v.empty() || to_idx(v.front().target()) >= tt.n_locations()
                        ? timezone_idx_t::invalid()
                        : tz_of(tt, v.front().target());
    return tz == timezone_idx_t::invalid()
               ? std::nullopt
               : std::optional{(to_local_time(tt.timezones_[tz], t) -
                                local_time{t.time_since_epoch()})
                                   .count()};
  };
  auto const a = offset_of(q.start_);
  auto const b = offset_of(q.destination_);
  return static_cast<std::int16_t>(a && b ? (*a + *b) / 2
                                          : a.value_or(b.value_or(0)));
}

}  // namespace nigiri::routing
