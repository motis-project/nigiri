#include "nigiri/rt/gtfsrt_resolve_trip.h"

#include "utl/parser/arg_parser.h"

#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/trip_update.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

trip gtfsrt_resolve_trip(timetable const& tt,
                         rt_timetable&,
                         source_idx_t const src,
                         transit_realtime::TripDescriptor const& td) {
  using loader::gtfs::hhmm_to_min;
  using loader::gtfs::parse_date;

  auto const trip_id = td.trip_id();
  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](pair<trip_id_idx_t, trip_idx_t> const& a, string const& b) {
        return std::tuple(tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()) <
               std::tuple(src, std::string_view{b});
      });

  auto const date = td.has_start_date()
                        ? std::make_optional(
                              parse_date(utl::parse<unsigned>(td.start_date())))
                        : std::nullopt;
  auto const time = td.has_start_time()
                        ? std::make_optional(hhmm_to_min(td.start_date()))
                        : std::nullopt;

  auto const id_matches = [&](trip_id_idx_t const t_id_idx) {
    return tt.trip_id_src_[t_id_idx] == src &&
           tt.trip_id_strings_[t_id_idx].view() == trip_id;
  };

  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    for (auto const [t, interval] : tt.trip_transport_ranges_[i->second]) {
      auto const gtfs_static_dep =
          tt.event_mam(t, interval.from_, event_type::kDep).as_duration() +
          tt.transport_first_dep_offset_[t];
      std::cout << gtfs_static_dep << "\n";
      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
      (void)date;
      (void)time;
      (void)id_matches;
      if (traffic_days.test(0U)) {
      }
    }
  }

  return {};
}

}  // namespace nigiri::rt