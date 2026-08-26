#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/resolve.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace std::chrono_literals;

namespace nigiri::loader::gtfs {

// Within a route, (traffic day, transport offset) lexicographic order must
// equal absolute event time order — GPU raptor's earliest-transport merge
// and any (day, offset)-keyed comparison rely on it. Equivalent invariant:
// at every stop, full event times (days * 1440 + mam) are non-decreasing in
// transport offset order and the first-to-last spread stays below one day.
// Routes mixing day-offset events (a trip crossing >24:00 relative to its
// service day) with plain trips of the next service day used to violate
// this (EU query Marseille->Ajaccio: the 42:06 arrival lost against the
// next day's earlier trip in (day, offset) order).
void expect_route_day_order(timetable const& tt) {
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const n_stops =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size());
    for (auto const ev_type : {event_type::kDep, event_type::kArr}) {
      auto const from = static_cast<stop_idx_t>(ev_type == event_type::kArr);
      auto const to = static_cast<stop_idx_t>(
          n_stops - (ev_type == event_type::kDep ? 1U : 0U));
      for (auto s = from; s != to; ++s) {
        auto const evs = tt.event_times_at_stop(r, s, ev_type);
        auto const full = [&](delta const d) {
          return d.days() * 1440 + d.mam();
        };
        for (auto i = 1U; i < evs.size(); ++i) {
          EXPECT_LE(full(evs[i - 1U]), full(evs[i]))
              << "route " << r << " stop " << s << " transport " << i;
        }
        if (!evs.empty()) {
          EXPECT_LT(full(evs.back()) - full(evs.front()), 1440)
              << "route " << r << " stop " << s;
        }
      }
    }
  }
}

// An overnight trip crossing 24:00 relative to its service day (day-offset
// arrival event, like the Toulon->Bastia ferry notated up to 42:06)
// coexists in one route with a plain trip of the next service day that is
// earlier in absolute time but later in (day, offset) order.
TEST(gtfs, route_day_order_day_offset_pair) {
  constexpr auto const kTimetable = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
A,Agency,https://example.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
S1,Stop 1,41.0,9.0
S2,Stop 2,41.1,9.1

# calendar_dates.txt
service_id,date,exception_type
MON,20250901,1
TUE,20250902,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R,A,R,Route,3

# trips.txt
route_id,service_id,trip_id
R,MON,T_OVERNIGHT
R,TUE,T_PLAIN

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_OVERNIGHT,17:50:00,17:50:00,S1,0
T_OVERNIGHT,42:06:00,42:06:00,S2,1
T_PLAIN,17:06:00,17:06:00,S1,0
T_PLAIN,17:16:00,17:16:00,S2,1
)";

  auto const tt = loader::load(
      {{.tag_ = "test",
        .path_ = kTimetable,
        .loader_config_ = {.default_tz_ = "Etc/UTC"}}},
      {}, {sys_days{2025_y / September / 1}, sys_days{2025_y / September / 8}});

  expect_route_day_order(tt);

  // splitting must not lose either trip
  EXPECT_EQ(2U, resolve(tt, nullptr, "T_OVERNIGHT", "20250901").size());
  EXPECT_EQ(2U, resolve(tt, nullptr, "T_PLAIN", "20250902").size());
}

// A trip crossing midnight (day-offset arrival, mam small) and a late
// same-day trip: time-of-day order is consistent, but the midnight-crosser
// arrives later in absolute time than its (day, offset)-successor — the
// full-time ORDER condition must reject (the spread here is tiny, so the
// spread condition alone would accept).
TEST(gtfs, route_day_order_midnight_cross) {
  constexpr auto const kTimetable = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
A,Agency,https://example.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
S1,Stop 1,41.0,9.0
S2,Stop 2,41.1,9.1

# calendar_dates.txt
service_id,date,exception_type
MON,20250901,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R,A,R,Route,3

# trips.txt
route_id,service_id,trip_id
R,MON,T_CROSS
R,MON,T_LATE

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_CROSS,23:00:00,23:00:00,S1,0
T_CROSS,24:30:00,24:30:00,S2,1
T_LATE,23:30:00,23:30:00,S1,0
T_LATE,23:59:00,23:59:00,S2,1
)";

  auto const tt = loader::load(
      {{.tag_ = "test",
        .path_ = kTimetable,
        .loader_config_ = {.default_tz_ = "Etc/UTC"}}},
      {}, {sys_days{2025_y / September / 1}, sys_days{2025_y / September / 8}});

  expect_route_day_order(tt);

  EXPECT_EQ(2U, resolve(tt, nullptr, "T_CROSS", "20250901").size());
  EXPECT_EQ(2U, resolve(tt, nullptr, "T_LATE", "20250901").size());
}

}  // namespace nigiri::loader::gtfs
