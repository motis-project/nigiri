#include "gtest/gtest.h"

#include "nigiri/loader/compress_stop_times.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;

// T1 + T2 share one pattern, T3 has another one, T4 crosses midnight.
TEST(loader, compress_stop_times) {
  constexpr auto const kTimetable = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
A,Agency,https://example.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
S1,Stop 1,41.0,9.0
S2,Stop 2,41.1,9.1
S3,Stop 3,41.2,9.2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
D,1,1,1,1,1,1,1,20250901,20250907

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R,A,R,Route,3

# trips.txt
route_id,service_id,trip_id
R,D,T1
R,D,T2
R,D,T3
R,D,T4

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,08:00:00,08:00:00,S1,0
T1,08:10:00,08:12:00,S2,1
T1,08:30:00,08:30:00,S3,2
T2,09:00:00,09:00:00,S1,0
T2,09:10:00,09:12:00,S2,1
T2,09:30:00,09:30:00,S3,2
T3,10:00:00,10:00:00,S1,0
T3,10:15:00,10:15:00,S2,1
T3,10:40:00,10:40:00,S3,2
T4,23:50:00,23:50:00,S1,0
T4,24:05:00,24:07:00,S2,1
T4,24:20:00,24:20:00,S3,2
)";

  auto tt = timetable{};
  tt.date_range_ = {sys_days{2025_y / September / 1},
                    sys_days{2025_y / September / 8}};
  register_special_stations(tt);
  gtfs::load_timetable({.default_tz_ = "Etc/UTC"}, source_idx_t{0},
                       mem_dir::read(kTimetable), tt);

  auto const ranges = tt.route_stop_time_ranges_;
  auto const times = tt.route_stop_times_;
  compress_stop_times(tt);
  EXPECT_TRUE(tt.route_stop_times_.empty());
  EXPECT_TRUE(tt.route_stop_time_ranges_.empty());
  ASSERT_EQ(4U, tt.transport_route_.size());

  auto n_patterns = 0U;
  auto has_day_change = false;
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    n_patterns += tt.route_times_[r].n_patterns_;
    auto const transports = tt.route_transport_ranges_[r];
    auto const n = static_cast<unsigned>(transports.size());
    auto const n_stops =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size());
    for (auto s = stop_idx_t{0U}; s != n_stops; ++s) {
      for (auto const ev_type : {event_type::kDep, event_type::kArr}) {
        if ((s == 0U && ev_type == event_type::kArr) ||
            (s == n_stops - 1U && ev_type == event_type::kDep)) {
          continue;
        }
        auto const raw = &times[ranges[r].from_ + event_idx(s, ev_type) * n];
        auto const evs = tt.event_times_at_stop(r, s, ev_type);
        ASSERT_EQ(n, evs.size());
        for (auto i = 0U; i != n; ++i) {
          auto const t = transports.from_ + i;
          EXPECT_EQ(raw[i], evs[i]);
          EXPECT_EQ(raw[i], tt.event_mam(t, s, ev_type));
          has_day_change |= raw[i].days() > tt.transport_first_dep_[t].days();
        }

        auto i = 0U;
        for (auto it = evs.begin(); it != evs.end(); ++it, ++i) {
          EXPECT_EQ(raw[i], *it);
        }
      }
    }
  }
  EXPECT_EQ(3U, n_patterns);
  EXPECT_TRUE(has_day_change);
}
