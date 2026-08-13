#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/raptor/pong.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

timetable get_tt() {
  static auto const files = mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20260601,20260607

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,05:00:00,05:00:00,S0,0,0,0
R0_MON,06:00:00,06:00:00,S1,1,0,0
R1_MON,08:00:00,08:00:00,S1,0,0,0
R1_MON,10:00:00,10:00:00,S2,1,0,0
)");

  timetable tt;
  tt.date_range_ = {sys_days{2026_y / June / 01}, sys_days{2026_y / June / 07}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, files, tt);
  finalize(tt);
  return tt;
}

TEST(routing, pong_respect_interval) {
  timetable tt = get_tt();

  auto const S0 = tt.find(location_id{"S0", source_idx_t{0}}).value();
  auto const S2 = tt.find(location_id{"S2", source_idx_t{0}}).value();

  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{sys_days{2026_y / June / 01},
                                          sys_days{2026_y / June / 02}},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = false,
      .start_ = {{S0, 0min, 0U}},
      .destination_ = {{S2, 0min, 0U}},
      .min_connection_count_ = 1U,
      .extend_interval_later_ = true
  };

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};
  auto const result =
      routing::pong_search(tt, nullptr, search_state, raptor_state,
                           std::move(q), direction::kForward);

  ASSERT_EQ(1U, result.journeys_->size());

  std::cout << to_string(tt, nullptr, *result.journeys_);
}

TEST(routing, pong_min_lookahead) {
  timetable tt = get_tt();

  auto const S0 = tt.find(location_id{"S0", source_idx_t{0}}).value();
  auto const S2 = tt.find(location_id{"S2", source_idx_t{0}}).value();

  auto q0 = routing::query{
      .start_time_ = unixtime_t{sys_days{2026_y / June / 01}},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = false,
      .start_ = {{S0, 0min, 0U}},
      .destination_ = {{S2, 0min, 0U}},
      .max_travel_time_ = 5h,
      .min_connection_count_ = 1U,
      .extend_interval_later_ = true,
  };

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};
  auto const result0 =
      routing::pong_search(tt, nullptr, search_state, raptor_state,
                           std::move(q0), direction::kForward);
  ASSERT_EQ(1U, result0.journeys_->size());
  auto const expected = R"(
[2026-06-01 04:00, 2026-06-01 09:00]
TRANSFERS: 1
     FROM: (S0, S0) [2026-06-01 04:00]
       TO: (S2, S2) [2026-06-01 09:00]
leg 0: (S0, S0) [2026-06-01 04:00] -> (S1, S1) [2026-06-01 05:00]
   0: S0      S0..............................................                               d: 01.06 04:00 [01.06 05:00]  [{name=R0, day=2026-06-01, id=R0_MON, src=0}]
   1: S1      S1.............................................. a: 01.06 05:00 [01.06 06:00]
leg 1: (S1, S1) [2026-06-01 05:00] -> (S1, S1) [2026-06-01 05:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2026-06-01 07:00] -> (S2, S2) [2026-06-01 09:00]
   0: S1      S1..............................................                               d: 01.06 07:00 [01.06 08:00]  [{name=R1, day=2026-06-01, id=R1_MON, src=0}]
   1: S2      S2.............................................. a: 01.06 09:00 [01.06 10:00]

)";
  EXPECT_EQ(expected, to_string(tt, nullptr, *result0.journeys_));

  auto q1 = routing::query{
      .start_time_ = unixtime_t{sys_days{2026_y / June / 01}},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = false,
      .start_ = {{S0, 0min, 0U}},
      .destination_ = {{S2, 0min, 0U}},
      .max_travel_time_ = 4h + 59min,
      .min_connection_count_ = 1U,
      .extend_interval_later_ = true,
  };
  auto const result1 =
      routing::pong_search(tt, nullptr, search_state, raptor_state,
                           std::move(q1), direction::kForward);
  EXPECT_EQ(0U, result1.journeys_->size());
}
} // namespace